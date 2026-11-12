"""Prepare recording-isolated dry/wet segments and labels for CNN training."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .audio import AudioConfig, read_audio, segment
from .chain import (catalog, chain_instance, checked_render, continuous, denormalize,
                     discrete, fingerprint, normalize)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def split_recordings(ids, seed=42, validation=0.15, test=0.15):
    unique = sorted(set(ids))
    if len(unique) < 3 or validation <= 0 or test <= 0 or validation + test >= 1:
        raise ValueError('Need >=3 original recordings and positive train/validation/test fractions')
    np.random.default_rng(seed).shuffle(unique)
    nv, nt = max(1, round(len(unique) * validation)), max(1, round(len(unique) * test))
    if nv + nt >= len(unique):
        raise ValueError('Not enough recordings for requested split fractions')
    return {rid: ('validation' if i < nv else 'test' if i < nv + nt else 'train')
            for i, rid in enumerate(unique)}


def load_manifest(path):
    import soundfile as sf

    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    metadata = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    if metadata['format_version'] != 1 or fingerprint(metadata['catalog']) != metadata['catalog_hash']:
        raise ValueError('Invalid manifest version or catalog fingerprint')
    groups, hash_splits, seen = {}, {}, set()
    definitions = metadata['catalog']
    if fingerprint(catalog()) != metadata['catalog_hash']:
        raise ValueError('Current renderer schema differs from prepared dataset')
    config = AudioConfig(**metadata['audio_config'])
    allowed = set(metadata.get('classes', ['bypass', *definitions]))
    if not rows:
        raise ValueError('Empty manifest')
    for row in rows:
        rid, split = row['recording_id'], row['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError(f'Unknown split: {split}')
        if groups.setdefault(rid, split) != split:
            raise ValueError(f'Original recording leaks across splits: {rid}')
        if metadata['splits'].get(rid) != split:
            raise ValueError('Manifest split differs from source split plan')
        source_hash = row['source_sha256']
        if hash_splits.setdefault(source_hash, split) != split:
            raise ValueError('Identical source file appears in different splits under different IDs')
        if row['sample_id'] in seen:
            raise ValueError('Duplicate sample ID')
        seen.add(row['sample_id'])
        items = row['ground_truth_chain']['chain']
        name = items[0]['effect_type'] if items else 'bypass'
        if name not in allowed:
            raise ValueError('Sample class is outside the declared dataset task')
        if len(items) > 1 or (items and items[0]['effect_type'] not in definitions):
            raise ValueError('Only zero or one supported drive is allowed')
        if items:
            definition = definitions[items[0]['effect_type']]
            values = items[0]['values']
            if set(values) != {p['name'] for p in definition['params']}:
                raise ValueError('Incomplete or extra ground-truth knobs')
            for p in continuous(definition):
                normalize(values[p['name']], p)
            for p in discrete(definition):
                options = [False, True] if p['type'] == 'boolean' else p['options']
                if values[p['name']] not in options:
                    raise ValueError('Invalid discrete knob')
            active_names = {p['name'] for p in definition['active_params']}
            if any(values[p['name']] != p['default'] for p in definition['params'] if p['name'] not in active_names):
                raise ValueError('Unrendered knobs must stay fixed at schema defaults')
        for key in ('dry_path', 'wet_path'):
            row[key] = str((path.parent / row[key]).resolve())
            info = sf.info(row[key])
            if info.samplerate != config.sample_rate or info.frames != config.samples or info.channels != 1:
                raise ValueError(f'Prepared segment format mismatch: {row[key]}')
    return rows, metadata


def pair_audio(row, config):
    # Prepared segments already encode crop start, padding and render alignment.
    return (segment(read_audio(row['dry_path'], config.sample_rate), config),
            segment(read_audio(row['wet_path'], config.sample_rate), config, reject_silence=False))


def prepare(sources, output, config=AudioConfig(), variations=3, seed=42,
            validation=0.15, test=0.15, legacy_labels=None, effect_types=None,
            include_bypass=True, task='joint'):
    import soundfile as sf

    if variations < 1:
        raise ValueError('variations must be positive')
    output = Path(output).resolve()
    manifest = output / 'manifest.jsonl'
    if manifest.exists():
        raise FileExistsError(f'Refusing to overwrite prepared dataset: {manifest}')
    definitions = catalog()
    effect_types = list(definitions) if effect_types is None else list(effect_types)
    if not effect_types or len(set(effect_types)) != len(effect_types) or set(effect_types) - set(definitions):
        raise ValueError('Expected unique supported effect types')
    if task not in ('joint', 'classification', 'regression'):
        raise ValueError('Unknown dataset task')
    if task == 'regression' and (include_bypass or len(effect_types) != 1):
        raise ValueError('Regression requires one known effect without bypass')
    splits = split_recordings([s['recording_id'] for s in sources], seed, validation, test)
    legacy = {}
    for path in sorted(Path(legacy_labels).glob('*.json')) if legacy_labels else []:
        label = json.loads(path.read_text(encoding='utf-8'))
        items = label['ground_truth_chain']['chain']
        if len(items) != 1 or items[0]['effect_type'] not in definitions:
            raise ValueError(f'Legacy label outside drive-only scope: {path}')
        legacy.setdefault(label['dry_sample_id'], []).append((items[0], str(path.resolve())))
    rng, rows, rejected = np.random.default_rng(seed), [], []
    output.mkdir(parents=True, exist_ok=True)
    for source_index, source in enumerate(sources):
        source_row_start = len(rows)
        try:
            source_hash = hashlib.sha256(Path(source['file_path']).read_bytes()).hexdigest()
            raw = read_audio(source['file_path'], config.sample_rate)
            for crop_index, start_sec in enumerate(source.get('starts_sec', [0.0])):
                if start_sec < 0:
                    raise ValueError('Negative crop start')
                start = round(start_sec * config.sample_rate)
                dry = segment(raw, config, start)
                prefix = f'{source_index:05d}_{crop_index:03d}'
                sf.write(output / f'{prefix}_dry.wav', dry, config.sample_rate, subtype='FLOAT')
                settings = [('bypass', {}, None)] if include_bypass else []
                if legacy_labels:
                    matches = legacy.get(source.get('sample_id', Path(source['file_path']).stem), [])
                    if not matches:
                        raise ValueError('No legacy labels matched source sample_id')
                    settings += [(item['effect_type'], item['values'], path) for item, path in matches
                                 if item['effect_type'] in effect_types]
                else:
                    for name, definition in definitions.items():
                        if name not in effect_types:
                            continue
                        for _ in range(variations):
                            values = {p['name']: p['default'] for p in definition['params']}
                            values.update({p['name']: denormalize(rng.uniform(), p) for p in continuous(definition)})
                            for p in discrete(definition):
                                options = [False, True] if p['type'] == 'boolean' else p['options']
                                values[p['name']] = options[int(rng.integers(len(options)))]
                            settings.append((name, values, None))
                for variant, (name, values, legacy_path) in enumerate(settings):
                    chain = chain_instance(name, values, definitions)
                    wet = checked_render(dry, config.sample_rate, chain)
                    sample_id = f'{prefix}_{variant:04d}'
                    wet_name = f'{sample_id}_wet.wav'
                    sf.write(output / wet_name, wet, config.sample_rate, subtype='FLOAT')
                    rows.append({'sample_id': sample_id, 'recording_id': source['recording_id'],
                                 'split': splits[source['recording_id']],
                                 'source_path': str(Path(source['file_path']).resolve()),
                                 'source_sha256': source_hash,
                                 'start_sample': start, 'valid_samples': min(config.samples, len(raw) - start),
                                 'dry_path': f'{prefix}_dry.wav', 'wet_path': wet_name,
                                 'ground_truth_chain': chain, 'legacy_label_path': legacy_path})
        except (ValueError, OSError, RuntimeError) as exc:
            del rows[source_row_start:]
            rejected.append({'source': source, 'error': str(exc)})
    metadata = {'format_version': 1, 'audio_config': config.to_dict(), 'catalog': definitions,
                'task': task, 'effect_types': effect_types, 'include_bypass': include_bypass,
                'classes': (['bypass'] if include_bypass else []) + effect_types,
                'catalog_hash': fingerprint(definitions), 'seed': seed, 'splits': splits,
                'source_index': sources, 'variations': variations,
                'versions': {p: importlib.metadata.version(p) for p in
                             ('numpy', 'librosa', 'soundfile', 'pedalboard')},
                'renderer': {'module': 'effects.render.render_chain', 'pedalboard': importlib.metadata.version('pedalboard'),
                             'amp_id': 'fixed_amp_v1', 'amp_rendered': False,
                             'compressor': None, 'eq': None, 'fresh_board_per_render': True,
                             'order': 'mono -> resample -> segment -> zero pad -> drive -> FLOAT WAV',
                             'normalization': 'none', 'augmentation': 'none'},
                'rejected': rejected}
    write_json(manifest.with_suffix('.meta.json'), metadata)
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
    load_manifest(manifest)
    present = {r['split'] for r in rows}
    if present != {'train', 'validation', 'test'}:
        raise ValueError(f'Usable sources missing splits: {present}; inspect rejected sources')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dry-dir', type=Path)
    source.add_argument('--source-index', type=Path, help='JSON list: recording_id, file_path, starts_sec; paths relative to index')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--audio-config', type=Path)
    p.add_argument('--legacy-labels', type=Path, help='Reuse existing label settings; regenerate aligned cropped wet')
    p.add_argument('--variations', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation', type=float, default=0.15)
    p.add_argument('--test', type=float, default=0.15)
    p.add_argument('--effects', nargs='+', choices=list(catalog()))
    p.add_argument('--no-bypass', action='store_true')
    p.add_argument('--task', choices=['joint', 'classification', 'regression'], default='joint')
    args = p.parse_args()
    if args.source_index:
        sources = json.loads(args.source_index.read_text(encoding='utf-8'))
        for s in sources:
            s['file_path'] = str((args.source_index.resolve().parent / s['file_path']).resolve())
    else:
        paths = sorted(path for path in args.dry_dir.rglob('*') if path.suffix.lower() in ('.wav', '.flac'))
        sources = [{'recording_id': path.relative_to(args.dry_dir).with_suffix('').as_posix(),
                    'file_path': str(path.resolve())} for path in paths]
    config = AudioConfig(**json.loads(args.audio_config.read_text())) if args.audio_config else AudioConfig()
    print(prepare(sources, args.output, config, args.variations, args.seed,
                  args.validation, args.test, args.legacy_labels, args.effects,
                  not args.no_bypass, args.task))


if __name__ == '__main__':
    main()
