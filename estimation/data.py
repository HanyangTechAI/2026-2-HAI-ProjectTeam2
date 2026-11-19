"""Import existing VST WAVs; never regenerate, renormalize or alter shared data."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audio import AudioConfig, log_mel, read_audio, segment, validate_rendered_pair
from .chain import DRIVES, catalog, continuous, fingerprint, normalize
from .distance import calibrate, raw_distances


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def split_recordings(ids, seed=42):
    unique = sorted(set(ids))
    if len(unique) < 3:
        raise ValueError('At least three independent dry originals are required')
    np.random.default_rng(seed).shuffle(unique)
    n = max(1, round(len(unique) * .15))
    if 2 * n >= len(unique):
        raise ValueError('Not enough originals for train/validation/test')
    return {rid: 'validation' if i < n else 'test' if i < 2*n else 'train'
            for i, rid in enumerate(unique)}


def validate_chain(chain, definitions):
    items = chain['chain']
    drives = [p['effect_type'] for p in items if p['effect_type'] in DRIVES]
    if len(drives) != 1:
        raise ValueError('Exactly one of the seven VST drives is required')
    if len(items) != 1 or items[0]['effect_type'] != drives[0]:
        raise ValueError('Drive-only dataset required: compressor/EQ/multiple stages are forbidden')
    for item in items:
        specs = continuous(definitions[item['effect_type']])
        if set(item['values']) != {p['name'] for p in specs}:
            raise ValueError(f"Schema/label knob mismatch: {item['effect_type']}")
        for p in specs:
            normalize(item['values'][p['name']], p)
    return drives[0]


def prepare(dataset, output, seed=42, config=None):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    dataset, output = Path(dataset).absolute(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(f'Use a new cache directory: {output}')
    manifest_path = dataset / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    audit_path = dataset/'render_audit.json'
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest() if audit_path.exists() else None
    if set(manifest.get('drives', [])) != set(DRIVES) or manifest.get('dry_peak_db') != -12:
        raise ValueError('Expected canonical VST7 dataset with -12 dBFS generated dry')
    definitions = catalog()
    mappings = manifest['dry_id_map']
    dry_map = {r['dry_sample_id']: r for r in mappings}
    if len(dry_map) != len(mappings):
        raise ValueError('Duplicate dry IDs')
    # Namespace + source-relative path groups all variants of the same recording.
    groups = {k: f"{v['namespace']}/{v['relative_path']}" for k, v in dry_map.items()}
    splits = split_recordings(groups.values(), seed)
    label_files = sorted((dataset / 'labels').glob('*.json'))
    wet_paths = {p.stem: p for p in (dataset / 'audio').glob('*.wav')}
    if (len(label_files) != manifest['completed_pairs'] or
            {p.stem for p in label_files} != set(wet_paths)):
        raise ValueError('Incomplete dataset: manifest, label and WAV counts disagree')
    records = []
    print(json.dumps({'loading_labels':0,'total':len(label_files)}),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded_labels = list(pool.map(lambda p: json.loads(p.read_text(encoding='utf-8')),label_files))
    for path,label in zip(label_files,loaded_labels):
        if path.stem != label['target_id'] or label['dry_sample_id'] not in dry_map:
            raise ValueError(f'Invalid target/dry identity: {path.name}')
        drive = validate_chain(label['ground_truth_chain'], definitions)
        rid = groups[label['dry_sample_id']]
        records.append({'sample_id': label['target_id'], 'dry_sample_id': label['dry_sample_id'],
                        'recording_id': rid, 'split': splits[rid], 'drive': drive,
                        'wet_path': str(wet_paths[path.stem]),
                        'ground_truth_chain': label['ground_truth_chain']})
    config = config or AudioConfig()
    print(json.dumps({'loading_labels':len(records),'preparing_generated_dry':len(dry_map)}),flush=True)
    output.mkdir(parents=True)
    (output/'native_dry').mkdir()
    dry_ids, dry_indices = sorted(dry_map), {}
    mel_shape = log_mel(np.ones(config.samples, dtype=np.float32) * .01, config).shape
    # Librosa lazily loads SciPy BLAS during the first feature extraction.
    threadpool_limits(limits=1, user_api='blas')
    dry_features = np.lib.format.open_memmap(output/'dry_features.npy', mode='w+',
                                             dtype='float32', shape=(len(dry_ids), *mel_shape))
    dry_audio = np.lib.format.open_memmap(output/'dry_audio.npy', mode='w+', dtype='float32',
                                          shape=(len(dry_ids), config.samples))
    hash_splits, hashes = {}, {}
    dry_details = []
    for i, dry_id in enumerate(dry_ids):
        mapping = dry_map[dry_id]
        relative = Path(mapping['render_input'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'dry':
            raise ValueError('render_input must point inside generated dry/')
        path = dataset / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        split = splits[groups[dry_id]]
        if hash_splits.setdefault(digest, split) != split:
            raise ValueError('Identical dry audio leaks across splits')
        wave, source_rate = sf.read(path, dtype='float32', always_2d=True)
        if source_rate != config.sample_rate:
            raise ValueError('Generated Dry and CNN must have the same sample rate')
        if not np.isfinite(wave).all():
            raise ValueError('Nonfinite generated dry')
        peak = float(np.max(np.abs(wave)))
        if not peak or abs(20*np.log10(peak) + 12) > .001:
            raise ValueError('Generated dry is not peak -12 dBFS')
        if len(wave) / source_rate > config.duration_sec + 1 / source_rate:
            raise ValueError('Configured duration would truncate this dataset')
        np.save(output/'native_dry'/f'{dry_id}.npy', wave[:,0] if wave.shape[1]==1 else wave.T)
        audio = segment(read_audio(path, config.sample_rate), config)
        dry_audio[i], dry_features[i] = audio, log_mel(audio, config)
        dry_indices[dry_id], hashes[dry_id] = i, digest
        dry_details.append({'dry_sample_id': dry_id, 'recording_id': groups[dry_id],
                            'render_input': mapping['render_input'], 'sha256': digest,
                            'source_rate': source_rate, 'peak_dbfs': float(20*np.log10(peak))})
    wet_features = np.lib.format.open_memmap(output/'wet_features.npy', mode='w+',
                                             dtype='float32', shape=(len(records), *mel_shape))
    wet_audio = np.lib.format.open_memmap(output/'wet_audio.npy', mode='w+', dtype='float32',
                                          shape=(len(records), config.samples))
    distances = []
    def read_wet(row):
        wave,sr = sf.read(row['wet_path'],dtype='float32',always_2d=True)
        if len(wave)/sr > config.duration_sec+1/sr:
            raise ValueError('Configured duration would truncate a wet recording')
        audio = wave.mean(1)
        if sr!=config.sample_rate:
            raise ValueError('Generated Wet and CNN must have the same sample rate')
        return segment(audio,config,reject_silence=False)
    pool = ThreadPoolExecutor(max_workers=8)
    def bounded_wet_audio():
        for start in range(0,len(records),32):
            yield from pool.map(read_wet,records[start:start+32])
    loaded_audio = bounded_wet_audio()
    for i, (row,audio) in enumerate(zip(records,loaded_audio)):
        row['dry_index'] = dry_indices[row['dry_sample_id']]
        validate_rendered_pair(dry_audio[row['dry_index']],audio,row['sample_id'])
        row['wet_index'] = i
        row['dry_path'] = str(dataset / dry_map[row['dry_sample_id']]['render_input'])
        wet_audio[i], wet_features[i] = audio, log_mel(audio, config)
        raw = raw_distances(dry_audio[row['dry_index']], audio, config.sample_rate)
        row['raw_dry_target_distance'] = raw
        distances.append(raw)
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            print(json.dumps({'preparing': i+1, 'total': len(records)}), flush=True)
    pool.shutdown(wait=True)
    for array in (dry_features, dry_audio, wet_features, wet_audio):
        array.flush()
    settings = calibrate(distances, [i for i, r in enumerate(records) if r['split'] == 'train'])
    metadata = {'format_version': 3, 'chain_policy': 'single_vst_drive_only',
                'data_quality_policy':'non_silent_non_passthrough_v1',
                'source_render_audit_sha256':audit_hash,
                'dataset_root': str(dataset), 'seed': seed,
                'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'source_effects_digest': manifest.get('effects_digest'),
                'source_algorithm': manifest.get('algorithm'),
                'audio_config': config.to_dict(), 'catalog': definitions,
                'catalog_hash': fingerprint(definitions), 'classes': list(DRIVES),
                'distance_settings': settings, 'splits': splits, 'dry_sources': dry_details,
                'input_policy': 'ONLY manifest.render_input under generated dry/; no renormalization',
                'counts': {s: {'originals': len({r['recording_id'] for r in records if r['split']==s}),
                               'pairs': sum(r['split']==s for r in records),
                               'per_drive': dict(Counter(r['drive'] for r in records if r['split']==s))}
                           for s in ('train','validation','test')}}
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    write_json(output/'dataset.json', metadata)
    return output


def cache_native_dry(cache):
    """Cache generated dry at the ORIGINAL rendering rate, never resampled first."""
    cache = Path(cache)
    metadata = json.loads((cache/'dataset.json').read_text(encoding='utf-8'))
    (cache/'native_dry').mkdir(exist_ok=True)
    for row in metadata['dry_sources']:
        path = cache/'native_dry'/f"{row['dry_sample_id']}.npy"
        if not path.exists():
            wave,sr = sf.read(Path(metadata['dataset_root'])/row['render_input'],dtype='float32',always_2d=True)
            if sr != row['source_rate']:
                raise ValueError('Native dry rate changed')
            np.save(path,wave[:,0] if wave.shape[1]==1 else wave.T)
    return cache


def load_manifest(cache):
    cache = Path(cache)
    metadata = json.loads((cache/'dataset.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (cache/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if (metadata['format_version'] != 3 or metadata.get('chain_policy') != 'single_vst_drive_only'
            or metadata.get('data_quality_policy') != 'non_silent_non_passthrough_v1'
            or set(metadata['catalog']) != set(DRIVES)
            or fingerprint(metadata['catalog']) != metadata['catalog_hash']):
        raise ValueError('Invalid VST cache')
    seen, hashes = {}, {}
    for r in metadata['dry_sources']:
        s = metadata['splits'][r['recording_id']]
        if hashes.setdefault(r['sha256'], s) != s:
            raise ValueError('Dry-content leakage')
    for row in rows:
        validate_chain(row['ground_truth_chain'],metadata['catalog'])
        if (seen.setdefault(row['recording_id'], row['split']) != row['split'] or
                metadata['splits'][row['recording_id']] != row['split']):
            raise ValueError('Original-recording split leakage')
    return rows, metadata


class PairDataset(Dataset):
    def __init__(self, cache, split):
        self.cache = Path(cache)
        all_rows, self.metadata = load_manifest(cache)
        self.rows = [r for r in all_rows if r['split']==split]
        self.dry = np.load(self.cache/'dry_features.npy', mmap_mode='r')
        self.wet = np.load(self.cache/'wet_features.npy', mmap_mode='r')
        if not self.rows:
            raise ValueError('Empty dataset split')

    def __len__(self):
        return len(self.rows)

    def close(self):
        for key in ('dry','wet'):
            array = getattr(self,key,None)
            if array is not None:
                array._mmap.close()
                setattr(self,key,None)

    def __getitem__(self, index):
        row = self.rows[index]
        items = {p['effect_type']: p['values'] for p in row['ground_truth_chain']['chain']}
        settings = self.metadata['distance_settings']
        target = np.array(row['raw_dry_target_distance'], dtype='float32')
        target /= [settings['spectral_scale'], settings['dynamics_scale']]
        return {'dry': torch.from_numpy(self.dry[row['dry_index']].copy()),
                'wet': torch.from_numpy(self.wet[row['wet_index']].copy()),
                'label': DRIVES.index(row['drive']),
                'knobs': {t: torch.tensor([normalize(items[t][p['name']],p) if t in items else 0
                                          for p in continuous(d)],dtype=torch.float32)
                          for t,d in self.metadata['catalog'].items()},
                'active': {t: t in items for t in self.metadata['catalog']},
                'distance': torch.from_numpy(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    print(prepare(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
