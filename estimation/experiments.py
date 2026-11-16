"""Separate CNN experiments: three-drive classification and known-overdrive regression."""
import argparse
import copy
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audio import AudioConfig
from .chain import chain_instance, checked_render, continuous, denormalize, normalize
from .data import load_manifest, pair_audio, prepare, write_json
from .evaluate import classification_metrics
from .infer import load_predictor
from .train import PairDataset, train


def overdrive_subset(manifest, destination):
    """Copy the exact same overdrive pairs/splits into a self-contained dataset."""
    rows, metadata = load_manifest(manifest)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Dataset already exists: {destination}')
    destination.mkdir(parents=True)
    copied, selected = set(), []
    for row in rows:
        items = row['ground_truth_chain']['chain']
        if not items or items[0]['effect_type'] != 'overdrive':
            continue
        item = copy.deepcopy(row)
        for key in ('dry_path', 'wet_path'):
            path = Path(row[key])
            if path.name not in copied:
                shutil.copy2(path, destination / path.name)
                copied.add(path.name)
            item[key] = path.name
        selected.append(item)
    if not selected:
        raise ValueError('No overdrive samples in source dataset')
    metadata.update(task='regression', effect_types=['overdrive'], include_bypass=False,
                    classes=['overdrive'], derived_from=str(Path(manifest).resolve()),
                    source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest())
    result = destination / 'manifest.jsonl'
    result.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
    write_json(result.with_suffix('.meta.json'), metadata)
    load_manifest(result)
    return result


def dataset_summary(manifest):
    rows, metadata = load_manifest(manifest)
    return {'task': metadata['task'], 'classes': metadata['classes'],
            'splits': {s: {'original_recordings': len({r['recording_id'] for r in rows if r['split'] == s}),
                           'samples': sum(r['split'] == s for r in rows),
                           'per_class': dict(Counter(r['ground_truth_chain']['chain'][0]['effect_type']
                                                     for r in rows if r['split'] == s))}
                       for s in ('train', 'validation', 'test')}}


def evaluate_experiment(manifest, checkpoint, output, split='validation', batch_size=16, threads=4):
    from metrics.spectral import METRIC_VERSION, spectral_distance

    torch.set_num_threads(threads)
    rows, metadata = load_manifest(manifest)
    model, config, state = load_predictor(checkpoint)
    if model.task not in ('classification', 'regression'):
        raise ValueError('This evaluator requires a single-task CNN checkpoint')
    if (model.task != metadata.get('task') or model.classes != metadata['classes'] or
            state['audio_config'] != metadata['audio_config'] or state['catalog_hash'] != metadata['catalog_hash']):
        raise ValueError('Checkpoint and dataset task/configuration do not match')
    dataset = PairDataset(rows, metadata, split, cache_features=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, index = [], 0
    start = time.perf_counter()
    with torch.inference_mode():
        for dry, wet, labels, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            prediction = model(dry, wet)
            probabilities = prediction['logits'].softmax(1).tolist() if prediction['logits'] is not None else None
            for i in range(len(labels)):
                row = dataset.rows[index]
                item = {'sample_id': row['sample_id'], 'recording_id': row['recording_id'], 'split': split}
                if model.task == 'classification':
                    predicted = int(np.argmax(probabilities[i]))
                    item.update(truth=model.classes[int(labels[i])], predicted=model.classes[predicted],
                                probabilities=dict(zip(model.classes, probabilities[i])))
                else:
                    effect = model.effect_types[0]
                    specs = continuous(model.definitions[effect])
                    values = row['ground_truth_chain']['chain'][0]['values']
                    normalized = prediction['continuous'][effect][i].tolist()
                    predicted_values = {p['name']: denormalize(v, p) for p, v in zip(specs, normalized)}
                    defaults = {p['name']: p['default'] for p in specs}
                    truth_norm = [normalize(values[p['name']], p) for p in specs]
                    default_norm = [normalize(defaults[p['name']], p) for p in specs]
                    source, target = pair_audio(row, config)
                    rendered = checked_render(source, config.sample_rate, chain_instance(effect, predicted_values))
                    default_audio = checked_render(source, config.sample_rate, chain_instance(effect))
                    item.update(effect_type=effect, parameter_order=[p['name'] for p in specs],
                                truth=values, predicted=predicted_values, defaults=defaults,
                                truth_normalized=truth_norm, predicted_normalized=normalized,
                                default_normalized=default_norm,
                                predicted_audio_distance=spectral_distance(rendered, target),
                                default_audio_distance=spectral_distance(default_audio, target))
                records.append(item)
                index += 1
    summary = {'task': model.task, 'split': split, 'samples': len(records),
               'original_recordings': len({r['recording_id'] for r in records}),
               'checkpoint_epoch': state['epoch'], 'checkpoint': str(Path(checkpoint).resolve()),
               'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               'manifest': str(Path(manifest).resolve()),
               'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
               'audio_config': config.to_dict(), 'seed': state['training_config']['seed'],
               'processing_seconds': time.perf_counter() - start}
    if model.task == 'classification':
        truth, predicted = [r['truth'] for r in records], [r['predicted'] for r in records]
        matrix = np.zeros((len(model.classes), len(model.classes)), dtype=int)
        for a, b in zip(truth, predicted):
            matrix[model.classes.index(a), model.classes.index(b)] += 1
        summary.update(classification_metrics(truth, predicted, model.classes))
        summary.update(classes=model.classes, confusion_matrix=matrix.tolist(),
                       confusion_matrix_axes='rows=truth, columns=prediction',
                       per_class_recall={t: float(matrix[i, i] / matrix[i].sum()) for i, t in enumerate(model.classes)},
                       balanced_random_accuracy=1 / len(model.classes))
    else:
        predicted = np.array([r['predicted_normalized'] for r in records])
        truth = np.array([r['truth_normalized'] for r in records])
        default = np.array([r['default_normalized'] for r in records])
        error, baseline = np.abs(predicted - truth), np.abs(default - truth)
        specs = continuous(model.definitions[model.effect_types[0]])
        per_knob = {}
        for j, p in enumerate(specs):
            name = p['name']
            per_knob[name] = {'unit': p['unit'], 'range': p['range'],
                              'normalized_mae': float(error[:, j].mean()),
                              'normalized_rmse': float(np.sqrt(np.mean(error[:, j] ** 2))),
                              'mae_in_original_units': float(np.mean([abs(r['predicted'][name] - r['truth'][name]) for r in records])),
                              'default_normalized_mae': float(baseline[:, j].mean()),
                              'default_mae_in_original_units': float(np.mean([abs(r['defaults'][name] - r['truth'][name]) for r in records]))}
        summary.update(effect_type=model.effect_types[0], classification_accuracy=None,
                       normalized_mae=float(error.mean()), default_normalized_mae=float(baseline.mean()),
                       all_knobs_within_0_1_rate=float(np.all(error <= 0.1, axis=1).mean()),
                       per_knob=per_knob,
                       mean_predicted_audio_distance=float(np.mean([r['predicted_audio_distance'] for r in records])),
                       mean_default_audio_distance=float(np.mean([r['default_audio_distance'] for r in records])),
                       metric={'version': METRIC_VERSION, 'direction': 'lower is better'},
                       note='Effect type is given. Knobs are predicted without CMA-ES.')
    (output / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    write_json(output / 'summary.json', summary)
    print(json.dumps({'task': model.task, 'split': split, 'samples': len(records),
                      'accuracy': summary.get('accuracy'), 'normalized_mae': summary.get('normalized_mae')}, ensure_ascii=False), flush=True)
    return summary


def prepare_experiments(sources, output_root, config=AudioConfig(), variations=5, seed=42):
    root = Path(output_root).resolve()
    data_root, results_root = root / 'datasets', root / 'experiments'
    for name in ('drive_classification', 'overdrive_regression'):
        if (data_root / name).exists() or (results_root / name).exists():
            raise FileExistsError(f'Experiment already exists: {name}; choose a new --output-root')
    classification = prepare(sources, data_root / 'drive_classification', config, variations, seed,
                             effect_types=['overdrive', 'distortion', 'fuzz'], include_bypass=False,
                             task='classification')
    regression = overdrive_subset(classification, data_root / 'overdrive_regression')
    # Both tasks use the same original recording splits and identical OD pairs.
    _, a = load_manifest(classification)
    _, b = load_manifest(regression)
    assert a['splits'] == b['splits']
    plan = {'seed': seed, 'variations_per_effect': variations, 'audio_config': config.to_dict(),
            'datasets': {'drive_classification': dataset_summary(classification),
                         'overdrive_regression': dataset_summary(regression)}}
    write_json(results_root / 'plan.json', plan)
    return plan


def train_experiments(output_root, epochs=30, seed=42, batch_size=16, threads=4):
    root = Path(output_root).resolve()
    results_root = root / 'experiments'
    classification = root / 'datasets/drive_classification/manifest.jsonl'
    regression = root / 'datasets/overdrive_regression/manifest.jsonl'
    plan = json.loads((results_root / 'plan.json').read_text(encoding='utf-8'))
    for name in ('drive_classification', 'overdrive_regression'):
        if (results_root / name).exists():
            raise FileExistsError(f'Results already exist: {name}; refusing to overwrite')
    plan.update(epochs=epochs, training_seed=seed, batch_size=batch_size, threads=threads)
    write_json(results_root / 'plan.json', plan)
    for name, manifest in [('drive_classification', classification), ('overdrive_regression', regression)]:
        print(f'TRAIN {name}', flush=True)
        checkpoint = train(manifest, results_root / name / 'model', epochs=epochs,
                           batch_size=batch_size, seed=seed, threads=threads, cache_features=True)
        for split in ('validation', 'test'):
            evaluate_experiment(manifest, checkpoint, results_root / name / split, split, batch_size, threads)
    return plan


def run_experiments(sources, output_root, config=AudioConfig(), variations=5, epochs=30,
                    seed=42, batch_size=16, threads=4):
    prepare_experiments(sources, output_root, config, variations, seed)
    return train_experiments(output_root, epochs, seed, batch_size, threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'prepare'):
        command = commands.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument('--dry-dir', type=Path)
        source.add_argument('--source-index', type=Path)
        command.add_argument('--output-root', type=Path, default=Path('outputs'))
        command.add_argument('--audio-config', type=Path)
        command.add_argument('--variations', type=int, default=5)
        command.add_argument('--seed', type=int, default=42)
        if name == 'run':
            command.add_argument('--epochs', type=int, default=30)
            command.add_argument('--batch-size', type=int, default=16)
            command.add_argument('--threads', type=int, default=4)
    training = commands.add_parser('train')
    training.add_argument('--output-root', type=Path, default=Path('outputs'))
    training.add_argument('--epochs', type=int, default=30)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--batch-size', type=int, default=16)
    training.add_argument('--threads', type=int, default=4)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--manifest', required=True)
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--split', choices=['validation', 'test'], default='validation')
    evaluate.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'evaluate':
        evaluate_experiment(args.manifest, args.checkpoint, args.output, args.split, threads=args.threads)
    elif args.command == 'train':
        train_experiments(args.output_root, args.epochs, args.seed, args.batch_size, args.threads)
    else:
        if args.source_index:
            sources = json.loads(args.source_index.read_text(encoding='utf-8'))
            for item in sources:
                item['file_path'] = str((args.source_index.resolve().parent / item['file_path']).resolve())
        else:
            paths = sorted(p for p in args.dry_dir.rglob('*') if p.suffix.lower() in ('.wav', '.flac'))
            sources = [{'recording_id': p.relative_to(args.dry_dir).with_suffix('').as_posix(),
                        'file_path': str(p.resolve())} for p in paths]
        config = AudioConfig(**json.loads(args.audio_config.read_text(encoding='utf-8'))) if args.audio_config else AudioConfig()
        if args.command == 'prepare':
            print(json.dumps(prepare_experiments(sources, args.output_root, config, args.variations, args.seed), indent=2))
        else:
            run_experiments(sources, args.output_root, config, args.variations, args.epochs,
                            args.seed, args.batch_size, args.threads)


if __name__ == '__main__':
    main()
