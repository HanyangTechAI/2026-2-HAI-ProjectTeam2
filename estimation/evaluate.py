"""Evaluate CNN knob predictions and inventory-constrained CMA-ES pipelines."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .audio import AudioConfig
from .chain import candidate, continuous, discrete, make_candidates, normalize
from .data import load_manifest, pair_audio, write_json
from .infer import load_predictor, predict_pair
from .refine import refine_candidates


def classification_metrics(truth, predicted, classes):
    if not truth:
        return {'count': 0, 'accuracy': None, 'macro_f1': None}
    f1 = []
    for t in classes:
        tp = sum(a == t and b == t for a, b in zip(truth, predicted))
        fp = sum(a != t and b == t for a, b in zip(truth, predicted))
        fn = sum(a == t and b != t for a, b in zip(truth, predicted))
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {'count': len(truth), 'accuracy': float(np.mean(np.array(truth) == np.array(predicted))),
            'macro_f1': float(np.mean(f1)), 'macro_f1_classes': classes,
            'zero_division': 0}


def evaluate_rows(rows, metadata, model, owned, output, k=3, budget=60,
                  seed=42, save_audio=False):
    """Evaluate a loaded PairCNN on prepared manifest rows.

    Truth is only consumed here, after inference, for metrics and oracle knobs.
    Full-pipeline candidate selection never receives ground truth.
    """
    definitions = metadata['catalog']
    classes = ['bypass', *definitions]
    config = AudioConfig(**metadata['audio_config'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        start = time.perf_counter()
        dry, reference = pair_audio(row, config)
        prediction = predict_pair(model, config, dry, reference)
        candidate_start = time.perf_counter()
        candidates = make_candidates(prediction, owned, k, definitions)
        candidate_seconds = time.perf_counter() - candidate_start
        # Same selected drive identities and discrete choices for both CMA arms.
        defaults = []
        for item in candidates:
            t = item['effect_type']
            fixed = {p['name']: item['chain']['chain'][0]['values'][p['name']]
                     for p in discrete(definitions[t])} if t != 'bypass' else None
            defaults.append(candidate(t, item['model_score'], fixed, definitions))
        methods = {
            'default_cma': refine_candidates(dry, reference, config.sample_rate, defaults, budget, seed),
            'cnn_only': refine_candidates(dry, reference, config.sample_rate, candidates, len(candidates), seed, optimize=False),
            'cnn_cma': refine_candidates(dry, reference, config.sample_rate, candidates, budget, seed)}
        assert methods['default_cma']['render_evaluations'] == methods['cnn_cma']['render_evaluations']
        items = row['ground_truth_chain']['chain']
        truth = items[0]['effect_type'] if items else 'bypass'
        values = items[0]['values'] if items else {}
        oracle = candidate(truth, prediction['probabilities'][truth], prediction['values'].get(truth), definitions)
        oracle_result = refine_candidates(dry, reference, config.sample_rate, [oracle], 1, seed, optimize=False)
        knob_errors = [] if truth == 'bypass' else [
            abs(normalize(prediction['values'][truth][p['name']], p) - normalize(values[p['name']], p))
            for p in continuous(definitions[truth])]
        discrete_correct = [] if truth == 'bypass' else [
            prediction['values'][truth][p['name']] == values[p['name']] for p in discrete(definitions[truth])]
        available = truth == 'bypass' or truth in owned
        record = {'sample_id': row['sample_id'], 'recording_id': row['recording_id'],
                  'truth': truth, 'truth_available': available, 'inventory': owned,
                  'prediction': prediction, 'selected_types': [c['effect_type'] for c in candidates],
                  'candidate_generation_seconds': candidate_seconds,
                  'predicted_type': max(prediction['probabilities'], key=prediction['probabilities'].get),
                  'constrained_predicted_type': candidates[0]['effect_type'],
                  'top_k_hit': truth in [c['effect_type'] for c in candidates],
                  'active_normalized_absolute_errors': knob_errors, 'discrete_correct': discrete_correct,
                  'oracle_knobs': oracle_result, 'methods': methods}
        for name, result in [*methods.items(), ('oracle_knobs', oracle_result)]:
            audio = result.pop('_audio')
            if save_audio:
                import soundfile as sf
                filename = f'{row["sample_id"]}_{name}.wav'
                sf.write(output / filename, audio, config.sample_rate, subtype='FLOAT')
                result['rendered_audio'] = filename
            result['total_seconds_including_inference'] = (result['processing_seconds'] +
                prediction['inference_seconds'] + (0 if name == 'oracle_knobs' else candidate_seconds))
        record['comparison_total_seconds'] = time.perf_counter() - start
        records.append(record)
    if not records:
        raise ValueError('No evaluation samples')
    summary = {'samples': len(records), 'k': k, 'budget_per_cma_method': budget, 'seed': seed,
               'metric': records[0]['methods']['cnn_cma']['metric'],
               'candidate_selection': 'same CNN-ranked inventory-constrained top-K for both CMA arms; discrete argmax shared',
               'cnn_only_selection': 'render all top-K initial candidates once; choose minimum spectral distance',
               'oracle_condition': 'ground-truth type only, all knobs predicted; one render; inventory ignored',
               'timing_definition': 'total includes inference (log-mel/device synchronization), candidate generation, render/metric/CMA; excludes loading checkpoint/audio and serialization; cold imports may affect first sample',
               'renderer': metadata['renderer'], 'catalog_hash': metadata['catalog_hash'],
               'audio_config': metadata['audio_config'], 'data_versions': metadata['versions']}
    for group, subset in [('all', records), ('truth_available', [r for r in records if r['truth_available']]),
                          ('truth_unavailable', [r for r in records if not r['truth_available']])]:
        truth = [r['truth'] for r in subset]
        errors = [e for r in subset for e in r['active_normalized_absolute_errors']]
        correct = [e for r in subset for e in r['discrete_correct']]
        summary[group] = dict(classification_metrics(truth, [r['predicted_type'] for r in subset], classes),
                              constrained_classification=classification_metrics(truth, [r['constrained_predicted_type'] for r in subset], classes),
                              top_k_recall=float(np.mean([r['top_k_hit'] for r in subset])) if subset else None,
                              active_normalized_mae=float(np.mean(errors)) if errors else None,
                              active_knob_count=len(errors), discrete_accuracy=float(np.mean(correct)) if correct else None,
                              oracle_initial_distance=float(np.mean([r['oracle_knobs']['distance'] for r in subset])) if subset else None)
    summary['methods'] = {}
    for method in ('default_cma', 'cnn_only', 'cnn_cma'):
        results = [r['methods'][method] for r in records]
        summary['methods'][method] = {key: float(np.mean([r[key] for r in results])) for key in
                                      ('distance', 'initial_best_distance', 'render_evaluations',
                                       'processing_seconds', 'total_seconds_including_inference')}
        # Unavailable labels still participate in sound reproduction, separately reported.
        for available in (True, False):
            subset = [r['methods'][method]['distance'] for r in records if r['truth_available'] == available]
            summary['methods'][method][f'distance_truth_{"available" if available else "unavailable"}'] = float(np.mean(subset)) if subset else None
    summary['mean_inference_seconds'] = float(np.mean([r['prediction']['inference_seconds'] for r in records]))
    summary['total_render_evaluations_all_conditions'] = sum(
        r['oracle_knobs']['render_evaluations'] + sum(m['render_evaluations'] for m in r['methods'].values()) for r in records)
    (output / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records), encoding='utf-8')
    write_json(output / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True, help='Trained CNN checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    p.add_argument('--owned', nargs='*', default=[])
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--budget', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--save-audio', action='store_true')
    args = p.parse_args()
    rows, metadata = load_manifest(args.manifest)
    rows = [r for r in rows if r['split'] == args.split]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('limit must be positive')
        rows = rows[:args.limit]
    predictor, _, state = load_predictor(args.checkpoint, args.device)
    if state['audio_config'] != metadata['audio_config'] or state['catalog_hash'] != metadata['catalog_hash']:
        raise ValueError('Checkpoint preprocessing or schema differs from manifest')
    summary = evaluate_rows(rows, metadata, predictor, args.owned, args.output,
                            args.k, args.budget, args.seed, args.save_audio)
    source = Path(args.checkpoint)
    summary['provenance'] = {'manifest': str(Path(args.manifest).resolve()),
                             'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                             'split': args.split, 'limit': args.limit,
                             'prediction_source': str(source.resolve()),
                             'prediction_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    write_json(Path(args.output) / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
