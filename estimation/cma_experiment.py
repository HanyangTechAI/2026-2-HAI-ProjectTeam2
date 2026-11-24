"""Prepare/run a paired CNN-seeded versus default-seeded VST7 CMA-ES pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .audio import segment
from .chain import DRIVES, make_candidates
from .data import load_manifest, write_json
from .distance import compare
from .evaluate import render_subset
from .infer import decode, load_predictor
from .refine import default_candidates, refine_candidates, to_search_inputs
from .renderer import activation_preflight as preflight
from .amp import context_from_record


def run(cache='outputs/step2_ir_v1/dataset', checkpoint='outputs/step2_ir_v1/model/best.pt',
        output='outputs/step2_ir_v1/cma_pilot', split='validation', samples=7, k=3,
        budget=180, restarts=2, seed=42, prepare_only=False, sample_id=None):
    torch.set_num_threads(4)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api='blas')
    cache, checkpoint, output = Path(cache), Path(checkpoint), Path(output)
    if k < 1 or k > 7 or budget < k or restarts < 1 or seed < 1:
        raise ValueError('Require 1<=k<=7, budget>=k, restarts>=1, seed>=1')
    if samples != -1 and samples < 1:
        raise ValueError('samples must be positive or -1 for all')
    rows, metadata = load_manifest(cache)
    model, config, state = load_predictor(checkpoint)
    if context_from_record(state) != context_from_record(metadata):
        raise ValueError('Checkpoint/cache amp contexts disagree')
    if any(state[key] != metadata[key] for key in
           ('source_manifest_sha256', 'source_render_audit_sha256', 'catalog_hash', 'audio_config', 'distance_settings')):
        raise ValueError('Checkpoint and cache provenance/preprocessing/distance settings disagree')
    split_rows = [r for r in rows if r['split'] == split]
    if sample_id:
        chosen = [r for r in split_rows if r['sample_id'] == sample_id]
        if len(chosen) != 1:
            raise ValueError('sample_id must identify exactly one sample in the requested split')
        policy = 'explicit single sample for wiring smoke check; not a representative result'
    else:
        ids = render_subset(split_rows, samples)
        chosen = [r for r in split_rows if r['sample_id'] in ids]
        policy = 'class-balanced round-robin originals, selected before inference/search scores'
    if not chosen:
        raise ValueError('No samples selected')
    if any((output/'results').glob('*.json')):
        raise FileExistsError('Results already exist; use a new --output to keep previous experiments')
    # Local mmap caches contain generated Dry only, at native and model rates.
    sources = {r['dry_sample_id']: r for r in metadata['dry_sources']}
    dry_features = np.load(cache/'dry_features.npy', mmap_mode='r')
    wet_features = np.load(cache/'wet_features.npy', mmap_mode='r')
    tasks = []
    with torch.inference_mode():
        for row in chosen:
            a = torch.from_numpy(dry_features[row['dry_index']].copy())[None]
            b = torch.from_numpy(wet_features[row['wet_index']].copy())[None]
            prediction = decode(model(a,b), 0, model.definitions, state['distance_settings'],model.amp_context['amp_id'])
            candidates = make_candidates(prediction, k=k, definitions=model.definitions)
            search_inputs = to_search_inputs(candidates, model.definitions)
            tasks.append({'sample_id': row['sample_id'], 'recording_id': row['recording_id'],
                          'dry_sample_id': row['dry_sample_id'], 'wet_index': row['wet_index'],
                          'source': sources[row['dry_sample_id']],
                          'ground_truth_chain': row['ground_truth_chain'],
                          'truth_drive': row['drive'], 'prediction': prediction,
                          'candidates': candidates, 'search_candidate_configs': search_inputs})
    plan = {'format_version': 1, 'status': 'prepared', 'split': split, 'sample_count': len(tasks),
            'amp_context':model.amp_context,
            'sample_policy': policy, 'samples_by_drive': {d: sum(t['truth_drive']==d for t in tasks) for d in DRIVES},
            'originals': len({t['recording_id'] for t in tasks}),
            'checkpoint': str(checkpoint.resolve()), 'checkpoint_epoch': state['epoch'],
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'source_manifest_sha256': state['source_manifest_sha256'], 'catalog_hash': state['catalog_hash'],
            'source_render_audit_sha256':state['source_render_audit_sha256'],
            'audio_config': state['audio_config'], 'distance_settings': state['distance_settings'],
            'k': k, 'budget_per_method_per_sample': budget, 'restarts': restarts, 'seed': seed,
            'input_policy': metadata['input_policy'],
            'search_engine': 'search.pipeline.run_cma_es_for_chain (main x0_norm API)',
            'comparison': 'same CNN structures, budgets and seeds; CNN knobs versus schema default knobs',
            'structure_policy': 'one VST drive per candidate; no compressor/EQ; CMA-ES changes drive knobs only',
            'evaluation_policy': 'initial seeds included in budget and best result; ground truth used for post-search reporting only',
            'max_search_renders': len(tasks)*2*budget,
            'additional_diagnostic_renders': 7+len(tasks),
            'render_limit_explanation': '7 activation checks plus 1 ground-truth-chain rerender per sample; no model retraining'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'plan.json', plan)
    (output/'tasks.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks), encoding='utf-8')
    if prepare_only:
        print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
        return plan
    from .renderer import Renderer
    import librosa
    renderer = Renderer(model.definitions,model.amp_context)
    first = tasks[0]
    native = np.load(cache/'native_dry'/f"{first['dry_sample_id']}.npy")
    status = preflight(renderer, native, first['source']['source_rate'], model.definitions)
    write_json(output/'renderer_preflight.json', status)
    if not status['available']:
        raise RuntimeError('VST activation check failed (dry pass-through or silence). Run in the licensed user environment; do not use these as search results.')
    target = np.load(cache/'wet_audio.npy', mmap_mode='r')
    results = []
    started = time.perf_counter()
    try:
        for index, task in enumerate(tasks):
            native = np.load(cache/'native_dry'/f"{task['dry_sample_id']}.npy")
            if abs(20*np.log10(max(float(np.max(np.abs(native))), 1e-30))+12) > .001:
                raise ValueError('Native cached Dry no longer has the generated -12 dBFS peak')
            rate = task['source']['source_rate']
            reference = np.asarray(target[task['wet_index']])
            before = time.perf_counter()
            common = dict(native_dry=native, reference=reference, native_rate=rate,
                          config=config, budget=budget, settings=state['distance_settings'],
                          renderer=renderer, seed=seed, restarts=restarts, definitions=model.definitions)
            cnn = refine_candidates(candidates=task['candidates'], **common)
            defaults = refine_candidates(candidates=default_candidates(task['candidates'],model.definitions), **common)
            # True knobs are never passed to either optimizer.
            oracle = renderer(native, rate, task['ground_truth_chain'])
            if oracle.ndim == 2:
                oracle = oracle.mean(0)
            if rate != config.sample_rate:
                oracle = librosa.resample(oracle, orig_sr=rate, target_sr=config.sample_rate)
            oracle_score = compare(segment(oracle,config,reject_silence=False),reference,
                                   config.sample_rate,state['distance_settings'])
            scores = {'cnn_top1_initial': cnn['initial_combined_d'][0],
                      'cnn_topk_best_initial': min(cnn['initial_combined_d']),
                      'default_topk_best_initial': min(defaults['initial_combined_d']),
                      'cnn_cma_best': cnn['distance']['combined'],
                      'default_cma_best': defaults['distance']['combined'],
                      'oracle_full_chain': oracle_score['combined']}
            if scores['cnn_cma_best'] > scores['cnn_topk_best_initial'] + 1e-10:
                raise RuntimeError('CMA search lost its initial seed')
            record = {'sample_id': task['sample_id'], 'recording_id': task['recording_id'],
                      'truth_drive': task['truth_drive'], 'scores': scores,
                      'cnn_seeded': cnn, 'default_seeded': defaults,
                      'oracle_distance': oracle_score,
                      'search_render_evaluations': cnn['render_evaluations']+defaults['render_evaluations'],
                      'seconds': time.perf_counter()-before}
            write_json(output/'results'/f'{index:04d}.json', record)
            results.append(record)
            print(json.dumps({'completed': len(results), 'total': len(tasks), 'sample_id': task['sample_id'],
                              'scores': scores, 'renders': record['search_render_evaluations']},ensure_ascii=False),flush=True)
    except Exception as error:
        write_json(output/'summary.json', {'status': 'failed', 'completed': len(results),
                   'planned': len(tasks), 'error': str(error), 'partial_results': 'results/*.json'})
        raise
    means = {name: float(np.mean([r['scores'][name] for r in results])) for name in results[0]['scores']}
    initial = means['cnn_topk_best_initial']
    summary = {'status': 'complete', 'samples': len(results), 'split': split, 'sample_policy': policy,
               'checkpoint_epoch': state['epoch'], 'mean_combined_d': means,
               'cnn_cma_reduction_vs_topk_initial': 1-means['cnn_cma_best']/initial if initial else None,
               'cnn_cma_wins_over_default_cma': sum(r['scores']['cnn_cma_best']<r['scores']['default_cma_best'] for r in results),
               'search_render_evaluations': sum(r['search_render_evaluations'] for r in results),
               'diagnostic_render_evaluations': 7+len(results), 'seconds': time.perf_counter()-started,
               'distance_settings': state['distance_settings'],
               'interpretation': 'Current renderer comparison, not a guarantee of exact saved-Wet reproduction. Oracle d is a reproduction reference, not a mathematical lower bound. Small pilots establish wiring only.'}
    write_json(output/'summary.json', summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='outputs/step2_ir_v1/dataset')
    parser.add_argument('--checkpoint', default='outputs/step2_ir_v1/model/best.pt')
    parser.add_argument('--output', default='outputs/step2_ir_v1/cma_pilot')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--samples', type=int, default=7, help='Multiple of 7; -1 for all')
    parser.add_argument('--sample-id', help='One explicit smoke sample; overrides balanced sampling')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=180, help='Total search renders per method/sample, shared by K candidates')
    parser.add_argument('--restarts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true', help='Export CNN tasks/plan without loading the VST')
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
