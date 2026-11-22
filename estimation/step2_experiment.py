"""Paired Step 2 domain evaluation and checkpoint-seeded team CMA-ES calls.

Uses the public search APIs without changing search/ or regenerating data.
External pairs are never used for training or model selection.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from .amp import context_from_record
from .audio import read_audio, segment
from .chain import DRIVES, continuous, fingerprint, normalize
from .data import load_manifest, prepare, write_json
from .distance import compare
from .evaluate import classification_metrics, evaluate, render_subset
from .infer import load_predictor
from .renderer import Renderer, activation_preflight


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_external(training_cache, external_cache):
    train_rows, train = load_manifest(training_cache)
    rows, external = load_manifest(external_cache)
    if (train['catalog_hash'] != external['catalog_hash'] or
            train['audio_config'] != external['audio_config'] or
            context_from_record(train) != context_from_record(external)):
        raise ValueError('External schema, audio settings, or cabinet differs')
    for field in ('sha256', 'dry_sample_id'):
        if {d[field] for d in train['dry_sources']} & {d[field] for d in external['dry_sources']}:
            raise ValueError(f'Training/external Dry overlap: {field}')
    return {'training_pairs':len(train_rows), 'external_pairs':len(rows),
            'external_originals':len(external['dry_sources']),
            'external_namespaces':sorted({r['recording_id'].split('/')[0] for r in rows}),
            'dry_id_overlap':0, 'dry_byte_hash_overlap':0,
            'scope':'new Dry recordings; Wet remains MultiDrive PRO + fixed IR, not hardware recordings'}


def evaluate_external(dataset,training_cache,checkpoint,output,render_samples=70):
    output = Path(output)
    _, config, state = load_predictor(checkpoint)
    cache = output/'dataset'
    if not (cache/'dataset.json').exists():
        prepare(dataset,cache,config=config)
    else:
        from .experiments import check_cache_source
        check_cache_source(dataset,cache)
    validation = validate_external(training_cache,cache)
    write_json(output/'external_data_check.json',validation)
    result = evaluate(cache,checkpoint,output/'all','all',render_samples=render_samples,
                      allow_external=True)
    if result['samples'] != validation['external_pairs']:
        raise RuntimeError('External evaluation did not cover every pair')
    write_json(output/'summary.json',result)
    return result


def run_search(cache,checkpoint,predictions,output,budget=40,k=3,limit=None,
               shard_index=0,shard_count=1):
    """Call the same checkpoint/candidate API as search.evaluate_dataset.

    The public pipeline accepts metric_fn/render_fn: supply the frozen train d
    and a renderer which rejects silent/pass-through VSTs before applying IR.
    Budget is PER candidate, matching search.evaluate_dataset --budget.
    """
    from search.evaluate_dataset import CnnPredictor, get_cnn_top_k_candidates
    from search.pipeline import search_and_rank_chains
    import scipy.linalg  # Materialize BLAS before limiting its worker pools.
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(4)
    threadpool_limits(limits=1,user_api='blas')
    rows, metadata = load_manifest(cache)
    predictor = CnnPredictor(checkpoint)
    state, config = predictor.state, predictor.config
    if (state['catalog_hash'] != metadata['catalog_hash'] or
            state['audio_config'] != metadata['audio_config'] or
            context_from_record(state) != context_from_record(metadata)):
        raise ValueError('Search checkpoint/cache context mismatch')
    settings = state['distance_settings']
    sources = {r['sample_id']:r for r in rows}
    examples = [json.loads(line) for line in Path(predictions).read_text(encoding='utf-8').splitlines()]
    if {r['sample_id'] for r in examples} != set(sources):
        raise ValueError('Predictions must cover this entire external dataset')
    if limit is not None:
        selected = render_subset(rows,limit)
        examples = [r for r in examples if r['sample_id'] in selected]
    if not 0 <= shard_index < shard_count:
        raise ValueError('Invalid shard index/count')
    examples = examples[shard_index::shard_count]
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    plan = {'checkpoint_sha256':sha(checkpoint),'source_manifest_sha256':metadata['source_manifest_sha256'],
            'prediction_sha256':sha(predictions),'runner_sha256':sha(__file__),
            'search_code_sha256':sha(Path(__file__).resolve().parents[1]/'search/pipeline.py'),
            'metric_code_sha256':{name:sha(Path(__file__).resolve().parents[1]/'metrics'/name)
                                  for name in ('spectral.py','dynamics.py','combined.py')},
            'samples':len(examples),'k':k,'budget_per_candidate':budget,'restarts':2,'seed':42,
            'shard_index':shard_index,'shard_count':shard_count,
            'selection':'all pairs' if limit is None else 'class-balanced round-robin Dry originals, before scores',
            'metric':'frozen training spectral+dynamics combined d','distance_settings':settings,
            'amp_context':context_from_record(metadata),'parameter_round_digits':None,
            'amp_policy':'known ground-truth fixed amp; never inject ground-truth drive',
            'timing_scope':'search call including candidate rendering/metric; excludes inference and WAV saving'}
    plan_hash = fingerprint(plan)
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('Existing search uses different inputs/settings; choose a new output')
    else:
        write_json(output/'plan.json',plan)
    renderer = Renderer(state['catalog'],context_from_record(state))
    first = sources[examples[0]['sample_id']]
    dry = read_audio(first['dry_path'],config.sample_rate)
    preflight = activation_preflight(renderer,dry,config.sample_rate,state['catalog'])
    write_json(output/'renderer_preflight.json',preflight)
    if not preflight['available']:
        raise RuntimeError('VST activation check failed')
    compare(segment(dry,config),segment(dry,config),config.sample_rate,settings)
    threadpool_limits(limits=1,user_api='blas')
    results = []
    for index, example in enumerate(examples):
        row = sources[example['sample_id']]
        folder = output/'samples'/row['sample_id']
        saved = folder/'result.json'
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
            if result['plan_hash'] != plan_hash or any(sha(folder/name)!=digest for name,digest in result['wav_sha256'].items()):
                raise ValueError('Search resume inputs or saved audio changed')
            results.append(result)
            continue
        native = read_audio(row['dry_path'],config.sample_rate)
        target = segment(read_audio(row['wet_path'],config.sample_rate),config,reject_silence=False)
        candidates, mode = get_cnn_top_k_candidates(
            Path(row['dry_path']),Path(row['wet_path']),config.sample_rate,row['sample_id'],
            list(DRIVES),k,predictions_json={row['sample_id']:example['prediction']})
        for candidate in candidates:
            candidate['chain']['amp_id'] = row['ground_truth_chain']['amp_id']
        attempts = {'total':0,'penalties':[]}
        scored = {'count':0,'best_loss':float('inf')}

        def render_checked(audio,rate,chain):
            attempts['total'] += 1
            scored['current_chain'] = copy.deepcopy(chain)
            try:
                wave = renderer(audio,rate,chain)
            except (ValueError,RuntimeError,OSError) as error:
                attempts['penalties'].append({'chain':copy.deepcopy(chain),'error':str(error)})
                raise
            return segment(wave,config,reject_silence=False)

        def metric_checked(x,y):
            try:
                score = compare(x,y,config.sample_rate,settings)
            except (ValueError,RuntimeError,OSError) as error:
                attempts['penalties'].append({'chain':scored['current_chain'],
                                              'stage':'metric','error':str(error)})
                raise
            if scored['count']==0:
                scored['initial_wave'] = x.copy()
                scored['initial_chain'] = scored['current_chain']
            if score['combined'] < scored['best_loss']:
                scored['best_loss'] = score['combined']
                scored['best_wave'] = x.copy()
            scored['count'] += 1
            return score['combined']

        started = time.perf_counter()
        ranked, dropped = search_and_rank_chains(
            native,config.sample_rate,target,candidate_chains=candidates,owned_effect_types=list(DRIVES),
            effects_def=state['catalog'],budget=budget,n_restarts=2,base_seed=42,
            parameter_round_digits=None,render_fn=render_checked,verbose=False,
            metric_fn=metric_checked)
        elapsed = time.perf_counter()-started
        if dropped or attempts['total'] > budget*len(candidates) or not scored['count'] or ranked[0]['loss'] >= 1e9:
            raise RuntimeError('Search failed or exceeded its render budget')
        expected_seed = candidates[0]['chain']['chain'][0]
        actual_seed = scored['initial_chain']['chain'][0]
        if (actual_seed['effect_type'] != expected_seed['effect_type'] or
                not all(np.isclose(actual_seed['values'][p],v,atol=1e-9,rtol=0)
                        for p,v in expected_seed['values'].items())):
            raise RuntimeError('CNN initial seed was penalized; rerun to resume this sample')
        if attempts['total'] != scored['count']+len(attempts['penalties']):
            raise RuntimeError('Unaccounted objective failures')
        # Preserve exactly the waveforms scored by CMA, avoiding VST drift on
        # a second render. Only the ground-truth oracle is an extra render.
        def render_final(chain):
            for attempt in range(3):
                try:
                    return segment(renderer(native,config.sample_rate,chain),config,reject_silence=False)
                except ValueError as error:
                    if attempt == 2 or not any(s in str(error) for s in ('pass-through','Silent VST')):
                        raise

        initial = scored['initial_wave']
        final = scored['best_wave']
        oracle = render_final(row['ground_truth_chain'])
        folder.mkdir(parents=True,exist_ok=True)
        waves = {'dry.wav':segment(native,config),'target.wav':target,
                 'cnn_initial.wav':initial,'cma_best.wav':final,'oracle.wav':oracle}
        for name,wave in waves.items():
            sf.write(folder/name,wave,config.sample_rate,subtype='FLOAT')
        truth = row['drive']
        result = {'sample_id':row['sample_id'],'recording_id':row['recording_id'],'truth':truth,
                  'plan_hash':plan_hash,'mode':mode,'candidates':candidates,
                  'gt_chain':row['ground_truth_chain'],'cnn_predicted':candidates[0]['effect_type'],
                  'cma_predicted':ranked[0]['effect_types'][0],
                  'topk_contains_truth':truth in [c['effect_type'] for c in candidates],
                  'cnn_distance':compare(initial,target,config.sample_rate,settings),
                  'cma_distance':compare(final,target,config.sample_rate,settings),
                  'oracle_distance':compare(oracle,target,config.sample_rate,settings),
                  'search_seconds':elapsed,'search_attempts':attempts['total'],
                  'valid_scored_renders':scored['count'],
                  'saved_audio_policy':'exact scored CNN seed and CMA best; oracle is an additional render',
                  'penalties':attempts['penalties'],
                  'ranked':[{'chain':r['chain'],'loss':r['loss'],'seconds':r['elapsed'],
                             'history':r['history'].tolist()} for r in ranked],
                  'wav_sha256':{name:sha(folder/name) for name in waves}}
        if result['cma_predicted']==truth:
            actual = ranked[0]['chain']['chain'][0]['values']
            expected = row['ground_truth_chain']['chain'][0]['values']
            result['correct_drive_normalized_knob_mae'] = float(np.mean([
                abs(normalize(actual[p['name']],p)-normalize(expected[p['name']],p))
                for p in continuous(state['catalog'][truth])]))
        else:
            result['correct_drive_normalized_knob_mae'] = None
        if not np.isclose(result['cma_distance']['combined'],ranked[0]['loss'],atol=1e-5,rtol=1e-4):
            raise RuntimeError('Saved best WAV does not reproduce the search score')
        if result['cma_distance']['combined'] > result['cnn_distance']['combined']+1e-5:
            raise RuntimeError('Search lost the CNN initial seed')
        write_json(saved,result)
        results.append(result)
        write_json(output/'progress.json',{'completed':len(results),'total':len(examples),
                   'last_sample':row['sample_id'],'last_search_seconds':elapsed})
        print(json.dumps({'search_completed':index+1,'total':len(examples),
                          'seconds':elapsed,'cnn_d':result['cnn_distance']['combined'],
                          'cma_d':result['cma_distance']['combined'] }),flush=True)
    summary = search_summary(results,plan)
    write_json(output/'summary.json',summary)
    return summary


def search_summary(results,plan):
    return {'samples':len(results),'plan':plan,
               'cnn':classification_metrics([r['truth'] for r in results],[r['cnn_predicted'] for r in results]),
               'cma':classification_metrics([r['truth'] for r in results],[r['cma_predicted'] for r in results]),
               'topk_recall':float(np.mean([r['topk_contains_truth'] for r in results])),
               'mean_search_seconds':float(np.mean([r['search_seconds'] for r in results])),
               'median_search_seconds':float(np.median([r['search_seconds'] for r in results])),
               'search_attempts':sum(r['search_attempts'] for r in results),
               'penalized_attempts':sum(len(r['penalties']) for r in results),
               'correct_drive_knob_mae_samples':sum(r['correct_drive_normalized_knob_mae'] is not None for r in results),
               'mean_correct_drive_normalized_knob_mae':float(np.mean([
                   r['correct_drive_normalized_knob_mae'] for r in results
                   if r['correct_drive_normalized_knob_mae'] is not None])) if any(
                       r['correct_drive_normalized_knob_mae'] is not None for r in results) else None,
               'mean_distances':{name:{key:float(np.mean([r[name][key] for r in results]))
                                      for key in ('combined','spectral','dynamics')}
                                 for name in ('cnn_distance','cma_distance','oracle_distance')}}


def parallel_search(cache,checkpoint,predictions,output,budget=40,k=3,workers=4):
    """Separate plugin processes, deterministic per-pair budgets and resumable WAVs."""
    if workers < 1:
        raise ValueError('workers must be positive')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    jobs, logs = [], []
    started = time.perf_counter()
    try:
        for index in range(workers):
            folder = output/'shards'/str(index)
            folder.mkdir(parents=True,exist_ok=True)
            log = (folder/'run.log').open('w',encoding='utf-8')
            logs.append(log)
            command = [sys.executable,'-u','-m','estimation.step2_experiment','search',
                       '--cache',str(cache),'--checkpoint',str(checkpoint),'--predictions',str(predictions),
                       '--output',str(folder),'--budget',str(budget),'--k',str(k),
                       '--shard-index',str(index),'--shard-count',str(workers)]
            env = dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            job = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            jobs.append(job)
        write_json(output/'processes.json',{'workers':workers,'pids':[p.pid for p in jobs],
                   'checkpoint_sha256':sha(checkpoint)})
        last = None
        while any(p.poll() is None for p in jobs):
            failed = [p.returncode for p in jobs if p.poll() not in (None,0)]
            if failed:
                raise RuntimeError(f'Search worker failed: {failed}; see shard run.log; rerun to resume')
            completed = sum(len(list((output/'shards'/str(i)/'samples').glob('*/result.json')))
                            for i in range(workers))
            if completed != last:
                write_json(output/'progress.json',{'completed':completed,'total':sum(
                    1 for _ in Path(predictions).open(encoding='utf-8')),'workers':workers,
                    'wall_seconds':time.perf_counter()-started})
                print(json.dumps({'parallel_search_completed':completed,'workers':workers,
                                  'wall_seconds':time.perf_counter()-started}),flush=True)
                last = completed
            time.sleep(1)
        if any(p.returncode != 0 for p in jobs):
            raise RuntimeError('Search worker failed; see shard run.log')
    finally:
        for process in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    results = []
    plans = []
    for index in range(workers):
        folder = output/'shards'/str(index)
        plans.append(json.loads((folder/'plan.json').read_text(encoding='utf-8')))
        results.extend(json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'samples').glob('*/result.json')))
    expected = {json.loads(line)['sample_id'] for line in Path(predictions).read_text(encoding='utf-8').splitlines()}
    if len(results) != len(expected) or {r['sample_id'] for r in results} != expected:
        raise RuntimeError('Sharded search omitted or duplicated samples')
    summary = search_summary(results,{'workers':workers,'shard_plans':plans,
                                    'budget_per_candidate':budget,'k':k,'seed':42})
    summary['wall_seconds'] = time.perf_counter()-started
    summary['throughput_seconds_per_pair'] = summary['wall_seconds']/len(results)
    summary['timing_note'] = 'per-pair search seconds include process contention; wall throughput is not single-query latency'
    write_json(output/'summary.json',summary)
    return summary


def report(training='outputs/step2_ir_v1',external='outputs/step2_real_v1',
           document='estimation/EXPERIMENTS.md'):
    """Write the canonical report from complete saved results, without refitting."""
    training, external = Path(training),Path(external)
    read = lambda p:json.loads(p.read_text(encoding='utf-8'))
    native = read(training/'test/summary.json')
    real = read(external/'all/summary.json')
    checks = read(external/'external_data_check.json')
    history = read(training/'model/history.json')
    best = min(history,key=lambda r:r['validation']['loss'])
    search_path = external/'search/summary.json'
    search = read(search_path) if search_path.exists() else None
    gap = native['accuracy']-real['accuracy']
    lines = ['## 2026-11-22 — Step 2: current-schema retraining and paired external guitars', '',
        '- 기존 CNN 구조, 현재 VST 7종 노브 범위, Marshall V30 IR(+12.5 dB)을 유지. 컴프/EQ 제외.',
        '- `step2_ir_v1`: 2,730쌍, Dry 원본 78개. train 1,890 / validation 420 / test 420쌍.',
        f'- 30 epoch, seed 42, best epoch **{best["epoch"]}** (전체 validation loss 최소). 외부 데이터로 모델을 선택하지 않음.',
        f'- best epoch loss: train {best["train"]["loss"]:.6f}, validation {best["validation"]["loss"]:.6f}.',
        f'- `step2_real_v1`: 전체 **{real["samples"]}쌍 / Dry 원본 {checks["external_originals"]}개**를 외부 평가. 임의 test 42쌍으로 제한하지 않음.',
        '- 학습/외부 Dry ID·내용 해시 중복 0. 대응 Dry는 각 생성 폴더 `manifest.render_input`의 `dry/`만 사용.',
        '- 외부 Wet도 VST+IR 합성이다. 새로운 기타/픽업/연주에 대한 평가이며 실제 페달/앰프 녹음 검증은 아니다.',
        '- namespace: '+', '.join(checks['external_namespaces']),
        f'- checkpoint SHA256: `{sha(training/"model/best.pt")}`.',
        '', '| 평가 | 쌍 수 | Top-1 | Top-3 | Macro F1 |', '|---|---:|---:|---:|---:|']
    for name, value in (('합성 test',native),('외부 기타 전체',real)):
        lines.append(f'| {name} | {value["samples"]} | {value["accuracy"]:.2%} | {value["top3_accuracy"]:.2%} | {value["macro_f1"]:.4f} |')
    lines += ['', f'- 합성 test → 외부 기타 Top-1 변화: **{-gap*100:+.2f} percentage points**.',
        '- 280쌍은 독립 녹음 280개가 아니라 Dry 40개의 7종 변형이다. 표본 독립성을 과장하지 않는다.',
        '- 모든 d는 이 모델의 train-only spectral/dynamics scale, 0.5/0.5 가중치로 고정. 외부 데이터의 scale로 다시 보정하지 않음.',
        '- 11/21 모델은 노브 범위·거리 구현/데이터가 달라 이번 수치와 직접적인 개선율을 계산하지 않는다.',
        '', '| 실제 렌더링 d (각 70쌍 균형 표본) | CNN Top-1 | CNN Top-3 최선 | 정답 체인 재현 oracle |',
        '|---|---:|---:|---:|']
    for name,value in (('합성 test',native),('외부 기타',real)):
        d = value.get('mean_rendered_distances')
        if d:
            lines.append(f'| {name} | {d["cnn_top1"]["combined"]:.6f} | {d["cnn_top3_best"]["combined"]:.6f} | {d["oracle_full_chain"]["combined"]:.6f} |')
    for name,value in (('합성 test',native),('외부 기타',real)):
        matrix = np.asarray(value['confusion_matrix'])
        pairs = sorted(((int(matrix[i,j]),DRIVES[i],DRIVES[j]) for i in range(7) for j in range(7) if i!=j),reverse=True)
        lines += ['', f'주요 혼동 ({name}, 정답 → 예측):']
        lines += [f'- `{a}` → `{b}`: {n}쌍' for n,a,b in pairs[:5] if n]
    if search:
        d = search['mean_distances']
        lines += ['', 'CMA-ES (외부 기타 전체, 공용 search API 호출; `search/` 수정 없음):',
            f'- 평가 완료: **{search["samples"]}쌍**, K=3, 후보당 40회(샘플당 최대 120회), restart=2, seed=42.',
            '- `search.evaluate_dataset.CnnPredictor`와 후보 생성 API, `search.pipeline.search_and_rank_chains`를 호출. 공개 `metric_fn`/`render_fn` 인자로 고정 combined d와 검증 렌더러를 전달.',
            '- 후보 amp_id는 정답의 알려진 고정 앰프로 설정. 정답 드라이브는 후보에 주입하지 않음.',
            f'- CNN Top-1 **{search["cnn"]["accuracy"]:.2%}** → CMA 최종 Top-1 **{search["cma"]["accuracy"]:.2%}**; Top-3 recall **{search["topk_recall"]:.2%}**.',
            f'- 동일 {search["samples"]}쌍 평균 d: CNN 초기 **{d["cnn_distance"]["combined"]:.6f}** → CMA **{d["cma_distance"]["combined"]:.6f}**; oracle **{d["oracle_distance"]["combined"]:.6f}**.',
            f'- 실제 최적화 render attempts {search["search_attempts"]:,}; penalty {search["penalized_attempts"]:,}. 초기/최종 WAV는 실제 점수화한 파형 그대로 저장. oracle 추가 렌더링과 preflight는 최적화 예산 밖.',
            f'- 4개 독립 VST 프로세스. 평균 샘플 탐색 {search["mean_search_seconds"]:.2f}초(동시 실행 경합 포함), 전체 wall {search["wall_seconds"]:.2f}초, 처리량 {search["throughput_seconds_per_pair"]:.2f}초/쌍.',
            '- 위 처리량은 단일 요청 지연시간이 아니며 이전 약 26초/샘플과 직접 비교하지 않는다.',
            '- 샘플마다 `dry.wav`, `target.wav`, `cnn_initial.wav`, `cma_best.wav`, `oracle.wav`, `result.json` 저장. FLOAT WAV 해시와 재계산 d로 결과를 확인.',
            '- 저장 위치: `outputs/step2_real_v1/search/shards/{0..3}/samples/<target_id>/`.']
    else:
        lines += ['', '- CMA-ES 전체 탐색 진행 중. 최종 `search/summary.json` 완료 후 이 보고서를 갱신한다.']
    lines += ['', '후속 판단:',
        f'- CNN 단독 Top-1의 {gap*100:.2f} percentage point 하락을 확인했으므로 입력 조건 다양화(232: 다른 기타/픽업·코드, 노이즈/게인)의 후속 통제 실험을 권장한다. 이번 실행은 현재 조건의 기준 실험으로 보존한다.',
        '- 외부 280쌍을 반복 튜닝용 validation으로 사용하지 않고, 추가 학습/validation용 Dry를 별도로 확보한다.',
        '- 데이터 생성은 팀 공용 `scripts/generate_step1_dataset.py`, IR 적용은 `scripts/apply_amp_dataset.py`를 사용. estimation 내부의 이전 생성·Demucs 비교 실행기는 제거했으며 현재 실험은 공유 Step 2 데이터를 사용.',
        '- 공유 드라이브 원본 수정 없음. 기존 모델/결과 보존. 기존 로컬 IR이 공유 IR과 동일한 해시여서 재사용하고 빠진 파일만 복사.',
        '', '재현 명령:', '```powershell',
        '.\\.conda-cnn\\python.exe -m estimation.experiments --dataset data/local/step2_ir_v1 --output outputs/step2_ir_v1 --epochs 30',
        '.\\.conda-cnn\\python.exe -m estimation.step2_experiment evaluate',
        '.\\.conda-cnn\\python.exe -m estimation.step2_experiment parallel-search --workers 4 --budget 40 --k 3',
        '.\\.conda-cnn\\python.exe -m estimation.step2_experiment report', '```', '']
    start, end = '<!-- step2-2026-11-22:start -->','<!-- step2-2026-11-22:end -->'
    document = Path(document)
    text = document.read_text(encoding='utf-8')
    section = start+'\n'+'\n'.join(lines)+'\n'+end
    if start in text:
        a,b = text.index(start),text.index(end)+len(end)
        text = text[:a]+section+text[b:]
    else:
        text = text.rstrip()+'\n\n'+section+'\n'
    document.write_text(text,encoding='utf-8',newline='\n')
    return {'document':str(document),'search_complete':search is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage',required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--dataset',default='data/local/step2_real_v1')
    e.add_argument('--training-cache',default='outputs/step2_ir_v1/dataset')
    e.add_argument('--checkpoint',default='outputs/step2_ir_v1/model/best.pt')
    e.add_argument('--output',default='outputs/step2_real_v1')
    e.add_argument('--render-samples',type=int,default=70)
    s = sub.add_parser('search')
    s.add_argument('--cache',default='outputs/step2_real_v1/dataset')
    s.add_argument('--checkpoint',default='outputs/step2_ir_v1/model/best.pt')
    s.add_argument('--predictions',default='outputs/step2_real_v1/all/samples.jsonl')
    s.add_argument('--output',default='outputs/step2_real_v1/search')
    s.add_argument('--budget',type=int,default=40)
    s.add_argument('--k',type=int,default=3)
    s.add_argument('--limit',type=int)
    s.add_argument('--shard-index',type=int,default=0)
    s.add_argument('--shard-count',type=int,default=1)
    parallel = sub.add_parser('parallel-search')
    parallel.add_argument('--cache',default='outputs/step2_real_v1/dataset')
    parallel.add_argument('--checkpoint',default='outputs/step2_ir_v1/model/best.pt')
    parallel.add_argument('--predictions',default='outputs/step2_real_v1/all/samples.jsonl')
    parallel.add_argument('--output',default='outputs/step2_real_v1/search')
    parallel.add_argument('--budget',type=int,default=40)
    parallel.add_argument('--k',type=int,default=3)
    parallel.add_argument('--workers',type=int,default=4)
    sub.add_parser('report')
    args = vars(p.parse_args())
    stage = args.pop('stage')
    function = {'evaluate':evaluate_external,'search':run_search,'parallel-search':parallel_search,'report':report}[stage]
    result = function(**args)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
