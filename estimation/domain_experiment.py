"""Paired synthetic domain test and UNPAIRED Demucs target-tone exploration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from .amp import context_from_record
from .audio import read_audio,segment
from .chain import DRIVES,continuous,fingerprint,make_candidates,normalize
from .data import PairDataset,load_manifest,write_json
from .distance import compare
from .evaluate import classification_metrics
from .infer import decode,load_predictor,predict_pair
from .refine import refine_candidates
from .renderer import Renderer,activation_preflight


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_grid(old,new,bare_cache,amp_cache,output):
    output = Path(output)
    baseline,base_meta = load_manifest(bare_cache)
    adapted,amp_meta = load_manifest(amp_cache)
    provenance = {'old_checkpoint_sha256':sha(old),'amp_checkpoint_sha256':sha(new),
                  'bare_manifest_sha256':base_meta['source_manifest_sha256'],
                  'amp_manifest_sha256':amp_meta['source_manifest_sha256'],'code_sha256':sha(__file__)}
    if (output/'provenance.json').exists():
        saved = json.loads((output/'provenance.json').read_text(encoding='utf-8'))
        if saved['inputs'] != provenance or any(sha(output/name)!=digest for name,digest in saved['result_sha256'].items()):
            raise ValueError('Existing synthetic comparison has different inputs or corrupted results')
        return json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert base_meta['catalog_hash'] == amp_meta['catalog_hash']
    assert base_meta['splits'] == amp_meta['splits']
    assert {d['recording_id']:d['sha256'] for d in base_meta['dry_sources']} == {d['recording_id']:d['sha256'] for d in amp_meta['dry_sources']}
    before = {r['sample_id']:r for r in baseline}
    for row in adapted:
        original = before[row['sample_id']]
        assert original['split'] == row['split'] and original['ground_truth_chain']['chain'] == row['ground_truth_chain']['chain']
    summary = {}
    for domain,cache in (('bare_drive',bare_cache),('fixed_cabinet',amp_cache)):
        dataset = PairDataset(cache,'test')
        for name,checkpoint in (('old',old),('amp',new)):
            model,config,state = load_predictor(checkpoint)
            assert state['catalog_hash'] == dataset.metadata['catalog_hash']
            assert state['audio_config'] == dataset.metadata['audio_config']
            predicted,truth,records,errors,top3 = [],[],[],[],0
            offset = 0
            with torch.inference_mode():
                for batch in DataLoader(dataset,batch_size=16,shuffle=False):
                    result = model(batch['dry'],batch['wet'])
                    for i in range(len(batch['label'])):
                        row = dataset.rows[offset+i]
                        pred = decode(result,i,model.definitions)
                        ranks = sorted(DRIVES,key=lambda d:-pred['probabilities'][d])
                        truth.append(row['drive']); predicted.append(ranks[0])
                        top3 += row['drive'] in ranks[:3]
                        target = row['ground_truth_chain']['chain'][0]['values']
                        error = float(np.mean([abs(normalize(pred['values'][row['drive']][p['name']],p)-normalize(target[p['name']],p)) for p in continuous(model.definitions[row['drive']])]))
                        errors.append(error)
                        records.append({'sample_id':row['sample_id'],'truth':row['drive'],'top3':ranks[:3],
                                        'probabilities':pred['probabilities'],'true_drive_knob_mae':error})
                    offset += len(batch['label'])
            key = domain+'__'+name
            summary[key] = {**classification_metrics(truth,predicted),'samples':len(records),
                            'top3_accuracy':top3/len(records),'normalized_knob_mae':float(np.mean(errors))}
            write_json(output/(key+'.json'),{'summary':summary[key],'samples':records})
        dataset.close()
    write_json(output/'summary.json',summary)
    write_json(output/'provenance.json',{'inputs':provenance,
               'result_sha256':{p.name:sha(p) for p in sorted(output.glob('*.json')) if p.name!='provenance.json'}})
    return summary


def run(old,new,bare_cache,amp_cache,targets,output,budget=90,k=3,seed=42):
    torch.set_num_threads(4)
    import scipy.linalg  # Load its BLAS before limiting the process pools.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1,user_api='blas')
    output,amp_cache = Path(output),Path(amp_cache)
    output.mkdir(parents=True,exist_ok=True)
    old_model,config,old_state = load_predictor(old)
    new_model,new_config,new_state = load_predictor(new)
    if old_state['catalog_hash'] != new_state['catalog_hash'] or config != new_config:
        raise ValueError('A controlled comparison requires the same vocabulary/knob bounds/audio config')
    context = context_from_record(new_state)
    target_paths = sorted(Path(targets).glob('*.wav'))
    if not target_paths or budget%k:
        raise ValueError('Need target WAVs and a budget divisible by K')
    rows,metadata = load_manifest(amp_cache)
    if (context_from_record(metadata) != context or
            metadata['source_manifest_sha256'] != new_state['source_manifest_sha256']):
        raise ValueError('New model/cache provenance or cabinet context disagrees')
    dry_row = sorted([r for r in rows if r['split']=='test'],key=lambda r:r['sample_id'])[0]
    source = next(s for s in metadata['dry_sources'] if s['dry_sample_id']==dry_row['dry_sample_id'])
    dry = np.load(amp_cache/'native_dry'/f"{dry_row['dry_sample_id']}.npy")
    rate = source['source_rate']
    if rate != config.sample_rate:
        raise ValueError('Generated Dry must already be at model/VST rate')
    dry_mono = segment(dry.mean(0) if dry.ndim==2 else dry,config)
    settings = old_state['distance_settings']  # One frozen baseline TRAIN scale for ALL comparisons.
    plan = {'old_checkpoint_sha256':sha(old),'amp_checkpoint_sha256':sha(new),
            'experiment_code_sha256':sha(__file__),
            'refine_code_sha256':sha(Path(__file__).with_name('refine.py')),
            'blas_threads':1,'library_warmup':'scipy.linalg loaded before limits; metric warmup before refinement',
            'target_sha256':{p.name:sha(p) for p in target_paths},'seed':seed,'k':k,'budget':budget,
            'amp_context':context,'common_distance_settings':settings,'dry_sample_id':dry_row['dry_sample_id'],
            'dry_recording_id':dry_row['recording_id'],'native_rate':rate,
            'has_ground_truth':False,'paired_dry':False,
            'scope':'UNPAIRED target-tone experiment; d includes different performance/content and source separation artifacts; NOT classification accuracy or generalization performance',
            'window_policy':'highest RMS 2.5-second nonoverlapping target window, selected before prediction; original target gain retained',
            'speed_policy':'K1 budget=budget/K uses the actually measured first branch of the same K3 search, not a separate timing benchmark'}
    plan_id = fingerprint(plan)
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text(encoding='utf-8')) != plan:
        raise ValueError('Output belongs to a different experiment')
    write_json(output/'plan.json',plan)
    synthetic = synthetic_grid(old,new,bare_cache,amp_cache,output/'synthetic')
    amp_renderer = Renderer(new_model.definitions,context)
    bare_renderer = Renderer(old_model.definitions,old_model.amp_context)
    preflight = activation_preflight(amp_renderer,dry,rate,new_model.definitions)
    write_json(output/'renderer_preflight.json',preflight)
    if not preflight['available']:
        raise RuntimeError('VST is not active')
    records = []
    for index,path in enumerate(target_paths):
        folder = output/'real'/f'{index:02d}'
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists():
            record = json.loads((folder/'result.json').read_text(encoding='utf-8'))
            if record['plan_id'] != plan_id:
                raise ValueError('Saved sample belongs to another plan')
            records.append(record)
            continue
        before = time.perf_counter()
        audio = read_audio(path,rate)
        starts = list(range(0,max(1,len(audio)-config.samples+1),config.samples))
        start = max(starts,key=lambda s:float(np.mean(audio[s:s+config.samples].astype('float64')**2)))
        reference = segment(audio,config,start)
        sf.write(folder/'target.wav',reference,rate,subtype='FLOAT')
        sf.write(folder/'dry.wav',dry_mono,rate,subtype='FLOAT')
        predictions = {'old':predict_pair(old_model,config,dry_mono,reference,old_state['distance_settings']),
                       'amp':predict_pair(new_model,config,dry_mono,reference,new_state['distance_settings'])}
        record = {'plan_id':plan_id,'target':path.name,'target_start_seconds':start/rate,'source_sha256':sha(path),
                  'ground_truth':None,'predictions':predictions,'initial':{},'search':{},'saved_audio':{}}
        def save_wave(name,chain,renderer):
            wave = np.asarray(renderer(dry,rate,chain))
            mono = segment(wave.mean(0) if wave.ndim==2 else wave,config,reject_silence=False)
            sf.write(folder/(name+'.wav'),mono,rate,subtype='FLOAT')
            score = compare(mono,reference,rate,settings)
            record['saved_audio'][name] = {'chain':chain,'distance':score,'file':name+'.wav','sha256':sha(folder/(name+'.wav'))}
            return score
        bare_candidates = make_candidates(predictions['old'],k=k,definitions=old_model.definitions)
        save_wave('old_no_amp_initial',bare_candidates[0]['chain'],bare_renderer)
        for name,model in (('old',old_model),('amp',new_model)):
            prediction = copy.deepcopy(predictions[name])
            prediction['amp_id'] = context['amp_id']
            candidates = make_candidates(prediction,k=k,definitions=model.definitions)
            record['initial'][name] = save_wave(name+'_ir_initial',candidates[0]['chain'],amp_renderer)
            options = dict(native_dry=dry,reference=reference,native_rate=rate,config=config,
                           candidates=candidates,budget=budget,settings=settings,renderer=amp_renderer,
                           seed=seed,restarts=1,definitions=model.definitions)
            search_started = time.perf_counter()
            result = refine_candidates(**options)
            result['wall_seconds'] = time.perf_counter()-search_started
            record['search'][name] = result
            save_wave(name+'_ir_cma',result['candidate']['chain'],amp_renderer)
            first = next(r for r in result['ranked_candidates'] if r['candidate']['effect_type']==candidates[0]['effect_type'])
            record['search'][name]['top1_budget_control'] = copy.deepcopy(first)
            if name=='amp':
                save_wave('amp_ir_cma_k1',first['candidate']['chain'],amp_renderer)
        record['seconds'] = time.perf_counter()-before
        write_json(folder/'result.json',record)
        records.append(record)
        print(json.dumps({'completed':len(records),'total':len(target_paths),'target':path.name,
                          'old_ir_cma_d':record['search']['old']['distance']['combined'],
                          'amp_ir_cma_d':record['search']['amp']['distance']['combined']},ensure_ascii=False),flush=True)
    summary = {'status':'complete','samples':len(records),'scope':plan['scope'],'has_ground_truth':False,
               'mean_saved_audio_d':{key:float(np.mean([r['saved_audio'][key]['distance']['combined'] for r in records])) for key in records[0]['saved_audio']},
               'mean_search_d':{name:float(np.mean([r['search'][name]['distance']['combined'] for r in records])) for name in ('old','amp')},
               'new_k3_search_mean_seconds':float(np.mean([sum(c['seconds'] for c in r['search']['amp']['ranked_candidates']) for r in records])),
               'new_k1_branch_mean_seconds':float(np.mean([r['search']['amp']['top1_budget_control']['seconds'] for r in records])),
               'new_k1_branch_mean_d':float(np.mean([r['search']['amp']['top1_budget_control']['distance']['combined'] for r in records])),
               'search_render_evaluations':sum(r['search'][m]['render_evaluations'] for r in records for m in ('old','amp')),
               'synthetic':synthetic,'plan_id':plan_id}
    write_json(output/'summary.json',summary)
    write_report(output,summary,plan,records,new)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def write_report(output,summary,plan,records,checkpoint):
    model,config,state = load_predictor(checkpoint)
    root = Path(checkpoint).parent.parent
    native = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    history = json.loads((root/'model/history.json').read_text(encoding='utf-8'))
    selected = next(h for h in history if h['epoch']==state['epoch'])
    lines = ['## 2026-11-21 캐비닛 포함 재학습과 Demucs 목표 톤 비교','',
        '### 고정 조건','',
        '- 앰프 구현: Marshall 4×12 V30의 EV MIX B 캐비닛 IR + 고정 게인 12.5 dB. 비선형 앰프 모델은 아니다.',
        '- 같은 Dry 78개·드라이브 7종·변형 5개, 총 2,730쌍의 기존 bare Wet에 공용 apply_amp를 적용했다.',
        '- Dry는 바이트 단위로 동일한 −12 dBFS 생성본이다. 원본별 train/validation/test=54/12/12, 쌍=1890/420/420을 유지했다.',
        '- 최근 main의 노브 범위 변경을 섞지 않고 기존 best checkpoint의 catalog를 고정했다. 같은 노브로 캐비닛 처리만 추가한 통제 비교다.',
        '- CNN encoder/fusion·head·학습 설정 유지: seed42, 30epochs, batch16, Adam lr0.001, cosine, CPU4threads. 새 가중치에서 학습했다.',
        '- 캐비닛 정의·IR hash를 캐시/checkpoint에 보존하고 CNN 후보와 CMA-ES까지 같은 amp_id를 전달한다. 컴프·EQ는 사용하지 않는다.',
        f'- IR SHA-256: `{plan["amp_context"]["ir_sha256"]}`.',
        f'- 새 best epoch: **{state["epoch"]}**(전체 validation loss 기준), checkpoint SHA-256: `{sha(checkpoint)}`.',
        f'- best train loss={selected["train"]["loss"]:.6f}, validation loss={selected["validation"]["loss"]:.6f}.',
        f'- 새 모델 경로: `{checkpoint}`. 이전 bare 모델은 보존했다.','',
        '### 정답이 있는 합성 test: 같은 420개에서 전후 비교','',
        '노브 MAE는 각 표본의 정답 드라이브 노브들을 평균한 뒤 420개 표본을 평균한다.',
        '노브 수가 다른 드라이브도 같은 표본 가중치를 가지며, 앞선 총 노브 수 가중 학습 MAE와 집계 방식이 다르다.','',
        '| 학습 모델 | 평가 데이터 | Top-1 | Top-3 | 정답 드라이브 조건부 노브 MAE |',
        '|---|---|---:|---:|---:|']
    for key,label in (('bare_drive__old','기존 / bare'),('fixed_cabinet__old','기존 / 캐비닛'),
                      ('bare_drive__amp','재학습 / bare'),('fixed_cabinet__amp','재학습 / 캐비닛')):
        s=summary['synthetic'][key]; model_name,domain=label.split(' / ')
        lines.append(f'| {model_name} | {domain} | {100*s["accuracy"]:.2f}% | {100*s["top3_accuracy"]:.2f}% | {s["normalized_knob_mae"]:.6f} |')
    lines += ['', '새 모델의 본래 캐비닛 데이터 평가:', '',
        '| Split | Top-1 | Top-3 | CNN Top-1 렌더 d | 같은 드라이브 기본 노브 d | 정답 재현 d |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ('validation','test'):
        s=native[split]; d=s['mean_rendered_distances']
        lines.append(f'| {split} | {100*s["accuracy"]:.2f}% | {100*s["top3_accuracy"]:.2f}% | {d["cnn_top1"]["combined"]:.6f} | {d["same_predicted_structure_default_knobs"]["combined"]:.6f} | {d["oracle_full_chain"]["combined"]:.6f} |')
    lines += ['', '위 렌더 d는 새 캐비닛 train에서만 계산한 scale을 사용한 split당 70개 평가다.',
        '앞선 bare 실험의 scale과 다르므로 두 절의 d 숫자를 직접 빼서 개선량으로 사용하지 않는다.','',
        '### 정답이 없는 Demucs 10개: 목표 톤 탐색','',
        '- 대응 Dry DI·정답 체인이 없다. 기존 합성 test Dry 1개를 공통 입력으로 사용했다.',
        '- 각 파일에서 예측 전에 RMS가 가장 큰 비중첩 2.5초 구간을 선택했다. 원본 목표 음량은 유지했다.',
        '- 기존 모델+같은 IR과 재학습 모델+IR을 비교한다. 구형 native no-amp 결과도 청취용으로 별도 저장했다.',
        '- 두 모델의 비교 d는 기존 bare 모델의 train-only scale 하나로 고정했다. 모델마다 다른 scale을 섞지 않았다.',
        '- K=3, 총 예산90(후보당30), n_runs=1, seed42를 사전에 고정했다.',
        '- 연주·음정·분리 아티팩트가 달라 d에는 톤 외 차이가 들어간다. 이 수치를 실제 녹음의 분류 정확도·노브 복원 오차·일반화 성능으로 해석하지 않는다.','',
        '| 파일 | 기존+IR 초기 d | 재학습+IR 초기 d | 기존+IR CMA d | 재학습+IR CMA d |',
        '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["target"]} | {r["initial"]["old"]["combined"]:.6f} | {r["initial"]["amp"]["combined"]:.6f} | {r["search"]["old"]["distance"]["combined"]:.6f} | {r["search"]["amp"]["distance"]["combined"]:.6f} |')
    lines += ['', '| 저장된 WAV 방법 | 평균 d |','|---|---:|']
    for name,value in summary['mean_saved_audio_d'].items():
        lines.append(f'| {name} | {value:.6f} |')
    lines += ['', '최종 WAV는 선택 체인을 다시 렌더링하여 실제 파일의 d와 hash를 따로 기록했다.',
        'VST 재현 차이가 있으면 탐색 당시 점수와 저장 WAV 점수가 다를 수 있다.','',
        '### K·예산과 시간','',
        '| 설정 | 평균 탐색 d | 평균 후보 블록 시간 |', '|---|---:|---:|',
        f'| 재학습 K3 / budget90 | {summary["mean_search_d"]["amp"]:.6f} | {summary["new_k3_search_mean_seconds"]:.3f}초 |',
        f'| 재학습 K1 / budget30 | {summary["new_k1_branch_mean_d"]:.6f} | {summary["new_k1_branch_mean_seconds"]:.3f}초 |','',
        'K1은 같은 실행에서 실제 측정한 첫 후보 블록이다. 독립 재실행 벤치마크가 아니며 모델 로드·특징 추출·라이브러리 준비·WAV 저장 시간을 제외한다.',
        'K3는 후보 블록 시간의 합이다. 사용자가 관측한 26초와 측정 범위가 같다고 가정하지 않는다.',
        f'총 탐색 렌더는 {summary["search_render_evaluations"]:,}회이며 진단·청취용 재렌더는 별도다.','',
        '### 원자료와 재현','',
        '- 데이터: `data/synthetic/vst7_amp_ir/`, 캐시·모델·학습 기록: `outputs/vst7_amp_ir/`.',
        f'- 비교 계획·원본 hash·합성 confusion matrix·표본별 JSON: `{output}`.',
        f'- 청취 WAV: `{output}/real/00`부터 `09`까지. 각 폴더에 target, dry, 기존/새 모델 초기·탐색 결과를 저장했다.',
        '- 연동 검사: 33 passed, 14 subtests passed(사용자 VST 환경). 격리 환경의 VST 검사는 Dry pass-through로 실패하여 실제 결과로 사용하지 않았다.',
        '- 기존 모델·데이터·이전 실험 결과와 공유 드라이브 원본은 보존했다.']
    pilot = root/'domain_comparison'
    pilot_files = sorted((pilot/'real').glob('*/result.json'))
    if pilot_files:
        pilot_plan=json.loads((pilot/'plan.json').read_text(encoding='utf-8'))
        assert all(pilot_plan[key]==plan[key] for key in ('old_checkpoint_sha256','amp_checkpoint_sha256','target_sha256','seed','k','budget','amp_context','common_distance_settings','dry_sample_id'))
        old_records=[json.loads(p.read_text(encoding='utf-8')) for p in pilot_files]
        matched={r['target']:r for r in records}
        initial_seconds=float(np.mean([r['search']['amp']['wall_seconds'] for r in old_records]))
        fixed_seconds=float(np.mean([matched[r['target']]['search']['amp']['wall_seconds'] for r in old_records]))
        lines += ['', '### 추가 진단: 지연 로딩된 BLAS 스레드 제한','',
                  '처음 실행에서는 SciPy BLAS가 늦게 로드되어 최초 스레드 제한을 피하고 12개 스레드를 사용했다.',
                  '저장 WAV의 동일 거리 계산 진단에서 추가 라이브러리 로드를 확인하고, 준비 후 BLAS를 1개 스레드로 고정했다.',
                  f'수정 전 완료된 {len(old_records)}개 표본을 보존한 뒤 수정 후 10개 전체를 다시 실행했다. 두 실행의 모델·입력·IR·seed·예산은 같다.',
                  f'중단 전 완료된 같은 표본들의 재학습 모델 K3/90 탐색 wall time: 평균 {initial_seconds:.3f}초 → {fixed_seconds:.3f}초.',
                  f'수정 전 원자료: `{pilot}`. 미완료 표본은 평균과 성능 비교에 넣지 않았다.',
                  '이 진단의 소수 표본과 실행 환경을 전체 데모 성능 보장으로 해석하지 않는다.']
    path=Path(__file__).resolve().parent/'EXPERIMENTS.md'
    text=path.read_text(encoding='utf-8')
    start,end='<!-- AMP_DOMAIN_START -->','<!-- AMP_DOMAIN_END -->'
    report=start+'\n\n'+'\n'.join(lines)+'\n\n'+end
    if start in text:
        before,rest=text.split(start,1);_previous,after=rest.split(end,1)
        text=before+report+after
    else:
        text=text.rstrip()+'\n\n'+report+'\n'
    path.write_text(text.rstrip()+'\n',encoding='utf-8',newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',default='outputs/vst7_drive_only/model/best.pt')
    parser.add_argument('--new',default='outputs/vst7_amp_ir/model/best.pt')
    parser.add_argument('--bare-cache',default='outputs/vst7_drive_only/dataset')
    parser.add_argument('--amp-cache',default='outputs/vst7_amp_ir/dataset')
    parser.add_argument('--targets',default='Demucs/output')
    parser.add_argument('--output',default='outputs/vst7_amp_ir/domain_comparison_blas1')
    parser.add_argument('--budget',type=int,default=90)
    parser.add_argument('--k',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    run(**vars(parser.parse_args()))
