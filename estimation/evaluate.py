"""Full-chain evaluation with exact combined d and explicit renderer preflight."""
import argparse
import copy
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

from .audio import AudioConfig, segment
from .chain import DRIVES, continuous, make_candidates, normalize
from .data import PairDataset, load_manifest, write_json
from .distance import compare
from .infer import decode, load_predictor


def classification_metrics(truth, predicted, classes=DRIVES):
    matrix = np.zeros((len(classes),len(classes)),dtype=int)
    for a,b in zip(truth,predicted):
        matrix[classes.index(a),classes.index(b)] += 1
    true_positive = matrix.diagonal()
    denominator = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*true_positive,denominator,out=np.zeros(len(classes)),where=denominator>0)
    return {'accuracy':float(true_positive.sum()/max(1,matrix.sum())), 'macro_f1':float(f1.mean()),
            'classes':list(classes),'confusion_matrix':matrix.tolist(),
            'confusion_matrix_axes':'rows=truth, columns=prediction',
            'per_class_recall':dict(zip(classes,np.divide(true_positive,matrix.sum(1),
                out=np.zeros(len(classes)),where=matrix.sum(1)>0).tolist()))}


class AudioEvaluation:
    def __init__(self,cache):
        from .renderer import Renderer
        self.cache = Path(cache)
        self.rows,self.metadata = load_manifest(cache)
        self.config = AudioConfig(**self.metadata['audio_config'])
        self.target = np.load(self.cache/'wet_audio.npy',mmap_mode='r')
        self.dry = np.load(self.cache/'dry_audio.npy',mmap_mode='r')
        self.sources = {r['dry_sample_id']:r for r in self.metadata['dry_sources']}
        from .amp import context_from_record
        self.renderer = Renderer(self.metadata['catalog'],context_from_record(self.metadata))
        self.render_attempts = 0
        self.transient_retries = 0

    def render(self,row,chain):
        source = self.sources[row['dry_sample_id']]
        native = np.load(self.cache/'native_dry'/f"{row['dry_sample_id']}.npy")
        for attempt in range(3):
            self.render_attempts += 1
            try:
                rendered = self.renderer(native,source['source_rate'],chain)
                break
            except ValueError as error:
                if attempt==2 or not any(t in str(error) for t in ('pass-through','Silent VST')):
                    raise
                self.transient_retries += 1
        if rendered.ndim==2:
            rendered = rendered.mean(0)
        if source['source_rate']!=self.config.sample_rate:
            rendered = librosa.resample(rendered,orig_sr=source['source_rate'],target_sr=self.config.sample_rate)
        return segment(rendered,self.config,reject_silence=False)

    def score(self,row,chain):
        return compare(self.render(row,chain),self.target[row['wet_index']],
                       self.config.sample_rate,self.metadata['distance_settings'])


def renderer_status(cache,refresh=False):
    cache = Path(cache)
    path = cache/'renderer_preflight.json'
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding='utf-8'))
    try:
        evaluator = AudioEvaluation(cache)
        checks = []
        for effect in DRIVES:
            row = next(r for r in evaluator.rows if r['drive']==effect and r['split']=='train')
            actual = evaluator.render(row,row['ground_truth_chain'])
            target = evaluator.target[row['wet_index']]
            relative = float(np.sqrt(np.sum((actual-target).astype('float64')**2)/
                                    max(float(np.sum(target.astype('float64')**2)),1e-30)))
            score = compare(actual,target,evaluator.config.sample_rate,evaluator.metadata['distance_settings'])
            checks.append({'drive':effect,'sample_id':row['sample_id'],
                           'relative_rms_error':relative,'oracle_d':score})
        success = all(r['relative_rms_error']<.001 for r in checks)
        # Availability and exact dataset reproduction are DIFFERENT questions.
        # Even with a renderer mismatch, d(rendered candidate, saved target) is
        # measurable. Report oracle d alongside it; never attribute all of that
        # discrepancy to the CNN or pretend that the oracle is zero.
        row = next(r for r in evaluator.rows if r['split']=='train')
        native = np.load(cache/'native_dry'/f"{row['dry_sample_id']}.npy")
        source_rate = evaluator.sources[row['dry_sample_id']]['source_rate']
        activation_checks = []
        for drive in DRIVES:
            values = {p['name']:p['default'] for p in continuous(evaluator.metadata['catalog'][drive])}
            audio = evaluator.renderer(native,source_rate,{'amp_id':evaluator.renderer.amp_context['amp_id'],
                                                          'chain':[{'effect_type':drive,'values':values}]})
            activation_checks.append({'drive':drive,'max_change_from_dry':float(np.max(np.abs(audio-native))),
                                      'output_peak':float(np.max(np.abs(audio)))})
        active = all(p['max_change_from_dry']>1e-7 and p['output_peak']>1e-7 for p in activation_checks)
        result = {'available':active,'dataset_reproduced':success,
                  'activation_checks':activation_checks,
                  'plugin_file':evaluator.renderer.plugin_file,'checks':checks,
                  'reason':'native-rate ground-truth reproduction passed' if success else
                           'rendered d includes a renderer/dataset mismatch; compare the reported oracle baseline',
                  'tolerance_relative_rms':.001}
        if not active:
            result['reason'] = 'VST returned silence or unchanged dry; check activation and process permissions'
    except (OSError,RuntimeError,ValueError,AttributeError) as exc:
        result = {'available':False,'reason':f'{type(exc).__name__}: {exc}'}
    write_json(path,result)
    return result


def render_validation_probe(model,cache,limit=14):
    """Fixed validation samples only; never use test for training/model choice."""
    dataset = PairDataset(cache,'validation')
    selected = []
    for drive_index,drive in enumerate(DRIVES):
        originals = {}
        for i,row in enumerate(dataset.rows):
            if row['drive']==drive:
                originals.setdefault(row['recording_id'],i)
        indices = list(originals.values())
        per_drive = min(max(1,limit//7),len(indices))
        selected += [indices[(drive_index*per_drive+j)%len(indices)] for j in range(per_drive)]
    batch = torch.utils.data.default_collate([dataset[i] for i in selected])
    evaluator = AudioEvaluation(cache)
    model.eval()
    records = []
    with torch.inference_mode():
        output = model(batch['dry'],batch['wet'])
        for i,idx in enumerate(selected):
            row = dataset.rows[idx]
            prediction = decode(output,i,model.definitions,amp_id=model.amp_context['amp_id'])
            candidate = make_candidates(prediction,k=1,definitions=model.definitions)[0]
            score = evaluator.score(row,candidate['chain'])
            baseline = copy.deepcopy(candidate['chain'])
            for item in baseline['chain']:
                item['values'] = {p['name']:p['default'] for p in continuous(model.definitions[item['effect_type']])}
            records.append({'sample_id':row['sample_id'],'cnn_d':score['combined'],
                            'recording_id':row['recording_id'],
                            'oracle_d':evaluator.score(row,row['ground_truth_chain'])['combined'],
                            'same_predicted_structure_default_knobs_d':evaluator.score(row,baseline)['combined']})
    dataset.close()
    return {'available':True,'samples':len(records),
            'transient_render_retries':evaluator.transient_retries,
            'mean_cnn_d':float(np.mean([r['cnn_d'] for r in records])),
            'mean_oracle_d':float(np.mean([r['oracle_d'] for r in records])),
            'mean_same_predicted_structure_default_knobs_d':float(np.mean([r['same_predicted_structure_default_knobs_d'] for r in records])),
            'selection':'fixed validation probe; not the complete validation split',
            'records':records}


def render_subset(rows,limit=70):
    """Fixed class-balanced sampling without looking at predictions or scores."""
    if limit<0 or limit>=len(rows):
        return {r['sample_id'] for r in rows}
    if limit==0:
        return set()
    if limit%len(DRIVES):
        raise ValueError('render_samples must be a multiple of seven, zero, or -1 for all')
    selected = set()
    for drive_index,drive in enumerate(DRIVES):
        groups = {}
        for row in rows:
            if row['drive']==drive:
                groups.setdefault(row['recording_id'],[]).append(row)
        originals = sorted(groups)
        count = min(limit//7,sum(len(v) for v in groups.values()))
        for j in range(count):
            group = groups[originals[(j+2*drive_index)%len(originals)]]
            variant = (j//len(originals)+drive_index)%len(group)
            selected.add(group[variant]['sample_id'])
    return selected


def evaluate(cache,checkpoint,output,split='test',batch_size=16,render_samples=70):
    torch.set_num_threads(4)
    model,config,state = load_predictor(checkpoint)
    dataset = PairDataset(cache,split)
    from .amp import context_from_record
    if model.amp_context != context_from_record(dataset.metadata):
        raise ValueError('Checkpoint/cache amp contexts disagree')
    selected_for_render = render_subset(dataset.rows,render_samples)
    if (state['source_manifest_sha256']!=dataset.metadata['source_manifest_sha256'] or
            state.get('source_render_audit_sha256')!=dataset.metadata['source_render_audit_sha256'] or
            state['catalog_hash']!=dataset.metadata['catalog_hash'] or
            state['audio_config']!=dataset.metadata['audio_config']):
        raise ValueError('Checkpoint belongs to another source dataset')
    status = renderer_status(cache)
    evaluator = AudioEvaluation(cache) if status['available'] else None
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1,user_api='blas')
    records,offset = [],0
    knob_errors = {t:[] for t in model.definitions}
    default_knob_errors = {t:[] for t in model.definitions}
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            result = model(batch['dry'],batch['wet'])
            for i in range(len(batch['label'])):
                row = dataset.rows[offset]
                prediction = decode(result,i,model.definitions,state['distance_settings'],model.amp_context['amp_id'])
                candidates = make_candidates(prediction,k=3,definitions=model.definitions)
                truth_items = {p['effect_type']:p['values'] for p in row['ground_truth_chain']['chain']}
                for t,values in truth_items.items():
                    for p in continuous(model.definitions[t]):
                        knob_errors[t].append(abs(normalize(prediction['values'][t][p['name']],p)-normalize(values[p['name']],p)))
                        default_knob_errors[t].append(abs(normalize(p['default'],p)-normalize(values[p['name']],p)))
                raw = row['raw_dry_target_distance']
                settings = state['distance_settings']
                true_d = (raw[0]/settings['spectral_scale']+raw[1]/settings['dynamics_scale'])/2
                item = {'sample_id':row['sample_id'],'recording_id':row['recording_id'],'dry_path':row['dry_path'],
                        'truth':row['drive'],'predicted':candidates[0]['effect_type'],
                        'ground_truth_chain':row['ground_truth_chain'], 'prediction':prediction,
                        'exact_structure_correct':candidates[0]['effect_type']==row['drive'],
                        'candidates':candidates,'measured_dry_target_d':true_d,
                        'rendered_distances':None}
                if evaluator and row['sample_id'] in selected_for_render:
                    scores = [evaluator.score(row,c['chain']) for c in candidates]
                    baseline = copy.deepcopy(candidates[0]['chain'])
                    for p in baseline['chain']:
                        p['values'] = {q['name']:q['default'] for q in continuous(model.definitions[p['effect_type']])}
                    item['rendered_distances'] = {'cnn_top1':scores[0],
                        'cnn_top3_best':min(scores,key=lambda d:d['combined']),
                        'same_predicted_structure_default_knobs':evaluator.score(row,baseline),
                        'oracle_full_chain':evaluator.score(row,row['ground_truth_chain'])}
                    for candidate,score in zip(candidates,scores):
                        candidate['audio_distance'] = score
                records.append(item)
                offset += 1
            if offset%64==0 or offset==len(dataset):
                print(json.dumps({'evaluating':split,'completed':offset,'total':len(dataset)}),flush=True)
    summary = classification_metrics([r['truth'] for r in records],[r['predicted'] for r in records])
    rendered_records = [r for r in records if r['rendered_distances'] is not None]
    summary.update(split=split,samples=len(records),checkpoint_epoch=state['epoch'],
                   original_recordings=len({r['recording_id'] for r in dataset.rows}),
                   top3_accuracy=float(np.mean([r['truth'] in [c['effect_type'] for c in r['candidates']] for r in records])),
                   conditional_normalized_knob_mae={t:float(np.mean(v)) if v else None for t,v in knob_errors.items()},
                   conditional_default_normalized_knob_mae={t:float(np.mean(v)) if v else None for t,v in default_knob_errors.items()},
                   exact_structure_accuracy=float(np.mean([r['exact_structure_correct'] for r in records])),
                   knob_metric_scope='each truth-active effect head, conditioned on true effect identity',
                   dry_target_d_estimate_mae=float(np.mean([abs(r['prediction']['dry_target_distance_estimate']['combined']-r['measured_dry_target_d']) for r in records])),
                   distance_settings=state['distance_settings'],renderer=status,
                   rendered_samples=len(rendered_records),
                   transient_render_retries=evaluator.transient_retries if evaluator else 0,
                   render_attempts=evaluator.render_attempts if evaluator else 0,
                   rendered_original_recordings=len({r['recording_id'] for r in rendered_records}),
                   render_sample_policy='class-balanced round-robin originals; selected before predictions',
                   mean_rendered_distances=None)
    if rendered_records:
        summary['mean_rendered_distances'] = {name:{metric:float(np.mean([r['rendered_distances'][name][metric] for r in rendered_records]))
                for metric in ('combined','spectral','dynamics')}
                for name in rendered_records[0]['rendered_distances']}
        a=summary['mean_rendered_distances']['cnn_top1']['combined']
        b=summary['mean_rendered_distances']['same_predicted_structure_default_knobs']['combined']
        summary['top1_d_reduction_vs_default_knobs'] = 1-a/b if b else None
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    write_json(output/'summary.json',summary)
    dataset.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--render-samples',type=int,default=70,
                        help='Class-balanced actual render comparisons; -1 for every sample')
    print(json.dumps(evaluate(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':
    main()
