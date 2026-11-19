"""Render fresh drive-only pairs with the team's unified generator.

Reuse only native generated Dry from an old local cache, never old mixed Wet.
The unified generator writes to a new physical local temporary directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from scripts.generate_step1_dataset import Config, DRIVES, Plan, Stage
from effects.render import render_chain, load_effects, VST_BUFFER_SIZE, VST_WARMUP_SECONDS
from .audio import validate_rendered_pair


def audit_existing(dataset, repair=False):
    """Audit every saved WAV and optionally retry transient bypass outputs.

    Writes only to the explicitly selected local generated folder. A Drive
    junction cannot pass the resolved path and plain-directory checks.
    """
    import tempfile
    from scripts.generate_step1_dataset import _check_plain_path
    root = Path(dataset).resolve()
    local = Path(__file__).resolve().parents[1]/'data/synthetic'
    temp = Path(tempfile.gettempdir()).resolve()
    if not (root.is_relative_to(local) or root.is_relative_to(temp)) or root in (local,temp):
        raise ValueError('Audit/repair is restricted to a generated local dataset')
    _check_plain_path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    mapping = {r['dry_sample_id']:r for r in manifest['dry_id_map']}
    dry = {}
    for key,row in mapping.items():
        x,rate = sf.read(root/row['render_input'],dtype='float32',always_2d=True)
        dry[key] = (x.T,rate)
    audit_path = root/'render_audit.json'
    previous = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
    checked,repairs = [],list(previous.get('repairs',[]))
    for label_path in sorted((root/'labels').glob('*.json')):
        label = json.loads(label_path.read_text(encoding='utf-8'))
        x,rate = dry[label['dry_sample_id']]
        path = root/'audio'/f"{label['target_id']}.wav"
        y,sr = sf.read(path,dtype='float32',always_2d=True)
        if rate!=sr:
            raise ValueError('Saved Dry/Wet sample rates disagree')
        y = y.T
        try:
            measured = validate_rendered_pair(x,y,label['target_id'])
        except ValueError as original_error:
            if not repair:
                raise
            old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            for attempt in range(1,4):
                replacement = np.asarray(render_chain(x.copy(),rate,label['ground_truth_chain']))
                try:
                    measured = validate_rendered_pair(x,replacement,label['target_id'])
                except ValueError:
                    if attempt==3:
                        raise
                    continue
                temporary = path.with_suffix('.repair.wav')
                with temporary.open('xb') as stream:
                    sf.write(stream,replacement.T,rate,format='WAV',subtype='FLOAT')
                temporary.replace(path)
                repairs.append({'sample_id':label['target_id'],'reason':str(original_error),
                                'attempts':attempt,'old_wav_sha256':old_hash,
                                'new_wav_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                break
        checked.append(measured)
    if len(checked)!=manifest['completed_pairs']:
        raise ValueError('Audit and manifest counts disagree')
    audit = {'status':'passed','pairs':len(checked),'originals':len(dry),
             'all_non_silent':True,'all_changed_from_dry':True,
             'minimum_wet_peak':min(r['peak'] for r in checked),
             'maximum_wet_peak':max(r['peak'] for r in checked),
             'minimum_max_change_from_dry':min(r['max_change_from_dry'] for r in checked),
             'repairs':repairs,'repair_count':len(repairs)}
    audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    provenance_path = root/'generation_provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        provenance['post_render_audit'] = audit
        provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
    return audit


def generate(source_cache, input_staging, output, variations=5, seed=42):
    cache, staging = Path(source_cache).resolve(), Path(input_staging).resolve()
    if staging.exists():
        raise FileExistsError('Use a new input staging directory')
    metadata = json.loads((cache/'dataset.json').read_text(encoding='utf-8'))
    staging.mkdir(parents=True)
    namespaces = set()
    sources = []
    for source in metadata['dry_sources']:
        relative = Path(source['relative_path']) if 'relative_path' in source else Path(source['recording_id'].split('/',1)[1])
        namespace = source['recording_id'].split('/',1)[0]
        if relative.is_absolute() or '..' in relative.parts or not namespace.isidentifier():
            raise ValueError('Unsafe source identity')
        path = staging/namespace/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        dry = np.load(cache/'native_dry'/f"{source['dry_sample_id']}.npy")
        if abs(20*np.log10(max(float(np.max(np.abs(dry))),1e-30))+12) > .001:
            raise ValueError('Expected cached generated Dry at -12 dBFS')
        with path.open('xb') as stream:
            sf.write(stream,dry if dry.ndim==1 else dry.T,source['source_rate'],format='WAV',subtype='FLOAT')
        namespaces.add(namespace)
        sources.append({'dry_sample_id':source['dry_sample_id'],'recording_id':source['recording_id'],
                        'source_rate':source['source_rate'],'original_generated_dry_sha256':source['sha256'],
                        'staged_wav_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    config = Config(variations=variations,seed=seed,effects=DRIVES,
                    policy=(Stage(DRIVES),),per_drive=True,dry_peak_db=-12.0)
    plan = Plan([(name,staging/name) for name in sorted(namespaces)],config,output)
    first = plan.sources[0]
    dry, rate = sf.read(first.path,dtype='float32',always_2d=True)
    dry = dry.T
    definitions = load_effects()
    checks = []
    for drive in DRIVES:
        values = {p['name']:p['default'] for p in definitions[drive]['params']}
        chain = {'amp_id':'fixed_amp_v1','chain':[{'effect_type':drive,'values':values}]}
        wet = np.asarray(render_chain(dry.copy(),rate,chain))
        peak, change = float(np.max(np.abs(wet))),float(np.max(np.abs(wet-dry)))
        if wet.shape!=dry.shape or not np.isfinite(wet).all() or peak<1e-7 or change<1e-6:
            raise RuntimeError(f'{drive}: VST activation/finite/shape check failed')
        repeated = np.asarray(render_chain(dry.copy(),rate,chain))
        relative = float(np.sqrt(np.sum((wet-repeated).astype('float64')**2)/max(np.sum(wet.astype('float64')**2),1e-30)))
        if relative>.001:
            raise RuntimeError(f'{drive}: repeated rendering is not stable ({relative})')
        checks.append({'drive':drive,'peak':peak,'max_change_from_dry':change,'repeat_relative_rms_error':relative})
    print(json.dumps({'preflight':'passed','originals':len(sources),'pairs':plan.summary()['pairs'],
                      'output':str(plan.output),'checks':checks}),flush=True)
    started = time.perf_counter()
    manifest = plan.execute()
    audit = audit_existing(plan.output,repair=True)
    provenance = {'source_cache_manifest_sha256':metadata['source_manifest_sha256'],
                  'reused':'native generated Dry only; all Wet freshly rendered',
                  'sources':sources,'preflight':checks,'seconds':time.perf_counter()-started,
                  'renderer_sha256':hashlib.sha256(Path('effects/render.py').read_bytes()).hexdigest(),
                  'generator_sha256':hashlib.sha256(Path('scripts/generate_step1_dataset.py').read_bytes()).hexdigest(),
                  'vst_buffer_size':VST_BUFFER_SIZE,'vst_warmup_seconds':VST_WARMUP_SECONDS,
                  'post_render_audit':audit}
    (plan.output/'generation_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'completed_pairs':manifest['completed_pairs'],'seconds':provenance['seconds'],
                      'output':str(plan.output)}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache',required=True)
    parser.add_argument('--input-staging',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--variations',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    generate(**vars(parser.parse_args()))


if __name__=='__main__':
    main()
