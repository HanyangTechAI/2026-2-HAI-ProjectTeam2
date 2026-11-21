"""Append the team's fixed cabinet to audited drive-only Wet; preserve Dry/splits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

from effects.render import apply_amp
from .amp import NO_AMP, current_context
from .audio import validate_rendered_pair
from .chain import catalog, fingerprint
from .data import validate_chain, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source,output,checkpoint):
    source,output = Path(source).resolve(),Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(repo/'data/synthetic') or output.exists():
        raise ValueError('Use a new physical local data/synthetic output directory')
    from scripts.generate_step1_dataset import _check_plain_path
    _check_plain_path(output)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['amp_id'] != NO_AMP:
        raise ValueError('Input must be drive-only WITHOUT a cabinet (prevent double IR)')
    audit = json.loads((source/'render_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed':
        raise ValueError('Source audit did not pass')
    from .infer import load_predictor
    _model,_config,state = load_predictor(checkpoint)
    if state['source_manifest_sha256'] != sha(source/'manifest.json'):
        raise ValueError('Baseline checkpoint does not describe this source dataset')
    context,definitions = current_context(),state['catalog']
    for name in ('dry','audio','labels'):
        (output/name).mkdir(parents=True)
    dry = {}
    for row in manifest['dry_id_map']:
        relative = Path(row['render_input'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'dry':
            raise ValueError('Invalid generated Dry path')
        shutil.copy2(source/relative,output/relative)
        x,rate = sf.read(source/relative,dtype='float32',always_2d=True)
        dry[row['dry_sample_id']] = (x.T,rate)
        assert sha(source/relative) == sha(output/relative)
    measurements = []
    for index,label_path in enumerate(sorted((source/'labels').glob('*.json'))):
        label = json.loads(label_path.read_text(encoding='utf-8'))
        validate_chain(label['ground_truth_chain'],definitions,NO_AMP)
        x,rate = dry[label['dry_sample_id']]
        y,sr = sf.read(source/'audio'/f"{label['target_id']}.wav",dtype='float32',always_2d=True)
        if sr != rate or x.shape != y.T.shape:
            raise ValueError('Source Dry/Wet rate or shape mismatch')
        validate_rendered_pair(x,y.T,label['target_id'])
        wet = np.asarray(apply_amp(y.T,rate,context['amp_id'],amp=context['definition']),dtype=np.float32)
        measurements.append(validate_rendered_pair(x,wet,label['target_id']))
        sf.write(output/'audio'/f"{label['target_id']}.wav",wet.T,rate,subtype='FLOAT')
        label['ground_truth_chain']['amp_id'] = context['amp_id']
        label['file_path'] = 'audio/'+label['target_id']+'.wav'
        write_json(output/'labels'/label_path.name,label)
        if (index+1)%250 == 0:
            print(json.dumps({'cabinet_pairs':index+1}),flush=True)
    if len(measurements) != manifest['completed_pairs']:
        raise ValueError('Source count mismatch')
    def update_amp(value):
        if isinstance(value,dict):
            return {k:(context['amp_id'] if k=='amp_id' else update_amp(v)) for k,v in value.items()}
        if isinstance(value,list):
            return [update_amp(v) for v in value]
        return value
    manifest = update_amp(copy.deepcopy(manifest))
    manifest.update(amp_context=context,amp_context_hash=fingerprint(context),
                    catalog=definitions,catalog_hash=fingerprint(definitions),
                    algorithm='drive-only-fixed-cabinet-postprocess-v1',
                    source_manifest_sha256=sha(source/'manifest.json'))
    write_json(output/'manifest.json',manifest)
    write_json(output/'render_audit.json',{
        'status':'passed','pairs':len(measurements),'originals':len(dry),
        'all_non_silent':True,'all_changed_from_dry':True,'repair_count':0,
        'maximum_wet_peak':max(v['peak'] for v in measurements),
        'minimum_wet_peak':min(v['peak'] for v in measurements),
        'source_render_audit_sha256':sha(source/'render_audit.json')})
    provenance = {'source':str(source),'source_manifest_sha256':sha(source/'manifest.json'),
                  'source_render_audit_sha256':sha(source/'render_audit.json'),
                  'operation':'team apply_amp on previously verified bare-drive Wet; no VST rerender',
                  'amp_context':context,'amp_context_hash':fingerprint(context),
                  'baseline_checkpoint_sha256':sha(checkpoint),'catalog_hash':fingerprint(definitions),
                  'dry_policy':'byte-identical generated Dry; no rescaling',
                  'split_policy':'same recording IDs and seed=42 as baseline',
                  'wet_policy':'FLOAT WAV; fixed IR gain only; no peak or loudness normalization',
                  'renderer_sha256':sha(repo/'effects/render.py'),
                  'pairs':len(measurements),'originals':len(dry)}
    write_json(output/'generation_provenance.json',provenance)
    print(json.dumps(provenance,indent=2),flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='data/synthetic/vst7_drive_only')
    parser.add_argument('--output',default='data/synthetic/vst7_amp_ir')
    parser.add_argument('--checkpoint',default='outputs/vst7_drive_only/model/best.pt')
    build(**vars(parser.parse_args()))
