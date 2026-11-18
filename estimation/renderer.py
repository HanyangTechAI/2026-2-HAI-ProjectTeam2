"""Optional inference renderer. Loading WAVs/training never needs a VST license."""
import os
from pathlib import Path

import numpy as np

from .chain import DRIVES, MODEL_NAMES, catalog, continuous


def plugin_path():
    folder = Path(os.environ.get('HAI_VST_DIR', r'C:\Program Files\Common Files\VST3\Audified'))
    if folder.is_file() or folder.suffix.lower() == '.vst3':
        candidates = [folder]
    else:
        bundle = folder / 'MultiDrive PRO.vst3'
        candidates = [bundle/'Contents/x86_64-win/MultiDrive PRO.vst3', bundle]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('MultiDrive PRO not installed; set HAI_VST_DIR. Trial activation is manual.')


def activation_preflight(renderer, dry, rate, definitions):
    """Fresh checks of all bare drives; never open UI or activate a trial."""
    checks = []
    for drive in DRIVES:
        chain = {'amp_id': 'fixed_amp_v1', 'chain': [{'effect_type': drive,
                 'values': {p['name']: p['default'] for p in continuous(definitions[drive])}}]}
        wave = np.asarray(renderer(dry, rate, chain))
        if wave.shape != dry.shape or not np.isfinite(wave).all():
            raise ValueError('VST preflight returned an invalid waveform')
        change, peak = float(np.max(np.abs(wave-dry))), float(np.max(np.abs(wave)))
        checks.append({'drive': drive, 'max_change_from_dry': change, 'output_peak': peak,
                       'active': change > 1e-6 and peak > 1e-7})
    return {'available': all(r['active'] for r in checks), 'checks': checks,
            'render_evaluations': len(checks), 'native_rate': rate,
            'plugin_file': getattr(renderer, 'plugin_file', None),
            'scope': 'fresh activation check; saved target reproduction is measured separately per sample'}


class Renderer:
    def __init__(self, definitions=None):
        from pedalboard import load_plugin
        self.definitions = definitions or catalog()
        self.plugin = load_plugin(str(plugin_path()))
        self.plugin_file = str(plugin_path())

    def __call__(self, dry, rate, chain):
        from pedalboard import Pedalboard
        from effects.render import build_effect
        plugins = []
        for item in chain['chain']:
            name, values = item['effect_type'], item['values']
            if name not in DRIVES:
                plugins.extend(build_effect(name, values))
                continue
            plugin = self.plugin
            if hasattr(plugin, 'bypass'):
                plugin.bypass = False
            plugin.pedal_model = MODEL_NAMES[name]
            # Reset known controls, including nonsampled input gain, each render.
            if hasattr(plugin, 'input_gain_db'):
                plugin.input_gain_db = 0.0
            for p in self.definitions[name]['params']:
                setattr(plugin, p['maps_to'].split('.')[-1], values[p['name']])
            plugins.append(plugin)
        audio = np.asarray(Pedalboard(plugins)(dry, rate, reset=True), dtype=np.float32)
        if audio.shape != dry.shape or not np.isfinite(audio).all():
            raise ValueError('Renderer changed shape or returned nonfinite values')
        return audio


def main():
    import argparse
    import json
    import soundfile as sf
    from .data import write_json
    from .distance import raw_distances
    parser = argparse.ArgumentParser(description='Compare seven saved targets to native-rate ground-truth rerenders')
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root = Path(args.dataset)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    source = manifest['dry_id_map'][0]
    path = root/source['render_input']
    dry,rate = sf.read(path,dtype='float32',always_2d=True)
    dry = dry[:,0] if dry.shape[1]==1 else dry.T
    renderer = Renderer()
    records = []
    for drive in DRIVES:
        name = 'wet_'+Path(source['render_input']).stem+'_'+drive+'_001'
        label = json.loads((root/'labels'/f'{name}.json').read_text(encoding='utf-8'))
        target,sr = sf.read(root/'audio'/f'{name}.wav',dtype='float32')
        actual = renderer(dry,rate,label['ground_truth_chain'])
        repeated = renderer(dry,rate,label['ground_truth_chain'])
        fresh = Renderer()(dry,rate,label['ground_truth_chain'])
        if sr!=rate or target.shape!=actual.shape:
            raise ValueError('Source/target native format mismatch')
        from scipy.signal import correlate
        correlations = correlate(actual,target,mode='full',method='fft')
        middle = len(target)-1
        lag = int(np.argmax(correlations[middle-256:middle+257])-256)
        records.append({'drive':drive,'relative_rms_error':float(np.linalg.norm(actual-target)/max(float(np.linalg.norm(target)),1e-12)),
                        'repeat_relative_rms_error':float(np.linalg.norm(actual-repeated)/max(float(np.linalg.norm(actual)),1e-12)),
                        'fresh_relative_rms_error':float(np.linalg.norm(fresh-target)/max(float(np.linalg.norm(target)),1e-12)),
                        'repeat_raw_distances':raw_distances(actual,repeated,rate),
                        'cross_correlation_lag_samples':lag,
                        'target_peak':float(np.max(np.abs(target))),
                        'raw_oracle_distances':raw_distances(actual,target,rate),
                        'output_peak':float(np.max(np.abs(actual)))})
    result = {'plugin_loaded':True,'dataset_reproduced':all(r['relative_rms_error']<.001 for r in records),
              'native_rate':rate,'plugin':renderer.plugin_file,'checks':records}
    write_json(args.output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
