"""Predict seven VST probabilities, full-chain knob seeds and top-K candidates."""
import argparse
import time

import numpy as np
import torch

from .audio import AudioConfig, log_mel, read_audio, segment
from .chain import AUXILIARY, DRIVES, continuous, denormalize, fingerprint, make_candidates
from .cnn import PairCNN
from .data import write_json


def load_predictor(checkpoint, device='cpu'):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if (state['format_version'] != 2 or fingerprint(state['catalog']) != state['catalog_hash'] or
            state['classes'] != list(DRIVES)):
        raise ValueError('Expected VST7 format_version=2 checkpoint')
    model = PairCNN(state['catalog'], **state['model_config']).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model, AudioConfig(**state['audio_config']), state


def decode(output, index, definitions, settings=None):
    result = {'probabilities': dict(zip(DRIVES, output['logits'][index].softmax(0).tolist())),
              'presence': dict(zip(AUXILIARY, output['presence'][index].sigmoid().tolist())),
              'values': {t: {p['name']: denormalize(float(v), p)
                             for p,v in zip(continuous(d), output['knobs'][t][index])}
                         for t,d in definitions.items()},
              'score_definition': 'drive softmax probability; not audio similarity'}
    if settings:
        values = output['distance'][index].tolist()
        result['dry_target_distance_estimate'] = {
            'spectral': values[0]*settings['spectral_scale'],
            'dynamics': values[1]*settings['dynamics_scale'],
            'combined': (values[0]+values[1])/2,
            'scope': 'estimate of d(input dry, reference); NOT a rendered candidate score'}
    return result


def predict_pair(model, config, dry, reference, settings=None):
    start = time.perf_counter()
    device = next(model.parameters()).device
    a = torch.from_numpy(log_mel(dry, config))[None].to(device)
    b = torch.from_numpy(log_mel(reference, config))[None].to(device)
    with torch.inference_mode():
        result = decode(model(a,b), 0, model.definitions, settings)
    result['inference_seconds'] = time.perf_counter()-start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dry', required=True, help='Generated dataset dry/ file, already peak -12 dBFS')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--owned', nargs='*', default=None)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--output', required=True)
    parser.add_argument('--refine-budget', type=int)
    parser.add_argument('--refine-restarts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    import soundfile as sf
    native, rate = sf.read(args.dry,dtype='float32',always_2d=True)
    native_peak = float(np.max(np.abs(native)))
    if not native_peak or abs(20*np.log10(native_peak)+12)>.001:
        raise ValueError('Use manifest.render_input from generated dry/, already peak -12 dBFS')
    native = native[:,0] if native.shape[1]==1 else native.T
    model, config, state = load_predictor(args.checkpoint)
    dry = segment(read_audio(args.dry, config.sample_rate), config)
    reference = segment(read_audio(args.reference, config.sample_rate), config, reject_silence=False)
    result = predict_pair(model, config, dry, reference, state['distance_settings'])
    result['candidates'] = make_candidates(result, args.owned, args.k, model.definitions)
    result['generated_dry_path'] = args.dry
    result['format_version'] = 2
    result['distance_settings'] = state['distance_settings']
    result['native_render_rate'] = rate
    if args.refine_budget is not None:
        from .refine import refine_candidates
        from .renderer import Renderer, activation_preflight
        if not result['candidates'] or args.refine_budget < len(result['candidates']):
            raise ValueError('Refinement requires at least one render per candidate')
        renderer = Renderer(model.definitions)
        status = activation_preflight(renderer, native, rate, model.definitions)
        if not status['available']:
            raise RuntimeError('VST activation check failed: dry pass-through or silence')
        result['renderer_preflight'] = status
        result['refined'] = refine_candidates(native, reference, rate, config,
                                               result['candidates'], args.refine_budget,
                                               state['distance_settings'], renderer,
                                               seed=args.seed, restarts=args.refine_restarts,
                                               definitions=model.definitions)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
