"""Predict drive candidates from a CNN checkpoint and optionally refine with CMA-ES."""
import argparse
import time

import torch

from .audio import AudioConfig, log_mel, read_audio, segment
from .cnn import PairCNN
from .chain import catalog, continuous, denormalize, discrete, fingerprint, make_candidates
from .data import write_json


def load_predictor(checkpoint, device='cpu'):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state['format_version'] != 1 or fingerprint(catalog()) != state['catalog_hash']:
        raise ValueError('Unsupported checkpoint or changed renderer schema')
    model = PairCNN(state['catalog'], **state['model_config']).to(device)
    if model.classes != state['classes']:
        raise ValueError('Checkpoint class order mismatch')
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model, AudioConfig(**state['audio_config']), state


def predict_pair(model, config, dry, reference):
    if model.task != 'joint':
        raise ValueError('Single-task checkpoints use estimation.experiments evaluate')
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    dry_mel = torch.from_numpy(log_mel(dry, config))[None].to(device)
    ref_mel = torch.from_numpy(log_mel(reference, config))[None].to(device)
    with torch.inference_mode():
        output = model(dry_mel, ref_mel)
    probabilities = output['logits'][0].softmax(0).cpu().tolist()
    values = {}
    for t, definition in model.definitions.items():
        values[t] = {p['name']: p['default'] for p in definition['params']}
        for p, value in zip(continuous(definition), output['continuous'].get(t, [[]])[0]):
            values[t][p['name']] = denormalize(float(value), p)
        for p in discrete(definition):
            logits = output['discrete'][f'{t}__{p["name"]}'][0]
            values[t][p['name']] = bool(logits[0] >= 0) if p['type'] == 'boolean' else p['options'][int(logits.argmax())]
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return {'probabilities': dict(zip(model.classes, probabilities)), 'values': values,
            'inference_seconds': time.perf_counter() - start,
            'score_definition': 'unconditional class softmax probability; not audio similarity'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dry', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--owned', nargs='*', default=[])
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--start-sec', type=float, default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--refine-budget', type=int, help='Run CMA-ES with this total render budget')
    p.add_argument('--rendered-output', help='Final FLOAT WAV path; requires --refine-budget')
    args = p.parse_args()
    model, config, _ = load_predictor(args.checkpoint, args.device)
    start = round(args.start_sec * config.sample_rate)
    dry = segment(read_audio(args.dry, config.sample_rate), config, start)
    ref = segment(read_audio(args.reference, config.sample_rate), config, start, reject_silence=False)
    prediction = predict_pair(model, config, dry, ref)
    prediction['candidates'] = make_candidates(prediction, args.owned, args.k, model.definitions)
    if args.rendered_output and args.refine_budget is None:
        raise ValueError('--rendered-output requires --refine-budget')
    if args.refine_budget is not None:
        from .refine import refine_candidates
        result = refine_candidates(dry, ref, config.sample_rate, prediction['candidates'], args.refine_budget)
        audio = result.pop('_audio')
        if args.rendered_output:
            from pathlib import Path
            import soundfile as sf
            Path(args.rendered_output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.rendered_output, audio, config.sample_rate, subtype='FLOAT')
            result['rendered_audio'] = args.rendered_output
        prediction['refined'] = result
    write_json(args.output, prediction)


if __name__ == '__main__':
    main()
