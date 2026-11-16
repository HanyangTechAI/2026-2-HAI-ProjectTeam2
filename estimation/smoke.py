"""Small end-to-end FUNCTIONAL check with temporary signals, not guitar accuracy."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .audio import AudioConfig
from .chain import chain_instance, checked_render, make_candidates
from .data import load_manifest, pair_audio, prepare, write_json
from .evaluate import evaluate_rows
from .infer import load_predictor, predict_pair
from .train import train


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = AudioConfig(sample_rate=16000, duration_sec=0.6, n_fft=512,
                         hop_length=128, n_mels=32)
    sources = []
    for i, freq in enumerate((110, 147, 196, 247, 294, 330)):
        t = np.arange(config.samples + 800) / config.sample_rate
        signal = (0.18 * np.sin(2 * np.pi * freq * t) +
                  0.04 * np.sin(2 * np.pi * freq * 2 * t)) * np.exp(-2 * t)
        path = output / f'functional_signal_{i}.wav'
        sf.write(path, signal.astype(np.float32), config.sample_rate, subtype='FLOAT')
        sources.append({'recording_id': f'functional-signal-{i}', 'file_path': str(path)})
    manifest = prepare(sources, output / 'prepared', config, variations=1)
    checkpoint = train(manifest, output / 'model', epochs=1, batch_size=4, threads=2)
    model, loaded_config, state = load_predictor(checkpoint)
    rows, metadata = load_manifest(manifest)
    tests = [r for r in rows if r['split'] == 'test']
    reproduction = 0.0
    for row in rows:
        dry, wet = pair_audio(row, config)
        regenerated = checked_render(dry, config.sample_rate, row['ground_truth_chain'])
        reproduction = max(reproduction, float(np.max(np.abs(regenerated - wet))))
    assert reproduction == 0, reproduction
    summary = evaluate_rows(tests, metadata, model, list(model.definitions),
                            output / 'evaluation', k=3, budget=15, seed=42, save_audio=True)
    # Missing inventory is explicitly tested as a separate sound reproduction condition.
    unavailable = evaluate_rows(tests, metadata, model, [], output / 'no_inventory',
                                k=3, budget=15, seed=42)
    dry, ref = pair_audio(tests[0], config)
    prediction = predict_pair(model, loaded_config, dry, ref)
    prediction['candidates'] = make_candidates(prediction, list(model.definitions))
    write_json(output / 'inference.json', prediction)
    # Quantify the class-identifiability limitation in actual renderer output.
    ambiguous = [chain_instance('overdrive', {'drive': 12, 'level': 0}),
                 chain_instance('distortion', {'dist': 12, 'tone': 0, 'level': 0}),
                 chain_instance('fuzz', {'fuzz': 12, 'tone': 0, 'level': 0})]
    rendered = [checked_render(dry, config.sample_rate, c) for c in ambiguous]
    overlaps = [float(np.max(np.abs(x - rendered[0]))) for x in rendered[1:]]
    report = {'kind': 'functional check using temporary harmonic signals; not guitar performance',
              'python': platform.python_version(), 'versions': {p: importlib.metadata.version(p) for p in
                ('torch', 'numpy', 'librosa', 'pedalboard', 'cma', 'soundfile')},
              'segments': len(rows), 'test_segments': len(tests), 'epochs': state['epoch'],
              'wet_reproduction_max_absolute_error': reproduction,
              'shared_drive_tone_zero_max_absolute_errors': overlaps,
              'evaluation': summary, 'empty_inventory_evaluation': unavailable,
              'checkpoint': str(checkpoint), 'manifest': str(manifest)}
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/cnn_smoke')
    run(parser.parse_args().output)


if __name__ == '__main__':
    main()
