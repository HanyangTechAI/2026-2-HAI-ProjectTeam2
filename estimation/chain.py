"""Canonical VST vocabulary and the candidate contract shared with search."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

DRIVES = ('808_scream', 'b_drive', 'o_drive', 'one_distortion',
          'bat_distortion', 'f_fuzz', 'm_fuzz')
AUXILIARY = ()  # Current task: one MultiDrive PRO model only, no optional stages.
MODEL_NAMES = dict(zip(DRIVES, ('808-Scream', 'B-Drive', 'O-Drive',
                              'One-Distortion', 'BAT-Distortion', 'F-Fuzz', 'M-Fuzz')))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def catalog():
    """Read team definitions without importing or loading any VST plugin.

    Older checkouts use display-name keys. Expose only the seven canonical
    dataset keys only. input_gain_db is fixed rather than sampled:
    in this dataset: keep it fixed at zero rather than inventing a label.
    """
    path = Path(__file__).resolve().parents[1] / 'schemas/effects.data.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for name in DRIVES:
        source = name if name in raw else MODEL_NAMES.get(name, name)
        if source not in raw:
            raise ValueError(f'Missing effect definition: {name}')
        definition = copy.deepcopy(raw[source])
        definition['params'] = [p for p in definition['params'] if p['name'] != 'input_gain_db']
        definition['active_params'] = definition['params']
        if name in DRIVES:
            if definition['source'] != 'vst':
                raise ValueError(f'{name} must be a VST definition')
            definition['render_fixed'] = {'input_gain_db': 0.0}
        result[name] = definition
    return result


def continuous(definition):
    return [p for p in definition['params'] if p['type'] == 'continuous']


def normalize(value, param):
    low, high = param['range']
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{param['name']}={value} outside [{low}, {high}]")
    return float(np.log(value / low) / np.log(high / low) if param.get('scale') == 'log'
                 else (value - low) / (high - low))


def denormalize(value, param):
    if not np.isfinite(value):
        raise ValueError('Nonfinite knob coordinate')
    low, high = param['range']
    value = float(np.clip(value, 0, 1))
    return float(low * (high / low) ** value if param.get('scale') == 'log'
                 else low + (high - low) * value)


def make_candidates(prediction, owned_effect_types=None, k=3, definitions=None):
    """One single-VST chain per permitted drive, with no compressor or EQ.

    model_score is a drive softmax probability, NOT an audio-distance score.
    vector_spec includes effect_index so CMA-ES can update every active stage.
    """
    definitions = definitions or catalog()
    if k < 1:
        raise ValueError('k must be positive')
    probabilities = prediction['probabilities']
    if (set(probabilities) != set(DRIVES) or
            any(not np.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()) or
            not np.isclose(sum(probabilities.values()), 1, atol=1e-5)):
        raise ValueError('Expected seven finite probabilities summing to one')
    allowed = set(DRIVES if owned_effect_types is None else owned_effect_types) & set(DRIVES)
    ranked = sorted(allowed, key=lambda t: (-probabilities[t], t))[:k]
    candidates = []
    for drive in ranked:
        names = [drive]
        items, specs, vector = [], [], []
        for name in names:
            values = prediction['values'][name]
            params = continuous(definitions[name])
            if set(values) != {p['name'] for p in params}:
                raise ValueError(f'Incomplete knobs for {name}')
            for p in params:
                vector.append(normalize(values[p['name']], p))
                specs.append(dict(p, effect_type=name, effect_index=len(items),
                                  normalized_bounds=[0.0, 1.0]))
            items.append({'effect_type': name, 'values': dict(values)})
        candidates.append({'effect_type': drive, 'model_score': float(probabilities[drive]),
                           'chain': {'amp_id': prediction.get('amp_id','fixed_amp_v1'), 'chain': items},
                           'x0': vector, 'vector_spec': specs})
    return candidates


def with_vector(item, vector):
    if len(vector) != len(item['vector_spec']) or not np.isfinite(vector).all():
        raise ValueError('Invalid CMA-ES vector')
    result = copy.deepcopy(item)
    result['x0'] = [float(v) for v in np.clip(vector, 0, 1)]
    for p, v in zip(result['vector_spec'], result['x0']):
        result['chain']['chain'][p['effect_index']]['values'][p['name']] = denormalize(v, p)
    return result
