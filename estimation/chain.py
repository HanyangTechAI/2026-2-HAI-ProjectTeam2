"""Drive definitions, knob coordinates and chain candidates for the CNN pipeline."""
import copy
import hashlib
import json

import numpy as np

from effects.render import load_effects, render_chain


def catalog():
    """Only rendered gain-stage knobs are learnable; retain ignored defaults."""
    result = {}
    for name, definition in load_effects().items():
        if definition['category'] != 'gain_stage' or definition['source'] != 'pedalboard':
            continue
        classes = definition.get('composite_of') or [definition['class_name']]
        params = [p for p in definition['params']
                  if p['maps_to'].partition('.')[0] in classes and '.' in p['maps_to']]
        result[name] = dict(definition, active_params=params)
    return result


def fingerprint(definitions):
    return hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest()


def continuous(definition):
    return [p for p in definition['active_params'] if p['type'] == 'continuous']


def discrete(definition):
    return [p for p in definition['active_params'] if p['type'] in ('categorical', 'boolean')]


def normalize(value, param):
    lo, hi = param['range']
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"Invalid {param['name']}: {value}, expected [{lo}, {hi}]")
    if param.get('scale') == 'log':
        return float((np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    return float((value - lo) / (hi - lo))


def denormalize(value, param):
    if not np.isfinite(value):
        raise ValueError('Non-finite parameter coordinate')
    value = float(np.clip(value, 0, 1))
    lo, hi = param['range']
    if param.get('scale') == 'log':
        return float(np.exp(np.log(lo) + value * (np.log(hi) - np.log(lo))))
    return float(lo + value * (hi - lo))


def chain_instance(effect_type, values=None, definitions=None):
    definitions = definitions if definitions is not None else catalog()
    items = []
    if effect_type != 'bypass':
        definition = definitions[effect_type]
        defaults = {p['name']: p['default'] for p in definition['params']}
        if set(values or {}) - set(defaults):
            raise ValueError('Unknown knob name')
        defaults.update(values or {})
        active_names = {p['name'] for p in definition['active_params']}
        for p in definition['params']:
            if p['name'] not in active_names:
                defaults[p['name']] = p['default']
        for p in continuous(definition):
            normalize(defaults[p['name']], p)
        for p in discrete(definition):
            options = [False, True] if p['type'] == 'boolean' else p['options']
            if defaults[p['name']] not in options:
                raise ValueError('Invalid discrete knob value')
        items = [{'effect_type': effect_type, 'values': defaults}]
    # Required schema identifier; renderer explicitly ignores amp_id.
    return {'amp_id': 'fixed_amp_v1', 'chain': items}


def candidate(effect_type, score, values=None, definitions=None):
    definitions = definitions if definitions is not None else catalog()
    chain = chain_instance(effect_type, values, definitions)
    specs = [] if effect_type == 'bypass' else continuous(definitions[effect_type])
    knobs = {} if not chain['chain'] else chain['chain'][0]['values']
    return {'effect_type': effect_type, 'model_score': float(score), 'chain': chain,
            'x0': [normalize(knobs[p['name']], p) for p in specs],
            'vector_spec': [dict(p, transform=p.get('scale', 'linear'),
                                 normalized_bounds=[0.0, 1.0]) for p in specs]}


def with_vector(item, vector):
    if len(vector) != len(item['vector_spec']):
        raise ValueError('Parameter vector length mismatch')
    if not np.isfinite(vector).all():
        raise ValueError('Non-finite parameter vector')
    vector = np.clip(vector, 0, 1)
    result = copy.deepcopy(item)
    result['x0'] = [float(v) for v in vector]
    for p, value in zip(result['vector_spec'], vector):
        result['chain']['chain'][0]['values'][p['name']] = denormalize(value, p)
    return result


def make_candidates(prediction, owned_effect_types, k=3, definitions=None):
    """One candidate per class. Discrete knobs use argmax, no enumeration."""
    if k < 1:
        raise ValueError('k must be positive')
    definitions = definitions if definitions is not None else catalog()
    expected = {'bypass', *definitions}
    probabilities = prediction['probabilities']
    if (set(probabilities) != expected or
            any(not np.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or
            not np.isclose(sum(probabilities.values()), 1, atol=1e-5)):
        raise ValueError('Expected finite class probabilities summing to one')
    if set(prediction['values']) != set(definitions):
        raise ValueError('Expected knob predictions for every supported drive')
    allowed = {'bypass'} | (set(owned_effect_types) & set(definitions))
    ranked = sorted(allowed, key=lambda t: (-prediction['probabilities'][t], t))[:k]
    return [candidate(t, prediction['probabilities'][t], prediction['values'].get(t), definitions)
            for t in ranked]


def checked_render(dry, sample_rate, chain):
    """render_chain constructs a fresh Pedalboard on every call (reset=True)."""
    wet = np.asarray(render_chain(dry, sample_rate, chain), dtype=np.float32)
    if wet.shape != dry.shape or not np.all(np.isfinite(wet)):
        raise ValueError('Renderer returned invalid or unaligned audio')
    return wet
