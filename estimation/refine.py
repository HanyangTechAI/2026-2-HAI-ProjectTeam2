"""Bridge CNN candidates to the team's search.pipeline Restart CMA-ES."""
import copy

import numpy as np

from .audio import segment
from .chain import catalog, continuous, normalize
from .distance import compare


def to_search_inputs(candidates, definitions=None):
    """Return chains and tuple-keyed initial_guesses containing REAL knob values.

    The team's engine normalizes initial_values internally, so x0 must never
    be passed as if it were real knobs. Validate both views before rendering.
    """
    from search.pipeline import ChainParameterSpec
    from .data import validate_chain

    definitions = definitions or catalog()
    chains, guesses = [], {}
    for item in candidates:
        chain = item['chain']
        drive = validate_chain(chain, definitions)
        if item['effect_type'] != drive:
            raise ValueError('Candidate drive does not match its chain')
        names = tuple(p['effect_type'] for p in chain['chain'])
        if names in guesses:
            raise ValueError('Duplicate candidate structure')
        spec = ChainParameterSpec(names, definitions, chain['amp_id'], round_digits=None)
        vector = np.asarray(item['x0'], dtype=float)
        if (vector.shape != (spec.dim,) or not np.isfinite(vector).all()
                or np.any(vector < 0) or np.any(vector > 1)):
            raise ValueError('Invalid normalized candidate x0')
        if not np.allclose(vector, spec.normalize(spec.chain_to_vector(chain)), atol=1e-7, rtol=0):
            raise ValueError('Candidate x0 disagrees with chain values or parameter order')
        expected = [(m[0], m[1], m[2]) for m in spec.param_meta]
        actual = [(p['effect_index'], p['effect_type'], p['name']) for p in item['vector_spec']]
        if actual != expected:
            raise ValueError('Candidate vector_spec disagrees with search parameter order')
        for supplied, (_index, name, knob, *_rest) in zip(item['vector_spec'], spec.param_meta):
            parameter = next(p for p in continuous(definitions[name]) if p['name'] == knob)
            if any(supplied.get(key) != parameter.get(key) for key in ('range', 'type', 'maps_to')):
                raise ValueError('Candidate vector_spec disagrees with search parameter ranges/mapping')
            if supplied.get('scale', 'linear') != parameter.get('scale', 'linear'):
                raise ValueError('Candidate vector_spec disagrees with search parameter scale')
        chains.append(copy.deepcopy(chain))
        guesses[names] = copy.deepcopy(chain)
    return chains, guesses


def default_candidates(candidates, definitions=None):
    """Keep CNN structures fixed and replace active knobs with their defaults."""
    definitions = definitions or catalog()
    result = copy.deepcopy(candidates)
    for item in result:
        for stage in item['chain']['chain']:
            stage['values'] = {p['name']: p['default'] for p in continuous(definitions[stage['effect_type']])}
        item['x0'] = [normalize(item['chain']['chain'][p['effect_index']]['values'][p['name']], p)
                      for p in item['vector_spec']]
    to_search_inputs(result, definitions)
    return result


def refine_candidates(native_dry, reference, native_rate, config, candidates, budget,
                      settings, renderer, seed=42, restarts=2, definitions=None):
    """Share a TOTAL render budget across K fixed candidate structures.

    Initial candidates count toward budget and remain eligible as the best result.
    VST processing uses native generated Dry; only metric inputs are resampled.
    """
    import librosa
    from search.pipeline import ChainParameterSpec, search_and_rank_chains

    definitions = definitions or catalog()
    chains, guesses = to_search_inputs(candidates, definitions)
    if not candidates or not isinstance(budget, int) or budget < len(candidates) or restarts < 1:
        raise ValueError('Positive restarts and a render budget covering every candidate are required')
    if not np.isfinite(native_dry).all() or np.asarray(reference).shape != (config.samples,):
        raise ValueError('Invalid native Dry or reference segment')
    if not np.isfinite(reference).all():
        raise ValueError('Non-finite reference')
    ranked, initial = [], []
    shares = [budget // len(candidates) + int(i < budget % len(candidates))
              for i in range(len(candidates))]
    for i, (item, chain, share) in enumerate(zip(candidates, chains, shares)):
        distances = []

        def render_fn(audio, rate, proposed_chain):
            wave = np.asarray(renderer(audio, rate, proposed_chain))
            if wave.shape != np.asarray(audio).shape or not np.isfinite(wave).all():
                raise ValueError('Renderer changed native shape or returned non-finite audio')
            return wave

        def metric_fn(wave, target):
            if wave.ndim == 2:
                wave = wave.mean(0)
            if native_rate != config.sample_rate:
                wave = librosa.resample(wave, orig_sr=native_rate, target_sr=config.sample_rate)
            score = compare(segment(wave, config, reject_silence=False), target,
                            config.sample_rate, settings)
            if not np.isfinite(score['combined']):
                raise ValueError('Non-finite combined distance')
            distances.append(score)
            return score['combined']

        rows, dropped = search_and_rank_chains(
            native_dry, native_rate, reference, [chain], effects_def=definitions,
            budget=share, n_restarts=restarts, n_seeds=1, metric_fn=metric_fn,
            initial_guesses=guesses, render_fn=render_fn, evaluate_initial=True,
            parameter_round_digits=None, strict=True, base_seed=seed+i)
        if dropped or len(rows) != 1 or not distances or len(distances) > share:
            raise RuntimeError('Search did not satisfy the candidate/render budget contract')
        found = rows[0]
        best_index = int(np.argmin([d['combined'] for d in distances]))
        if not np.isclose(found['loss'], distances[best_index]['combined']):
            raise RuntimeError('Search loss does not match actual rendered scores')
        candidate = copy.deepcopy(item)
        candidate['chain'] = found['chain']
        spec = ChainParameterSpec(found['effect_types'], definitions, round_digits=None)
        candidate['x0'] = spec.normalize(spec.chain_to_vector(found['chain'])).tolist()
        ranked.append({'candidate': candidate, 'distance': distances[best_index],
                       'initial_distance': distances[0], 'render_evaluations': len(distances),
                       'render_budget': share, 'seed': seed+i,
                       'history_combined_d': np.minimum.accumulate([d['combined'] for d in distances]).tolist()})
        initial.append(distances[0]['combined'])
    ranked.sort(key=lambda r: r['distance']['combined'])
    to_search_inputs([r['candidate'] for r in ranked], definitions)
    return dict(candidate=ranked[0]['candidate'], distance=ranked[0]['distance'],
                ranked_candidates=ranked, initial_combined_d=initial,
                render_evaluations=sum(r['render_evaluations'] for r in ranked),
                render_budget=budget, restarts=restarts, seed=seed,
                engine='search.pipeline.search_and_rank_chains/run_cma_es_for_chain',
                distance_settings=settings, native_render_rate=native_rate)
