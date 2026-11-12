"""Refine CNN drive candidates with CMA-ES and an exact render budget.

The original pipeline.run_cma_es assumes two overdrive dimensions and always
starts from the centre. This adapter keeps the same bounded pycma ask/tell
algorithm but accepts the shared candidate x0 and arbitrary active knob counts.
"""
import time

import numpy as np

from .chain import checked_render, with_vector


def refine_candidates(dry, reference, sample_rate, candidates, budget, seed=42,
                      optimize=True, sigma=0.2):
    from metrics.spectral import METRIC_VERSION, spectral_distance

    if not candidates or budget < len(candidates):
        raise ValueError('Budget must cover every candidate pre-evaluation, including bypass')
    start = time.perf_counter()
    n, errors, best, initial = 0, [], None, []

    def evaluate(item):
        nonlocal n, best
        n += 1
        try:
            audio = checked_render(dry, sample_rate, item['chain'])
            distance = spectral_distance(audio, reference)
            if not np.isfinite(distance):
                raise ValueError('Non-finite spectral distance')
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append(str(exc))
            return 1e6
        if best is None or distance < best['distance']:
            best = {'candidate': item, 'distance': distance, '_audio': audio}
        return distance

    for item in candidates:
        initial.append({'effect_type': item['effect_type'], 'distance': evaluate(item)})
    active = [c for c in candidates if c['x0']]
    if optimize and active:
        import cma

        # Deterministic equal allocations; both initialization methods receive
        # the same candidate list, allocations, seeds and total render budget.
        remaining = budget - n
        for index, item in enumerate(active):
            allocation = remaining // len(active) + int(index < remaining % len(active))
            used, restart = 0, 0
            while used < allocation:
                x0 = np.clip(item['x0'], 1e-6, 1 - 1e-6).tolist()
                es = cma.CMAEvolutionStrategy(x0, sigma, {
                    'bounds': [0, 1], 'popsize': 6,
                    'seed': seed + index * 1000 + restart + 1,
                    'verbose': -9, 'verb_log': 0})
                while not es.stop() and used < allocation:
                    xs = es.ask()
                    batch = xs[:allocation - used]
                    losses = [evaluate(with_vector(item, x)) for x in batch]
                    used += len(batch)
                    if len(batch) == len(xs):
                        es.tell(xs, losses)
                restart += 1
    if best is None:
        raise ValueError(f'Every render failed: {errors}')
    return dict(best, initial=initial, initial_best_distance=min(i['distance'] for i in initial),
                render_evaluations=n, requested_budget=budget,
                unused_budget=budget - n,
                unused_reason='no continuous candidates' if optimize and not active else None,
                render_errors=errors, processing_seconds=time.perf_counter() - start,
                metric={'version': METRIC_VERSION, 'direction': 'lower is better',
                        'units': 'natural-log magnitude difference', 'sample_rate': sample_rate})
