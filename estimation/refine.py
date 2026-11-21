"""Drive-only CNN candidates using main's CMA-ES API without editing search/."""
import copy
import time

import numpy as np

from search.pipeline import ChainParameterSpec, extract_effect_types, run_cma_es_for_chain
from .audio import segment
from .chain import catalog, continuous, normalize
from .distance import compare


class DriveParameterSpec(ChainParameterSpec):
    """Use the team's coordinate order/scales while retaining exact audio knobs."""
    def vector_to_chain(self, real_params):
        if np.asarray(real_params).shape != (self.dim,):
            raise ValueError('Wrong knob vector length')
        stages = [{'effect_type':name,'values':dict(self.fixed_values[i])}
                  for i,name in enumerate(self.effect_types)]
        for value,(index,_name,knob,*_rest) in zip(real_params,self.param_meta):
            stages[index]['values'][knob] = float(value)
        return {'amp_id':self.amp_id,'chain':stages}


def to_search_inputs(candidates, definitions=None):
    """Validate/export main-compatible candidate_configs, including normalized x0."""
    from .data import validate_chain
    definitions = definitions or catalog()
    seen = set()
    for item in candidates:
        chain = item['chain']
        drive = validate_chain(chain,definitions)
        if item['effect_type'] != drive or extract_effect_types(item) != [drive] or drive in seen:
            raise ValueError('Candidate identity is inconsistent or duplicated')
        seen.add(drive)
        spec = DriveParameterSpec([drive],definitions,chain['amp_id'])
        x0 = np.asarray(item['x0'],dtype=float)
        if x0.shape != (spec.dim,) or not np.isfinite(x0).all() or np.any(x0<0) or np.any(x0>1):
            raise ValueError('Invalid normalized candidate x0')
        if not np.allclose(x0,spec.normalize(spec.chain_to_vector(chain)),atol=1e-7,rtol=0):
            raise ValueError('Candidate x0 disagrees with chain values or parameter order')
        expected = [(m[0],m[1],m[2]) for m in spec.param_meta]
        actual = [(p['effect_index'],p['effect_type'],p['name']) for p in item['vector_spec']]
        if actual != expected:
            raise ValueError('Candidate vector_spec disagrees with parameter order')
        for supplied,parameter in zip(item['vector_spec'],continuous(definitions[drive])):
            if any(supplied.get(k)!=parameter.get(k) for k in ('range','type','maps_to')):
                raise ValueError('Candidate vector_spec disagrees with parameter ranges/mapping')
            if supplied.get('scale','linear')!=parameter.get('scale','linear'):
                raise ValueError('Candidate vector_spec disagrees with parameter scale')
    return copy.deepcopy(candidates)


def default_candidates(candidates, definitions=None):
    definitions = definitions or catalog()
    result = copy.deepcopy(candidates)
    for item in result:
        drive = item['effect_type']
        item['chain']['chain'][0]['values'] = {p['name']:p['default'] for p in continuous(definitions[drive])}
        item['x0'] = [normalize(item['chain']['chain'][0]['values'][p['name']],p) for p in item['vector_spec']]
    return to_search_inputs(result,definitions)


def refine_candidates(native_dry, reference, native_rate, config, candidates, budget,
                      settings, renderer, seed=42, restarts=2, definitions=None):
    """Use main's normalized x0 API with a total render cap shared by K drives.

    Evaluate and retain the exact CNN seed before the team's optimizer (which
    clips boundary seeds). Both evaluations count toward the same render budget.
    """
    definitions = definitions or catalog()
    candidates = to_search_inputs(candidates,definitions)
    if not candidates or not isinstance(budget,int) or budget<len(candidates) or restarts<1:
        raise ValueError('Render budget must cover every candidate; restarts must be positive')
    if native_rate != config.sample_rate:
        raise ValueError('CNN, renderer and distance must use the same sample rate')
    if np.asarray(reference).shape != (config.samples,) or not np.isfinite(reference).all():
        raise ValueError('Invalid reference segment')
    if not np.isfinite(native_dry).all():
        raise ValueError('Non-finite Dry')
    # SciPy loads another BLAS library lazily: initialize metrics before
    # limiting pools, otherwise that library retains its default threads.
    compare(reference,reference,native_rate,settings)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1,user_api='blas')
    shares = [budget//len(candidates)+int(i<budget%len(candidates)) for i in range(len(candidates))]
    ranked, initial = [], []
    for index,(item,share) in enumerate(zip(candidates,shares)):
        candidate_started = time.perf_counter()
        spec = DriveParameterSpec([item['effect_type']],definitions,item['chain']['amp_id'])
        distances, rendered_chains = [], []

        def objective(chain):
            if len(distances)>=share:
                raise RuntimeError('Render budget exceeded')
            wave = np.asarray(renderer(native_dry,native_rate,chain))
            if wave.shape!=np.asarray(native_dry).shape or not np.isfinite(wave).all():
                raise ValueError('Renderer changed shape or returned non-finite audio')
            if wave.ndim==2:
                wave = wave.mean(0)
            score = compare(segment(wave,config,reject_silence=False),reference,native_rate,settings)
            if not np.isfinite(score['combined']):
                raise ValueError('Non-finite combined distance')
            distances.append(score)
            rendered_chains.append(copy.deepcopy(chain))
            return score['combined']

        objective(item['chain'])
        if share>1:
            _chain,_vector,loss,_history = run_cma_es_for_chain(
                objective,spec,budget=share-1,seed=seed+index,n_runs=restarts,
                x0_norm=np.asarray(item['x0'],dtype=float))
            if not np.isclose(loss,min(d['combined'] for d in distances[1:])):
                raise RuntimeError('Team optimizer result disagrees with rendered scores')
        best = int(np.argmin([d['combined'] for d in distances]))
        candidate = copy.deepcopy(item)
        candidate['chain'] = rendered_chains[best]
        candidate['x0'] = spec.normalize(spec.chain_to_vector(candidate['chain'])).tolist()
        ranked.append({'candidate':candidate,'distance':distances[best],
                       'seconds':time.perf_counter()-candidate_started,
                       'initial_distance':distances[0],'render_evaluations':len(distances),
                       'render_budget':share,'seed':seed+index,
                       'history_combined_d':np.minimum.accumulate([d['combined'] for d in distances]).tolist()})
        initial.append(distances[0]['combined'])
    ranked.sort(key=lambda r:r['distance']['combined'])
    to_search_inputs([r['candidate'] for r in ranked],definitions)
    return {'candidate':ranked[0]['candidate'],'distance':ranked[0]['distance'],
            'ranked_candidates':ranked,'initial_combined_d':initial,
            'render_evaluations':sum(r['render_evaluations'] for r in ranked),
            'render_budget':budget,'restarts':restarts,'seed':seed,
            'engine':'search.pipeline.run_cma_es_for_chain (main x0_norm API)',
            'blas_threads':1,'distance_warmup':'reference/reference, no VST render',
            'distance_settings':settings,'native_render_rate':native_rate}
