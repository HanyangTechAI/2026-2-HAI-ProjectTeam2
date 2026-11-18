"""Real CMA-ES contract checks without requiring an installed/licensed VST."""
import copy
import unittest

import librosa
import numpy as np

from estimation.audio import AudioConfig
from estimation.chain import DRIVES, catalog, continuous, make_candidates, with_vector
from estimation.cma_experiment import preflight
from estimation.refine import default_candidates, refine_candidates, to_search_inputs
from search.pipeline import ChainParameterSpec, run_cma_es_for_chain, search_and_rank_chains


class CnnCmaTest(unittest.TestCase):
    def setUp(self):
        self.definitions = catalog()
        prediction = {'probabilities': dict(zip(DRIVES,[.4,.2,.15,.1,.05,.05,.05])),
                      'presence': {'compressor': .9, 'eq_3band': .9},
                      'values': {name: {p['name']: p['default'] for p in continuous(d)}
                                 for name,d in self.definitions.items()}}
        self.candidates = make_candidates(prediction,k=3,definitions=self.definitions)

    def test_cnn_vectors_and_team_vectors_agree_for_full_chains(self):
        candidates = [with_vector(c,np.linspace(.123456789,.87654321,len(c['x0']))) for c in self.candidates]
        chains, guesses = to_search_inputs(candidates,self.definitions)
        for item,chain in zip(candidates,chains):
            names = tuple(s['effect_type'] for s in chain['chain'])
            self.assertEqual(guesses[names],chain)
            spec = ChainParameterSpec(names,self.definitions,round_digits=None)
            np.testing.assert_allclose(spec.normalize(spec.chain_to_vector(chain)),item['x0'],atol=1e-12)
            restored = spec.vector_to_chain(spec.denormalize(item['x0']))
            np.testing.assert_allclose(spec.chain_to_vector(restored),spec.chain_to_vector(chain),atol=1e-12)
        broken = copy.deepcopy(candidates)
        broken[0]['x0'][0] += .01
        with self.assertRaisesRegex(ValueError,'disagrees'):
            to_search_inputs(broken,self.definitions)
        broken = copy.deepcopy(candidates)
        broken[0]['vector_spec'].reverse()
        with self.assertRaisesRegex(ValueError,'parameter order'):
            to_search_inputs(broken,self.definitions)
        broken = copy.deepcopy(candidates)
        broken[0]['vector_spec'][0]['range'] = [-100,0]
        with self.assertRaisesRegex(ValueError,'ranges/mapping'):
            to_search_inputs(broken,self.definitions)

    def test_default_control_preserves_structure_and_updates_all_seed_coordinates(self):
        altered = [with_vector(c,[.7]*len(c['x0'])) for c in self.candidates]
        defaults = default_candidates(altered,self.definitions)
        for original,control in zip(altered,defaults):
            self.assertEqual([s['effect_type'] for s in original['chain']['chain']],
                             [s['effect_type'] for s in control['chain']['chain']])
            for stage in control['chain']['chain']:
                self.assertEqual(stage['values'],{p['name']:p['default'] for p in continuous(self.definitions[stage['effect_type']])})
        self.assertTrue(all(v==.7 for v in altered[0]['x0']))

    def test_team_cma_evaluates_exact_initial_knobs_first_and_counts_seed_in_budget(self):
        spec = ChainParameterSpec(['808_scream'],self.definitions,round_digits=None)
        norm = np.array([.123456789,.7654321,.456789123])
        real = spec.denormalize(norm)
        seen = []

        def objective(chain):
            x = spec.chain_to_vector(chain)
            seen.append(x)
            return float(np.sum((spec.normalize(x)-norm)**2))

        chain,vector,loss,history = run_cma_es_for_chain(
            objective,spec,budget=41,n_runs=2,seed=42,initial_values=real,
            evaluate_initial=True,strict=True)
        np.testing.assert_allclose(seen[0],real,rtol=0,atol=1e-12)
        self.assertEqual(len(seen),41)
        self.assertLess(loss,1e-25)
        self.assertEqual(len(history),41)
        np.testing.assert_allclose(vector,real,atol=1e-12)
        self.assertTrue(np.all(np.diff(history)<=0))
        self.assertEqual(chain['chain'][0]['effect_type'],'808_scream')

    def test_native_rate_adapter_actual_cma_budget_repeatability_and_seed_retention(self):
        prediction = {'probabilities':dict(zip(DRIVES,[.4,.2,.15,.1,.05,.05,.05])),
                      'presence':{'compressor':0,'eq_3band':0},
                      'values':{name:{p['name']:p['default'] for p in continuous(d)}
                                for name,d in self.definitions.items()}}
        candidates = make_candidates(prediction,k=2,definitions=self.definitions)
        rate = 16000
        x = (.2*np.sin(2*np.pi*173*np.arange(9600)/rate)).astype('float32')
        config = AudioConfig(sample_rate=8000,duration_sec=.6,n_fft=256,hop_length=64,n_mels=32)
        reference = librosa.resample(x*.7,orig_sr=rate,target_sr=config.sample_rate)
        seen = []

        def renderer(audio,sr,chain):
            self.assertEqual(sr,rate)
            self.assertEqual(audio.shape,x.shape)
            seen.append(copy.deepcopy(chain))
            stage = chain['chain'][0]
            name = 'level' if 'level' in stage['values'] else 'volume'
            return audio*float((stage['values'][name]-45)/10+.25)

        options = dict(native_dry=x,reference=reference,native_rate=rate,config=config,
                       candidates=candidates,budget=53,settings={'spectral_weight':.5,
                       'dynamics_weight':.5,'spectral_scale':1.,'dynamics_scale':1.},
                       renderer=renderer,seed=42,restarts=1,definitions=self.definitions)
        first = refine_candidates(**options)
        self.assertEqual(len(seen),53)
        self.assertEqual(first['render_evaluations'],53)
        self.assertLessEqual(first['distance']['combined'],min(first['initial_combined_d']))
        np.testing.assert_allclose(list(seen[0]['chain'][0]['values'].values()),
                                   list(candidates[0]['chain']['chain'][0]['values'].values()),atol=1e-12)
        self.assertEqual(sorted(r['render_budget'] for r in first['ranked_candidates']),[26,27])
        second = refine_candidates(**options)
        self.assertAlmostEqual(first['distance']['combined'],second['distance']['combined'],places=12)
        self.assertEqual(first['candidate'],second['candidate'])

        def broken_renderer(*args):
            raise RuntimeError('simulated plugin failure')

        with self.assertRaisesRegex(RuntimeError,'simulated plugin failure'):
            refine_candidates(**dict(options,renderer=broken_renderer))

    def test_failed_activation_is_detected_for_dry_pass_through(self):
        x = np.sin(np.arange(1024)*.1).astype('float32')*.2
        status = preflight(lambda audio,rate,chain:audio.copy(),x,44100,self.definitions)
        self.assertFalse(status['available'])
        self.assertEqual(status['render_evaluations'],7)
        self.assertTrue(all(not r['active'] for r in status['checks']))

    def test_invalid_seed_and_nonfinite_metric_fail_instead_of_reporting_success(self):
        spec = ChainParameterSpec(['808_scream'],self.definitions)
        with self.assertRaises(ValueError):
            run_cma_es_for_chain(lambda c:0.,spec,initial_values=[.5],strict=True)
        with self.assertRaisesRegex(ValueError,'non-finite'):
            run_cma_es_for_chain(lambda c:float('nan'),spec,budget=1,
                                 evaluate_initial=True,strict=True)

    def test_existing_team_pipeline_default_renderer_and_rounding_still_work(self):
        from effects.render import build_effect
        from pedalboard import Pedalboard
        x = (.1*np.sin(np.arange(4800)*.1)).astype('float32')
        spec = ChainParameterSpec(['overdrive'])
        values = spec.denormalize(np.linspace(.123456789,.87654321,spec.dim))
        chain = spec.vector_to_chain(values)
        for actual, real in zip(spec.chain_to_vector(chain),values):
            self.assertEqual(actual,round(float(real),4))
        wet = Pedalboard(build_effect('overdrive',chain['chain'][0]['values']))(x,8000)
        ranked,dropped = search_and_rank_chains(x,8000,wet,[chain],budget=6,n_restarts=1)
        self.assertFalse(dropped)
        self.assertEqual(len(ranked),1)
        self.assertEqual(ranked[0]['effect_types'],['overdrive'])
        self.assertTrue(np.isfinite(ranked[0]['loss']))


if __name__=='__main__':
    unittest.main()
