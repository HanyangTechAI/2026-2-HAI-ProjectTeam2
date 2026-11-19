"""Drive-only candidate contracts against main's unchanged x0_norm CMA API."""
import copy
import unittest

import numpy as np

from estimation.audio import AudioConfig
from estimation.chain import DRIVES, catalog, continuous, make_candidates, with_vector
from estimation.refine import DriveParameterSpec, default_candidates, refine_candidates, to_search_inputs
from estimation.renderer import activation_preflight
from search.pipeline import run_cma_es_for_chain, search_and_rank_chains


class CnnCmaTest(unittest.TestCase):
    def setUp(self):
        self.definitions = catalog()
        prediction = {'probabilities':dict(zip(DRIVES,[.4,.2,.15,.1,.05,.05,.05])),
                      'values':{name:{p['name']:p['default'] for p in continuous(d)}
                                for name,d in self.definitions.items()}}
        self.candidates = make_candidates(prediction,k=3,definitions=self.definitions)

    def test_team_coordinates_and_candidate_configs_agree(self):
        candidates = [with_vector(c,np.linspace(.123456789,.87654321,len(c['x0']))) for c in self.candidates]
        self.assertEqual(to_search_inputs(candidates,self.definitions),candidates)
        for item in candidates:
            spec = DriveParameterSpec([item['effect_type']],self.definitions)
            real = spec.chain_to_vector(item['chain'])
            np.testing.assert_allclose(spec.normalize(real),item['x0'],atol=1e-12)
            np.testing.assert_allclose(spec.chain_to_vector(spec.vector_to_chain(real)),real,atol=1e-12)
        for field in ('x0','order','range'):
            broken = copy.deepcopy(candidates)
            if field=='x0':
                broken[0]['x0'][0] += .01
            elif field=='order':
                broken[0]['vector_spec'].reverse()
            else:
                broken[0]['vector_spec'][0]['range']=[-100,100]
            with self.assertRaises(ValueError):
                to_search_inputs(broken,self.definitions)

    def test_default_control_changes_only_knobs(self):
        altered = [with_vector(c,[.7]*len(c['x0'])) for c in self.candidates]
        defaults = default_candidates(altered,self.definitions)
        for original,control in zip(altered,defaults):
            self.assertEqual(original['effect_type'],control['effect_type'])
            self.assertEqual(len(control['chain']['chain']),1)
            self.assertEqual(control['chain']['chain'][0]['values'],
                             {p['name']:p['default'] for p in continuous(self.definitions[control['effect_type']])})
        self.assertTrue(all(v==.7 for v in altered[0]['x0']))

    def test_main_cma_normalized_seed_is_evaluated_and_budgeted(self):
        spec = DriveParameterSpec(['808_scream'],self.definitions)
        norm = np.array([.123456789,.7654321,.456789123])
        seen = []
        def objective(chain):
            real = spec.chain_to_vector(chain)
            seen.append(real)
            return float(np.sum((spec.normalize(real)-norm)**2))
        _chain,vector,loss,history = run_cma_es_for_chain(
            objective,spec,budget=41,n_runs=2,seed=42,x0_norm=norm)
        np.testing.assert_allclose(seen[0],spec.denormalize(norm),rtol=0,atol=1e-12)
        self.assertEqual(len(seen),41)
        self.assertLess(loss,1e-25)
        self.assertEqual(len(history),41)
        self.assertTrue(np.all(np.diff(history)<=0))
        np.testing.assert_allclose(vector,spec.denormalize(norm),atol=1e-12)

    def test_actual_cma_budget_rate_repeatability_and_exact_seed_retention(self):
        candidates = self.candidates[:2]
        rate = 8000
        x = (.2*np.sin(2*np.pi*173*np.arange(4800)/rate)).astype('float32')
        config = AudioConfig(sample_rate=rate,duration_sec=.6,n_fft=256,hop_length=64,n_mels=32)
        reference = x*.7
        seen = []
        def renderer(audio,sr,chain):
            self.assertEqual(sr,rate)
            self.assertEqual(audio.shape,x.shape)
            self.assertEqual(len(chain['chain']),1)
            seen.append(copy.deepcopy(chain))
            return audio*((chain['chain'][0]['values']['level']-45)/10+.25)
        options = dict(native_dry=x,reference=reference,native_rate=rate,config=config,
                       candidates=candidates,budget=53,settings={'spectral_weight':.5,
                       'dynamics_weight':.5,'spectral_scale':1.,'dynamics_scale':1.},
                       renderer=renderer,seed=42,restarts=1,definitions=self.definitions)
        first = refine_candidates(**options)
        self.assertEqual(len(seen),53)
        self.assertEqual(first['render_evaluations'],53)
        self.assertLessEqual(first['distance']['combined'],min(first['initial_combined_d']))
        self.assertEqual(seen[0],candidates[0]['chain'])
        second = refine_candidates(**options)
        self.assertEqual(first['candidate'],second['candidate'])
        self.assertAlmostEqual(first['distance']['combined'],second['distance']['combined'],places=12)
        with self.assertRaisesRegex(ValueError,'same sample rate'):
            refine_candidates(**dict(options,native_rate=16000))
        def broken_renderer(*args):
            raise RuntimeError('simulated plugin failure')
        with self.assertRaisesRegex(RuntimeError,'simulated plugin failure'):
            refine_candidates(**dict(options,renderer=broken_renderer))

    def test_activation_rejects_dry_pass_through(self):
        x = np.sin(np.arange(1024)*.1).astype('float32')*.2
        status = activation_preflight(lambda audio,rate,chain:audio.copy(),x,44100,self.definitions)
        self.assertFalse(status['available'])
        self.assertEqual(status['render_evaluations'],7)

    def test_main_high_level_api_accepts_cnn_candidate_wrapper(self):
        # A native Gain stands in for a licensed VST to exercise main's parser.
        effects = {'808_scream':{'class_name':'Gain','source':'pedalboard',
                   'params':[{'name':'level','type':'continuous','range':[-12,12],
                              'maps_to':'Gain.gain_db','default':0}]}}
        x = (.1*np.sin(np.arange(4800)*.1)).astype('float32')
        candidate = {'effect_type':'808_scream','model_score':.9,'x0':[.5],
                     'chain':{'amp_id':'fixed_amp_v1','chain':[{'effect_type':'808_scream','values':{'level':0}}]}}
        rows,dropped = search_and_rank_chains(x,8000,x,candidate_chains=[candidate],
                                             effects_def=effects,budget=6,n_restarts=1)
        self.assertFalse(dropped)
        self.assertEqual(rows[0]['effect_types'],['808_scream'])
        self.assertTrue(np.isfinite(rows[0]['loss']))


if __name__=='__main__':
    unittest.main()
