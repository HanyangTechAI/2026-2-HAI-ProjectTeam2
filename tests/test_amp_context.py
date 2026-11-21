"""Cabinet identity must survive CNN-to-CMA handoff; IR must not hide VST bypass."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from estimation.amp import NO_AMP,context_from_record,no_amp,validate_context
from estimation.chain import DRIVES,catalog,continuous,fingerprint,make_candidates
from estimation.data import validate_chain
from estimation.refine import default_candidates,to_search_inputs
from estimation.renderer import Renderer


class AmpContextTest(unittest.TestCase):
    def setUp(self):
        self.context = {'amp_id':'cab_test','definition':{'amp_id':'cab_test',
            'render_method':'impulse_response','fixed':True,'file_path':'cab.wav','gain_db':12.5},
            'ir_sha256':'a'*64}
        self.definitions = catalog()
        self.prediction = {'amp_id':'cab_test','probabilities':dict(zip(DRIVES,[.4,.2,.15,.1,.05,.05,.05])),
            'values':{d:{p['name']:p['default'] for p in continuous(v)} for d,v in self.definitions.items()}}

    def test_legacy_no_amp_and_context_integrity(self):
        self.assertEqual(context_from_record({}),no_amp())
        saved = {'amp_context':self.context,'amp_context_hash':fingerprint(self.context)}
        self.assertEqual(context_from_record(saved),self.context)
        broken = copy.deepcopy(saved)
        broken['amp_context']['definition']['gain_db'] = 0
        with self.assertRaisesRegex(ValueError,'hash'):
            context_from_record(broken)
        broken = dict(self.context,ir_sha256='z'*64)
        with self.assertRaises(ValueError):
            validate_context(broken)

    def test_candidate_cma_and_defaults_preserve_amp(self):
        candidates = make_candidates(self.prediction)
        defaults = default_candidates(candidates,self.definitions)
        to_search_inputs(candidates,self.definitions)
        for item in candidates+defaults:
            self.assertEqual(item['chain']['amp_id'],'cab_test')
            validate_chain(item['chain'],self.definitions,'cab_test')
            with self.assertRaisesRegex(ValueError,'amp'):
                validate_chain(item['chain'],self.definitions,NO_AMP)

    def test_bypassed_vst_is_rejected_before_cabinet(self):
        x = (.1*np.sin(np.arange(1024)*.1)).astype('float32')
        candidate = make_candidates(self.prediction)[0]
        with patch('effects.render.find_plugin',return_value='unused'), \
             patch('estimation.renderer.verify_ir',return_value=self.context), \
             patch('effects.render.render_chain',return_value=x.copy()), \
             patch('effects.render.apply_amp',return_value=x*.5) as cabinet:
            renderer = Renderer(self.definitions,self.context)
            with self.assertRaisesRegex(ValueError,'pass-through'):
                renderer(x,44100,candidate['chain'])
            cabinet.assert_not_called()

    def test_renderer_cannot_silently_drop_cabinet(self):
        x = (.1*np.sin(np.arange(1024)*.1)).astype('float32')
        chain = make_candidates(self.prediction)[0]['chain']
        with patch('effects.render.find_plugin',return_value='unused'), \
             patch('estimation.renderer.verify_ir',return_value=self.context), \
             patch('effects.render.render_chain',return_value=x*.5) as bare, \
             patch('effects.render.apply_amp',return_value=x*.25) as cabinet:
            result = Renderer(self.definitions,self.context)(x,44100,chain)
            self.assertEqual(bare.call_args.args[2]['amp_id'],NO_AMP)
            self.assertEqual(cabinet.call_args.args[2],'cab_test')
            np.testing.assert_array_equal(result,x*.25)
