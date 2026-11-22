"""External evaluation relaxes source identity, never schema/input safeguards."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from estimation.evaluate import AudioEvaluation, check_evaluation_source
from estimation.step2_experiment import validate_external


class ExternalEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.state = {'catalog_hash':'schema','audio_config':{'sample_rate':44100},
                      'source_manifest_sha256':'train','source_render_audit_sha256':'audit'}

    def test_external_source_requires_explicit_opt_in(self):
        external = dict(self.state,source_manifest_sha256='external')
        with self.assertRaisesRegex(ValueError,'allow-external'):
            check_evaluation_source(self.state,external)
        self.assertFalse(check_evaluation_source(self.state,external,allow_external=True))
        self.assertTrue(check_evaluation_source(self.state,self.state))

    def test_opt_in_does_not_relax_schema_or_audio(self):
        for field,value in (('catalog_hash','different'),('audio_config',{'sample_rate':22050})):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError,'schema or audio'):
                    check_evaluation_source(self.state,dict(self.state,**{field:value}),True)

    def test_external_overlap_rejected_even_with_different_ids(self):
        base = dict(self.state,dry_sources=[{'sha256':'same-audio','dry_sample_id':'train-id'}])
        external = copy.deepcopy(base)
        external['dry_sources'][0]['dry_sample_id'] = 'new-id'
        with patch('estimation.step2_experiment.load_manifest',side_effect=[([],base),([],external)]), \
             patch('estimation.step2_experiment.context_from_record',return_value={'amp_id':'same'}):
            with self.assertRaisesRegex(ValueError,'sha256'):
                validate_external('training','external')

    def test_rendered_external_d_uses_frozen_training_scales(self):
        evaluator = AudioEvaluation.__new__(AudioEvaluation)
        evaluator.distance_settings = {'training_scales':True}
        evaluator.metadata = {'distance_settings':{'external_scales':True}}
        evaluator.config = type('Config',(),{'sample_rate':44100})()
        evaluator.target = [np.ones(10)]
        with patch.object(evaluator,'render',return_value=np.zeros(10)), \
             patch('estimation.evaluate.compare',return_value={'combined':1}) as metric:
            evaluator.score({'wet_index':0},{})
        self.assertIs(metric.call_args.args[3],evaluator.distance_settings)


if __name__=='__main__':
    unittest.main()
