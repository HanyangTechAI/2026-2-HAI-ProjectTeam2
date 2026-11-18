"""VST7 contracts and learning checks; no installed/activated plugin required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from estimation.chain import (AUXILIARY,DRIVES,catalog,continuous,denormalize,
                               make_candidates,normalize,with_vector)
from estimation.cnn import PairCNN,supervised_loss
from estimation.data import split_recordings,validate_chain
from estimation.distance import calibrate,compare
from estimation.audio import AudioConfig
from estimation.data import PairDataset,prepare
from estimation.evaluate import render_subset


class EstimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.definitions = catalog()

    def prediction(self):
        return {'probabilities':dict(zip(DRIVES,[.4,.2,.15,.1,.05,.05,.05])),
                'presence':{'compressor':.9,'eq_3band':.8},
                'values':{t:{p['name']:p['default'] for p in continuous(d)}
                          for t,d in self.definitions.items()}}

    def test_catalog_only_seven_vst_and_two_stages(self):
        self.assertEqual(set(self.definitions),set(DRIVES)|set(AUXILIARY))
        for t in DRIVES:
            self.assertEqual(self.definitions[t]['source'],'vst')
            self.assertNotIn('input_gain_db',[p['name'] for p in continuous(self.definitions[t])])

    def test_normalized_vectors_include_every_chain_stage(self):
        item = make_candidates(self.prediction(),k=1)[0]
        self.assertEqual([p['effect_type'] for p in item['chain']['chain']],
                         ['compressor','808_scream','eq_3band'])
        changed = with_vector(item,[.37]*len(item['x0']))
        for spec in item['vector_spec']:
            actual = changed['chain']['chain'][spec['effect_index']]['values'][spec['name']]
            self.assertAlmostEqual(normalize(actual,spec),.37)
        self.assertEqual(item['chain']['chain'][0]['values']['ratio'],4)

    def test_inventory_top_k_and_presence_threshold(self):
        prediction = self.prediction()
        self.assertEqual(make_candidates(prediction,[],3),[])
        items = make_candidates(prediction,['m_fuzz','b_drive'],3)
        self.assertEqual([p['effect_type'] for p in items],['b_drive','m_fuzz'])
        prediction['presence'] = {'compressor':.49,'eq_3band':.1}
        self.assertEqual(len(make_candidates(prediction,k=1)[0]['chain']['chain']),1)
        prediction['probabilities']['b_drive'] = float('nan')
        with self.assertRaises(ValueError):
            make_candidates(prediction)

    def test_optional_chain_and_unknown_knobs_are_validated(self):
        chain = make_candidates(self.prediction(),k=1)[0]['chain']
        self.assertEqual(validate_chain(chain,self.definitions),'808_scream')
        altered = copy.deepcopy(chain)
        altered['chain'].reverse()
        with self.assertRaises(ValueError):
            validate_chain(altered,self.definitions)
        altered = copy.deepcopy(chain)
        altered['chain'][1]['values']['input_gain_db'] = 0
        with self.assertRaises(ValueError):
            validate_chain(altered,self.definitions)

    def test_group_split_and_train_only_distance_scaling(self):
        ids = [f'dry{i}' for i in range(78)]
        split = split_recordings(ids+ids)
        self.assertEqual(split,split_recordings(list(reversed(ids))))
        self.assertEqual([list(split.values()).count(s) for s in ('train','validation','test')],[54,12,12])
        raw = [[1,2],[3,6],[1e10,1e10]]
        settings = calibrate(raw,[0,1])
        self.assertEqual(settings['spectral_scale'],2)
        self.assertEqual(settings['dynamics_scale'],4)
        x = np.sin(np.arange(4000)*.1)*.1
        self.assertEqual(compare(x,x,8000,settings)['combined'],0)
        self.assertGreater(compare(x,x*.5,8000,settings)['combined'],0)

    def test_render_sampling_is_balanced_and_deterministic(self):
        rows = [{'sample_id':f'{t}_{i}_{v}','recording_id':str(i),'drive':t}
                for t in DRIVES for i in range(12) for v in range(5)]
        selected = render_subset(rows,70)
        self.assertEqual(len(selected),70)
        self.assertEqual(selected,render_subset(rows,70))
        for drive in DRIVES:
            per_class = [r for r in rows if r['sample_id'] in selected and r['drive']==drive]
            self.assertEqual(len(per_class),10)
            self.assertEqual(len({r['recording_id'] for r in per_class}),10)
        self.assertEqual(len(render_subset(rows,-1)),420)

    def test_joint_loss_masks_absent_stages_and_unselected_drives(self):
        model = PairCNN(self.definitions,channels=(8,8,16,16),hidden=32)
        output = model(torch.randn(2,1,64,64),torch.randn(2,1,64,64))
        self.assertEqual(output['logits'].shape,(2,7))
        self.assertEqual(output['presence'].shape,(2,2))
        batch = {'label':torch.zeros(2,dtype=torch.long),'presence':torch.zeros(2,2),
                 'distance':torch.ones(2,2),
                 'active':{t:torch.full((2,),t==DRIVES[0]) for t in self.definitions},
                 'knobs':{t:torch.zeros(2,len(continuous(d))) for t,d in self.definitions.items()}}
        first,_ = supervised_loss(output,batch)
        altered = copy.deepcopy(batch)
        altered['knobs']['compressor'][:] = 1e5
        altered['knobs'][DRIVES[1]][:] = 1e5
        second,_ = supervised_loss(output,altered)
        torch.testing.assert_close(first,second)
        first.backward()
        self.assertIsNone(model.knobs['compressor'].weight.grad)
        self.assertIsNone(model.knobs[DRIVES[1]].weight.grad)
        self.assertIsNotNone(model.knobs[DRIVES[0]].weight.grad)
        self.assertIsNotNone(model.distance.weight.grad)

    def test_import_uses_render_input_and_cache_works_without_shared_drive(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'generated'
            for folder in ('audio','labels','dry'):
                (source/folder).mkdir(parents=True,exist_ok=True)
            mappings = []
            for i in range(3):
                x = np.sin(2*np.pi*(150+17*i)*np.arange(4800)/8000).astype('float32')
                x *= 10**(-12/20)/np.max(np.abs(x))
                relative = f'dry/source{i}.wav'
                sf.write(source/relative,x,8000,subtype='FLOAT')
                mappings.append({'dry_sample_id':f'dry{i}','namespace':'guitar',
                                 'relative_path':f'source{i}.wav','render_input':relative})
                for drive in DRIVES:
                    name = f'wet{i}_{drive}'
                    values = self.prediction()['values'][drive]
                    label = {'target_id':name,'dry_sample_id':f'dry{i}',
                             'file_path':'Z:/other-computer/not-readable.wav',
                             'ground_truth_chain':{'amp_id':'fixed_amp_v1','chain':[
                                 {'effect_type':drive,'values':values}]}}
                    (source/'labels'/f'{name}.json').write_text(json.dumps(label),encoding='utf-8')
                    sf.write(source/'audio'/f'{name}.wav',x*.7,8000,subtype='FLOAT')
            manifest = {'drives':list(DRIVES),'dry_peak_db':-12,'completed_pairs':21,
                        'dry_id_map':mappings}
            (source/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            config = AudioConfig(sample_rate=8000,duration_sec=.6,n_fft=256,hop_length=64,n_mels=32)
            cache = prepare(source,root/'cache',config=config)
            source.rename(root/'unavailable_source')
            dataset = PairDataset(cache,'train')
            self.assertEqual(len(dataset),7)
            batch = dataset[0]
            self.assertEqual(batch['dry'].shape[0:2],(1,32))
            self.assertTrue(torch.isfinite(batch['distance']).all())
            self.assertTrue((cache/'native_dry'/'dry0.npy').exists())
            dataset.close()


if __name__=='__main__':
    unittest.main()
