"""Meaningful invariants for split isolation, masked loss and render budgets."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from effects import render_chain
from estimation.audio import AudioConfig, log_mel, segment
from estimation.cnn import PairCNN, masked_loss
from estimation.chain import (candidate, catalog, chain_instance, continuous,
                               denormalize, make_candidates, normalize, with_vector)
from estimation.data import load_manifest, prepare, split_recordings
from estimation.experiments import overdrive_subset
from estimation.refine import refine_candidates


class EstimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.definitions = catalog()
        cls.sr = 8000
        cls.config = AudioConfig(sample_rate=cls.sr, duration_sec=0.6, n_fft=256,
                                hop_length=64, n_mels=32)
        t = np.arange(cls.config.samples) / cls.sr
        cls.dry = (0.2 * np.sin(2 * np.pi * 233 * t) * np.exp(-2 * t)).astype(np.float32)

    def prediction(self):
        return {'probabilities': {'bypass': 0.1, 'overdrive': 0.5, 'distortion': 0.3, 'fuzz': 0.1},
                'values': {t: {p['name']: p['default'] for p in d['params']}
                           for t, d in self.definitions.items()}}

    def test_normalization_roundtrip_and_vector_order(self):
        for t, definition in self.definitions.items():
            item = candidate(t, 0.5)
            for coordinate in (0, 0.37, 1):
                result = with_vector(item, [coordinate] * len(item['x0']))
                for p in continuous(definition):
                    value = result['chain']['chain'][0]['values'][p['name']]
                    self.assertAlmostEqual(normalize(value, p), coordinate)
        log_param = {'name': 'freq', 'range': [10, 1000], 'scale': 'log'}
        self.assertAlmostEqual(denormalize(0.5, log_param), 100)
        self.assertAlmostEqual(normalize(100, log_param), 0.5)

    def test_inventory_and_unrendered_mode(self):
        prediction = self.prediction()
        self.assertEqual([c['effect_type'] for c in make_candidates(prediction, [], k=3)], ['bypass'])
        items = make_candidates(prediction, ['fuzz', 'compressor', 'unknown'], k=99)
        self.assertEqual({c['effect_type'] for c in items}, {'fuzz', 'bypass'})
        self.assertEqual(len(candidate('fuzz', 1)['x0']), 3)
        self.assertEqual(chain_instance('fuzz', {'mode': 'Modern'})['chain'][0]['values']['mode'], 'Vintage')
        self.assertEqual(candidate('bypass', 1)['x0'], [])
        prediction['probabilities']['fuzz'] = float('nan')
        with self.assertRaises(ValueError):
            make_candidates(prediction, ['fuzz'])

    def test_recording_split_is_grouped_and_deterministic(self):
        ids = ['take-a', 'take-a', 'take-b', 'take-c', 'take-d']
        splits = split_recordings(ids)
        self.assertEqual(splits, split_recordings(list(reversed(ids))))
        self.assertEqual(set(splits.values()), {'train', 'validation', 'test'})
        self.assertEqual(len(splits), 4)

    def test_silence_nonfinite_and_gain_information(self):
        with self.assertRaises(ValueError):
            segment(np.zeros(100), self.config)
        with self.assertRaises(ValueError):
            segment(np.array([np.nan]), self.config)
        with self.assertRaises(ValueError):
            log_mel(np.full(self.config.samples, np.inf), self.config)
        self.assertFalse(np.allclose(log_mel(self.dry, self.config), log_mel(self.dry * 0.2, self.config)))

    def test_default_four_second_preprocessing_and_forward(self):
        config = AudioConfig()
        dry = segment(self.dry, config)
        mel = log_mel(dry, config)
        self.assertEqual(mel.shape, (1, 64, 345))
        model = PairCNN(self.definitions)
        with torch.inference_mode():
            output = model(torch.from_numpy(mel)[None], torch.from_numpy(mel)[None])
        self.assertEqual(output['logits'].shape, (1, 4))
        self.assertEqual(output['continuous']['fuzz'].shape, (1, 3))

    def test_masked_loss_ignores_inactive_heads_and_bypass(self):
        model = PairCNN(self.definitions)
        output = model(torch.randn(3, 1, 32, 48), torch.randn(3, 1, 32, 48))
        knobs = {t: torch.zeros(3, len(continuous(d))) for t, d in self.definitions.items()}
        labels = torch.tensor([0, 0, 0])
        loss, stats = masked_loss(output, labels, knobs, {}, self.definitions, model.classes)
        self.assertEqual(stats['knob_huber'], 0)
        loss.backward()
        for head in model.knobs.values():
            self.assertIsNone(head.weight.grad)
        labels = torch.tensor([1, 0, 1])
        first, _ = masked_loss(output, labels, knobs, {}, self.definitions, model.classes)
        altered = {t: x.clone() for t, x in knobs.items()}
        altered['distortion'][:] = 1e4
        altered['overdrive'][1] = 1e4
        second, _ = masked_loss(output, labels, altered, {}, self.definitions, model.classes)
        torch.testing.assert_close(first, second)

    def test_classification_and_regression_use_separate_heads_and_losses(self):
        dry, wet = torch.randn(3, 1, 32, 48), torch.randn(3, 1, 32, 48)
        classifier = PairCNN(self.definitions, task='classification', include_bypass=False)
        prediction = classifier(dry, wet)
        self.assertEqual(classifier.classes, ['overdrive', 'distortion', 'fuzz'])
        self.assertEqual(prediction['logits'].shape, (3, 3))
        self.assertEqual(prediction['continuous'], {})
        loss, stats = masked_loss(prediction, torch.tensor([0, 1, 2]), {}, {},
                                  classifier.definitions, classifier.classes)
        self.assertEqual(stats['knob_huber'], 0)
        loss.backward()
        self.assertIsNotNone(classifier.classifier.weight.grad)
        regressor = PairCNN(self.definitions, task='regression', effect_types=['overdrive'], include_bypass=False)
        prediction = regressor(dry, wet)
        self.assertIsNone(prediction['logits'])
        self.assertEqual(set(prediction['continuous']), {'overdrive'})
        target = {'overdrive': torch.zeros(3, 2)}
        loss, stats = masked_loss(prediction, torch.zeros(3, dtype=torch.long), target, {},
                                  regressor.definitions, regressor.classes)
        self.assertEqual(stats['classification'], 0)
        loss.backward()
        self.assertIsNotNone(regressor.knobs['overdrive'].weight.grad)

    def test_task_datasets_share_exact_overdrive_pairs_and_splits(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for i in range(3):
                path = root / f'take{i}.wav'
                sf.write(path, self.dry * (1 + i * 0.1), self.sr, subtype='FLOAT')
                sources.append({'recording_id': str(i), 'file_path': str(path)})
            manifest = prepare(sources, root / 'classification', self.config, variations=2,
                               task='classification', include_bypass=False)
            classification, a = load_manifest(manifest)
            self.assertEqual(len(classification), 18)
            self.assertNotIn('bypass', a['classes'])
            regression, b = load_manifest(overdrive_subset(manifest, root / 'regression'))
            self.assertEqual(len(regression), 6)
            self.assertEqual(b['task'], 'regression')
            self.assertEqual(b['classes'], ['overdrive'])
            self.assertEqual(a['splits'], b['splits'])
            source = {r['sample_id']: r for r in classification}
            for row in regression:
                original = source[row['sample_id']]
                self.assertEqual(row['split'], original['split'])
                self.assertEqual(Path(row['wet_path']).read_bytes(), Path(original['wet_path']).read_bytes())

    def test_fresh_renderer_and_class_ambiguity(self):
        # At zero tone gain, the distinct PeakFilter frequencies disappear.
        settings = [('overdrive', {'drive': 12, 'level': 0}),
                    ('distortion', {'dist': 12, 'tone': 0, 'level': 0}),
                    ('fuzz', {'fuzz': 12, 'tone': 0, 'level': 0})]
        audios = [render_chain(self.dry, self.sr, chain_instance(t, v)) for t, v in settings]
        np.testing.assert_allclose(audios[0], audios[1], atol=2e-6)
        np.testing.assert_allclose(audios[0], audios[2], atol=2e-6)
        render_chain(self.dry * 0.8, self.sr, chain_instance('fuzz'))
        again = render_chain(self.dry, self.sr, chain_instance('overdrive', settings[0][1]))
        np.testing.assert_array_equal(audios[0], again)

    def test_cma_budget_and_bypass_fast_path(self):
        reference = render_chain(self.dry, self.sr, chain_instance('overdrive', {'drive': 10, 'level': -2}))
        candidates = [candidate('overdrive', 0.8), candidate('bypass', 0.2)]
        result = refine_candidates(self.dry, reference, self.sr, candidates, budget=11)
        self.assertEqual(result['render_evaluations'], 11)
        self.assertLessEqual(result['distance'], result['initial_best_distance'])
        bypass = refine_candidates(self.dry, self.dry, self.sr, [candidate('bypass', 1)], budget=11)
        self.assertEqual(bypass['render_evaluations'], 1)
        self.assertEqual(bypass['distance'], 0)
        with self.assertRaises(ValueError):
            refine_candidates(self.dry, reference, self.sr, candidates, budget=1)

    def test_manifest_reproduction_and_rejects_split_leakage(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for i in range(3):
                path = root / f'take{i}.wav'
                sf.write(path, self.dry * (1 + i * 0.1), self.sr, subtype='FLOAT')
                sources.append({'recording_id': f'take{i}', 'file_path': str(path)})
            manifest = prepare(sources, root / 'prepared', self.config, variations=1)
            rows, _ = load_manifest(manifest)
            self.assertEqual(len(rows), 12)
            for row in rows:
                dry, _ = sf.read(row['dry_path'], dtype='float32')
                wet, _ = sf.read(row['wet_path'], dtype='float32')
                np.testing.assert_array_equal(wet, render_chain(dry, self.sr, row['ground_truth_chain']))
            serialized = [json.loads(line) for line in manifest.read_text().splitlines()]
            serialized[1]['split'] = 'test' if serialized[0]['split'] != 'test' else 'train'
            manifest.write_text('\n'.join(json.dumps(r) for r in serialized))
            with self.assertRaises(ValueError):
                load_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
