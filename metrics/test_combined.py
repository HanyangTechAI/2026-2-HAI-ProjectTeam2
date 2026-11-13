"""Numerical safety/compatibility checks, not perceptual calibration.

Every weight and scale below is an arbitrary TEST setting, not a recommendation.
Run: python -m unittest discover -s metrics -p test_combined.py -v
"""

from dataclasses import asdict
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from metrics.combined import combined_distance, combined_distance_details
from metrics.dynamics import dynamics_distance
from metrics.spectral import spectral_distance


TEST_SETTINGS = dict(spectral_weight=3.0, dynamics_weight=2.0,
                     spectral_scale=0.2, dynamics_scale=8.0)


def synthetic_note(rate=16000, attack=0.08):
    """Two-second 440 Hz note, peak .5, 80 ms attack and 500 ms release."""
    time = np.arange(2 * rate) / rate
    envelope = np.zeros_like(time)
    rising = (time >= 0.1) & (time < 0.5)
    envelope[rising] = 0.5 * np.minimum((time[rising] - 0.1) / attack, 1)
    envelope[(time >= 0.5) & (time < 1.2)] = 0.5
    falling = time >= 1.2
    envelope[falling] = 0.5 * np.maximum(1 - (time[falling] - 1.2) / 0.5, 0)
    return envelope * np.sin(2 * np.pi * 440 * time)


class CombinedDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rate = 16000
        cls.x = synthetic_note(cls.rate)

    def details(self, x, y, **settings):
        return combined_distance_details(x, y, self.rate, **(TEST_SETTINGS | settings))

    def assert_consistent(self, x, y, **settings):
        config = TEST_SETTINGS | settings
        result = self.details(x, y, **settings)
        self.assertEqual(result.spectral, spectral_distance(x, y))
        self.assertEqual(result.dynamics, dynamics_distance(x, y, self.rate))
        sw = config['spectral_weight'] / (config['spectral_weight'] + config['dynamics_weight'])
        dw = config['dynamics_weight'] / (config['spectral_weight'] + config['dynamics_weight'])
        expected = (sw * result.spectral / config['spectral_scale']
                    + dw * result.dynamics / config['dynamics_scale'])
        self.assertAlmostEqual(result.combined, expected, places=12)
        self.assertAlmostEqual(result.spectral_weight + result.dynamics_weight, 1.0)
        self.assertEqual(result.combined, combined_distance(x, y, self.rate, **config))
        for value in asdict(result).values():
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        return result

    def test_identical_waveform(self):
        result = self.assert_consistent(self.x, self.x.copy())
        for name in ('combined', 'spectral', 'dynamics', 'scaled_spectral', 'scaled_dynamics'):
            self.assertEqual(getattr(result, name), 0.0)

    def test_symmetry_including_all_components(self):
        y = synthetic_note(attack=0.35) * 0.4
        forward = self.assert_consistent(self.x, y)
        self.assertGreater(forward.combined, 0.0)
        self.assertEqual(forward, self.details(y, self.x))

    def test_spectral_only_and_unit_scale_compatibility(self):
        for scale in (1.0, 0.3, 7.0):
            with self.subTest(scale=scale):
                result = self.assert_consistent(self.x, self.x * 0.5,
                                                spectral_weight=7, dynamics_weight=0,
                                                spectral_scale=scale)
                self.assertEqual(result.combined, spectral_distance(self.x, self.x * 0.5) / scale)
                self.assertGreater(result.dynamics, 0)

    def test_dynamics_only_and_unit_scale_compatibility(self):
        for scale in (1.0, 0.3, 7.0):
            with self.subTest(scale=scale):
                result = self.assert_consistent(self.x, self.x * 0.5,
                                                spectral_weight=0, dynamics_weight=7,
                                                dynamics_scale=scale)
                self.assertEqual(result.combined, dynamics_distance(self.x, self.x * 0.5, self.rate) / scale)
                self.assertGreater(result.spectral, 0)

    def test_arbitrary_weights_scales_and_recorded_settings(self):
        result = self.assert_consistent(self.x, synthetic_note(attack=0.25),
                                        spectral_weight=2, dynamics_weight=7,
                                        spectral_scale=0.37, dynamics_scale=4.2)
        self.assertAlmostEqual(result.spectral_weight, 2 / 9)
        self.assertAlmostEqual(result.dynamics_weight, 7 / 9)
        self.assertEqual(result.scaled_spectral, result.spectral / 0.37)
        self.assertEqual(result.scaled_dynamics, result.dynamics / 4.2)
        self.assertEqual(result.spectral_scale, 0.37)
        self.assertEqual(result.dynamics_scale, 4.2)

    def test_weight_change_moves_toward_selected_scaled_score(self):
        x, y = np.full(4096, 0.4), np.full(4096, 0.2)
        scores = [self.details(x, y, spectral_weight=1 - w, dynamics_weight=w,
                               spectral_scale=2, dynamics_scale=3)
                  for w in (0, 0.25, 0.75, 1)]
        self.assertLess(scores[0].scaled_spectral, scores[0].scaled_dynamics)
        self.assertTrue(all(a.combined < b.combined for a, b in zip(scores, scores[1:])))
        self.assertEqual(scores[0].combined, scores[0].scaled_spectral)
        self.assertEqual(scores[-1].combined, scores[-1].scaled_dynamics)

    def test_common_weight_multiplier_does_not_change_result(self):
        expected = self.details(self.x, self.x * 0.5)
        for sw, dw in ((30, 20), (0.3, 0.2), (3e307, 2e307)):
            result = self.details(self.x, self.x * 0.5, spectral_weight=sw, dynamics_weight=dw)
            self.assertAlmostEqual(expected.combined, result.combined, places=12)

    def test_huge_weights_do_not_overflow_their_sum(self):
        result = self.details(self.x, self.x * 0.5, spectral_weight=1e308, dynamics_weight=1e308)
        self.assertEqual(result.spectral_weight, 0.5)
        self.assertEqual(result.dynamics_weight, 0.5)
        self.assertTrue(math.isfinite(result.combined))

    def test_scale_change_divides_component_and_preserves_raw_scores(self):
        a = self.details(self.x, self.x * 0.5)
        b = self.details(self.x, self.x * 0.5, spectral_scale=0.4, dynamics_scale=16)
        self.assertEqual((a.spectral, a.dynamics), (b.spectral, b.dynamics))
        self.assertEqual(b.scaled_spectral, a.scaled_spectral / 2)
        self.assertEqual(b.scaled_dynamics, a.scaled_dynamics / 2)
        self.assertEqual(b.combined, a.combined / 2)

    def test_gain_only_is_retained_without_input_normalization(self):
        result = self.assert_consistent(self.x, self.x * 0.5)
        self.assertGreater(result.spectral, 0)
        self.assertAlmostEqual(result.dynamics, 20 * np.log10(2), places=10)
        self.assertGreater(result.combined, 0)

    def test_synthetic_compression(self):
        wet = np.sign(self.x) * np.minimum(np.abs(self.x),
                                         0.1 * (np.abs(self.x) / 0.1) ** 0.25)
        result = self.assert_consistent(self.x, wet)
        self.assertGreater(result.spectral, 0)
        self.assertGreater(result.dynamics, 0)
        self.assertGreater(result.combined, 0)

    def test_attack_and_sustain_changes(self):
        sustain = self.x.copy()
        sustain[int(0.5 * self.rate):int(1.2 * self.rate)] *= 0.5
        for y in (synthetic_note(attack=0.35), sustain):
            with self.subTest(change='attack' if y is not sustain else 'sustain'):
                result = self.assert_consistent(self.x, y)
                self.assertGreater(result.spectral, 0)
                self.assertGreater(result.dynamics, 0)
                self.assertGreater(result.combined, 0)

    def test_silence_and_small_noise(self):
        noise = np.random.default_rng(42).uniform(-1e-6, 1e-6, self.x.size)
        silence = np.zeros_like(noise)
        for x, y in ((silence, silence), (silence, noise), (noise, -noise)):
            result = self.assert_consistent(x, y)
            self.assertEqual(result.combined, 0)

    def test_two_floors_are_not_treated_as_the_same_threshold(self):
        time = np.arange(self.rate) / self.rate
        quiet = 5e-5 * np.sin(2 * np.pi * (self.rate * 41 / 2048) * time)
        result = self.assert_consistent(np.zeros_like(quiet), quiet)
        self.assertEqual(result.spectral, 0)
        self.assertGreater(result.dynamics, 0)
        self.assertGreater(result.combined, 0)

    def test_long_appended_common_silence_including_partial_windows(self):
        x, y = self.x[:19573], synthetic_note(attack=0.3)[:19573]
        baseline = self.details(x, y)
        for length in (1, 159, 513, self.rate, 20 * self.rate + 17):
            result = self.details(np.pad(x, (0, length)), np.pad(y, (0, length)))
            with self.subTest(length=length):
                for field in ('spectral', 'dynamics', 'combined'):
                    self.assertAlmostEqual(getattr(result, field), getattr(baseline, field), places=12)

    def test_one_sided_sound(self):
        result = self.assert_consistent(self.x, np.zeros_like(self.x))
        self.assertGreater(result.spectral, 0)
        self.assertGreater(result.dynamics, 0)
        self.assertEqual(result, self.details(np.zeros_like(self.x), self.x))

    def test_existing_calls_numeric_types_and_no_mutation(self):
        for dtype in (np.float32, np.float64, np.int16, np.uint16):
            x = np.array([0, 1, 2, 3, 2, 1], dtype=dtype)
            y = x * 0.5
            originals = x.copy(), y.copy()
            self.assert_consistent(x, y)
            np.testing.assert_array_equal(x, originals[0])
            np.testing.assert_array_equal(y, originals[1])
        self.assert_consistent([0.0, 0.4], [0.0, 0.2])

    def test_supported_sample_rates_and_short_signals(self):
        for rate in (1, 8000, 16000.0, np.int64(44100), 192000, 768000):
            with self.subTest(rate=rate):
                result = combined_distance_details([0.4], [0.2], rate, **TEST_SETTINGS)
                self.assertAlmostEqual(result.dynamics, 20 * np.log10(2), places=12)
                self.assertEqual(result.spectral, spectral_distance([0.4], [0.2]))
                self.assertGreater(result.combined, 0)

    def test_zero_weight_still_calls_both_original_functions(self):
        for sw, dw in ((1, 0), (0, 1)):
            with self.subTest(weights=(sw, dw)), \
                    patch('metrics.combined.spectral_distance', wraps=spectral_distance) as spectral, \
                    patch('metrics.combined.dynamics_distance', wraps=dynamics_distance) as dynamics:
                self.details(self.x, self.x, spectral_weight=sw, dynamics_weight=dw)
                spectral.assert_called_once_with(self.x, self.x)
                dynamics.assert_called_once_with(self.x, self.x, self.rate)


class CombinedInputTests(unittest.TestCase):
    def distance(self, x=(0.4,), y=(0.2,), rate=16000, **settings):
        return combined_distance(x, y, rate, **(TEST_SETTINGS | settings))

    def test_all_configuration_is_explicit(self):
        for missing in TEST_SETTINGS:
            settings = {k: v for k, v in TEST_SETTINGS.items() if k != missing}
            for function in (combined_distance, combined_distance_details):
                with self.subTest(missing=missing, function=function.__name__), self.assertRaises(TypeError):
                    function([0.4], [0.2], 16000, **settings)

    def test_invalid_weights(self):
        invalid = (-1, float('nan'), float('inf'), -float('inf'), True, np.bool_(False),
                   '1', None, [1], np.array(1), 1j, 10 ** 400)
        for name in ('spectral_weight', 'dynamics_weight'):
            for value in invalid:
                with self.subTest(name=name, value=repr(value)), self.assertRaisesRegex(ValueError, name):
                    self.distance(**{name: value})
        with self.assertRaisesRegex(ValueError, 'At least one'):
            self.distance(spectral_weight=0, dynamics_weight=0)

    def test_invalid_scales_including_zero_weight_component(self):
        invalid = (0, -1, float('nan'), float('inf'), -float('inf'), True, np.bool_(True),
                   '1', None, [1], np.array(1), 1j, 10 ** 400)
        for name in ('spectral_scale', 'dynamics_scale'):
            for value in invalid:
                for sw, dw in ((1, 0), (0, 1)):
                    with self.subTest(name=name, value=repr(value), weights=(sw, dw)), \
                            self.assertRaisesRegex(ValueError, name):
                        self.distance(**{name: value}, spectral_weight=sw, dynamics_weight=dw)

    def test_invalid_sample_rates_even_for_spectral_only(self):
        for rate in (0, -1, 0.5, 16000.5, True, np.bool_(True), '16000', None,
                     float('nan'), float('inf'), 10 ** 400, 768001,
                     (16000, 48000), np.array(16000), 16000 + 0j):
            for sw, dw in ((1, 0), (0, 1), (3, 2)):
                with self.subTest(rate=repr(rate), weights=(sw, dw)), self.assertRaises(ValueError):
                    self.distance(rate=rate, spectral_weight=sw, dynamics_weight=dw)

    def test_abnormal_inputs_on_either_side_even_with_zero_weight(self):
        invalid = ([], 1.0, None, [[1.0]], np.zeros((2, 2, 1)), ['1'], [True],
                   [1j], [None], {'x': 1}, [[1], [1, 2]], [float('nan')],
                   [float('inf')], [-float('inf')], [10 ** 400], np.array([1], dtype=object))
        for bad in invalid:
            for x, y in ((bad, [0.0]), ([0.0], bad)):
                for sw, dw in ((1, 0), (0, 1), (3, 2)):
                    with self.subTest(bad=repr(bad), weights=(sw, dw)), self.assertRaises(ValueError):
                        self.distance(x, y, spectral_weight=sw, dynamics_weight=dw)

    def test_mismatched_lengths(self):
        for x, y in (([0.0], [0.0, 0.0]), ([0.0, 0.0], [0.0])):
            with self.assertRaisesRegex(ValueError, 'same length'):
                self.distance(x, y)

    def test_identical_extreme_values_do_not_bypass_existing_validation(self):
        for sw, dw in ((1, 0), (0, 1), (3, 2)):
            for value in (1e160, 1e308, -1e308):
                with self.subTest(value=value, weights=(sw, dw)), self.assertRaisesRegex(ValueError, 'RMS'):
                    self.distance([value], [value], spectral_weight=sw, dynamics_weight=dw)

    def test_scaled_overflow_is_rejected_even_for_zero_weight(self):
        for name in ('spectral_scale', 'dynamics_scale'):
            for sw, dw in ((1, 0), (0, 1)):
                with self.subTest(name=name, weights=(sw, dw)), self.assertRaisesRegex(ValueError, 'scaled_'):
                    self.distance(**{name: float(np.nextafter(0.0, 1.0))},
                                  spectral_weight=sw, dynamics_weight=dw)

    def test_numpy_real_configuration_becomes_python_floats(self):
        result = combined_distance_details([0.4], [0.2], 16000,
                                          spectral_weight=np.int64(3), dynamics_weight=np.float32(2),
                                          spectral_scale=np.float64(0.2), dynamics_scale=np.int32(8))
        for value in asdict(result).values():
            self.assertIs(type(value), float)


@unittest.skipUnless(importlib.util.find_spec('pedalboard'), 'Optional pedalboard is not installed')
class CombinedCompressorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The renderer creates directories on import; isolate those side effects.
        source = Path(__file__).resolve().parents[1] / 'scripts' / 'Create_Synthetic_pairs.py'
        spec = importlib.util.spec_from_file_location('combined_team_renderer', source)
        module = importlib.util.module_from_spec(spec)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                spec.loader.exec_module(module)
            finally:
                os.chdir(previous)
        cls.render = staticmethod(module.apply_effects)
        cls.rate = 16000
        cls.dry = synthetic_note(cls.rate)
        cls.params = dict(threshold_db=-24.0, ratio=4.0, attack_ms=5.0, release_ms=100.0)
        cls.drive = dict(drive=0.0, level=0.0)

    def check_scores(self, x, y):
        result = combined_distance_details(x, y, self.rate, **TEST_SETTINGS)
        self.assertEqual(result.spectral, spectral_distance(x, y))
        self.assertEqual(result.dynamics, dynamics_distance(x, y, self.rate))
        self.assertGreater(result.spectral, 0)
        self.assertGreater(result.dynamics, 0)
        self.assertTrue(math.isfinite(result.combined))
        self.assertGreater(result.combined, 0)
        self.assertAlmostEqual(result.combined,
                               0.6 * result.spectral / 0.2 + 0.4 * result.dynamics / 8, places=12)
        self.assertEqual(result, combined_distance_details(y, x, self.rate, **TEST_SETTINGS))

    def test_actual_compressor_before_after(self):
        from pedalboard import Compressor, Pedalboard
        wet = Pedalboard([Compressor(**self.params)])(self.dry, self.rate)
        self.assertEqual(wet.shape, self.dry.shape)
        self.check_scores(self.dry, wet)

    def test_team_renderer_ratio_one_vs_four_and_repeatability(self):
        before = self.render(self.dry, self.rate, self.params | {'ratio': 1.0}, 'overdrive', self.drive)
        after = self.render(self.dry, self.rate, self.params, 'overdrive', self.drive)
        repeated = self.render(self.dry, self.rate, self.params, 'overdrive', self.drive)
        self.check_scores(before, after)
        self.assertEqual(combined_distance(after, repeated, self.rate, **TEST_SETTINGS), 0.0)

    def test_team_renderer_attack_and_release_changes(self):
        target = self.render(self.dry, self.rate, self.params, 'overdrive', self.drive)
        for changes in ({'attack_ms': 100.0}, {'release_ms': 1000.0}):
            with self.subTest(changes=changes):
                wet = self.render(self.dry, self.rate, self.params | changes, 'overdrive', self.drive)
                self.check_scores(target, wet)


if __name__ == '__main__':
    unittest.main()
