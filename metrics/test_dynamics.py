"""Synthetic behavior checks; these do not establish perceptual similarity.

Run: python -m unittest discover -s metrics -p test_dynamics.py -v
"""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from metrics.dynamics import dynamics_distance


def note(rate=16000, attack=0.08, sustain=0.5, release=0.5):
    """A two-second note; changes stay within the named time region."""
    time = np.arange(2 * rate) / rate
    envelope = np.zeros_like(time)
    onset, sustain_start, release_start = 0.1, 0.5, 1.2
    rising = (time >= onset) & (time < sustain_start)
    envelope[rising] = sustain * np.minimum((time[rising] - onset) / attack, 1)
    held = (time >= sustain_start) & (time < release_start)
    envelope[held] = sustain
    falling = time >= release_start
    envelope[falling] = sustain * np.maximum(1 - (time[falling] - release_start) / release, 0)
    return envelope * np.sin(2 * np.pi * 440 * time)


class DynamicsDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rate = 16000
        cls.x = note(cls.rate)

    def test_identical_signal_and_python_float(self):
        distance = dynamics_distance(self.x, self.x.copy(), self.rate)
        self.assertIs(type(distance), float)
        self.assertEqual(distance, 0.0)

    def test_symmetry_for_different_envelopes(self):
        y = note(attack=0.3, sustain=0.2, release=0.7)
        self.assertGreater(dynamics_distance(self.x, y, self.rate), 0)
        self.assertEqual(dynamics_distance(self.x, y, self.rate),
                         dynamics_distance(y, self.x, self.rate))

    def test_gain_difference_matches_decibels(self):
        x = np.full(16321, 0.4)
        for gain in (0.25, 0.5, 2.0):
            with self.subTest(gain=gain):
                self.assertAlmostEqual(dynamics_distance(x, x * gain, self.rate),
                                       abs(20 * np.log10(gain)), places=12)

    def test_attack_difference(self):
        self.assertGreater(dynamics_distance(self.x, note(attack=0.35), self.rate), 0.1)

    def test_sustain_difference_only(self):
        y = self.x.copy()
        y[int(0.5 * self.rate):int(1.2 * self.rate)] *= 0.5
        self.assertGreater(dynamics_distance(self.x, y, self.rate), 1.0)

    def test_release_difference(self):
        self.assertGreater(dynamics_distance(self.x, note(release=0.15), self.rate), 1.0)

    def test_synthetic_compression_changes_loudness_envelope(self):
        # Memoryless synthetic compression; real attack/release tested below.
        y = np.sign(self.x) * np.minimum(np.abs(self.x),
                                       0.1 * (np.abs(self.x) / 0.1) ** 0.25)
        self.assertGreater(dynamics_distance(self.x, y, self.rate), 1.0)

    def test_silence_and_bounded_tiny_noise(self):
        noise = np.random.default_rng(42).uniform(-1e-6, 1e-6, self.x.size)
        silence = np.zeros_like(noise)
        for x, y in ((silence, silence), (silence, noise), (noise, -noise)):
            self.assertEqual(dynamics_distance(x, y, self.rate), 0.0)

    def test_small_noise_on_sound_is_smaller_than_gain_change(self):
        noise = np.random.default_rng(43).normal(0, 1e-6, self.x.size)
        self.assertLess(dynamics_distance(self.x, self.x + noise, self.rate),
                        dynamics_distance(self.x, self.x * 0.5, self.rate))

    def test_quiet_useful_tone_above_floor_and_lost_subfloor_tone(self):
        t = np.arange(self.rate) / self.rate
        for amplitude, retained in ((1e-4, True), (1e-5, False), (1e-6, False)):
            quiet = amplitude * np.sin(2 * np.pi * 440 * t)
            distance = dynamics_distance(np.zeros_like(quiet), quiet, self.rate)
            with self.subTest(amplitude=amplitude):
                if retained:
                    self.assertGreater(distance, 0)
                else:
                    self.assertEqual(distance, 0)

    def test_long_common_silence_and_non_hop_aligned_tails(self):
        # Trim inside the note so incomplete boundary windows are exercised.
        x, y = self.x[:19573], note(attack=0.3, release=0.2)[:19573]
        baseline = dynamics_distance(x, y, self.rate)
        for length in (1, 159, 161, self.rate, 20 * self.rate + 17):
            with self.subTest(length=length):
                self.assertAlmostEqual(dynamics_distance(np.pad(x, (0, length)),
                                                        np.pad(y, (0, length)), self.rate),
                                       baseline, places=12)

    def test_one_sided_sound_is_symmetric_and_not_diluted(self):
        x = self.x[:19573]
        silence = np.zeros_like(x)
        baseline = dynamics_distance(x, silence, self.rate)
        self.assertGreater(baseline, 0)
        for length in (0, 1, 5 * self.rate + 3):
            a, b = np.pad(x, (0, length)), np.pad(silence, (0, length))
            self.assertEqual(dynamics_distance(a, b, self.rate),
                             dynamics_distance(b, a, self.rate))
            self.assertAlmostEqual(dynamics_distance(a, b, self.rate), baseline, places=12)

    def test_quiet_extra_sound_after_common_loud_sound(self):
        t = np.arange(self.rate) / self.rate
        tone = np.sin(2 * np.pi * 440 * t)
        x = np.concatenate([0.5 * tone, np.zeros(2 * self.rate)])
        y = x.copy()
        y[2 * self.rate:] = 1e-4 * tone
        self.assertGreater(dynamics_distance(x, y, self.rate), 1.0)
        self.assertEqual(dynamics_distance(x, y, self.rate), dynamics_distance(y, x, self.rate))

    def test_disjoint_sounds(self):
        silence = np.zeros_like(self.x)
        self.assertGreater(dynamics_distance(np.concatenate([self.x, silence]),
                                            np.concatenate([silence, self.x]), self.rate), 0)

    def test_common_silence_on_hop_grid(self):
        y = self.x * 0.5
        self.assertAlmostEqual(dynamics_distance(np.pad(self.x, (1600, 0)),
                                                np.pad(y, (1600, 0)), self.rate),
                               dynamics_distance(self.x, y, self.rate), places=12)

    def test_multiple_common_rates_keep_gain_difference(self):
        for rate in (8000, 16000, 22050, 44100, 48000, 96000, 192000):
            with self.subTest(rate=rate):
                x = note(rate)
                self.assertAlmostEqual(dynamics_distance(x, x * 0.5, rate),
                                       20 * np.log10(2), places=10)

    def test_two_source_rates_cannot_be_passed_as_common_rate(self):
        with self.assertRaisesRegex(ValueError, "one common"):
            dynamics_distance(self.x, self.x, (16000, 48000))
        # Arrays from different rates with equal duration also differ in length.
        with self.assertRaisesRegex(ValueError, "same length"):
            dynamics_distance(note(16000), note(48000), 16000)
        # Equal-length arrays have no metadata: source-rate equality is the
        # caller's responsibility, explicitly documented in the public API.

    def test_no_mutation_and_supported_numeric_types(self):
        for dtype in (np.float32, np.float64, np.int16, np.uint16):
            x = np.array([0, 1, 2, 3, 2, 1], dtype=dtype)
            original = x.copy()
            self.assertEqual(dynamics_distance(x, x, np.int64(16000)), 0.0)
            np.testing.assert_array_equal(x, original)
        self.assertEqual(dynamics_distance([0.0, 1.0], [0.0, 1.0], 16000.0), 0.0)

    def test_short_inputs_and_low_supported_rate(self):
        for rate in (1, 10, 16000, 768000):
            self.assertEqual(dynamics_distance([0.0], [0.0], rate), 0.0)
            self.assertAlmostEqual(dynamics_distance([0.4], [0.2], rate),
                                   20 * np.log10(2), places=12)

    def test_equal_rms_does_not_mean_equal_waveforms(self):
        self.assertEqual(dynamics_distance(self.x, -self.x, self.rate), 0.0)


class DynamicsInputTests(unittest.TestCase):
    def test_invalid_waveforms_on_either_side(self):
        bad_values = ([], 1.0, None, [[1.0]], np.zeros((2, 2, 1)),
                      ["1"], [True], [1j], [None], {"x": 1}, [[1], [1, 2]],
                      [float("nan")], [float("inf")], [-float("inf")],
                      [10 ** 400], np.array([1], dtype=object))
        for bad in bad_values:
            for x, y in ((bad, [0.0]), ([0.0], bad)):
                with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                    dynamics_distance(x, y, 16000)

    def test_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            dynamics_distance([0.0], [0.0, 0.0], 16000)

    def test_invalid_sample_rates(self):
        for rate in (0, -1, 0.5, 16000.5, True, np.bool_(True), "16000", None,
                     float("nan"), float("inf"), 1e308, 10 ** 400, 768001,
                     [16000], np.array(16000), 16000 + 0j):
            with self.subTest(rate=repr(rate)), self.assertRaises(ValueError):
                dynamics_distance([0.0], [0.0], rate)

    def test_extreme_amplitudes_raise_clear_error(self):
        for value in (1e160, 1e308, -1e308):
            for x, y in (([value], [0.0]), ([0.0], [value]), ([value], [value])):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "RMS"):
                    dynamics_distance(x, y, 16000)

    def test_large_but_representable_amplitudes(self):
        x = np.full(1000, 1e150)
        distance = dynamics_distance(x, x * 0.5, 16000)
        self.assertTrue(np.isfinite(distance))
        self.assertAlmostEqual(distance, 20 * np.log10(2), places=10)


@unittest.skipUnless(importlib.util.find_spec("pedalboard"), "Optional pedalboard is not installed")
class DynamicsCompressorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The team's module creates data directories at import time. Load it
        # in a temporary cwd so no repository data or WAV files are created.
        source = Path(__file__).resolve().parents[1] / "scripts" / "Create_Synthetic_pairs.py"
        spec = importlib.util.spec_from_file_location("dynamics_team_renderer", source)
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
        cls.dry = note(cls.rate)
        cls.params = {"threshold_db": -24.0, "ratio": 4.0,
                      "attack_ms": 5.0, "release_ms": 100.0}

    def test_pedalboard_compressor_before_after(self):
        from pedalboard import Compressor, Pedalboard
        wet = Pedalboard([Compressor(**self.params)])(self.dry, self.rate)
        self.assertEqual(wet.shape, self.dry.shape)
        self.assertGreater(dynamics_distance(self.dry, wet, self.rate), 1.0)
        self.assertEqual(dynamics_distance(self.dry, wet, self.rate),
                         dynamics_distance(wet, self.dry, self.rate))

    def test_team_renderer_compression_enabled_vs_ratio_one(self):
        # Keep the renderer's required overdrive stage identical in both runs.
        drive = {"drive": 0.0, "level": 0.0}
        before = self.render(self.dry, self.rate, {**self.params, "ratio": 1.0}, "overdrive", drive)
        after = self.render(self.dry, self.rate, self.params, "overdrive", drive)
        repeated = self.render(self.dry, self.rate, self.params, "overdrive", drive)
        self.assertGreater(dynamics_distance(before, after, self.rate), 1.0)
        self.assertEqual(dynamics_distance(after, repeated, self.rate), 0.0)

    def test_team_renderer_attack_and_release_settings(self):
        drive = {"drive": 0.0, "level": 0.0}
        target = self.render(self.dry, self.rate, self.params, "overdrive", drive)
        for changes in ({"attack_ms": 100.0}, {"release_ms": 1000.0}):
            wet = self.render(self.dry, self.rate, {**self.params, **changes}, "overdrive", drive)
            with self.subTest(changes=changes):
                self.assertGreater(dynamics_distance(target, wet, self.rate), 0.01)


if __name__ == "__main__":
    unittest.main()
