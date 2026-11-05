"""Check spectral distance with deterministic signals (original seven kept)."""

import unittest
import warnings

import numpy as np

from spectral import spectral_distance


class SpectralDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample_rate = 22050
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        cls.x = 0.5 * np.sin(2 * np.pi * 440 * time)
        cls.y = 0.5 * np.sin(2 * np.pi * 880 * time)

    def test_identical_signal(self):
        self.assertEqual(spectral_distance(self.x, self.x.copy()), 0.0)

    def test_different_signals(self):
        self.assertGreater(spectral_distance(self.x, self.y), 0.0)

    def test_symmetry(self):
        self.assertEqual(
            spectral_distance(self.x, self.y),
            spectral_distance(self.y, self.x),
        )

    def test_amplitude_difference(self):
        self.assertGreater(spectral_distance(self.x, 0.5 * self.x), 0.0)

    def test_different_lengths(self):
        for x, y in ((self.x, self.y[:-1]), (self.x[:-1], self.y)):
            with self.subTest(x_length=len(x), y_length=len(y)):
                with self.assertRaises(ValueError):
                    spectral_distance(x, y)

    def test_requires_one_dimensional_waveforms(self):
        for x, y in ((self.x[:, None], self.y), (self.x, self.y[:, None])):
            with self.subTest(x_shape=x.shape, y_shape=y.shape):
                with self.assertRaises(ValueError):
                    spectral_distance(x, y)

    def test_identical_silence(self):
        silence = np.zeros_like(self.x)
        self.assertEqual(spectral_distance(silence, silence.copy()), 0.0)


class SpectralFloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time = np.arange(22050, dtype=np.float64) / 22050
        cls.tone = 0.5 * np.sin(2 * np.pi * 440 * cls.time)
        cls.noise = np.random.default_rng(42).normal(0, 1e-6, cls.time.size)

    def test_silence_and_tiny_noise(self):
        self.assertEqual(spectral_distance(np.zeros_like(self.noise), self.noise), 0.0)

    def test_tone_and_tiny_noise(self):
        tiny = spectral_distance(self.tone, self.tone + self.noise)
        volume = spectral_distance(self.tone, 0.5 * self.tone)
        self.assertLess(tiny, volume)

    def test_quiet_coherent_tone_remains_distinguishable(self):
        # Preserve an above-floor quiet tone; the old 1e-5 tone is now lost.
        # That exact loss is checked separately below rather than hidden.
        quiet = 1e-3 * np.sin(2 * np.pi * (22050 * 41 / 2048) * self.time)
        self.assertGreater(spectral_distance(np.zeros_like(quiet), quiet), 1e-3)

    def test_meaningful_broadband_volume_difference(self):
        noise = np.random.default_rng(43).normal(0, 0.1, self.time.size)
        # Some random bins now cross the -80 dB floor. For a gain of 1/2,
        # each clamped log error is in [0, ln(2)], not identically ln(2).
        # An unclipped broadband control below still requires exact ln(2).
        distance = spectral_distance(noise, noise * 0.5)
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, np.log(2))
        self.assertAlmostEqual(distance, 0.6924778711500804, places=10)

    def test_frequency_difference_exceeds_tiny_noise(self):
        other = 0.5 * np.sin(2 * np.pi * 880 * self.time)
        self.assertGreater(spectral_distance(self.tone, other),
                           spectral_distance(self.tone, self.tone + self.noise))

    def test_silence_tail_noise(self):
        signal = np.concatenate([self.tone, np.zeros_like(self.tone)])
        altered = signal.copy()
        altered[len(self.tone):] = self.noise
        self.assertLess(spectral_distance(signal, altered), spectral_distance(signal, signal * 0.5))

    def test_empty_input(self):
        for x, y in ((np.array([]), np.array([])), (np.array([]), self.tone)):
            with self.subTest(lengths=(len(x), len(y))), self.assertRaises(ValueError):
                spectral_distance(x, y)

    def test_nonfinite_inputs_on_either_side(self):
        for value in (np.nan, np.inf, -np.inf):
            bad = self.tone.copy()
            bad[123] = value
            for x, y in ((bad, self.tone), (self.tone, bad)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    spectral_distance(x, y)

    def test_scalar_and_higher_dimensions(self):
        for bad in (np.array(1.0), np.zeros((1, 1, 1))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                spectral_distance(bad, bad)

    def test_nonnumeric_and_complex_inputs(self):
        for bad in (np.array(["1"]), np.array([True]), np.array([1j]), np.array([None])):
            with self.subTest(dtype=bad.dtype), self.assertRaises(ValueError):
                spectral_distance(bad, bad)

    def test_float32_and_float64_finite_float_return(self):
        for dtype in (np.float32, np.float64):
            x = self.tone.astype(dtype)
            distance = spectral_distance(x, self.noise.astype(dtype))
            self.assertIs(type(distance), float)
            self.assertTrue(np.isfinite(distance))
            self.assertGreaterEqual(distance, 0)
            self.assertEqual(distance, spectral_distance(self.noise.astype(dtype), x))

    def test_short_nonempty_input(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # librosa warns about n_fft > length
            self.assertEqual(spectral_distance(np.zeros(1), np.zeros(1)), 0.0)

    def test_stft_overflow_rejected(self):
        huge = np.full(4096, 1e308)
        with self.assertRaises(ValueError):
            spectral_distance(huge, huge)


class SpectralSilenceDurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rate = 22050
        time = np.arange(cls.rate) / cls.rate
        cls.tone = 0.5 * np.sin(2 * np.pi * 440 * time)

    def test_appended_silence_does_not_dilute_gain_difference(self):
        baseline = spectral_distance(self.tone, self.tone * 0.5)
        for seconds in (1, 2, 5, 20):
            padded = np.pad(self.tone, (0, seconds * self.rate))
            with self.subTest(seconds=seconds):
                self.assertAlmostEqual(spectral_distance(padded, padded * 0.5), baseline, places=12)

    def test_non_hop_aligned_silence_lengths(self):
        baseline = spectral_distance(self.tone, self.tone * 0.5)
        for samples in (1, 99, 511, 513, 22051):
            padded = np.pad(self.tone, (0, samples))
            with self.subTest(samples=samples):
                self.assertAlmostEqual(spectral_distance(padded, padded * 0.5), baseline, places=12)

    def test_leading_silence_on_the_same_frame_grid(self):
        baseline = spectral_distance(self.tone, self.tone * 0.5)
        for samples in (512, 4096, 512 * 100):
            padded = np.pad(self.tone, (samples, 0))
            with self.subTest(samples=samples):
                self.assertAlmostEqual(spectral_distance(padded, padded * 0.5), baseline, places=12)

    def test_one_sided_signal_is_included_and_symmetric(self):
        silence = np.zeros_like(self.tone)
        baseline = spectral_distance(self.tone, silence)
        self.assertGreater(baseline, 0)
        for seconds in (0, 2, 5):
            a, b = np.pad(self.tone, (0, seconds * self.rate)), np.pad(silence, (0, seconds * self.rate))
            distance = spectral_distance(a, b)
            self.assertIs(type(distance), float)
            self.assertEqual(distance, spectral_distance(b, a))
            self.assertAlmostEqual(distance, baseline, places=12)

    def test_extra_sound_after_a_common_burst_is_not_discarded(self):
        silence = np.zeros_like(self.tone)
        truth = np.concatenate([self.tone, silence, silence])
        extra_sound = np.concatenate([self.tone, silence, self.tone])
        self.assertEqual(spectral_distance(truth, truth), 0)
        self.assertGreater(spectral_distance(truth, extra_sound), 0)
        self.assertEqual(spectral_distance(truth, extra_sound), spectral_distance(extra_sound, truth))

    def test_disjoint_one_sided_bursts_are_not_ignored(self):
        silence = np.zeros_like(self.tone)
        a = np.concatenate([self.tone, silence, silence])
        b = np.concatenate([silence, silence, self.tone])
        self.assertGreater(spectral_distance(a, b), 0)

    def test_quiet_one_sided_burst_has_nonzero_weight(self):
        silence = np.zeros_like(self.tone)
        a = np.concatenate([self.tone, silence, silence])
        # Peak amplitude 1e-3 remains above the new floor. The old 1e-5
        # event is explicitly expected to disappear in a separate test.
        b = np.concatenate([self.tone, silence, self.tone * 2e-3])
        self.assertGreater(spectral_distance(a, b), 0)

    def test_long_silence_and_independent_subfloor_noise(self):
        rng = np.random.default_rng(45)
        silence = np.zeros(self.rate * 6)
        a, b = rng.normal(0, 1e-6, silence.size), rng.normal(0, 1e-6, silence.size)
        for x, y in ((silence, silence), (silence, a), (a, b)):
            self.assertEqual(spectral_distance(x, y), 0.0)

    def test_subfloor_noise_tail_has_small_effect(self):
        baseline = spectral_distance(self.tone, self.tone * 0.5)
        rng = np.random.default_rng(46)
        for seconds in (1, 2, 5):
            a = np.concatenate([self.tone, rng.normal(0, 1e-6, self.rate * seconds)])
            b = np.concatenate([self.tone * 0.5, rng.normal(0, 1e-6, self.rate * seconds)])
            self.assertAlmostEqual(spectral_distance(a, b), baseline, delta=1e-5)

    def test_large_finite_inputs_do_not_overflow_energy_weights(self):
        distance = spectral_distance(self.tone * 1e160, self.tone * 0.5e160)
        self.assertIs(type(distance), float)
        self.assertTrue(np.isfinite(distance))
        self.assertGreater(distance, 0)


class SpectralFloor80Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time = np.arange(22050) / 22050
        cls.bin_tone = np.sin(2 * np.pi * (22050 * 41 / 2048) * cls.time)

    def test_requested_floor_and_configuration_id(self):
        from spectral import MAGNITUDE_FLOOR, METRIC_VERSION, NOISE_FLOOR_DB
        self.assertEqual(NOISE_FLOOR_DB, -80.0)
        self.assertAlmostEqual(MAGNITUDE_FLOOR, 0.0512, places=15)
        self.assertEqual(METRIC_VERSION, "log_stft_joint_active_energy_v2_floor_m80db")

    def test_previous_quiet_tone_is_exactly_lost(self):
        quiet = 1e-5 * self.bin_tone
        self.assertGreater(float(np.max(np.abs(quiet))), 0)
        self.assertEqual(spectral_distance(np.zeros_like(quiet), quiet), 0.0)

    def test_quiet_tones_above_and_below_new_floor(self):
        silence = np.zeros_like(self.bin_tone)
        self.assertEqual(spectral_distance(silence, 5e-5 * self.bin_tone), 0.0)
        self.assertGreater(spectral_distance(silence, 2e-4 * self.bin_tone), 0.0)

    def test_previous_one_sided_quiet_event_is_lost(self):
        head = 0.5 * np.sin(2 * np.pi * 440 * self.time)
        silence = np.zeros_like(head)
        x = np.concatenate([head, silence, silence])
        y = np.concatenate([head, silence, 1e-5 * self.bin_tone])
        self.assertEqual(spectral_distance(x, y), 0.0)
        self.assertEqual(spectral_distance(y, x), 0.0)

    def test_unclipped_broadband_gain_retains_exact_log_ratio(self):
        # An aligned impulse has equal magnitudes in every bin. Its active
        # Hann windows have magnitudes 1 or 0.5, so BOTH gains exceed 0.0512.
        impulse = np.zeros(4096)
        impulse[2048] = 1.0
        self.assertAlmostEqual(spectral_distance(impulse, impulse * 0.5), np.log(2), places=12)

    def test_two_different_subfloor_frequencies_are_indistinguishable(self):
        x = 1e-5 * self.bin_tone
        y = 1e-5 * np.sin(2 * np.pi * (22050 * 82 / 2048) * self.time)
        self.assertFalse(np.array_equal(x, y))
        self.assertEqual(spectral_distance(x, y), 0.0)

    def test_silence_extension_preserves_lost_and_retained_quiet_cases(self):
        for amplitude in (1e-5, 1e-3):
            tone = amplitude * self.bin_tone
            silence = np.zeros_like(tone)
            baseline = spectral_distance(silence, tone)
            for seconds in (2, 5):
                with self.subTest(amplitude=amplitude, seconds=seconds):
                    self.assertAlmostEqual(spectral_distance(np.pad(silence, (0, seconds * 22050)),
                                                           np.pad(tone, (0, seconds * 22050))),
                                           baseline, places=12)


if __name__ == "__main__":
    unittest.main()
