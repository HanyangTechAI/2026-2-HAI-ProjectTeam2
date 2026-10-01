"""Check the baseline spectral distance with deterministic signals."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
