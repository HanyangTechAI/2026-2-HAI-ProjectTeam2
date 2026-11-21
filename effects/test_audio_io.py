"""effects/audio_io.py 테스트: python -m unittest effects.test_audio_io"""
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from effects.audio_io import STANDARD_SAMPLE_RATE, read_mono, to_mono


class ReadMonoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = np.arange(4410, dtype=np.float32) / 44100
        self.wave = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, audio, sr):
        path = os.path.join(self.tmp.name, name)
        sf.write(path, audio, sr, subtype="FLOAT")
        return path

    def test_mono_44k_unchanged(self):
        audio, sr = read_mono(self.write("mono.wav", self.wave, 44100))
        self.assertEqual(sr, STANDARD_SAMPLE_RATE)
        np.testing.assert_array_equal(audio, self.wave)

    def test_stereo_is_channel_average(self):
        stereo = np.column_stack([self.wave, 0.5 * self.wave])
        audio, _ = read_mono(self.write("stereo.wav", stereo, 44100))
        self.assertEqual(audio.ndim, 1)
        np.testing.assert_allclose(audio, 0.75 * self.wave, atol=1e-7)

    def test_resampled_to_standard_rate(self):
        audio, sr = read_mono(self.write("48k.wav", self.wave, 48000))
        self.assertEqual(sr, 44100)
        self.assertEqual(len(audio), round(len(self.wave) * 44100 / 48000))
        self.assertEqual(audio.dtype, np.float32)

    def test_keep_native_rate(self):
        _, sr = read_mono(self.write("48k.wav", self.wave, 48000), sample_rate=None)
        self.assertEqual(sr, 48000)

    def test_rejects_empty_and_nonfinite(self):
        bad = self.wave.copy()
        bad[10] = np.nan
        with self.assertRaises(ValueError):
            read_mono(self.write("nan.wav", bad, 44100))
        with self.assertRaises(ValueError):
            read_mono(self.write("empty.wav", np.zeros(0, np.float32), 44100))

    def test_to_mono_shapes(self):
        self.assertEqual(to_mono(np.zeros((10, 2))).shape, (10,))
        with self.assertRaises(ValueError):
            to_mono(np.zeros((2, 3, 4)))


if __name__ == "__main__":
    unittest.main()
