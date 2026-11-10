"""effects/render.py 테스트: python -m unittest discover -s effects -p "test_*.py" """
import unittest

import numpy as np
from pedalboard import Pedalboard, Compressor, Distortion, Gain, PeakFilter, LowShelfFilter, HighShelfFilter

from effects import build_effect, load_effects, render_chain

SR = 44100
t = np.arange(SR, dtype=np.float32) / SR
DRY = (0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-3 * t)).astype(np.float32)


class RenderChainTest(unittest.TestCase):
    def assert_same(self, chain, board):
        np.testing.assert_array_equal(render_chain(DRY, SR, chain), board(DRY, SR))

    def test_drive_types_match_manual_pedalboard(self):
        # Step 0 정답 데이터를 만든 기존 apply_effects 구성과 같은 소리인지 확인
        self.assert_same(
            [{"effect_type": "overdrive", "values": {"drive": 7.5, "level": -3.0}}],
            Pedalboard([Distortion(drive_db=7.5), Gain(gain_db=-3.0)]),
        )
        self.assert_same(
            [{"effect_type": "distortion", "values": {"dist": 20.0, "tone": 4.0, "level": 2.0}}],
            Pedalboard([Distortion(drive_db=20.0), PeakFilter(cutoff_frequency_hz=1000.0, gain_db=4.0), Gain(gain_db=2.0)]),
        )
        self.assert_same(
            [{"effect_type": "fuzz", "values": {"fuzz": 30.0, "tone": -5.0, "level": 1.0, "mode": "Vintage"}}],
            Pedalboard([Distortion(drive_db=30.0), PeakFilter(cutoff_frequency_hz=800.0, gain_db=-5.0), Gain(gain_db=1.0)]),
        )

    def test_chain_order_and_dict_input(self):
        comp = {"threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}
        eq = {"low_gain_db": 3.0, "mid_freq_hz": 1500.0, "mid_gain_db": -2.0, "high_gain_db": 1.0}
        chain = {"amp_id": "fixed_amp_v1", "chain": [
            {"effect_type": "compressor", "values": comp},
            {"effect_type": "overdrive", "values": {"drive": 10.0, "level": 0.0}},
            {"effect_type": "eq_3band", "values": eq},
        ]}
        self.assert_same(chain, Pedalboard([
            Compressor(**comp),
            Distortion(drive_db=10.0), Gain(gain_db=0.0),
            LowShelfFilter(gain_db=3.0), PeakFilter(cutoff_frequency_hz=1500.0, gain_db=-2.0), HighShelfFilter(gain_db=1.0),
        ]))

    def test_every_schema_effect_builds_with_defaults(self):
        for effect_type, definition in load_effects().items():
            defaults = {p["name"]: p["default"] for p in definition["params"]}
            self.assertTrue(build_effect(effect_type, defaults), effect_type)

    def test_rejects_unknown_effect_and_wrong_values(self):
        with self.assertRaises(ValueError):
            build_effect("wah", {})
        with self.assertRaises(ValueError):
            build_effect("overdrive", {"drive": 5.0})

    def test_vst_source_not_supported_yet(self):
        effects = {"plugin_od": {"category": "gain_stage", "source": "vst", "class_name": "OverDiver",
                                 "composite_of": None, "params": [
                                     {"name": "drive", "maps_to": "OverDiver.drive", "type": "continuous",
                                      "range": [0.0, 1.0], "default": 0.5}]}}
        with self.assertRaises(NotImplementedError):
            build_effect("plugin_od", {"drive": 0.5}, effects)


if __name__ == "__main__":
    unittest.main()
