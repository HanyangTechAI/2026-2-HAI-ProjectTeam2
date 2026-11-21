"""effects/render.py 테스트: python -m unittest effects.test_render

VST 테스트는 MultiDrive PRO 플러그인이 있는 PC에서만 실행되고, 없으면 건너뜁니다.
(플러그인 위치: 저장소 루트, effects/VST_files, 기본 설치 폴더, 또는 환경변수 HAI_VST_DIR)
"""
import unittest

import numpy as np
from pedalboard import (Pedalboard, Clipping, Compressor, Distortion, Gain, HighpassFilter, HighShelfFilter,
                        LowpassFilter, LowShelfFilter, PeakFilter)

from effects import build_effect, load_effects, render_chain
from effects.render import find_plugin

SR = 44100
t = np.arange(SR, dtype=np.float32) / SR
DRY = (0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-3 * t)).astype(np.float32)

OD = {"hpf_freq": 200.0, "drive": 30.0, "lpf_freq": 4000.0, "level": -1.0}


def has_plugin():
    try:
        find_plugin("MultiDrivePro")
        return True
    except FileNotFoundError:
        return False


def defaults(definition):
    return {p["name"]: p["default"] for p in definition["params"]}


class RenderChainTest(unittest.TestCase):
    def assert_same(self, chain, board):
        np.testing.assert_array_equal(render_chain(DRY, SR, chain), board(DRY, SR))

    def test_drive_types_match_manual_pedalboard(self):
        self.assert_same(
            [{"effect_type": "overdrive", "values": OD}],
            Pedalboard([HighpassFilter(cutoff_frequency_hz=200.0), Distortion(drive_db=30.0),
                        LowpassFilter(cutoff_frequency_hz=4000.0), Gain(gain_db=-1.0)]),
        )
        self.assert_same(
            [{"effect_type": "distortion", "values": {"hpf_freq": 40.0, "dist": 30.0, "lpf_freq": 5000.0, "level": 0.5}}],
            Pedalboard([HighpassFilter(cutoff_frequency_hz=40.0), Distortion(drive_db=30.0), Clipping(threshold_db=-4.0),
                        LowpassFilter(cutoff_frequency_hz=5000.0), Gain(gain_db=0.5)]),
        )
        self.assert_same(
            [{"effect_type": "fuzz", "values": {"fuzz": 30.0, "notch_freq": 1000.0, "notch_gain": -7.0,
                                                "lpf_freq": 2500.0, "level": 1.0}}],
            Pedalboard([Distortion(drive_db=30.0), Clipping(threshold_db=-6.0),
                        PeakFilter(cutoff_frequency_hz=1000.0, gain_db=-7.0),
                        LowpassFilter(cutoff_frequency_hz=2500.0), Gain(gain_db=1.0)]),
        )

    def test_chain_order_and_dict_input(self):
        comp = {"threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}
        eq = {"low_gain_db": 3.0, "mid_freq_hz": 1500.0, "mid_gain_db": -2.0, "high_gain_db": 1.0}
        chain = {"amp_id": "fixed_amp_v1", "chain": [
            {"effect_type": "compressor", "values": comp},
            {"effect_type": "overdrive", "values": OD},
            {"effect_type": "eq_3band", "values": eq},
        ]}
        self.assert_same(chain, Pedalboard([
            Compressor(**comp),
            HighpassFilter(cutoff_frequency_hz=200.0), Distortion(drive_db=30.0),
            LowpassFilter(cutoff_frequency_hz=4000.0), Gain(gain_db=-1.0),
            LowShelfFilter(gain_db=3.0), PeakFilter(cutoff_frequency_hz=1500.0, gain_db=-2.0), HighShelfFilter(gain_db=1.0),
        ]))

    def test_every_pedalboard_effect_builds_with_defaults(self):
        for effect_type, definition in load_effects().items():
            if definition["source"] == "pedalboard":
                self.assertTrue(build_effect(effect_type, defaults(definition)), effect_type)

    def test_rejects_unknown_effect_and_wrong_values(self):
        with self.assertRaises(ValueError):
            build_effect("wah", {})
        with self.assertRaises(ValueError):
            build_effect("overdrive", {"drive": 5.0})
        with self.assertRaises(ValueError):
            build_effect("808_scream", {"overdrive": 50.0})  # 노브 값이 빠지면 플러그인을 불러오기 전에 거부

    def test_unknown_source_not_supported(self):
        effects = {"nam_od": {"category": "gain_stage", "source": "nam", "class_name": "Capture",
                              "composite_of": None, "params": [
                                  {"name": "drive", "maps_to": "Capture.drive", "type": "continuous",
                                   "range": [0.0, 1.0], "default": 0.5}]}}
        with self.assertRaises(NotImplementedError):
            build_effect("nam_od", {"drive": 0.5}, effects)

    def test_same_vst_twice_in_chain_rejected(self):
        values = defaults(load_effects()["808_scream"])
        with self.assertRaises(ValueError):
            render_chain(DRY, SR, [{"effect_type": "808_scream", "values": values},
                                   {"effect_type": "808_scream", "values": values}])


class AmpRenderTest(unittest.TestCase):
    """앰프(캐비닛 IR) 적용. 임시 IR 파일을 만들어 쓰므로 공유 드라이브 없이 실행됩니다."""

    def setUp(self):
        import os
        import tempfile
        import soundfile as sf

        self.tmp = tempfile.TemporaryDirectory()
        ir = np.zeros(2400, dtype=np.float32)
        ir[0], ir[40], ir[400] = 1.0, 0.4, -0.2
        sf.write(os.path.join(self.tmp.name, "test_ir.wav"), ir, 48000, subtype="FLOAT")
        self.amp = {"amp_id": "test_ir", "reference_amp": None, "render_method": "impulse_response",
                    "file_path": "test_ir.wav", "fixed": True, "gain_db": 6.0}
        self.old_env = os.environ.get("HAI_IR_DIR")
        os.environ["HAI_IR_DIR"] = self.tmp.name
        self.items = [{"effect_type": "overdrive", "values": OD}]

    def tearDown(self):
        import os

        if self.old_env is None:
            os.environ.pop("HAI_IR_DIR", None)
        else:
            os.environ["HAI_IR_DIR"] = self.old_env
        self.tmp.cleanup()

    def test_ir_applied_after_chain_with_gain(self):
        from pedalboard import Convolution

        dry_wet = render_chain(DRY, SR, self.items)
        expected = Pedalboard([Convolution(impulse_response_filename=f"{self.tmp.name}/test_ir.wav"),
                               Gain(gain_db=6.0)])(dry_wet, SR, reset=True)
        got = render_chain(DRY, SR, {"amp_id": "test_ir", "chain": self.items}, amp=self.amp)
        np.testing.assert_allclose(got, expected, atol=1e-6)
        self.assertGreater(np.max(np.abs(got - dry_wet)), 1e-3)

    def test_no_amp_for_step1_id_and_plain_list(self):
        plain = render_chain(DRY, SR, self.items)
        np.testing.assert_array_equal(
            render_chain(DRY, SR, {"amp_id": "fixed_amp_v1", "chain": self.items}, amp=self.amp), plain)

    def test_unknown_amp_and_missing_ir_rejected(self):
        with self.assertRaises(ValueError):
            render_chain(DRY, SR, {"amp_id": "other_amp", "chain": self.items}, amp=self.amp)
        with self.assertRaises(FileNotFoundError):
            render_chain(DRY, SR, {"amp_id": "test_ir", "chain": self.items},
                         amp=dict(self.amp, file_path="missing.wav"))


@unittest.skipUnless(has_plugin(), "MultiDrive PRO 플러그인이 없는 PC")
class VstRenderTest(unittest.TestCase):
    def vst_effects(self):
        return {k: d for k, d in load_effects().items() if d["source"] == "vst"}

    def test_every_vst_effect_changes_sound_and_knobs_respond(self):
        for effect_type, definition in self.vst_effects().items():
            with self.subTest(effect_type=effect_type):
                low = {p["name"]: p["range"][0] for p in definition["params"]}
                high = {p["name"]: p["range"][1] for p in definition["params"]}
                a = render_chain(DRY, SR, [{"effect_type": effect_type, "values": low}])
                b = render_chain(DRY, SR, [{"effect_type": effect_type, "values": high}])
                self.assertTrue(np.isfinite(b).all())
                self.assertGreater(np.max(np.abs(b - DRY)), 1e-3, "드라이 그대로 나옴 (체험판 미활성화?)")
                self.assertGreater(np.max(np.abs(a - b)), 1e-4, "노브가 소리에 반영되지 않음")

    def test_short_clip_is_not_silent(self):
        # 기본 버퍼(8192)로 렌더링하면 첫 0.19초가 무음이 되던 문제 확인
        short = DRY[:4410]
        for effect_type, definition in self.vst_effects().items():
            with self.subTest(effect_type=effect_type):
                wet = render_chain(short, SR, [{"effect_type": effect_type, "values": defaults(definition)}])
                self.assertGreater(np.max(np.abs(wet[:2048])), 1e-4)

    def test_models_are_distinct_and_switching_does_not_leak(self):
        # 모델을 번갈아 써도 각 모델 소리가 유지되는지 (불러온 뒤 모델을 바꾸던 예전 버그 확인)
        effects = self.vst_effects()
        first = {k: render_chain(DRY, SR, [{"effect_type": k, "values": defaults(d)}]) for k, d in effects.items()}
        again = {k: render_chain(DRY, SR, [{"effect_type": k, "values": defaults(d)}])
                 for k, d in reversed(list(effects.items()))}
        for k in effects:
            np.testing.assert_allclose(first[k], again[k], atol=1e-3, err_msg=k)
        keys = list(effects)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                self.assertGreater(np.max(np.abs(first[a] - first[b])), 1e-3, f"{a}와 {b}가 같은 소리")


if __name__ == "__main__":
    unittest.main()
