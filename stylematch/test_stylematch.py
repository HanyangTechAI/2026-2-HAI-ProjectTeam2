"""stylematch 테스트: python -m unittest stylematch.test_stylematch
AFx-Rep 가중치가 없는 PC에서는 임베딩 테스트를 건너뜁니다."""
import unittest

import numpy as np

from effects import build_effect, render_chain
from estimation.chain import DRIVES
from stylematch.match import CRUNCH, match_effects


def has_afx():
    try:
        from stylematch.afx import find_afx_dir
        find_afx_dir()
        return True
    except FileNotFoundError:
        return False


SR = 44100
t = np.arange(SR, dtype=np.float32) / SR
DRY = (0.25 * np.sin(2 * np.pi * 196 * t) * np.exp(-2 * t)).astype(np.float32)


class MatchEffectsTest(unittest.TestCase):
    def test_ranges_and_crunch(self):
        eff = match_effects()
        for d in DRIVES:
            for p in eff[d]["params"]:
                if p["name"] in ("level", "volume"):
                    self.assertEqual(p["range"][0], p["range"][1])
                else:
                    self.assertEqual(p["range"], [0.0, 100.0])
        self.assertEqual(eff[CRUNCH]["params"][0]["range"], [0.0, 15.0])

    def test_crunch_adds_saturation(self):
        eff = match_effects()
        from stylematch.match import apply_crunch
        quiet = (DRY * 0.05).astype(np.float32)  # 드라이브 뒤처럼 작은 신호에서도 찌그러져야 함
        self.assertTrue(np.array_equal(apply_crunch(quiet, 0.0), quiet))
        hot = apply_crunch(quiet, 12.0)
        self.assertAlmostEqual(float(np.max(np.abs(hot))), float(np.max(np.abs(quiet))), places=5)  # 최대 음량 유지
        self.assertGreater(np.max(np.abs(hot - quiet)), 0.1 * np.max(np.abs(quiet)))
        self.assertTrue(build_effect(CRUNCH, {"drive_db": 3.0}, eff))


@unittest.skipUnless(has_afx(), "AFx-Rep 가중치가 없는 PC")
class AfxTest(unittest.TestCase):
    def test_embedding_unit_and_identity(self):
        from stylematch.afx import afx_distance, embed
        a = embed(DRY, SR)
        self.assertEqual(a.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=4)
        self.assertAlmostEqual(afx_distance(a, embed(DRY * 0.3, SR)), 0.0, places=3)  # 음량 무관

    def test_distortion_moves_embedding(self):
        from stylematch.afx import afx_distance, embed
        eff = match_effects()
        from stylematch.match import apply_crunch
        hot = apply_crunch(DRY, 15.0)
        self.assertGreater(afx_distance(embed(DRY, SR), embed(hot, SR)), 0.01)


if __name__ == "__main__":
    unittest.main()
