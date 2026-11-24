"""search/comp_candidates.py 테스트 (VST 불필요): python -m unittest search.test_comp_candidates"""
import unittest

from search.comp_candidates import choose_compressor, compressor_defaults, expand_with_compressor


def chain(*types):
    return {"amp_id": "marshall_v30_ir_v1", "chain": [{"effect_type": t, "values": {}} for t in types]}


class ExpandTest(unittest.TestCase):
    def test_adds_compressor_variant_in_front(self):
        out = expand_with_compressor([chain("808_scream"), chain("b_drive")])
        self.assertEqual([[i["effect_type"] for i in c["chain"]] for c in out],
                         [["808_scream"], ["b_drive"], ["compressor", "808_scream"], ["compressor", "b_drive"]])
        self.assertEqual(out[2]["chain"][0]["values"], compressor_defaults())
        self.assertEqual(out[2]["amp_id"], "marshall_v30_ir_v1")

    def test_accepts_cnn_candidate_format_and_skips_existing_compressor(self):
        cnn = {"effect_type": "o_drive", "x0": [0.5, 0.5], "chain": chain("o_drive")}
        out = expand_with_compressor([cnn, chain("compressor", "m_fuzz")])
        self.assertEqual(len(out), 3)  # o_drive, comp+m_fuzz(그대로), comp+o_drive


class ChooseTest(unittest.TestCase):
    def ranked(self, plain, comp):
        return [{"effect_types": ["808_scream"], "loss": plain},
                {"effect_types": ["compressor", "808_scream"], "loss": comp}]

    def test_compressor_needs_clear_margin(self):
        self.assertFalse(choose_compressor(self.ranked(0.100, 0.095))["compressor_used"])  # 5% 개선 → 없음
        self.assertTrue(choose_compressor(self.ranked(0.100, 0.050))["compressor_used"])   # 50% 개선 → 사용

    def test_best_drive_wins_after_decision(self):
        ranked = self.ranked(0.2, 0.19) + [{"effect_types": ["b_drive"], "loss": 0.1},
                                          {"effect_types": ["compressor", "b_drive"], "loss": 0.02}]
        best = choose_compressor(ranked)
        self.assertEqual((best["effect_types"], best["compressor_used"]), (["compressor", "b_drive"], True))


if __name__ == "__main__":
    unittest.main()
