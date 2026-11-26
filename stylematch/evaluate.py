"""
stylematch/evaluate.py

같은 연주 dry 없이 목표 톤 맞추기 평가 (AFx-Rep 격자 → CMA-ES).
- dry: 공통 dry (IDMT Fender Strat 프레이즈 FS_Lick1_FN, 학습에 안 쓴 실제 연주)
- synthetic: 정답 있는 합성 목표 (예: step2_real_v1), 드라이브별 N개
- real: step3_targets_v1의 기타 단독 톤(isolated) – 원본 녹음에서 5.5초 구간 여러 개 임베딩 평균
결과: <out>/results.json, 목표별 target/final wav

사용 예
-------
python -m stylematch.evaluate synthetic --data data/local/step2_real_v1 --per-drive 2 --out outputs/stylematch/synth
python -m stylematch.evaluate real --targets data/local/step3_targets_v1 --raw <드라이브>/data/step3_val_data --out outputs/stylematch/real
"""
import argparse
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from effects.audio_io import read_mono, to_mono
from estimation.chain import DRIVES
from stylematch.afx import embed, embed_long
from stylematch.match import match_tone

ROOT = Path(__file__).resolve().parent.parent
IDMT_ZIP = ROOT / "data" / "raw" / "IDMT-SMT-GUITAR_V2.zip"
DRY_NAME = "IDMT-SMT-GUITAR_V2/dataset2/audio/FS_Lick1_FN.wav"
FAMILY = {"808_scream": "OD", "b_drive": "OD", "o_drive": "OD", "one_distortion": "DS", "bat_distortion": "DS",
          "f_fuzz": "FZ", "m_fuzz": "FZ"}
PEDAL_FAMILY = {"Boss SD1": "OD", "JRAD Archer": "OD"}


def common_dry(seconds, sr=44100):
    with zipfile.ZipFile(IDMT_ZIP) as z:
        x, rate = sf.read(io.BytesIO(z.read(DRY_NAME)), dtype="float32", always_2d=True)
    x = to_mono(x)
    n = int(seconds * sr)
    s = max(range(0, max(1, len(x) - n + 1), 11025), key=lambda i: np.mean(x[i:i + n] ** 2))
    d = x[s:s + n]
    return (d * (10 ** (-12 / 20) / np.max(np.abs(d)))).astype(np.float32), sr


def summarize(rows):
    scored = [r for r in rows if r["truth"]]
    out = {"n": len(rows), "n_scored": len(scored)}
    for stage in ("screen", "final"):
        if scored:
            ranks = np.array([r[f"{stage}_rank"] for r in scored])
            out[stage] = {"top1": float(np.mean(ranks == 1)), "top3": float(np.mean(ranks <= 3)), "mean_rank": float(ranks.mean())}
        fam = [r[f"{stage}_family_ok"] for r in rows if r["family"]]
        if fam:
            out.setdefault(stage, {})["family"] = float(np.mean(fam))
    return out


def run(items, dry, sr, out, budget, k):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, target, te, truth, family) in enumerate(items):
        t0 = time.perf_counter()
        res = match_tone(dry, target, sr, k=k, budget=budget, target_embedding=te)
        best = res["best"]
        final_order = [c["effect_type"] for c in res["candidates"]]
        final_order += [d for d in res["screen_order"] if d not in final_order]
        row = {"name": name, "truth": truth, "family": family,
               "screen_top": res["screen_order"][0], "final": best["effect_type"],
               "screen_rank": res["screen_order"].index(truth) + 1 if truth else None,
               "final_rank": final_order.index(truth) + 1 if truth else None,
               "screen_family_ok": FAMILY[res["screen_order"][0]] == family if family else None,
               "final_family_ok": FAMILY[best["effect_type"]] == family if family else None,
               "final_chain": best["chain"], "final_distance": best["afx_distance"], "seconds": time.perf_counter() - t0}
        rows.append(row)
        from stylematch.match import render_match
        sf.write(out / f"{name}_target.wav", np.asarray(target, np.float32), sr, subtype="FLOAT")
        sf.write(out / f"{name}_final.wav", np.asarray(render_match(dry, sr, best["chain"]), np.float32), sr, subtype="FLOAT")
        print(f"{i + 1}/{len(items)} {name[:38]:38s} 정답 {str(truth or family)[:12]:12s} | 격자 {row['screen_top']:15s}"
              f" → 최종 {row['final']:15s} ({row['seconds']:.0f}s)", flush=True)
    result = {"summary": summarize(rows), "rows": rows}
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["synthetic", "real"])
    p.add_argument("--data"); p.add_argument("--per-drive", type=int, default=2)
    p.add_argument("--targets"); p.add_argument("--raw")
    p.add_argument("--out", required=True); p.add_argument("--budget", type=int, default=30); p.add_argument("--k", type=int, default=3)
    a = p.parse_args(argv)
    items = []
    if a.mode == "synthetic":
        dry, sr = common_dry(2.5)
        data = Path(a.data)
        by = {}
        for lb in sorted((data / "labels").glob("*.json")):
            L = json.loads(lb.read_text(encoding="utf-8"))
            by.setdefault(L["ground_truth_chain"]["chain"][0]["effect_type"], []).append(lb)
        for d in DRIVES:
            pick = by[d][:: max(1, len(by[d]) // a.per_drive)][: a.per_drive]
            for lb in pick:
                x, _ = read_mono(data / "audio" / f"{lb.stem}.wav")
                items.append((lb.stem, x, embed(x, sr), d, FAMILY[d]))
    else:
        dry, sr = common_dry(5.5)
        for r in csv.DictReader(open(Path(a.targets) / "targets.csv", encoding="utf-8-sig")):
            if r["kind"] != "isolated":
                continue
            raw, _ = read_mono(Path(a.raw) / r["source_file"])
            seg, _ = read_mono(Path(a.targets) / "isolated" / f"{r['id']}.wav")
            fam = FAMILY[r["drive_in_catalog"]] if r["drive_in_catalog"] else PEDAL_FAMILY.get(r["pedal"])
            items.append((r["id"], seg, embed_long(raw, sr), r["drive_in_catalog"] or None, fam))
    run(items, dry, sr, Path(a.out), a.budget, a.k)


if __name__ == "__main__":
    main()
