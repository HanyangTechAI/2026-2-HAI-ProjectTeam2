"""
stylematch/room_eval.py

공간 리버브 흡수 단계 검증 (합성, 정답 있음, 다른 연주).

목표 = step2_real_v1의 wet(다른 연주) + 인공 리버브. 리버브는 탐색에 쓰는 pedalboard Reverb와
다른 방식(지수 감쇠 노이즈 IR 컨볼루션)으로 입혀, 같은 알고리즘이라 쉽게 맞히는 이점을 없앰.
조건: none(리버브 없음) / mild(RT60 0.6 s, mix 0.25) / heavy(RT60 1.5 s, mix 0.45)
방법: 격자 판별(room 단계 없음 vs 있음) → 정답 페달 순위, 격자가 고른 리버브 양
(페달 순위는 격자에서 정해지고 CMA-ES는 노브만 다듬으므로 격자 비교로 판별 영향 확인 가능)

사용 예
-------
python -m stylematch.room_eval --data data/local/step2_real_v1 --per-drive 4 --out outputs/stylematch/room_synth.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from effects.audio_io import read_mono
from estimation.chain import DRIVES
from stylematch.afx import afx_distance, embed
from stylematch.evaluate import FAMILY, common_dry
from stylematch.match import ROOM, match_effects, screen

CONDITIONS = {"none": None, "mild": (0.6, 0.25), "heavy": (1.5, 0.45)}


def noise_reverb(x, sr, rt60, mix, seed=0):
    """지수 감쇠 노이즈 IR(초기 20 ms 사전 지연)로 만든 리버브. 출력 길이·RMS는 입력과 같게."""
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)
    ir = np.concatenate([np.zeros(int(0.02 * sr)), ir])
    wet = fftconvolve(x, ir)[: len(x)]
    rms = lambda v: np.sqrt(np.mean(v ** 2)) + 1e-12
    wet *= rms(x) / rms(wet)
    y = (1 - mix) * x + mix * wet
    return (y * rms(x) / rms(y)).astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True); p.add_argument("--per-drive", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    data = Path(a.data)
    dry, sr = common_dry(2.5)
    eff = match_effects()
    t0 = time.perf_counter()
    dummy = np.ones(512) / np.sqrt(512)
    screen(dry, sr, dummy, eff, room=False)
    t1 = time.perf_counter()
    screen(dry, sr, dummy, eff, room=True)
    print(f"격자 준비: room 없음 {t1 - t0:.0f}s, room 있음 {time.perf_counter() - t1:.0f}s", flush=True)

    by = {}
    for lb in sorted((data / "labels").glob("*.json")):
        L = json.loads(lb.read_text(encoding="utf-8"))
        by.setdefault(L["ground_truth_chain"]["chain"][0]["effect_type"], []).append(lb)
    rows = []
    for d in DRIVES:
        for lb in by[d][:: max(1, len(by[d]) // a.per_drive)][: a.per_drive]:
            clean, _ = read_mono(data / "audio" / f"{lb.stem}.wav")
            e_clean = embed(clean, sr)
            for cond, spec in CONDITIONS.items():
                target = clean if spec is None else noise_reverb(clean, sr, *spec)
                te = embed(target, sr)
                row = {"id": lb.stem, "truth": d, "condition": cond, "afx_shift": afx_distance(te, e_clean)}
                for room in (False, True):
                    starts, order = screen(dry, sr, te, eff, room=room, k=3)
                    tag = "room" if room else "base"
                    row[f"{tag}_rank"] = order.index(d) + 1
                    row[f"{tag}_family_ok"] = FAMILY[order[0]] == FAMILY[d]
                    row[f"{tag}_top"] = order[0]
                    if room:
                        row["room_mix"] = next(i["values"]["mix"] for i in starts[0]["chain"]["chain"]
                                               if i["effect_type"] == ROOM)
                rows.append(row)
                print(f"{lb.stem[-30:]:30s} {cond:5s} 정답 {d:14s} | 기본 {row['base_rank']} ({row['base_top']}) "
                      f"| room {row['room_rank']} ({row['room_top']}, mix {row['room_mix']})", flush=True)
    summary = {}
    for cond in CONDITIONS:
        sel = [r for r in rows if r["condition"] == cond]
        summary[cond] = {"n": len(sel), "afx_shift": float(np.mean([r["afx_shift"] for r in sel])),
                         "room_mix_mean": float(np.mean([r["room_mix"] for r in sel]))}
        for tag in ("base", "room"):
            rk = np.array([r[f"{tag}_rank"] for r in sel])
            summary[cond][tag] = {"top1": float(np.mean(rk == 1)), "top3": float(np.mean(rk <= 3)),
                                  "mean_rank": float(rk.mean()),
                                  "family": float(np.mean([r[f"{tag}_family_ok"] for r in sel]))}
    out = {"summary": summary, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
