"""
stylematch/reverb_est.py

잔향을 AFx-Rep 거리와 따로 추정 (리버브가 연주 차이를 덮는 해킹 방지).

지표 = 소리가 줄어드는 가장 빠른 속도 (dB/s).
리버브가 있으면 음이 끝나도 소리가 천천히 줄어 "가장 빠른 감쇠"가 느려짐. 연주 내용과 덜 관련됨.
- 엔벨로프: 300 Hz 이상, 20 ms 창 / 5 ms 간격, dB
- 50 ms 구간 기울기 중 가장 가파른 쪽(하위 10%)의 평균. 최대 음량 −45 dB 아래 구간은 제외
추정: 우리 렌더(dry → 드라이브 → 크런치 → IR → 리버브 mix)의 지표가 목표와 같아지는 mix를 고름.

사용 예
-------
python -m stylematch.reverb_est --data data/local/step2_real_v1 --per-drive 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from stylematch.match import ROOM, ROOM_SIZE_DEFAULT, apply_room, render_match

MIX_STEPS = np.round(np.arange(0.0, 0.501, 0.05), 2)


def decay_rate(x, sr):
    """가장 빠른 감쇠 속도 (dB/s, 음수). 0에 가까울수록 잔향이 김."""
    x = sosfilt(butter(2, 300, "highpass", fs=sr, output="sos"), np.asarray(x, np.float64).reshape(-1))
    win, hop = int(0.02 * sr), int(0.005 * sr)
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    env = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    env -= env.max()
    n = 10  # 50 ms
    t = np.arange(n) * hop / sr
    seg = np.lib.stride_tricks.sliding_window_view(env, n)
    slopes = np.polyfit(t, seg.T, 1)[0]
    ok = seg.min(axis=1) > -45
    s = np.sort(slopes[ok])
    if len(s) < 10:
        return float("nan")
    return float(np.mean(s[: max(3, len(s) // 10)]))


def estimate_room(dry, sr, chain, target, base=None):
    """목표의 감쇠 속도와 같아지는 리버브 mix. chain은 리버브 없는 체인(드라이브+크런치).
    반환: (mix, {mix: 감쇠 속도}, 목표 감쇠 속도)"""
    base = render_match(dry, sr, chain) if base is None else base
    rt = decay_rate(target, sr)
    curve = {float(m): decay_rate(apply_room(base, sr, m), sr) for m in MIX_STEPS}
    if rt <= curve[0.0]:  # 우리 dry보다 더 빨리 줄어듦 → 리버브 없음
        return 0.0, curve, rt
    ms = np.array(list(curve)); vs = np.array(list(curve.values()))
    order = np.argsort(vs)  # 감쇠 속도는 mix가 클수록 0에 가까워짐(단조 가정, 정렬 후 보간)
    return float(np.clip(np.interp(rt, vs[order], ms[order]), 0, MIX_STEPS[-1])), curve, rt


def with_room(chain, mix):
    items = [i for i in chain["chain"] if i["effect_type"] != ROOM]
    if mix > 0:
        items.append({"effect_type": ROOM, "values": {"mix": float(mix), "size": ROOM_SIZE_DEFAULT}})
    return {**chain, "chain": items}


def main(argv=None):
    """합성 검증: 목표(다른 연주 + 인공 리버브)의 mix 추정이 리버브 양을 따라가는지."""
    from effects.audio_io import read_mono
    from estimation.chain import DRIVES
    from stylematch.afx import embed
    from stylematch.evaluate import common_dry
    from stylematch.match import match_effects, screen
    from stylematch.room_eval import CONDITIONS, noise_reverb

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True); p.add_argument("--per-drive", type=int, default=4)
    p.add_argument("--out", default="outputs/stylematch/reverb_est_synth.json")
    a = p.parse_args(argv)
    data = Path(a.data)
    dry, sr = common_dry(2.5)
    eff = match_effects()
    by = {}
    for lb in sorted((data / "labels").glob("*.json")):
        L = json.loads(lb.read_text(encoding="utf-8"))
        by.setdefault(L["ground_truth_chain"]["chain"][0]["effect_type"], []).append(lb)
    rows = []
    for d in DRIVES:
        for lb in by[d][:: max(1, len(by[d]) // a.per_drive)][: a.per_drive]:
            clean, _ = read_mono(data / "audio" / f"{lb.stem}.wav")
            starts, _ = screen(dry, sr, embed(clean, sr), eff, k=1)
            base = render_match(dry, sr, starts[0]["chain"])
            for cond, spec in CONDITIONS.items():
                target = clean if spec is None else noise_reverb(clean, sr, *spec)
                mix, curve, rt = estimate_room(dry, sr, starts[0]["chain"], target, base)
                rows.append({"id": lb.stem, "truth": d, "condition": cond, "mix": mix, "target_rate": rt,
                             "dry_rate": curve[0.0]})
                print(f"{lb.stem[-28:]:28s} {cond:5s} 목표 {rt:7.0f} dB/s | 우리(mix 0) {curve[0.0]:7.0f} → mix {mix:.2f}",
                      flush=True)
    summary = {c: {"mix_mean": float(np.mean([r["mix"] for r in rows if r["condition"] == c])),
                   "mix_zero_share": float(np.mean([r["mix"] < 0.05 for r in rows if r["condition"] == c])),
                   "rate_median": float(np.median([r["target_rate"] for r in rows if r["condition"] == c]))}
               for c in CONDITIONS}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
