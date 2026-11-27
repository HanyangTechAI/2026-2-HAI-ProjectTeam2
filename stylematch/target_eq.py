"""
stylematch/target_eq.py

목표 EQ 정규화: 탐색 전에 목표의 대역 균형(녹음 EQ · 마이크 · 믹싱)을 우리 렌더 쪽에 맞춰,
거리가 녹음 EQ 차이가 아니라 찌그러짐 종류 · 양을 비교하게 함. 결과에는 원래 목표 기준 EQ 후보정(spectrum_eq).

모드
- none  : 정규화 없음 (기존)
- tilt  : 큰 기울기만 – 목표와 기준 스펙트럼 차이를 log 주파수 2차 곡선으로 근사해 보정 (페달 고유 음색은 남김)
- full  : 1/3옥타브 전체 맞춤 (비교용, 페달 고유 음색까지 지워질 수 있음)
- tilt2 : 2단계 – tilt로 1차 격자 → 1위 체인 렌더를 기준으로 다시 tilt
- align : 목표는 그대로, 격자 후보마다 목표 쪽으로 큰 기울기를 맞춘 뒤 지문 비교 (후보별 EQ 정렬, align_screen)
기준 스펙트럼: 공통 dry를 7종(기본 노브, 크런치 6 dB) + 캐비닛 IR로 렌더한 평균 (dry별 캐시)

평가 (합성, 정답 있음): python -m stylematch.target_eq synth --data data/local/step2_real_v1 --per-drive 4
평가 (실물 데모, 정답 6개): python -m stylematch.target_eq real --targets data/local/step3_targets_v1 --raw <드라이브>/data/step3_val_data
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from effects import render_chain
from estimation.chain import DRIVES
from stylematch.assess import BANDS, avg_spectrum_db

MODES = ("none", "tilt", "full", "tilt2", "align")
MAX_DB = 12.0
TAPS = 4096
_REF = {}
_GRID_AUDIO = {}


def apply_band_gains(audio, sr, gains_db):
    """1/3옥타브 대역별 보정량(dB)을 선형 위상 FIR로 적용, 음량(RMS)은 유지."""
    x = np.asarray(audio, np.float64).reshape(-1)
    f = np.fft.rfftfreq(TAPS, 1 / sr)
    g = np.interp(np.log(np.maximum(f, 1.0)), np.log(BANDS), np.asarray(gains_db, np.float64))
    h = np.fft.fftshift(np.fft.irfft(10 ** (g / 20), TAPS)) * np.hanning(TAPS)
    y = fftconvolve(x, h)[TAPS // 2: TAPS // 2 + len(x)]
    y *= (np.sqrt(np.mean(x ** 2)) + 1e-12) / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    return y.astype(np.float32)


def tilt_fit(diff_db, degree=2):
    """대역 차이를 log 주파수 다항식으로 근사 (큰 기울기만)."""
    lf = np.log2(BANDS / 1000.0)
    return np.polyval(np.polyfit(lf, diff_db, degree), lf)


def curve(target, reference_db, mode):
    d = reference_db - avg_spectrum_db(target)
    d = d - d.mean()
    if mode in ("tilt", "tilt2"):
        d = tilt_fit(d)
    return np.clip(d, -MAX_DB, MAX_DB)


def reference_db(dry, sr, eff=None, amp_id="marshall_v30_ir_v1"):
    """공통 dry → 7종(기본 노브) → 크런치 6 dB → IR 렌더의 평균 스펙트럼 (dB)."""
    from stylematch.match import CRUNCH, match_effects, render_match
    eff = eff or match_effects()
    key = (hashlib.sha1(np.asarray(dry, np.float32).tobytes()).hexdigest(), int(sr), amp_id)
    if key not in _REF:
        specs = []
        for d in DRIVES:
            chain = {"amp_id": amp_id, "chain": [
                {"effect_type": d, "values": {p["name"]: p["default"] for p in eff[d]["params"]}},
                {"effect_type": CRUNCH, "values": {"drive_db": 6.0}}]}
            specs.append(avg_spectrum_db(render_match(dry, sr, chain, eff)))
        _REF[key] = np.mean(specs, axis=0)
    return _REF[key]


def normalize_target(dry, sr, target, mode, eff=None, long_audio=None, allowed=None):
    """목표(와 지문용 긴 구간)를 정규화. 반환: (정규화 목표, 정규화 긴 구간 또는 None, 보정 곡선 dB)"""
    if mode == "none":
        return np.asarray(target, np.float32), long_audio, np.zeros(len(BANDS))
    from stylematch.afx import embed, embed_long
    from stylematch.match import match_effects, render_match, screen
    eff = eff or match_effects()
    src = long_audio if long_audio is not None else target
    ref = reference_db(dry, sr, eff)
    c = curve(src, ref, "full" if mode == "full" else "tilt")
    if mode == "tilt2":  # 1차 결과의 1위 체인 렌더를 기준으로 다시
        first = apply_band_gains(src, sr, c)
        te = embed_long(first, sr) if long_audio is not None else embed(first, sr)
        starts, _ = screen(dry, sr, te, eff, k=1, allowed=allowed)
        ref = avg_spectrum_db(render_match(dry, sr, starts[0]["chain"], eff))
        c = curve(src, ref, "tilt")
    t = apply_band_gains(target, sr, c)
    la = apply_band_gains(long_audio, sr, c) if long_audio is not None else None
    return t, la, c


def grid_audio(dry, sr, eff):
    """격자 후보(screen과 같은 순서)의 렌더 소리와 평균 스펙트럼 (dry별 캐시)."""
    from stylematch.match import _SCREEN_CACHE, apply_crunch, screen
    from effects.render import apply_amp
    key = hashlib.sha1(np.asarray(dry, np.float32).tobytes()).hexdigest()
    if key not in _GRID_AUDIO:
        screen(dry, sr, np.ones(512) / np.sqrt(512), eff)
        grid = _SCREEN_CACHE[(key, int(sr), "marshall_v30_ir_v1", False)]
        bares, items = {}, []
        for drive, chain, _ in grid:
            k = json.dumps(chain["chain"][0], sort_keys=True)
            if k not in bares:
                bares[k] = render_chain(dry, sr, {"amp_id": "fixed_amp_v1", "chain": chain["chain"][:1]}, effects=eff)
            w = np.asarray(apply_amp(apply_crunch(bares[k], chain["chain"][1]["values"]["drive_db"]), sr,
                                     chain["amp_id"]), np.float32)
            items.append((drive, chain, w, avg_spectrum_db(w)))
        _GRID_AUDIO[key] = items
    return _GRID_AUDIO[key]


def align_screen(dry, sr, target_audio, target_embedding, eff, k=3, allowed=None):
    """후보별 EQ 정렬 격자: 각 후보를 목표 쪽으로 큰 기울기만 맞춘 뒤 AFx 거리 → 드라이브 순위."""
    from stylematch.afx import afx_distance, embed
    tdb = avg_spectrum_db(target_audio)
    best = {}
    for drive, chain, w, wdb in grid_audio(dry, sr, eff):
        if allowed is not None and drive not in allowed:
            continue
        d = tdb - wdb
        g = np.clip(tilt_fit(d - d.mean()), -MAX_DB, MAX_DB)
        dist = afx_distance(embed(apply_band_gains(w, sr, g), sr), target_embedding)
        if drive not in best or dist < best[drive][0]:
            best[drive] = (dist, chain, g)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])
    return ranked[:k], [d for d, _ in ranked]


def random_recording_eq(rng):
    """합성 평가용 녹음 EQ: 큰 기울기(2차, 최대 약 ±8 dB) + 무작위 피크 1개(±6 dB, 1옥타브 폭)."""
    lf = np.log2(BANDS / 1000.0)
    g = rng.uniform(-2.5, 2.5) * lf + rng.uniform(-0.8, 0.8) * lf ** 2
    c, a = rng.uniform(-2, 3), rng.uniform(-6, 6)
    g += a * np.exp(-0.5 * ((lf - c) / 0.5) ** 2)
    return np.clip(g - g.mean(), -10, 10)


def ranks(dry, sr, eff, target, truth, long_audio=None, allowed=None):
    from stylematch.afx import embed, embed_long
    from stylematch.match import screen
    out = {}
    for mode in MODES:
        if mode == "align":
            src = long_audio if long_audio is not None else target
            te = embed_long(src, sr) if long_audio is not None else embed(target, sr)
            top, order = align_screen(dry, sr, src, te, eff, allowed=allowed)
            out[mode] = {"rank": order.index(truth) + 1 if truth else None, "top": order[0],
                         "curve_range_db": float(np.ptp(top[0][1][2]))}
            continue
        t, la, c = normalize_target(dry, sr, target, mode, eff, long_audio, allowed)
        te = embed_long(la, sr) if la is not None else embed(t, sr)
        _, order = screen(dry, sr, te, eff, k=3, allowed=allowed)
        out[mode] = {"rank": order.index(truth) + 1 if truth else None, "top": order[0],
                     "curve_range_db": float(np.ptp(c))}
    return out


def summarize(rows, key=lambda r: True):
    sel = [r for r in rows if key(r)]
    out = {}
    for m in MODES:
        rk = np.array([r[m]["rank"] for r in sel if r[m]["rank"]])
        out[m] = {"n": int(len(rk)), "top1": float(np.mean(rk == 1)), "top3": float(np.mean(rk <= 3)),
                  "mean_rank": float(rk.mean())}
    return out


def main(argv=None):
    from effects.audio_io import read_mono
    from stylematch.evaluate import FAMILY, common_dry
    from stylematch.match import match_effects
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["synth", "real"])
    p.add_argument("--data"); p.add_argument("--per-drive", type=int, default=4)
    p.add_argument("--targets"); p.add_argument("--raw")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    eff = match_effects()
    rows = []
    if a.mode == "synth":
        dry, sr = common_dry(2.5)
        rng = np.random.default_rng(0)
        by = {}
        for lb in sorted((Path(a.data) / "labels").glob("*.json")):
            L = json.loads(lb.read_text(encoding="utf-8"))
            by.setdefault(L["ground_truth_chain"]["chain"][0]["effect_type"], []).append(lb)
        for d in DRIVES:
            for lb in by[d][:: max(1, len(by[d]) // a.per_drive)][: a.per_drive]:
                clean, _ = read_mono(Path(a.data) / "audio" / f"{lb.stem}.wav")
                for cond in ("clean", "eq1", "eq2"):
                    target = clean if cond == "clean" else apply_band_gains(clean, sr, random_recording_eq(rng))
                    r = {"id": lb.stem, "truth": d, "condition": cond, **ranks(dry, sr, eff, target, d)}
                    rows.append(r)
                    print(f"{lb.stem[-28:]:28s} {cond:5s} {d:14s} " +
                          " ".join(f"{m} {r[m]['rank']}" for m in MODES), flush=True)
        summary = {"clean": summarize(rows, lambda r: r["condition"] == "clean"),
                   "recording_eq": summarize(rows, lambda r: r["condition"] != "clean")}
    else:
        dry, sr = common_dry(5.5)
        for r0 in csv.DictReader(open(Path(a.targets) / "targets.csv", encoding="utf-8-sig")):
            if r0["kind"] != "isolated":
                continue
            seg, _ = read_mono(Path(a.targets) / "isolated" / f"{r0['id']}.wav")
            raw, _ = read_mono(Path(a.raw) / r0["source_file"])
            truth = r0["drive_in_catalog"] or None
            r = {"id": r0["id"], "truth": truth, **ranks(dry, sr, eff, seg, truth, raw)}
            fam = FAMILY[truth] if truth else {"Boss SD1": "OD", "JRAD Archer": "OD"}.get(r0["pedal"])
            for m in MODES:
                r[m]["family_ok"] = FAMILY[r[m]["top"]] == fam if fam else None
            rows.append(r)
            print(f"{r0['id'][:30]:30s} 정답 {str(truth):14s} " +
                  " | ".join(f"{m} {r[m]['top'][:8]}({r[m]['rank']})" for m in MODES), flush=True)
        summary = summarize(rows, lambda r: r["truth"])
        for m in MODES:
            fam = [r[m]["family_ok"] for r in rows if r[m]["family_ok"] is not None]
            summary[m]["family_12"] = float(np.mean(fam))
    out = Path(a.out or f"outputs/stylematch/target_eq_{a.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
