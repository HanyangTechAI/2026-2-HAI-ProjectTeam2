"""
stylematch/assess.py

톤 맞추기 결과 평가 – "정답 페달 일치" 대신 소리 차이로 평가하고, 거리 자체의 신뢰성을 측정합니다.

1) results : run_targets 결과 폴더를 독립 지표로 채점
   - 독립 지표 = 탐색에 쓰지 않은 지표 (metric hacking 확인용)
       · 평균 스펙트럼 거리: 2.5초 전체의 1/3옥타브 평균 스펙트럼 모양 차이(dB, 음량 무관)
       · 찌그러짐 통계 거리: 스펙트럼 평탄도·크레스트 팩터 차이 (찌그러짐 정도)
   - 기준선: ① 페달 없음(캐비닛만) ② 1위 페달 기본 노브(보정 전) ③ 7종 기본 노브 평균
   - 결과: 기준선 대비 거리 감소율, 목표별 표
2) reliability : 정답 있는 합성 데이터에서 "AFx-Rep 거리가 작을수록 실제로 정답 소리에 가까운가"
   - 실제 가까움 = 같은 dry로 렌더한 정답 소리와 후보의 시점별 스펙트럼 거리 (같은 연주라 신뢰 가능)
   - 목표별 스피어만 상관(AFx-Rep 거리 vs 실제 가까움) 평균, 평균 스펙트럼 거리와 비교

사용 예
-------
python -m stylematch.assess results --dir outputs/stylematch/step3_v1
python -m stylematch.assess reliability --data data/local/step2_real_v1 --per-drive 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from effects import load_effects, render_chain
from effects.audio_io import read_mono
from effects.render import apply_amp
from estimation.chain import DRIVES
from metrics.spectral import spectral_distance
from stylematch.afx import afx_distance, embed
from stylematch.evaluate import common_dry
from stylematch.match import GRID_CRUNCH, apply_crunch, match_effects, screen

SR = 44100
BANDS = 80 * 2 ** (np.arange(0, np.log2(10000 / 80) * 3 + 1) / 3)


def avg_spectrum_db(x):
    x = np.asarray(x, np.float64).reshape(-1)
    n = 4096
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::n // 2] * np.hanning(n)
    p = np.mean(np.abs(np.fft.rfft(fr, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    return np.array([10 * np.log10(np.mean(p[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))]) + 1e-20) for c in BANDS])


def spectrum_shape_distance(a, b):
    d = avg_spectrum_db(a) - avg_spectrum_db(b)
    return float(np.mean(np.abs(d - d.mean())))


def saturation_stats(x):
    x = np.asarray(x, np.float64).reshape(-1)
    n = 2048
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::n // 2] * np.hanning(n)
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-20
    flat = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    return np.array([10 * np.log10(np.median(flat)), 20 * np.log10(np.max(np.abs(x)) / rms)])


def saturation_distance(a, b):
    return float(np.mean(np.abs(saturation_stats(a) - saturation_stats(b))))


METRICS = {"spectrum_shape": spectrum_shape_distance, "saturation": saturation_distance}


def assess_results(folder):
    folder = Path(folder)
    eff = match_effects()
    dry, sr = common_dry(5.5)
    amp = "marshall_v30_ir_v1"
    base_none = apply_amp(dry, sr, amp)
    defaults = {d: np.asarray(render_chain(dry, sr, {"amp_id": amp, "chain": [
        {"effect_type": d, "values": {p["name"]: p["default"] for p in load_effects()[d]["params"]}}]}), np.float32)
        for d in DRIVES}
    rows = []
    for rj in sorted(folder.glob("*/*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        target, _ = read_mono(rj.parent / "target.wav")
        ours, _ = read_mono(rj.parent / "top1.wav")
        top = r["candidates"][0]["drive"]
        row = {"id": r["id"], "kind": r["kind"], "top1": top}
        for m, fn in METRICS.items():
            ours_d = fn(ours, target)
            b = {"none": fn(base_none, target), "top1_default": fn(defaults[top], target),
                 "avg_default": float(np.mean([fn(defaults[d], target) for d in DRIVES]))}
            row[m] = {"ours": ours_d, **b,
                      "reduction_vs_top1_default": 1 - ours_d / max(b["top1_default"], 1e-9),
                      "reduction_vs_avg_default": 1 - ours_d / max(b["avg_default"], 1e-9)}
        rows.append(row)
    summary = {}
    for kind in sorted({r["kind"] for r in rows}) + ["all"]:
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        summary[kind] = {m: {k: float(np.median([r[m][k] for r in sel])) for k in
                             ("ours", "none", "top1_default", "avg_default", "reduction_vs_top1_default",
                              "reduction_vs_avg_default")} | {"better_than_avg_default": float(np.mean(
                                 [r[m]["ours"] < r[m]["avg_default"] for r in sel]))} for m in METRICS} | {"n": len(sel)}
    out = {"metrics": "독립 지표 (탐색에 사용 안 함), 값은 목표별 중앙값", "summary": summary, "rows": rows}
    (folder / "assessment.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind, s in summary.items():
        print(f"[{kind}] n={s['n']}")
        for m in METRICS:
            v = s[m]
            print(f"  {m:15s} 우리 {v['ours']:.3f} | 페달 없음 {v['none']:.3f} · 1위 기본노브 {v['top1_default']:.3f} · "
                  f"7종 평균 {v['avg_default']:.3f} | 7종 평균 대비 감소 {v['reduction_vs_avg_default'] * 100:.0f}% "
                  f"(더 나은 비율 {v['better_than_avg_default'] * 100:.0f}%)")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def assess_reliability(data, per_drive, head_path=None):
    data = Path(data)
    head = None
    if head_path:
        from stylematch.head import Head
        head = Head(head_path)
    eff = match_effects()
    dry, sr = common_dry(2.5)
    screen(dry, sr, np.ones(512) / np.sqrt(512), eff)  # 격자 렌더·임베딩 캐시 채우기
    from stylematch.match import _SCREEN_CACHE
    grid = next(iter(_SCREEN_CACHE.values()))
    # 격자 후보의 실제 소리 (같은 dry) – 정답 소리와의 시점별 거리 계산용
    grid_audio, bares = [], {}
    for drive, chain, _ in grid:
        key = json.dumps(chain["chain"][0], sort_keys=True)  # 같은 드라이브 설정은 VST 렌더 1번 (크런치만 다름)
        if key not in bares:
            bares[key] = render_chain(dry, sr, {"amp_id": "fixed_amp_v1", "chain": chain["chain"][:1]}, effects=eff)
        grid_audio.append(apply_amp(apply_crunch(bares[key], chain["chain"][1]["values"]["drive_db"]), sr, chain["amp_id"]))
    by = {}
    for lb in sorted((data / "labels").glob("*.json")):
        if head is not None and "chords" in lb.stem:
            continue  # 보정층 학습(wetonly)에 같은 코드 녹음이 들어 있어 공정 비교를 위해 제외
        L = json.loads(lb.read_text(encoding="utf-8"))
        by.setdefault(L["ground_truth_chain"]["chain"][0]["effect_type"], []).append(lb)
    rho = {}
    z = lambda v: (np.asarray(v) - np.mean(v)) / max(np.std(v), 1e-12)  # 목표별 표준화 후 합산
    for d in DRIVES:
        for lb in by[d][:: max(1, len(by[d]) // per_drive)][:per_drive]:
            L = json.loads(lb.read_text(encoding="utf-8"))
            target, _ = read_mono(data / "audio" / f"{lb.stem}.wav")      # 다른 연주(목표)
            oracle = np.asarray(render_chain(dry, sr, L["ground_truth_chain"]), np.float32)  # 같은 dry로 정답 설정
            truth = [spectral_distance(np.asarray(w, np.float32), oracle) for w in grid_audio]
            te = embed(target, sr)
            afx = [afx_distance(e, te) for _, _, e in grid]
            spec = [spectrum_shape_distance(w, target) for w in grid_audio]
            sat = [saturation_distance(w, target) for w in grid_audio]
            if head is not None:  # 보정층 특징 거리
                gf = head.features(np.stack([e for _, _, e in grid]))
                hf = head.features(te)[0]
                hd = list(1 - gf @ hf)
            combos = {"afx": afx, "spectrum": spec, "saturation": sat,
                      "afx+spectrum": z(afx) + z(spec), "afx+saturation": z(afx) + z(sat),
                      "afx+spectrum+saturation": z(afx) + z(spec) + z(sat),
                      "2afx+spectrum+saturation": 2 * z(afx) + z(spec) + z(sat)}
            if head is not None:
                combos["head"] = hd
                combos["head+spectrum+saturation"] = z(hd) + z(spec) + z(sat)
            for k, v in combos.items():
                rho.setdefault(k, []).append(spearman(v, truth))
            print(f"{lb.stem[-34:]:34s} " + " | ".join(f"{k} {rho[k][-1]:+.2f}" for k in ("afx", "afx+spectrum+saturation", "head") if k in rho),
                  flush=True)
    out = {"n_targets": len(rho["afx"]), "grid_candidates": len(grid),
           "spearman_mean": {k: float(np.mean(v)) for k, v in rho.items()},
           "spearman_positive_share": {k: float(np.mean(np.array(v) > 0)) for k, v in rho.items()},
           "meaning": "거리 순위와 '같은 dry 기준 정답 소리와의 실제 거리' 순위의 상관 (1에 가까울수록 신뢰)"}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["results", "reliability"])
    p.add_argument("--dir"); p.add_argument("--data"); p.add_argument("--per-drive", type=int, default=3)
    p.add_argument("--out"); p.add_argument("--head", help="보정층 head.pt (reliability에 함께 측정)")
    a = p.parse_args(argv)
    if a.mode == "results":
        assess_results(a.dir)
    else:
        out = assess_reliability(a.data, a.per_drive, a.head)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
