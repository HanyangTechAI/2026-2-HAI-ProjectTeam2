"""
stylematch/run_targets.py

Step 3 목표 소리(step3_targets_v1)에 톤 맞추기를 실행하고, 데모·청취용 결과를 저장합니다.

목표마다 <out>/<kind>/<id>/ 에:
  target.wav          목표 2.5초 구간 (음량 정규화됨)
  dry.wav             공통 dry (IDMT Fender Strat 프레이즈 5.5초)
  top1.wav~top3.wav   후보 페달 1~3위로 dry를 렌더링한 결과 (음량 목표에 맞춤)
  result.json         후보별 페달·실제 페달명·노브·크런치·AFx 거리, (알면) 정답 페달
그리고 <out>/summary.json (정답 있는 목표의 1위·상위 3 정확도)

사용 예 (저장소 루트에서)
-------------------------
python -m stylematch.run_targets --targets data/local/step3_targets_v1
    --raw-separated <드라이브>/data/step3_real_data --raw-isolated <드라이브>/data/step3_val_data
    --out outputs/stylematch/step3_v1
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from effects import load_effects
from effects.audio_io import read_mono
from stylematch.afx import embed_long
from stylematch.evaluate import FAMILY, PEDAL_FAMILY, common_dry
from stylematch.match import CRUNCH, ROOM, match_tone, render_match


def describe(chain, effects):
    """사람이 읽는 형태: 페달 이름 · 노브 · 크런치."""
    drive = next(i for i in chain["chain"] if i["effect_type"] not in (CRUNCH, ROOM))
    crunch = next((i["values"]["drive_db"] for i in chain["chain"] if i["effect_type"] == CRUNCH), 0.0)
    room = next((i["values"] for i in chain["chain"] if i["effect_type"] == ROOM), None)
    ref = effects[drive["effect_type"]].get("reference_pedal") or {}
    out = {"drive": drive["effect_type"], "pedal": f"{ref.get('brand', '')} {ref.get('model', '')}".strip(),
           "knobs": {k: round(float(v), 1) for k, v in drive["values"].items()},
           "amp_crunch_db": round(float(crunch), 1)}
    if room is not None:  # 목표 녹음의 공간감 흡수용 (추천 대상 아님)
        out["room_reverb"] = {k: round(float(v), 2) for k, v in room.items()}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", required=True)
    p.add_argument("--raw-separated"); p.add_argument("--raw-isolated")
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=30); p.add_argument("--k", type=int, default=3)
    p.add_argument("--only", choices=["separated", "isolated"])
    p.add_argument("--resume", action="store_true", help="result.json이 있는 목표는 건너뛰고 이어서 실행")
    p.add_argument("--owned", nargs="+", help="보유 드라이브만 후보로 (예: 808_scream b_drive). 없으면 7종 전체")
    p.add_argument("--embed-window", type=float,
                   help="목표 지문을 원본 전체 대신 start_sec부터 이 길이(초)만으로 계산 (예: 11)")
    p.add_argument("--target-eq", choices=["none", "tilt", "full", "tilt2"], default="none",
                   help="탐색 전 목표 EQ 정규화 (target_eq.py). target.wav는 원본 그대로 저장")
    p.add_argument("--room", action="store_true", help="리버브 게이트를 넘는 목표만 공간 리버브 단계로 흡수 (실험용)")
    a = p.parse_args(argv)

    effects = load_effects()
    dry, sr = common_dry(5.5)
    out = Path(a.out)
    raw_dirs = {"separated": a.raw_separated, "isolated": a.raw_isolated}
    rows = list(csv.DictReader(open(Path(a.targets) / "targets.csv", encoding="utf-8-sig")))
    if a.only:
        rows = [r for r in rows if r["kind"] == a.only]
    summary_rows = []
    for i, r in enumerate(rows, 1):
        done = out / r["kind"] / r["id"] / "result.json"
        if a.resume and done.is_file():  # 중단 후 재개: 끝난 목표는 결과만 읽어 요약에 포함
            prev = json.loads(done.read_text(encoding="utf-8"))
            cands = prev["candidates"]
            truth, family = prev["truth_drive"], prev["truth_family"]
            order = [c["drive"] for c in cands] + [d for d in prev["screen_order"] if d not in [c["drive"] for c in cands]]
            summary_rows.append({"id": r["id"], "kind": r["kind"], "truth": truth, "family": family,
                                 "top": cands[0]["drive"], "rank": order.index(truth) + 1 if truth else None,
                                 "family_ok": FAMILY[cands[0]["drive"]] == family if family else None})
            print(f"{i}/{len(rows)} {r['kind'][:3]} {r['id'][:36]:36s} (이미 완료, 건너뜀)", flush=True)
            continue
        t0 = time.perf_counter()
        target, _ = read_mono(Path(a.targets) / r["kind"] / f"{r['id']}.wav")
        raw_dir = raw_dirs.get(r["kind"])
        te, raw = None, None
        if raw_dir:
            raw = read_mono(Path(raw_dir) / r["source_file"])[0]
            if a.embed_window:  # 선택 구간부터 일정 길이만 (곡 전체의 큰 구간 대신)
                s0 = int(float(r["start_sec"]) * sr)
                raw = raw[s0:s0 + int(a.embed_window * sr)]
        search_target = target
        eq_curve = None
        if a.target_eq != "none":
            from stylematch.target_eq import normalize_target
            search_target, raw, c = normalize_target(dry, sr, target, a.target_eq, long_audio=raw, allowed=a.owned)
            eq_curve = [round(float(v), 2) for v in c]
        if raw is not None:
            te = embed_long(raw, sr)
        res = match_tone(dry, search_target, sr, k=a.k, budget=a.budget, target_embedding=te, allowed=a.owned, room=a.room)
        folder = out / r["kind"] / r["id"]
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "target.wav", target, sr, subtype="FLOAT")
        sf.write(folder / "dry.wav", dry, sr, subtype="FLOAT")
        cands = []
        for rank, c in enumerate(res["candidates"], 1):
            sf.write(folder / f"top{rank}.wav", np.asarray(render_match(dry, sr, c["chain"]), np.float32), sr,
                     subtype="FLOAT")
            cands.append({"rank": rank, **describe(c["chain"], effects), "afx_distance": round(c["afx_distance"], 4),
                          "screen_distance": round(c["screen_distance"], 4), "chain": c["chain"]})
        truth = r.get("drive_in_catalog") or None
        family = FAMILY[truth] if truth else PEDAL_FAMILY.get(r.get("pedal", ""))
        result = {"id": r["id"], "kind": r["kind"], "source_file": r["source_file"],
                  "reference_pedal_label": r.get("pedal") or None, "knobs_clock": r.get("knobs_clock") or None,
                  "truth_drive": truth, "truth_family": family, "screen_order": res["screen_order"],
                  "candidates": cands, "room_gate": res.get("room_gate"),
                  "target_eq": {"mode": a.target_eq, "curve_db": eq_curve}, "seconds": round(time.perf_counter() - t0, 1),
                  "method": "AFx-Rep grid -> CMA-ES within pedal (AFx-Rep distance, amp crunch"
                            + (", room reverb" if a.room else "") + ") -> volume match"}
        (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        order = [c["drive"] for c in cands] + [d for d in res["screen_order"] if d not in [c["drive"] for c in cands]]
        summary_rows.append({"id": r["id"], "kind": r["kind"], "truth": truth, "family": family,
                             "top": cands[0]["drive"], "rank": order.index(truth) + 1 if truth else None,
                             "family_ok": FAMILY[cands[0]["drive"]] == family if family else None})
        print(f"{i}/{len(rows)} {r['kind'][:3]} {r['id'][:36]:36s} → " +
              " / ".join(f"{c['pedal'] or c['drive']}" for c in cands) + f"  ({result['seconds']:.0f}s)", flush=True)
    scored = [s for s in summary_rows if s["truth"]]
    fam = [s["family_ok"] for s in summary_rows if s["family_ok"] is not None]
    summary = {"n": len(summary_rows), "n_scored": len(scored),
               "top1": float(np.mean([s["rank"] == 1 for s in scored])) if scored else None,
               "top3": float(np.mean([s["rank"] <= 3 for s in scored])) if scored else None,
               "family": float(np.mean(fam)) if fam else None, "rows": summary_rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
