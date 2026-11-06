"""
search/evaluate_dataset.py

Step 0 검증: 정답 데이터(wav + json)에서 Overdrive 노브(drive, level)를 복원합니다.
탐색 알고리즘은 search/pipeline.py의 Random Search / Restart-CMA-ES를 그대로 사용합니다.

Usage (프로젝트 루트에서)
-------------------------
python -m search.evaluate_dataset                  # OD 정답 데이터 전체
python -m search.evaluate_dataset --limit 10       # 앞에서 10개만 (빠른 확인용)
python -m search.evaluate_dataset --budget 200 --restarts 4 --seeds 1
python -m search.evaluate_dataset --examples       # 저장된 결과로 오디오 예시(dry / 정답 / 찾은 소리) 생성
"""
import os
import csv
import glob
import json
import argparse
import numpy as np
import soundfile as sf

from metrics.spectral import spectral_distance
from scripts.Create_Synthetic_pairs import apply_effects
from search.pipeline import (
    DRY_AUDIO_DIR, OUTPUT_DIR, PARAM_NAMES, normalize, run_random_search, run_cma_es,
)

LABEL_DIR = "data/synthetic/Fender_Strat_Clean/labels"
EFFECT_TYPE = "overdrive"
KNOBS = ["drive", "level"]  # PARAM_SPECS 순서와 같아야 합니다.

# Step 0 통과 기준: 노브 오차가 모두 5% 이내 "또는" 소리 거리가 기준 이하이면 성공
# (다른 노브 조합으로 같은 소리를 내는 경우도 성공으로 봄. 예: drive가 낮으면 drive와 level이 같은 역할)
KNOB_THRESHOLD_PCT = 5.0
AUDIO_THRESHOLD = 0.00935930329251289  # metrics/README.md의 -80dB 하한 기준 잠정 참고값
PASS_RATE = 0.9


# ----------------------------------------------------------------------
# 정답 데이터 로딩
# ----------------------------------------------------------------------
def load_pairs(limit=None):
    """Overdrive 정답 json을 읽어 (target_id, dry_path, wet_path, 정답 노브) 목록을 반환합니다."""
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))
    if not label_files:
        raise FileNotFoundError(
            f"[ERROR] '{LABEL_DIR}'에 정답 json이 없습니다. DRIVE_SETUP.md대로 드라이브를 연결했는지 확인하세요."
        )

    pairs = []
    for path in label_files:
        with open(path, encoding="utf-8") as f:
            label = json.load(f)
        chain = label["ground_truth_chain"]["chain"]
        if len(chain) != 1 or chain[0]["effect_type"] != EFFECT_TYPE:
            continue
        values = chain[0]["values"]
        pairs.append((
            label["target_id"],
            os.path.join(DRY_AUDIO_DIR, label["dry_sample_id"] + ".wav"),
            label["file_path"],
            np.array([values[k] for k in KNOBS], dtype=float),
        ))
        if limit and len(pairs) >= limit:
            break
    return pairs


def load_mono(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def loudest_window(dry, wet, sr, duration_sec=1.0):
    """dry 기준으로 에너지가 가장 큰 구간을 골라 dry와 wet을 같은 위치에서 자릅니다.
    컴프가 없는 Step 0 체인은 시간에 따라 상태가 변하지 않으므로, 잘라서 렌더링해도 결과가 같습니다."""
    win = int(sr * duration_sec)
    if len(dry) <= win:
        return dry, wet
    hop = max(1, sr // 10)
    starts = np.arange(0, len(dry) - win + 1, hop)
    s = int(starts[int(np.argmax([np.sum(dry[i:i + win] ** 2) for i in starts]))])
    return np.ascontiguousarray(dry[s:s + win]), np.ascontiguousarray(wet[s:s + win])


# ----------------------------------------------------------------------
# 메인 평가
# ----------------------------------------------------------------------
def evaluate(budget=200, n_restarts=4, n_seeds=1, limit=None):
    pairs = load_pairs(limit)
    print(f"[INFO] {EFFECT_TYPE} 정답 데이터 {len(pairs)}개, 예산 {budget}회, 시드 {n_seeds}개")

    methods = {
        "Random Search": lambda f, s: run_random_search(f, budget, s),
        f"CMA-ES ({n_restarts} restarts)": lambda f, s: run_cma_es(f, budget, s, n_runs=n_restarts),
    }
    rows = []

    for i, (target_id, dry_path, wet_path, truth) in enumerate(pairs, 1):
        dry, sr = load_mono(dry_path)
        wet, _ = load_mono(wet_path)
        dry, wet = loudest_window(dry, wet, sr)

        def objective(params):
            # 정답 데이터와 같은 조건(컴프 없음)으로 렌더링
            rendered = apply_effects(dry, sr, None, EFFECT_TYPE, dict(zip(KNOBS, map(float, params))))
            return spectral_distance(rendered, wet)

        for name, fn in methods.items():
            for seed in range(n_seeds):
                found, loss, _ = fn(objective, seed)
                err = np.abs(normalize(found) - normalize(truth)) * 100.0
                rows.append({
                    "target_id": target_id, "method": name, "seed": seed, "loss": loss,
                    **{f"truth_{k}": t for k, t in zip(KNOBS, truth)},
                    **{f"found_{k}": v for k, v in zip(KNOBS, found)},
                    **{f"err_{k}_pct": e for k, e in zip(KNOBS, err)},
                    "err_mean_pct": err.mean(),
                    "err_max_pct": err.max(),
                })
        print(f"[{i}/{len(pairs)}] {target_id} 완료")

    print_summary(rows)
    save_rows(rows)
    return rows


def print_summary(rows):
    width = 84
    print("\n" + "=" * width)
    print(f"[요약] 정규화 오차(%) = |찾은 값 - 정답| / 노브 범위  ({', '.join(n.strip() for n in PARAM_NAMES)})")
    print("-" * width)
    print("성공 비율 = 모든 노브의 오차가 기준 이내인 파일 비율")
    print("-" * width)
    print(f"{'Method':<22} | {'평균':>6} | {'drive':>6} | {'level':>6} | {'2% 이내':>7} | {'5% 이내':>7} | {'10% 이내':>8}")
    print("-" * width)
    for method in dict.fromkeys(r["method"] for r in rows):
        sub = [r for r in rows if r["method"] == method]
        max_err = np.array([r["err_max_pct"] for r in sub])
        print(f"{method:<22} | {np.mean([r['err_mean_pct'] for r in sub]):>6.1f} | "
              f"{np.mean([r['err_drive_pct'] for r in sub]):>6.1f} | "
              f"{np.mean([r['err_level_pct'] for r in sub]):>6.1f} | "
              + " | ".join(f"{np.mean(max_err <= th) * 100:>6.0f}%" for th in (2.0, 5.0, 10.0)))
    print("=" * width)

    print(f"\n[Step 0 판정] 노브 오차 {KNOB_THRESHOLD_PCT:g}% 이내 또는 소리 거리 {AUDIO_THRESHOLD:.5f} 이하 → 성공, "
          f"성공률 {PASS_RATE * 100:.0f}% 이상 → 통과")
    print("-" * width)
    print(f"{'Method':<22} | {'노브 성공':>9} | {'소리 성공':>9} | {'최종 성공':>9} | 판정")
    print("-" * width)
    for method in dict.fromkeys(r["method"] for r in rows):
        sub = [r for r in rows if r["method"] == method]
        knob_ok = np.array([r["err_max_pct"] <= KNOB_THRESHOLD_PCT for r in sub])
        audio_ok = np.array([r["loss"] <= AUDIO_THRESHOLD for r in sub])
        rate = np.mean(knob_ok | audio_ok)
        print(f"{method:<22} | {np.mean(knob_ok) * 100:>8.0f}% | {np.mean(audio_ok) * 100:>8.0f}% | "
              f"{rate * 100:>8.0f}% | {'통과' if rate >= PASS_RATE else '불통과'}")
    print("=" * width)


def save_rows(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"dataset_{EFFECT_TYPE}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] 상세 결과 저장: {path}")


# ----------------------------------------------------------------------
# 결과 기록용 오디오 예시
# ----------------------------------------------------------------------
EXAMPLE_DIR = os.path.join(OUTPUT_DIR, "examples")


def pick_examples(rows):
    """대표 사례 3개: CMA-ES 노브 성공 / CMA-ES 소리만 성공 / Random Search 실패"""
    cma = [r for r in rows if r["method"].startswith("CMA")]
    rnd = [r for r in rows if r["method"] == "Random Search"]
    knob_ok = lambda r: r["err_max_pct"] <= KNOB_THRESHOLD_PCT
    audio_ok = lambda r: r["loss"] <= AUDIO_THRESHOLD
    picks = {
        "1_cma_knob_ok": min((r for r in cma if knob_ok(r)), key=lambda r: r["err_max_pct"]),
        "2_cma_audio_only": max((r for r in cma if audio_ok(r) and not knob_ok(r)), key=lambda r: r["err_max_pct"]),
    }
    failed = [r for r in rnd if not (knob_ok(r) or audio_ok(r))]
    if failed:
        picks["3_random_fail"] = max(failed, key=lambda r: r["loss"])
    return picks


def save_examples(csv_path=None):
    """저장된 결과 CSV에서 대표 사례를 골라 dry / 정답(wet) / 찾은 소리를 wav로 저장합니다."""
    csv_path = csv_path or os.path.join(OUTPUT_DIR, f"dataset_{EFFECT_TYPE}.csv")
    with open(csv_path, encoding="utf-8") as f:
        rows = [{k: (v if k in ("target_id", "method") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    pairs = {p[0]: p for p in load_pairs()}
    os.makedirs(EXAMPLE_DIR, exist_ok=True)

    for name, r in pick_examples(rows).items():
        _, dry_path, wet_path, truth = pairs[r["target_id"]]
        dry, sr = load_mono(dry_path)
        wet, _ = load_mono(wet_path)
        found = {k: r[f"found_{k}"] for k in KNOBS}
        rendered = apply_effects(dry, sr, None, EFFECT_TYPE, found)
        for tag, audio in (("dry", dry), ("truth", wet), ("found", rendered)):
            sf.write(os.path.join(EXAMPLE_DIR, f"{name}_{tag}.wav"), audio, sr, subtype="FLOAT")
        print(f"[{name}] {r['target_id']} ({r['method']})\n"
              f"    정답 { {k: round(float(t), 2) for k, t in zip(KNOBS, truth)} } / 찾은 값 { {k: round(v, 2) for k, v in found.items()} } "
              f"/ 노브 오차 {r['err_max_pct']:.1f}% / 소리 거리 {r['loss']:.4f}")
    print(f"[INFO] 오디오 예시 저장: {EXAMPLE_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description="Step 0: 정답 데이터에서 Overdrive 노브 복원 평가")
    p.add_argument("--budget", type=int, default=200, help="파일당 평가 횟수 예산 (기본값: 200)")
    p.add_argument("--restarts", type=int, default=4, help="CMA-ES 분할 실행 횟수 (기본값: 4)")
    p.add_argument("--seeds", type=int, default=1, help="파일당 반복 시드 수 (기본값: 1)")
    p.add_argument("--limit", type=int, default=None, help="앞에서 N개 파일만 평가 (빠른 확인용)")
    p.add_argument("--examples", action="store_true", help="평가 대신, 저장된 결과로 오디오 예시만 생성")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.examples:
        save_examples()
    else:
        evaluate(budget=args.budget, n_restarts=args.restarts, n_seeds=args.seeds, limit=args.limit)
