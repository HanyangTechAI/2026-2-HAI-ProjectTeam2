"""
search/evaluate_dataset.py

Step 1 데이터셋 평가 스크립트
- 생성 폴더(--dataset, 기본값: data/synthetic/step1_vst_v0_임시)의 dry/ 및 labels/를 읽어
  VST 후보 체인 탐색 정확도, Loss, 소요 시간을 측정합니다.
"""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

from effects import load_effects
from search.pipeline import (
    DEFAULT_DATASET_DIR,
    search_and_rank_chains,
    extract_effect_types,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_generated_dry_path(base_dir: Path, dry_sample_id: str, dry_map: dict) -> Path:
    if dry_sample_id in dry_map:
        rel = dry_map[dry_sample_id]
        for candidate in (base_dir / rel, PROJECT_ROOT / rel):
            if candidate.exists() and candidate.is_file():
                return candidate

    for p in base_dir.glob(f"**/dry/*{dry_sample_id}*.wav"):
        return p

    all_gen_dry = sorted(base_dir.glob("**/dry/*.wav"))
    if all_gen_dry:
        return all_gen_dry[0]

    raise FileNotFoundError(
        f"생성 폴더('{base_dir}') 내에 dry 음원({dry_sample_id})이 없습니다. "
        "원본 dry 대신 생성 폴더의 dry/가 필요합니다."
    )


def evaluate_dataset(dataset_dir=DEFAULT_DATASET_DIR, limit=3, budget=40):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        synth_root = PROJECT_ROOT / "data" / "synthetic"
        if synth_root.exists():
            dataset_dir = synth_root

    effects_def = load_effects()

    dry_map = {}
    manifest_files = sorted(dataset_dir.glob("**/manifest.json"))
    if manifest_files:
        with open(manifest_files[0], encoding="utf-8") as f:
            manifest = json.load(f)
        dry_map = {m["dry_sample_id"]: m["render_input"] for m in manifest.get("dry_id_map", [])}
        base_dir = manifest_files[0].parent
    else:
        base_dir = dataset_dir

    label_files = sorted(base_dir.glob("**/labels/*.json"))
    if not label_files:
        raise FileNotFoundError(f"'{base_dir}' 경로에서 평가할 labels/*.json 파일을 찾을 수 없습니다.")

    print("=" * 88)
    print(f"[Dataset Evaluation] 생성 폴더 데이터셋 평가 (경로: {base_dir}, budget={budget})")
    print("=" * 88)

    results_log = []
    for lb_path in label_files:
        if len(results_log) >= limit:
            break
        with open(lb_path, encoding="utf-8") as f:
            label = json.load(f)

        target_id = label.get("target_id", lb_path.stem)
        gt_chain = label.get("ground_truth_chain")
        if not gt_chain:
            continue

        gt_etypes = extract_effect_types(gt_chain)
        if any(e not in effects_def for e in gt_etypes):
            continue

        dry_id = label.get("dry_sample_id", "")
        dry_path = find_generated_dry_path(base_dir, dry_id, dry_map)

        wet_candidates = list(base_dir.glob(f"**/audio/{target_id}.wav"))
        if not wet_candidates:
            wet_candidates = list(base_dir.glob(f"**/audio/*{target_id}*.wav"))
        if not wet_candidates and label.get("file_path"):
            p = PROJECT_ROOT / label["file_path"]
            if p.exists():
                wet_candidates = [p]
        if not wet_candidates:
            continue

        wet_path = wet_candidates[0]
        dry, sr = sf.read(str(dry_path), dtype="float32")
        wet_target, _ = sf.read(str(wet_path), dtype="float32")
        if dry.ndim > 1:
            dry = dry.mean(axis=1)
        if wet_target.ndim > 1:
            wet_target = wet_target.mean(axis=1)

        alt_drive = "b_drive" if gt_etypes != ["b_drive"] else "808_scream"
        candidates = [gt_etypes, [alt_drive]]

        start_time = time.perf_counter()
        ranked, _ = search_and_rank_chains(
            dry_audio=dry,
            sr=sr,
            target_audio=wet_target,
            candidate_configs=candidates,
            owned_effect_types=None,
            effects_def=effects_def,
            budget=budget,
            n_restarts=2,
        )
        elapsed = time.perf_counter() - start_time

        best = ranked[0]
        results_log.append({
            "target_id": target_id,
            "gt_chain": "->".join(gt_etypes),
            "pred_chain": "->".join(best["effect_types"]),
            "match": best["effect_types"] == gt_etypes,
            "loss": best["loss"],
            "time": elapsed,
        })

    if not results_log:
        raise RuntimeError(f"'{base_dir}' 내에 현재 스키마와 일치하는 평가 가능 샘플이 없습니다.")

    print("\n" + "=" * 88)
    print(f"{'Target ID':<38} | {'정답':<14} | {'예측 1위':<14} | {'일치':<5} | {'Loss':>8} | {'시간':>6}")
    print("-" * 88)
    for r in results_log:
        print(f"{r['target_id'][:38]:<38} | {r['gt_chain']:<14} | {r['pred_chain']:<14} | "
              f"{str(r['match']):<5} | {r['loss']:>8.4f} | {r['time']:>5.1f}s")
    print("-" * 88)
    acc = np.mean([r["match"] for r in results_log]) * 100.0
    avg_loss = np.mean([r["loss"] for r in results_log])
    avg_time = np.mean([r["time"] for r in results_log])
    print(f"이펙터 맞힘 정확도: {acc:.1f}% | 평균 Loss: {avg_loss:.6f} | 타겟당 평균 시간: {avg_time:.2f}초")
    print("=" * 88)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 1: Evaluate search pipeline on generated dataset")
    p.add_argument("--dataset", "--dir", dest="dataset", type=str, default=str(DEFAULT_DATASET_DIR),
                   help="생성 데이터셋 폴더 경로 (기본값: data/synthetic/step1_vst_v0_임시)")
    p.add_argument("--limit", type=int, default=3, help="평가할 샘플 수")
    p.add_argument("--budget", type=int, default=40, help="후보당 CMA-ES 평가 횟수 예산")
    args = p.parse_args()
    evaluate_dataset(dataset_dir=args.dataset, limit=args.limit, budget=args.budget)