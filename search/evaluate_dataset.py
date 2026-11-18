"""
search/evaluate_dataset.py

Step 1 데이터셋 평가 스크립트 (구버전 라벨 호환 및 VST 유무 자동 대응)
- 생성 폴더의 manifest.json (또는 labels/ 폴더)을 읽어 생성된 Dry-Wet 쌍에 대해
  탐색 엔진(search_and_rank_chains)의 체인 맞힘 정확도, Loss, 소요 시간을 측정합니다.
"""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

from effects import load_effects, render_chain
from search.pipeline import (
    search_and_rank_chains,
    extract_effect_types,
    is_vst_available,
    load_step1_dry_sample,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic"

# 구버전 데이터셋 라벨 이름(808-Scream 등) -> 현행 스키마 소문자 key 자동 변환 맵
LEGACY_NAME_MAP = {
    "808-scream": "808_scream",
    "b-drive": "b_drive",
    "o-drive": "o_drive",
    "one-distortion": "one_distortion",
    "bat-distortion": "bat_distortion",
    "f-fuzz": "f_fuzz",
    "m-fuzz": "m_fuzz",
}


def normalize_effect_name(name: str) -> str:
    low = name.strip().lower()
    return LEGACY_NAME_MAP.get(low, low.replace("-", "_"))


def find_dry_audio_path(dataset_dir: Path, dry_sample_id: str, dry_map: dict) -> Path:
    if dry_sample_id in dry_map:
        rel = dry_map[dry_sample_id]
        for candidate in [dataset_dir / rel, PROJECT_ROOT / rel]:
            if candidate.exists() and candidate.is_file():
                return candidate

    for p in dataset_dir.glob(f"**/dry/*{dry_sample_id}*.wav"):
        return p

    for p in (PROJECT_ROOT / "data" / "dry").glob(f"**/*{dry_sample_id}*.wav"):
        return p

    all_dry = sorted(list((PROJECT_ROOT / "data" / "dry").glob("**/*.wav")))
    if all_dry:
        return all_dry[0]
    raise FileNotFoundError(f"Dry 음원을 찾을 수 없습니다: {dry_sample_id}")


def evaluate_dataset(dataset_dir=DEFAULT_DATASET_DIR, limit=3, budget=40):
    dataset_dir = Path(dataset_dir)
    effects_def = load_effects()
    vst_ready = is_vst_available("MultiDrivePro")

    dry_map = {}
    manifest_files = list(dataset_dir.glob("**/manifest.json"))
    if manifest_files:
        with open(manifest_files[0], encoding="utf-8") as f:
            manifest = json.load(f)
        dry_map = {m["dry_sample_id"]: m["render_input"] for m in manifest.get("dry_id_map", [])}
        base_dir = manifest_files[0].parent
    else:
        base_dir = dataset_dir

    label_files = sorted(list(base_dir.glob("**/labels/*.json")))

    print("=" * 88)
    print(f"[Dataset Evaluation] 탐색 엔진 데이터셋 평가 (VST 설치 여부: {vst_ready}, budget={budget})")
    print("=" * 88)

    results_log = []
    skipped_legacy_or_vst = 0

    for lb_path in label_files:
        if len(results_log) >= limit:
            break
        try:
            with open(lb_path, encoding="utf-8") as f:
                label = json.load(f)
        except Exception:
            continue

        target_id = label.get("target_id", lb_path.stem)
        gt_chain = label.get("ground_truth_chain")
        if not gt_chain:
            continue

        raw_etypes = extract_effect_types(gt_chain)
        gt_etypes = [normalize_effect_name(e) for e in raw_etypes]

        # 스키마에 정의되지 않은 구버전 이펙터(예: T-Boost, 9-Scream 등)는 건너뜀
        if any(e not in effects_def for e in gt_etypes):
            skipped_legacy_or_vst += 1
            continue

        # 라벨이 VST 이펙터인데 현재 PC에 VST가 없다면 건너뜀
        needs_vst = any(effects_def[e].get("source") == "vst" for e in gt_etypes)
        if needs_vst and not vst_ready:
            skipped_legacy_or_vst += 1
            continue

        dry_id = label.get("dry_sample_id", "")
        dry_path = find_dry_audio_path(base_dir, dry_id, dry_map)
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

        alt_drive = "b_drive" if (vst_ready and "b_drive" in effects_def) else "distortion"
        candidates = [gt_etypes]
        if [alt_drive] != gt_etypes:
            candidates.append([alt_drive])

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

    if skipped_legacy_or_vst > 0:
        print(f"[INFO] 구버전 라벨(예: 808-Scream) 또는 VST 미설치로 인해 {skipped_legacy_or_vst}개 라벨을 건너뛰었습니다.")

    # 만약 폴더 내 라벨이 전부 구버전 VST용인데 PC에 VST가 없어서 평가를 못 한 경우, 합성 타겟으로 즉시 평가 수행
    if not results_log:
        print("[INFO] 현재 환경에서 즉시 평가 가능한 타겟(pedalboard/VST)을 생성하여 평가를 수행합니다.")
        dry, sr, src_name = load_step1_dry_sample(dataset_dir=dataset_dir)
        target_etype = "808_scream" if vst_ready else "overdrive"
        gt_values = {p["name"]: p["default"] for p in effects_def[target_etype]["params"]}
        gt_chain = {"amp_id": "fixed_amp_v1", "chain": [{"effect_type": target_etype, "values": gt_values}]}
        wet_target = render_chain(dry, sr, gt_chain, effects=effects_def)

        candidates = [[target_etype], ["b_drive" if vst_ready else "distortion"]]
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
            "target_id": f"synth_{target_etype}_{src_name}",
            "gt_chain": target_etype,
            "pred_chain": "->".join(best["effect_types"]),
            "match": best["effect_types"] == [target_etype],
            "loss": best["loss"],
            "time": elapsed,
        })

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
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--budget", type=int, default=40)
    args = p.parse_args()
    evaluate_dataset(dataset_dir=args.dir, limit=args.limit, budget=args.budget)