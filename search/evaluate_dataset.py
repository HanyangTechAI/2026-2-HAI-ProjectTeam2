"""
search/evaluate_dataset.py

Step 1 데이터셋 평가 스크립트 (CNN 상위 K개 후보 연동 & 무편향 정확도 측정)
- 정답(gt_etypes)을 후보에 강제로 포함하던 기존 방식을 제거
- 5번(도영님) CNN 모델 추론 결과(또는 예측 JSON)로부터 상위 K개 후보(make_candidates)를 생성하여 탐색
- Step 1 검증 지표 2종 동시 출력:
  1) 상위 K개 후보 내 정답 포함 비율 (Top-K Recall)
  2) CMA-ES 탐색 후 최종 1위 이펙터 맞힘 정확도 (Top-1 Accuracy)
"""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

from effects import load_effects
from effects.render import find_plugin
from estimation.chain import DRIVES, catalog, make_candidates
from search.pipeline import (
    DEFAULT_DATASET_DIR,
    search_and_rank_chains,
    extract_effect_types,
    load_user_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_vst_available(class_name="MultiDrivePro"):
    try:
        find_plugin(class_name)
        return True
    except FileNotFoundError:
        return False


class CnnPredictor:
    """5번(도영님) CNN: 생성 dry + 목표 wet을 함께 넣어 7종 확률과 노브 초기값을 예측합니다."""
    def __init__(self, checkpoint_path):
        from estimation.infer import load_predictor  # torch 필요 (.conda-cnn 환경)
        self.model, self.config, self.state = load_predictor(checkpoint_path)

    def __call__(self, dry_path, wet_path, sr):
        from estimation.audio import read_audio, segment
        from estimation.infer import predict_pair
        if sr != self.config.sample_rate:
            raise ValueError(f"CNN 샘플레이트({self.config.sample_rate})와 데이터({sr})가 다릅니다.")
        dry = segment(read_audio(str(dry_path), self.config.sample_rate), self.config)
        ref = segment(read_audio(str(wet_path), self.config.sample_rate), self.config, reject_silence=False)
        return predict_pair(self.model, self.config, dry, ref, self.state["distance_settings"])


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


def get_cnn_top_k_candidates(
    dry_path: Path,
    wet_path: Path,
    sr: int,
    target_id: str,
    owned_effects: list,
    top_k: int = 3,
    predictor=None,
    predictions_json: dict = None,
):
    """
    정답(gt_etypes)을 참조하지 않고 CNN 예측만으로 상위 K개 후보(x0 포함)를 생성합니다.
    1순위: 사전 예측 JSON(target_id → infer.py 출력) / 2순위: CNN checkpoint 직접 추론
    CNN이 없으면 순위를 매길 수 없으므로 보유 드라이브 전체를 기본 노브로 탐색합니다.
    """
    defs = catalog()

    pred = None
    if predictions_json and target_id in predictions_json:
        pred = predictions_json[target_id]
    elif predictor is not None:
        pred = predictor(dry_path, wet_path, sr)

    if pred is not None:
        return make_candidates(pred, owned_effect_types=owned_effects, k=top_k, definitions=defs), "CNN-TopK"

    uniform = {
        "probabilities": {d: 1.0 / len(DRIVES) for d in DRIVES},
        "values": {d: {p["name"]: p["default"] for p in defs[d]["params"] if p["type"] == "continuous"}
                   for d in DRIVES},
    }
    return make_candidates(uniform, owned_effect_types=owned_effects, k=len(DRIVES), definitions=defs), "All-Drives"


def evaluate_dataset(
    dataset_dir=DEFAULT_DATASET_DIR,
    limit=5,
    budget=40,
    top_k=3,
    checkpoint_path=None,
    predictions_path=None,
):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        synth_root = PROJECT_ROOT / "data" / "synthetic"
        if synth_root.exists():
            dataset_dir = synth_root

    effects_def = load_effects()
    vst_ready = is_vst_available("MultiDrivePro")
    if not vst_ready:
        print("[WARN] MultiDrive PRO 플러그인이 없어 VST 데이터셋을 평가할 수 없습니다.")
        return
    source_filter = "vst"

    # 평가 시 보유 이펙터: VST 드라이브 7종 전체 보유로 가정
    owned_effects = load_user_inventory()
    for d in DRIVES:
        if d not in owned_effects:
            owned_effects.append(d)

    predictions_json = None
    if predictions_path:
        with open(predictions_path, encoding="utf-8") as f:
            predictions_json = json.load(f)
    predictor = CnnPredictor(checkpoint_path) if checkpoint_path else None
    if predictor is None and predictions_json is None:
        print("[WARN] --checkpoint/--predictions 없음 -> CNN 순위 없이 보유 드라이브 전체를 탐색합니다.")

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

    print("=" * 96)
    print(f"[Dataset Evaluation] CNN 상위 K개 후보 기반 탐색 평가 (K={top_k}, budget={budget}, source={source_filter})")
    print("=" * 96)

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
        # CNN 후보는 드라이브 1개짜리이므로 정답 체인의 드라이브 종류로 비교 (컴프·EQ 제외)
        gt_drive = next((e for e in gt_etypes if e in DRIVES), None)
        if gt_drive is None:
            continue
        if any(effects_def[e].get("source") == "vst" for e in gt_etypes) and not vst_ready:
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

        # 핵심 수정: 정답(gt_etypes)을 후보에 강제로 넣지 않고 CNN 상위 K개 후보만 생성
        candidates, cand_mode = get_cnn_top_k_candidates(
            dry_path=dry_path,
            wet_path=wet_path,
            sr=sr,
            target_id=target_id,
            owned_effects=owned_effects,
            top_k=top_k,
            predictor=predictor,
            predictions_json=predictions_json,
        )

        cand_drives = [c["effect_type"] for c in candidates]
        in_top_k = gt_drive in cand_drives

        start_time = time.perf_counter()
        ranked, _ = search_and_rank_chains(
            dry_audio=dry,
            sr=sr,
            target_audio=wet_target,
            candidate_chains=candidates,
            owned_effect_types=owned_effects,
            effects_def=effects_def,
            budget=budget,
            n_restarts=2,
        )
        elapsed = time.perf_counter() - start_time

        best = ranked[0]
        results_log.append({
            "target_id": target_id,
            "mode": cand_mode,
            "gt_chain": "->".join(gt_etypes),
            "pred_chain": "->".join(best["effect_types"]),
            "in_top_k": in_top_k,
            "top1_match": best["effect_types"] == [gt_drive],
            "loss": best["loss"],
            "time": elapsed,
        })

    if not results_log:
        print(f"[WARN] '{base_dir}' 내에 현재 환경(VST={vst_ready})에서 평가 가능한 샘플이 없습니다.")
        return

    print("\n" + "=" * 96)
    print(f"{'Target ID':<34} | {'정답':<14} | {'예측 1위':<14} | {'Top-K포함':<9} | {'1위일치':<7} | {'Loss':>8} | {'시간':>6}")
    print("-" * 96)
    for r in results_log:
        print(f"{r['target_id'][:34]:<34} | {r['gt_chain']:<14} | {r['pred_chain']:<14} | "
              f"{str(r['in_top_k']):<9} | {str(r['top1_match']):<7} | {r['loss']:>8.4f} | {r['time']:>5.1f}s")
    print("-" * 96)
    topk_recall = np.mean([r["in_top_k"] for r in results_log]) * 100.0
    top1_acc = np.mean([r["top1_match"] for r in results_log]) * 100.0
    avg_loss = np.mean([r["loss"] for r in results_log])
    avg_time = np.mean([r["time"] for r in results_log])
    if all(r["mode"] == "All-Drives" for r in results_log):
        print(f"[검증 요약] (CNN 없음: 7종 전체 탐색, Top-K 포함률은 의미 없음) "
              f"최종 1위 맞힘 정확도(Top-1 Acc): {top1_acc:.1f}%")
    else:
        print(f"[검증 요약] 상위 {top_k}개 내 정답 포함 비율(Top-{top_k} Recall): {topk_recall:.1f}% | "
              f"최종 1위 맞힘 정확도(Top-1 Acc): {top1_acc:.1f}%")
    print(f"            평균 Loss: {avg_loss:.6f} | 타겟당 평균 소요 시간: {avg_time:.2f}초")
    print("=" * 96)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 1: Evaluate search pipeline with CNN Top-K candidates")
    p.add_argument("--dataset", "--dir", dest="dataset", type=str, default=str(DEFAULT_DATASET_DIR),
                   help="생성 데이터셋 폴더 경로 (기본값: data/synthetic/step1_vst_v0_임시)")
    p.add_argument("--limit", type=int, default=5, help="평가할 샘플 수")
    p.add_argument("--budget", type=int, default=40, help="후보당 CMA-ES 평가 횟수 예산")
    p.add_argument("--top-k", type=int, default=3, help="CNN에서 추출할 상위 K개 후보 수 (기본값: 3)")
    p.add_argument("--checkpoint", type=str, default=None, help="학습된 CNN 모델 가중치 경로")
    p.add_argument("--predictions", type=str, default=None, help="CNN 사전 예측 결과 JSON 경로")
    args = p.parse_args()
    evaluate_dataset(
        dataset_dir=args.dataset,
        limit=args.limit,
        budget=args.budget,
        top_k=args.top_k,
        checkpoint_path=args.checkpoint,
        predictions_path=args.predictions,
    )