"""
search/evaluate_dataset.py

Step 1 데이터셋 평가 스크립트 (CNN 상위 K개 후보 연동 + 시연용 샘플 저장 옵션 추가)
"""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

from effects import load_effects, render_chain
from search.pipeline import (
    DEFAULT_DATASET_DIR,
    search_and_rank_chains,
    extract_effect_types,
    load_user_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_default_checkpoint() -> Path | None:
    """저장소 내 학습된 CNN 체크포인트(.pt / .pth)를 자동 탐색합니다."""
    search_dirs = [
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "estimation",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in ("*.pt", "*.pth"):
            files = sorted(d.glob(f"**/{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return files[0]
    return None


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


def build_cnn_top_k_inputs(
    dry_path: Path,
    wet_path: Path,
    owned_effects: list | None,
    top_k: int,
    predictor_bundle=None,
    effects_def=None,
):
    if predictor_bundle is not None:
        from estimation.audio import read_audio, segment
        from estimation.chain import make_candidates
        from estimation.infer import predict_pair
        from estimation.refine import to_search_inputs

        model, config, state = predictor_bundle
        dry_seg = segment(read_audio(str(dry_path), config.sample_rate), config)
        ref_seg = segment(read_audio(str(wet_path), config.sample_rate), config, reject_silence=False)
        pred = predict_pair(model, config, dry_seg, ref_seg, state.get("distance_settings"))
        candidates = make_candidates(pred, owned_effects, k=top_k, definitions=model.definitions)
        raw_inputs = to_search_inputs(candidates, model.definitions)
        candidate_chains = [item["chain"] for item in raw_inputs]
        initial_guesses = {(item["effect_type"],): item["chain"] for item in raw_inputs}
        return candidate_chains, initial_guesses, model.definitions, "CNN-TopK"

    try:
        from estimation.chain import catalog, make_candidates, DRIVES, AUXILIARY
        from estimation.refine import to_search_inputs

        defs = catalog()
        uniform_p = 1.0 / len(DRIVES)
        unbiased_pred = {
            "probabilities": {d: uniform_p for d in DRIVES},
            "presence": {a: 0.0 for a in AUXILIARY},
            "values": {
                t: {p["name"]: p["default"] for p in defs[t]["params"] if p["type"] == "continuous"}
                for t in defs
            },
        }
        candidates = make_candidates(unbiased_pred, owned_effects, k=top_k, definitions=defs)
        raw_inputs = to_search_inputs(candidates, defs)
        candidate_chains = [item["chain"] for item in raw_inputs]
        initial_guesses = {(item["effect_type"],): item["chain"] for item in raw_inputs}
        return candidate_chains, initial_guesses, defs, "Unbiased-Catalog-TopK"
    except Exception:
        effects_def = effects_def or load_effects()
        vst_drives = [
            k for k, v in effects_def.items()
            if v.get("category") == "gain_stage" and v.get("source") == "vst"
        ]
        if owned_effects is not None:
            vst_drives = [d for d in vst_drives if d in owned_effects]
        selected = vst_drives[:top_k]
        chains = [
            {
                "amp_id": "fixed_amp_v1",
                "chain": [
                    {
                        "effect_type": d,
                        "values": {p["name"]: p["default"] for p in effects_def[d]["params"]},
                    }
                ],
            }
            for d in selected
        ]
        guesses = {(d,): ch for d, ch in zip(selected, chains)}
        return chains, guesses, effects_def, "Unbiased-VST-TopK"


def evaluate_dataset(
    dataset_dir=DEFAULT_DATASET_DIR,
    limit=5,
    budget=40,
    top_k=3,
    checkpoint_path=None,
    owned_override=None,
    save_audio=False,
):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        synth_root = PROJECT_ROOT / "data" / "synthetic"
        if synth_root.exists():
            dataset_dir = synth_root

    effects_def = load_effects()

    ckpt = Path(checkpoint_path) if checkpoint_path else find_default_checkpoint()
    predictor_bundle = None
    if ckpt and ckpt.exists():
        try:
            from estimation.infer import load_predictor
            predictor_bundle = load_predictor(str(ckpt))
            print(f"[INFO] CNN 체크포인트 로드 완료: {ckpt}")
        except Exception as e:
            print(f"[WARN] CNN 체크포인트 로드 실패 ({e}) -> 무편향(Unbiased) 후보 모드로 진행합니다.")
    else:
        print("[INFO] 지정된 CNN 체크포인트가 없어 무편향(Unbiased) 상위 K개 후보 모드로 평가합니다.")

    if save_audio:
        out_path = PROJECT_ROOT / "demo" / "samples"
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 시연용 샘플 저장 경로: {out_path}")

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

    print("=" * 98)
    print(f"[Dataset Evaluation] CNN 상위 K개 후보 기반 탐색 평가 (K={top_k}, 후보당 budget={budget})")
    print("=" * 98)

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

        candidate_chains, initial_guesses, active_defs, mode_tag = build_cnn_top_k_inputs(
            dry_path=dry_path,
            wet_path=wet_path,
            owned_effects=owned_override,
            top_k=top_k,
            predictor_bundle=predictor_bundle,
            effects_def=effects_def,
        )

        cand_etypes_list = [extract_effect_types(c) for c in candidate_chains]
        in_top_k = gt_etypes in cand_etypes_list

        start_time = time.perf_counter()
        ranked, _ = search_and_rank_chains(
            dry_audio=dry,
            sr=sr,
            target_audio=wet_target,
            candidate_chains=candidate_chains,
            owned_effect_types=owned_override,
            effects_def=active_defs,
            budget=budget,
            n_restarts=2,
            initial_guesses=initial_guesses,
            evaluate_initial=True,
        )
        elapsed = time.perf_counter() - start_time

        best = ranked[0]
        results_log.append({
            "target_id": target_id,
            "mode": mode_tag,
            "gt_chain": "->".join(gt_etypes),
            "pred_chain": "->".join(best["effect_types"]),
            "in_top_k": in_top_k,
            "top1_match": best["effect_types"] == gt_etypes,
            "loss": best["loss"],
            "time": elapsed,
        })

        if save_audio:
            sf.write(str(out_path / f"{target_id}_dry.wav"), dry, sr)
            sf.write(str(out_path / f"{target_id}_target.wav"), wet_target, sr)
            try:
                found_wet = render_chain(dry, sr, best["chain"], effects=active_defs)
            except TypeError:
                found_wet = render_chain(dry, sr, best["chain"])
            sf.write(str(out_path / f"{target_id}_found.wav"), found_wet, sr)

            chain_data = best.get("chain", [])
            if isinstance(chain_data, dict):
                chain_data = chain_data.get("chain", [])

            clean_result = {
                "target_id": target_id,
                "mode": mode_tag,
                "gt_chain": gt_etypes,
                "effect_types": best["effect_types"],
                "chain": chain_data,
                "loss": float(best["loss"]),
                "elapsed_sec": round(elapsed, 2),
            }
            with open(out_path / f"{target_id}_result.json", "w", encoding="utf-8") as f:
                json.dump(clean_result, f, indent=2, ensure_ascii=False)

    if not results_log:
        raise RuntimeError(f"'{base_dir}' 내에 평가 가능한 샘플이 없습니다.")

    print("\n" + "=" * 98)
    print(f"{'Target ID':<34} | {'정답 체인':<16} | {'예측 1위':<16} | {'Top-K포함':<9} | {'1위일치':<7} | {'Loss':>8} | {'시간':>6}")
    print("-" * 98)
    for r in results_log:
        print(f"{r['target_id'][:34]:<34} | {r['gt_chain'][:16]:<16} | {r['pred_chain'][:16]:<16} | "
              f"{str(r['in_top_k']):<9} | {str(r['top1_match']):<7} | {r['loss']:>8.4f} | {r['time']:>5.1f}s")
    print("-" * 98)
    topk_recall = np.mean([r["in_top_k"] for r in results_log]) * 100.0
    top1_acc = np.mean([r["top1_match"] for r in results_log]) * 100.0
    avg_loss = np.mean([r["loss"] for r in results_log])
    avg_time = np.mean([r["time"] for r in results_log])
    print(f"[Step 1 검증 요약] 상위 {top_k}개 내 정답 포함 비율(Top-{top_k} Recall): {topk_recall:.1f}% | "
          f"최종 1위 맞힘 정확도(Top-1 Acc): {top1_acc:.1f}%")
    print(f"                  평균 Loss: {avg_loss:.6f} | 타겟당 평균 소요 시간: {avg_time:.2f}초")
    print("=" * 98)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 1: Evaluate search pipeline using CNN Top-K candidates")
    p.add_argument("--dataset", "--dir", dest="dataset", type=str, default=str(DEFAULT_DATASET_DIR),
                   help="생성 데이터셋 폴더 경로")
    p.add_argument("--checkpoint", type=str, default=None, help="학습된 CNN 모델 체크포인트(.pt) 경로")
    p.add_argument("--top-k", type=int, default=3, help="CNN에서 추출할 상위 K개 후보 수 (기본값: 3)")
    p.add_argument("--limit", type=int, default=5, help="평가할 샘플 수")
    p.add_argument("--budget", type=int, default=40, help="후보당 CMA-ES 평가 횟수 예산")
    p.add_argument("--owned", nargs="*", default=None, help="보유 이펙터 필터 (미지정 시 전체 허용)")
    p.add_argument("--save_audio", action="store_true", help="시연용 샘플을 demo/samples/에 저장")
    args = p.parse_args()
    evaluate_dataset(
        dataset_dir=args.dataset,
        limit=args.limit,
        budget=args.budget,
        top_k=args.top_k,
        checkpoint_path=args.checkpoint,
        owned_override=args.owned,
        save_audio=args.save_audio,
    )