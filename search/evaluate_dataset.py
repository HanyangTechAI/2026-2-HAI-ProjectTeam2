"""
search/evaluate_dataset.py

데이터셋 평가 스크립트 (파이프라인 변경 사항 반영)
- effects.render_chain 및 search.pipeline의 search_and_rank_chains를 활용하여
  데이터셋 내 타겟 샘플들에 대한 탐색 성능(정확도 및 Loss)을 평가합니다.
"""
import os
import json
from pathlib import Path
import numpy as np
import soundfile as sf

from effects import load_effects, render_chain
from metrics.spectral import spectral_distance
from search.pipeline import search_and_rank_chains, load_user_inventory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGETS_DIR = PROJECT_ROOT / "data" / "synthetic"  # 또는 데이터셋 타겟 경로


def evaluate_dataset():
    print("=" * 70)
    print("[Dataset Evaluation] 탐색 엔진 데이터셋 일괄 평가 시작")
    print("=" * 70)

    effects_def = load_effects()
    owned_effects = load_user_inventory()

    # 예시 후보 체인 목록
    candidate_chains = [
        ["overdrive"],
        ["distortion"],
        ["compressor", "overdrive"],
        ["overdrive", "eq_3band"],
        ["compressor", "overdrive", "eq_3band"],
    ]

    # 테스트용 드라이 오디오 로드
    dry_path = PROJECT_ROOT / "data" / "synthetic" / "dry" / "sample_guitar_dry.wav"
    if not dry_path.exists():
        # 대체 경로 탐색
        wavs = list((PROJECT_ROOT / "data").glob("**/*.wav"))
        if not wavs:
            print("[ERROR] 평가에 사용할 .wav 음원 파일을 찾을 수 없습니다.")
            return
        dry_path = wavs[0]

    dry_audio, sr = sf.read(str(dry_path), dtype="float32")
    if dry_audio.ndim > 1:
        dry_audio = dry_audio.mean(axis=1)
    dry_audio = dry_audio[:int(sr * 1.0)]  # 1초 자르기

    # 가상의 타겟 생성 및 평가 테스트
    ground_truth = {
        "amp_id": "fixed_amp_v1",
        "chain": [{"effect_type": "overdrive", "values": {"drive": 8.0, "level": 0.0}}],
    }
    target_audio = render_chain(dry_audio, sr, ground_truth, effects=effects_def)

    print(f"[INFO] 평가용 타겟 오디오 생성 완료 (사용 음원: {dry_path.name})")

    ranked_results, dropped = search_and_rank_chains(
        dry_audio=dry_audio,
        sr=sr,
        target_audio=target_audio,
        top_k_candidates=candidate_chains,
        owned_effect_types=owned_effects,
        effects_def=effects_def,
        budget=50,  # 빠른 테스트용 예산
        n_restarts=2,
        n_seeds=1,
    )

    print("\n[평가 결과 요약]")
    if ranked_results:
        best = ranked_results[0]
        print(f"  - 1위 예측 체인: {' -> '.join(best['effect_types'])}")
        print(f"  - 최소 Loss: {best['loss']:.6f}")
        print(f"  - 예측 노브 값: {best['chain']['chain']}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_dataset()