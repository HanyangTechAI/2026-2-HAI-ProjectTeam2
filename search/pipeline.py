"""
search/pipeline.py

Step 1: 스키마 기반 다중 후보 체인 탐색 엔진 (VST / 생성 데이터 연동)
- 학습·탐색 입력을 생성 폴더의 dry/ (data/synthetic/dry)로 사용
- 모델의 상위 K개 후보 및 초기 추정값(top_k_candidates) 입력 지원
- VST 이펙터 탐색 실행 및 연산 소요 시간 측정
- 사용자 보유 이펙터(user_inventory) 필터링 및 Loss 오름차순(유사도순) 정렬
"""
import os
import glob
import json
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import cma

from effects import load_effects, render_chain
from metrics.spectral import spectral_distance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Task 1 요구사항: 원본 dry가 아닌 생성 폴더의 dry/ 경로 사용
DRY_AUDIO_DIR = PROJECT_ROOT / "data" / "synthetic" / "dry"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "search"
PENALTY = 1e3


# ----------------------------------------------------------------------
# [1] 스키마 기반 체인 파라미터 스펙(ChainParameterSpec) 동적 빌더
# ----------------------------------------------------------------------
class ChainParameterSpec:
    def __init__(self, effect_types, effects_def=None, amp_id="fixed_amp_v1"):
        self.effects_def = effects_def or load_effects()
        self.effect_types = list(effect_types)
        self.amp_id = amp_id

        self.param_meta = []  # (effect_idx, effect_type, param_name, low, high, is_log)
        self.fixed_values = [{} for _ in self.effect_types]

        for idx, etype in enumerate(self.effect_types):
            if etype not in self.effects_def:
                raise ValueError(f"스키마에 정의되지 않은 effect_type: {etype}")
            edef = self.effects_def[etype]
            for p in edef["params"]:
                pname = p["name"]
                ptype = p["type"]
                if ptype == "continuous":
                    low, high = float(p["range"][0]), float(p["range"][1])
                    is_log = (p.get("scale") == "log")
                    self.param_meta.append((idx, etype, pname, low, high, is_log))
                else:
                    self.fixed_values[idx][pname] = p["default"]

        self.dim = len(self.param_meta)
        self.lows = np.array([m[3] for m in self.param_meta], dtype=float)
        self.highs = np.array([m[4] for m in self.param_meta], dtype=float)
        self.is_log = np.array([m[5] for m in self.param_meta], dtype=bool)
        self.names = [f"{m[1]}.{m[2]}" for m in self.param_meta]

    def denormalize(self, x_norm):
        if self.dim == 0:
            return np.array([], dtype=float)
        x = np.clip(np.asarray(x_norm, dtype=float), 0.0, 1.0)
        diff = np.where(self.highs == self.lows, 1.0, self.highs - self.lows)
        lin = self.lows + x * diff
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_lows = np.where(self.is_log, np.maximum(self.lows, 1e-12), 1.0)
            safe_highs = np.where(self.is_log, np.maximum(self.highs, 1e-12), 1.0)
            log_lo, log_hi = np.log(safe_lows), np.log(safe_highs)
            log_diff = np.where(log_hi == log_lo, 1.0, log_hi - log_lo)
            logv = np.exp(log_lo + x * log_diff)
        return np.where(self.is_log, logv, lin)

    def normalize(self, real_params):
        if self.dim == 0:
            return np.array([], dtype=float)
        p = np.asarray(real_params, dtype=float)
        diff = np.where(self.highs == self.lows, 1.0, self.highs - self.lows)
        lin = (p - self.lows) / diff
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_p = np.where(self.is_log, np.maximum(p, 1e-12), 1.0)
            safe_lows = np.where(self.is_log, np.maximum(self.lows, 1e-12), 1.0)
            safe_highs = np.where(self.is_log, np.maximum(self.highs, 1e-12), 1.0)
            log_lo, log_hi = np.log(safe_lows), np.log(safe_highs)
            log_diff = np.where(log_hi == log_lo, 1.0, log_hi - log_lo)
            logv = (np.log(safe_p) - log_lo) / log_diff
        return np.clip(np.where(self.is_log, logv, lin), 0.0, 1.0)

    def vector_to_chain(self, real_params):
        chain_list = [
            {"effect_type": etype, "values": dict(self.fixed_values[idx])}
            for idx, etype in enumerate(self.effect_types)
        ]
        for val, (idx, _, pname, _, _, _) in zip(real_params, self.param_meta):
            chain_list[idx]["values"][pname] = round(float(val), 4)
        return {"amp_id": self.amp_id, "chain": chain_list}

    def chain_to_vector(self, chain_data):
        items = chain_data["chain"] if isinstance(chain_data, dict) else chain_data
        vals = []
        for idx, _, pname, _, _, _ in self.param_meta:
            vals.append(float(items[idx]["values"][pname]))
        return np.array(vals, dtype=float)


# ----------------------------------------------------------------------
# [2] 사용자 보유 이펙터(user_inventory) 필터링
# ----------------------------------------------------------------------
def load_user_inventory(inventory_path=None):
    path = Path(inventory_path) if inventory_path else (SCHEMAS_DIR / "user_inventory.data.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("owned_effect_types", [])


def extract_effect_types(candidate):
    """
    후보 체인 입력 형식을 유연하게 처리:
    - 리스트 형태: ["overdrive"] 또는 [{"effect_type": "overdrive", "values": {...}}]
    - 딕셔너리 형태: {"effect_types": [...]} 또는 {"chain": [...]} 또는 모델이 주는 상위 K개 후보 포맷
    """
    if isinstance(candidate, dict):
        if "effect_types" in candidate:
            return list(candidate["effect_types"])
        if "chain" in candidate:
            return [item["effect_type"] for item in candidate["chain"]]
    if isinstance(candidate, list):
        if not candidate:
            return []
        if isinstance(candidate[0], dict):
            return [item["effect_type"] for item in candidate]
        return [str(x) for x in candidate]
    raise ValueError(f"지원하지 않는 후보 체인 형식: {candidate}")


def filter_candidates_by_inventory(candidate_chains, owned_effect_types):
    owned_set = set(owned_effect_types)
    kept, dropped = [], []
    for cand in candidate_chains:
        etypes = extract_effect_types(cand)
        missing = [e for e in etypes if e not in owned_set]
        if not missing:
            kept.append(cand)  # 후보 전체 구조(초기값 포함)를 보존
        else:
            dropped.append((etypes, missing))
    return kept, dropped


# ----------------------------------------------------------------------
# [3] 평가 추적기 및 탐색 알고리즘
# ----------------------------------------------------------------------
class Tracker:
    def __init__(self, objective_fn, spec: ChainParameterSpec, budget: int):
        self.objective_fn = objective_fn
        self.spec = spec
        self.budget = budget
        self.n = 0
        self.best_loss = float("inf")
        self.best_norm = np.full(spec.dim, 0.5, dtype=float)
        self.history = []

    @property
    def exhausted(self):
        return self.n >= self.budget

    def __call__(self, x_norm):
        if self.exhausted:
            return PENALTY
        x_norm = np.asarray(x_norm, dtype=float)
        try:
            real_params = self.spec.denormalize(x_norm)
            chain_dict = self.spec.vector_to_chain(real_params)
            loss = float(self.objective_fn(chain_dict))
            if not np.isfinite(loss):
                loss = PENALTY
        except Exception as e:
            print(f"[WARN] 렌더링/평가 실패 -> penalty: {e}")
            loss = PENALTY

        self.n += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_norm = np.clip(x_norm.copy(), 0.0, 1.0)
        self.history.append(self.best_loss)
        return loss

    def best_result(self):
        best_real = self.spec.denormalize(self.best_norm)
        best_chain = self.spec.vector_to_chain(best_real)
        h = list(self.history)
        if not h:
            h = [PENALTY] * self.budget
        elif len(h) < self.budget:
            h += [h[-1]] * (self.budget - len(h))
        return best_chain, best_real, self.best_loss, np.array(h[:self.budget])


def run_cma_es_for_chain(objective_fn, spec: ChainParameterSpec, budget=200, seed=0,
                         n_runs=4, sigma0=0.25, popsize=None, initial_values=None):
    rng = np.random.default_rng(seed)
    tr = Tracker(objective_fn, spec, budget)

    if spec.dim == 0:
        tr(np.array([]))
        return tr.best_result()

    if spec.dim == 1:
        for g in np.linspace(0.0, 1.0, budget):
            if tr.exhausted:
                break
            tr(np.array([g]))
        return tr.best_result()

    pop = popsize or max(6, 4 + int(3 * np.log(spec.dim)))
    base_evals = budget // n_runs
    rem = budget % n_runs
    run_budgets = [base_evals + (1 if i < rem else 0) for i in range(n_runs)]

    for run_idx, sub_budget in enumerate(run_budgets):
        if tr.exhausted or sub_budget <= 0:
            break

        if run_idx == 0 and initial_values is not None:
            x0 = spec.normalize(initial_values)
        elif run_idx == 0:
            x0 = np.full(spec.dim, 0.5)
        else:
            x0 = rng.uniform(0.1, 0.9, spec.dim)

        opts = {
            "bounds": [0.0, 1.0],
            "popsize": pop,
            "maxfevals": sub_budget,
            "seed": int(seed + 1000 * run_idx + 1),
            "verbose": -9,
            "tolfun": 1e-12,
            "tolx": 1e-12,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        evals_in_run = 0

        while not es.stop() and not tr.exhausted and evals_in_run < sub_budget:
            xs = es.ask()
            remaining = sub_budget - evals_in_run
            if len(xs) > remaining:
                xs = xs[:remaining]
            losses = [tr(x) for x in xs]
            evals_in_run += len(xs)
            if len(xs) == pop:
                es.tell(xs, losses)
            else:
                break

    return tr.best_result()


# ----------------------------------------------------------------------
# [4] 상위 K개 후보 입력(Task 4) 및사도순 정렬 탐색 메인 함수
# ----------------------------------------------------------------------
def search_and_rank_chains(
    dry_audio,
    sr,
    target_audio,
    top_k_candidates,
    owned_effect_types=None,
    effects_def=None,
    budget=200,
    n_restarts=4,
    n_seeds=1,
    metric_fn=spectral_distance,
):
    """
    Task 4 연동 인터페이스:
    - top_k_candidates는 도영님(5번 모델)이 주는 상위 K개 후보 리스트입니다.
      형식 예시:
      [
        {
          "effect_types": ["overdrive"],
          "initial_chain": {"amp_id": "fixed_amp_v1", "chain": [...]}, # 선택적 초기값 (Warm-start)
          "score": 0.85 # 모델 예측 점수 등
        },
        ... 또는 단순 리스트 ["overdrive"] 형태도 자동 지원
      ]
    """
    effects_def = effects_def or load_effects()

    # 1. 인벤토리 필터링
    if owned_effect_types is not None:
        valid_candidates, dropped = filter_candidates_by_inventory(top_k_candidates, owned_effect_types)
    else:
        valid_candidates = list(top_k_candidates)
        dropped = []

    def objective_fn(chain_dict):
        wet = render_chain(dry_audio, sr, chain_dict, effects=effects_def)
        return metric_fn(wet, target_audio)

    ranked_results = []
    
    # Task 3 연동: VST 탐색 전체 소요 시간 측정 시작
    total_start_time = time.time()

    for cand in valid_candidates:
        etypes = extract_effect_types(cand)
        etypes_tuple = tuple(etypes)
        spec = ChainParameterSpec(etypes, effects_def=effects_def)

        # 모델이 제공한 초기값(initial_chain) 추출 시도
        init_vec = None
        if isinstance(cand, dict) and "initial_chain" in cand:
            try:
                init_vec = spec.chain_to_vector(cand["initial_chain"])
                print(f"[INFO] 후보 {etypes}에 대해 지도학습 모델의 초기 추정값(Warm-start)을 적용합니다.")
            except Exception as e:
                print(f"[WARN] 초기값 변환 실패 ({etypes}): {e}")

        best_overall = None
        seed_losses = []

        for seed in range(n_seeds):
            best_chain, best_vec, best_loss, hist = run_cma_es_for_chain(
                objective_fn,
                spec,
                budget=budget,
                seed=seed,
                n_runs=n_restarts,
                initial_values=init_vec if seed == 0 else None,
            )
            seed_losses.append(best_loss)
            if best_overall is None or best_loss < best_overall["loss"]:
                best_overall = {
                    "effect_types": etypes,
                    "chain": best_chain,
                    "best_vector": best_vec,
                    "loss": float(best_loss),
                    "spec": spec,
                    "history": hist,
                }

        best_overall["loss_mean"] = float(np.mean(seed_losses))
        best_overall["loss_std"] = float(np.std(seed_losses))
        ranked_results.append(best_overall)

    total_elapsed = time.time() - total_start_time
    print(f"\n[INFO] VST 탐색 총 소요 시간: {total_elapsed:.2f}초 (후보 {len(valid_candidates)}개 검사 완료)")

    # Loss 오름차순(유사도 높은 순) 정렬
    ranked_results.sort(key=lambda r: r["loss"])
    for rank_idx, item in enumerate(ranked_results, start=1):
        item["rank"] = rank_idx

    return ranked_results, dropped


# ----------------------------------------------------------------------
# [5] 데이터 로딩 및 리포트 유틸리티
# ----------------------------------------------------------------------
def load_synthetic_dry_sample(duration_sec=1.0):
    """Task 1: 생성 폴더 내 dry/ 오디오 로드"""
    wav_files = sorted(glob.glob(str(DRY_AUDIO_DIR / "*.wav")))
    if not wav_files:
        # 혹시 원본 경로에 있다면 안내 메시지 출력
        fallback_dir = PROJECT_ROOT / "data" / "dry" / "Fender_Strat_Clean" / "audio"
        wav_files = sorted(glob.glob(str(fallback_dir / "*.wav")))
        if not wav_files:
            raise FileNotFoundError(
                f"\n[ERROR] '{DRY_AUDIO_DIR}' 또는 대체 경로에 .wav 파일이 없습니다!\n"
                "아림님 데이터셋 생성 스크립트를 통해 생성된 dry/ 폴더 위치를 확인해 주세요."
            )
        path = wav_files[0]
        print(f"[WARN] 지정된 경로에 없어 대체 경로의 파일을 사용합니다: {path}")
    else:
        path = wav_files[0]

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    win = int(sr * duration_sec)
    if len(audio) > win:
        hop = max(1, sr // 10)
        starts = np.arange(0, len(audio) - win + 1, hop)
        energies = [np.sum(audio[s:s + win] ** 2) for s in starts]
        s = int(starts[int(np.argmax(energies))])
        audio = audio[s:s + win]

    print(f"[INFO] 생성 폴더 Dry 오디오 로드 완료: {path} (sr={sr}, 길이={len(audio) / sr:.2f}s)")
    return np.ascontiguousarray(audio, dtype=np.float32), sr, os.path.basename(path)


def format_chain_label(effect_types):
    return " -> ".join(effect_types) if effect_types else "(bypass)"


def print_ranking_report(ranked_results, dropped, owned_effect_types):
    width = 96
    print("\n" + "=" * width)
    print("[Step 1 탐색 엔진] 모델 상위 K개 후보 기반 VST 탐색 및 유사도순 정렬 리포트")
    print("-" * width)
    print(f"보유 이펙터 (user_inventory): {owned_effect_types}")
    if dropped:
        print("인벤토리 부족으로 제외된 후보 체인:")
        for etypes, missing in dropped:
            print(f"  - [제외] {format_chain_label(etypes)} (미보유: {missing})")
    else:
        print("제외된 후보 체인 없음")

    print("-" * width)
    print(f"{'순위':<6} | {'후보 체인 (Effect Chain)':<38} | {'차원':>4} | {'최종 Loss (유사도)':>18} | {'시드 평균 ± 표준편차':>20}")
    print("-" * width)
    for r in ranked_results:
        label = format_chain_label(r["effect_types"])
        print(f"#{r['rank']:<5} | {label:<38} | {r['spec'].dim:>4} | {r['loss']:>18.6f} | "
              f"{r['loss_mean']:>9.4f} ± {r['loss_std']:<7.4f}")
    print("=" * width)


# ----------------------------------------------------------------------
# [6] 메인 파이프라인 실행
# ----------------------------------------------------------------------
def run_search_pipeline(budget=200, n_restarts=4, n_seeds=1, inventory_override=None):
    effects_def = load_effects()
    dry_audio, sr, _ = load_synthetic_dry_sample()

    owned_effects = inventory_override if inventory_override is not None else load_user_inventory()

    # 테스트용 상위 K개 후보 리스트 (도영님 모델 연동 규격 호환)
    top_k_candidates = [
        {"effect_types": ["overdrive"]},
        {"effect_types": ["distortion"]},
        {"effect_types": ["compressor", "overdrive"]},
        {"effect_types": ["overdrive", "eq_3band"]},
        {"effect_types": ["compressor", "overdrive", "eq_3band"]},
    ]

    # 가상의 타겟 오디오 생성 (테스트용)
    ground_truth_chain = {
        "amp_id": "fixed_amp_v1",
        "chain": [{"effect_type": "overdrive", "values": {"drive": 6.0, "level": 0.0}}],
    }
    target_audio = render_chain(dry_audio, sr, ground_truth_chain, effects=effects_def)

    ranked_results, dropped = search_and_rank_chains(
        dry_audio=dry_audio,
        sr=sr,
        target_audio=target_audio,
        top_k_candidates=top_k_candidates,
        owned_effect_types=owned_effects,
        effects_def=effects_def,
        budget=budget,
        n_restarts=n_restarts,
        n_seeds=n_seeds,
    )

    print_ranking_report(ranked_results, dropped, owned_effects)

    if ranked_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = OUTPUT_DIR / "best_chain_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ranked_results[0]["chain"], f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 1위 최적 체인 JSON 저장 완료: {out_path}")

    return ranked_results


def parse_args():
    p = argparse.ArgumentParser(description="Step 1: Top-K Model Candidates VST Search & Ranking")
    p.add_argument("--budget", type=int, default=200, help="후보 체인당 평가 횟수 예산")
    p.add_argument("--restarts", type=int, default=4, help="CMA-ES 재시작 횟수")
    p.add_argument("--seeds", type=int, default=1, help="시드 수")
    p.add_argument("--inventory", type=str, nargs="*", default=None, help="보유 이펙터 오버라이드")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search_pipeline(
        budget=args.budget,
        n_restarts=args.restarts,
        n_seeds=args.seeds,
        inventory_override=args.inventory,
    )