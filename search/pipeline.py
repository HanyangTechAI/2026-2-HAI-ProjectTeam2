"""
search/pipeline.py

Step 1: 스키마 기반 다중 후보 체인 탐색 엔진
- 사용자 보유 이펙터(user_inventory) 기반 후보 체인 필터링
- 후보 체인별 동적 파라미터 공간 구성 (schemas/effects.data.json 연동)
- 후보 체인별 Restart CMA-ES 실행 및 유사도(Loss 오름차순) 정렬
- 지도학습 모델 초기 추정값(initial_guesses) Warm-start 연동 지원

Usage (프로젝트 루트에서)
-------------------------
python -m search.pipeline
python -m search.pipeline --budget 200 --restarts 4 --seeds 1
python -m search.pipeline --inventory compressor overdrive eq_3band
"""
import os
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import cma

from effects import load_effects, render_chain
from metrics.spectral import spectral_distance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRY_AUDIO_DIR = PROJECT_ROOT / "data" / "dry" / "Fender_Strat_Clean" / "audio"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "search"
PENALTY = 1e3


# ----------------------------------------------------------------------
# [1] 스키마 기반 체인 파라미터 스펙(ChainParameterSpec) 동적 빌더
# ----------------------------------------------------------------------
class ChainParameterSpec:
    """
    주어진 체인 구조(예: ["compressor", "overdrive"])에 대해
    schemas/effects.data.json을 읽어 연속형 노브들의 탐색 범위와 스케일(linear/log)을 동적으로 구성합니다.
    """
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
                    # categorical(예: fuzz의 mode) 또는 boolean은 스키마 default 값으로 고정
                    self.fixed_values[idx][pname] = p["default"]

        self.dim = len(self.param_meta)
        self.lows = np.array([m[3] for m in self.param_meta], dtype=float)
        self.highs = np.array([m[4] for m in self.param_meta], dtype=float)
        self.is_log = np.array([m[5] for m in self.param_meta], dtype=bool)
        self.names = [f"{m[1]}.{m[2]}" for m in self.param_meta]

    def denormalize(self, x_norm):
        """[0, 1]^D 정규화 벡터 -> 실제 노브 값 배열 (0 나눗셈 방어 포함)"""
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
        """실제 노브 값 배열 -> [0, 1]^D 정규화 벡터 (0 나눗셈 방어 포함)"""
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
        """실제 노브 값 1차원 배열을 chain.schema.json 규격의 딕셔너리로 변환"""
        chain_list = [
            {"effect_type": etype, "values": dict(self.fixed_values[idx])}
            for idx, etype in enumerate(self.effect_types)
        ]
        for val, (idx, _, pname, _, _, _) in zip(real_params, self.param_meta):
            chain_list[idx]["values"][pname] = round(float(val), 4)
        return {"amp_id": self.amp_id, "chain": chain_list}

    def chain_to_vector(self, chain_data):
        """chain.schema.json 규격 데이터에서 연속형 노브 값만 추출하여 1차원 배열로 반환"""
        items = chain_data["chain"] if isinstance(chain_data, dict) else chain_data
        vals = []
        for idx, _, pname, _, _, _ in self.param_meta:
            vals.append(float(items[idx]["values"][pname]))
        return np.array(vals, dtype=float)


# ----------------------------------------------------------------------
# [2] 사용자 보유 이펙터(user_inventory) 필터링
# ----------------------------------------------------------------------
def load_user_inventory(inventory_path=None):
    """schemas/user_inventory.data.json에서 보유 이펙터 목록을 읽습니다."""
    path = Path(inventory_path) if inventory_path else (SCHEMAS_DIR / "user_inventory.data.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("owned_effect_types", [])


def extract_effect_types(candidate):
    """체인 후보에서 effect_type 문자열 리스트를 추출합니다."""
    if isinstance(candidate, dict) and "chain" in candidate:
        return [item["effect_type"] for item in candidate["chain"]]
    if isinstance(candidate, list):
        if not candidate:
            return []
        if isinstance(candidate[0], dict):
            return [item["effect_type"] for item in candidate]
        return [str(x) for x in candidate]
    raise ValueError(f"지원하지 않는 후보 체인 형식: {candidate}")


def filter_candidates_by_inventory(candidate_chains, owned_effect_types):
    """사용자가 보유한 이펙터(owned_effect_types)만으로 구성된 후보 체인만 남깁니다."""
    owned_set = set(owned_effect_types)
    kept, dropped = [], []
    for cand in candidate_chains:
        etypes = extract_effect_types(cand)
        missing = [e for e in etypes if e not in owned_set]
        if not missing:
            kept.append(etypes)
        else:
            dropped.append((etypes, missing))
    return kept, dropped


# ----------------------------------------------------------------------
# [3] 평가 추적기 및 탐색 알고리즘 (Restart CMA-ES & Random Search)
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


def run_random_search_for_chain(objective_fn, spec: ChainParameterSpec, budget=200, seed=0):
    rng = np.random.default_rng(seed)
    tr = Tracker(objective_fn, spec, budget)
    if spec.dim == 0:
        tr(np.array([]))
        return tr.best_result()
    while not tr.exhausted:
        tr(rng.uniform(0.0, 1.0, spec.dim))
    return tr.best_result()


def run_cma_es_for_chain(objective_fn, spec: ChainParameterSpec, budget=200, seed=0,
                         n_runs=4, sigma0=0.25, popsize=None, initial_values=None):
    """단일 후보 체인에 대해 Restart CMA-ES(예: 50회 x 4번)를 실행합니다."""
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
# [4] 다중 후보 체인 탐색 및 유사도순 정렬 (Warm-start 지원 포함)
# ----------------------------------------------------------------------
def search_and_rank_chains(
    dry_audio,
    sr,
    target_audio,
    candidate_chains,
    owned_effect_types=None,
    effects_def=None,
    budget=200,
    n_restarts=4,
    n_seeds=1,
    metric_fn=spectral_distance,
    initial_guesses=None,
):
    """
    1) owned_effect_types(user_inventory)로 후보 체인을 필터링합니다.
    2) 각 후보 체인별로 CMA-ES를 실행해 최적 노브 값과 최소 Loss를 구합니다.
       (지도학습 모델이 예측한 initial_guesses가 있으면 첫 시드 시작점으로 활용)
    3) Loss 오름차순(유사도 높은 순)으로 정렬하여 결과를 반환합니다.
    """
    effects_def = effects_def or load_effects()
    initial_guesses = initial_guesses or {}

    if owned_effect_types is not None:
        valid_candidates, dropped = filter_candidates_by_inventory(candidate_chains, owned_effect_types)
    else:
        valid_candidates = [extract_effect_types(c) for c in candidate_chains]
        dropped = []

    def objective_fn(chain_dict):
        wet = render_chain(dry_audio, sr, chain_dict, effects=effects_def)
        return metric_fn(wet, target_audio)

    ranked_results = []
    for etypes in valid_candidates:
        etypes_tuple = tuple(etypes)
        spec = ChainParameterSpec(etypes, effects_def=effects_def)

        init_vec = None
        if etypes_tuple in initial_guesses:
            try:
                init_vec = spec.chain_to_vector(initial_guesses[etypes_tuple])
                print(f"[INFO] 후보 {etypes}에 대해 지도학습 초기 추정값(Warm-start)을 적용합니다.")
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

    ranked_results.sort(key=lambda r: r["loss"])
    for rank_idx, item in enumerate(ranked_results, start=1):
        item["rank"] = rank_idx

    return ranked_results, dropped


# ----------------------------------------------------------------------
# [5] 데이터 로드 및 결과 출력 유틸리티
# ----------------------------------------------------------------------
def load_real_dry_sample(duration_sec=1.0):
    wav_files = sorted(glob.glob(str(DRY_AUDIO_DIR / "*.wav")))
    if not wav_files:
        raise FileNotFoundError(
            f"\n[ERROR] '{DRY_AUDIO_DIR}' 폴더에 실제 .wav 파일이 없습니다!\n"
            "구글 드라이브 공유 폴더(Fender_Strat_Clean/audio)에서 .wav 샘플을 다운받아 위 경로에 넣어주세요."
        )

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

    print(f"[INFO] 실제 Dry 오디오 로드 완료: {path} (sr={sr}, 길이={len(audio) / sr:.2f}s)")
    return np.ascontiguousarray(audio, dtype=np.float32), sr, os.path.basename(path)


def format_chain_label(effect_types):
    return " -> ".join(effect_types) if effect_types else "(bypass)"


def print_ranking_report(ranked_results, dropped, owned_effect_types, ground_truth_chain=None):
    width = 96
    print("\n" + "=" * width)
    print("[Step 1 탐색 엔진] 사용자 인벤토리 필터링 및 후보 체인 유사도순 정렬 결과")
    print("-" * width)
    print(f"보유 이펙터 (user_inventory): {owned_effect_types}")
    if dropped:
        print("인벤토리 부족으로 제외된 후보 체인:")
        for etypes, missing in dropped:
            print(f"  - [제외] {format_chain_label(etypes)} (미보유: {missing})")
    else:
        print("제외된 후보 체인 없음 (모든 후보가 보유 이펙터 내에 있음)")

    print("-" * width)
    print(f"{'순위':<6} | {'후보 체인 (Effect Chain)':<38} | {'차원':>4} | {'최종 Loss (유사도)':>18} | {'시드 평균 ± 표준편차':>20}")
    print("-" * width)
    for r in ranked_results:
        label = format_chain_label(r["effect_types"])
        print(f"#{r['rank']:<5} | {label:<38} | {r['spec'].dim:>4} | {r['loss']:>18.6f} | "
              f"{r['loss_mean']:>9.4f} ± {r['loss_std']:<7.4f}")
    print("=" * width)

    if not ranked_results:
        return

    best = ranked_results[0]
    print(f"\n[1위 체인 상세 노브 값] 체인: {format_chain_label(best['effect_types'])} (Loss: {best['loss']:.6f})")
    for item in best["chain"]["chain"]:
        print(f"  * {item['effect_type']:<12}: {item['values']}")

    if ground_truth_chain is not None:
        gt_types = extract_effect_types(ground_truth_chain)
        if gt_types == best["effect_types"]:
            spec = best["spec"]
            gt_vec = spec.chain_to_vector(ground_truth_chain)
            found_vec = best["best_vector"]
            norm_err = np.abs(spec.normalize(gt_vec) - spec.normalize(found_vec)) * 100.0
            print("\n" + "-" * width)
            print("[정답 체인 vs 1위 탐색 체인 노브 비교]")
            print(f"{'파라미터 (Knob)':<26} | {'정답(Target)':>12} | {'찾은 값(Found)':>14} | {'절대 오차':>10} | {'정규화 오차(%)':>14}")
            print("-" * width)
            for name, t_val, f_val, ne in zip(spec.names, gt_vec, found_vec, norm_err):
                print(f"{name:<26} | {t_val:>12.2f} | {f_val:>14.2f} | {abs(t_val - f_val):>10.2f} | {ne:>14.1f}")
            print("-" * width)
            print(f"{'평균 정규화 오차(%)':<26}   {'':>12}   {'':>14}   {'':>10}   {norm_err.mean():>14.1f}")
            print("=" * width)


# ----------------------------------------------------------------------
# [6] 메인 파이프라인 실행
# ----------------------------------------------------------------------
DEFAULT_CANDIDATE_CHAINS = [
    ["overdrive"],
    ["distortion"],
    ["fuzz"],
    ["compressor", "overdrive"],
    ["overdrive", "eq_3band"],
    ["compressor", "overdrive", "eq_3band"],
    ["compressor", "distortion", "eq_3band"],
]


def run_search_pipeline(budget=200, n_restarts=4, n_seeds=1, inventory_override=None):
    effects_def = load_effects()
    dry_audio, sr, _ = load_real_dry_sample()

    owned_effects = inventory_override if inventory_override is not None else load_user_inventory()

    ground_truth_chain = {
        "amp_id": "fixed_amp_v1",
        "chain": [
            {"effect_type": "overdrive", "values": {"drive": 10.0, "level": 0.0}},
        ],
    }
    target_audio = render_chain(dry_audio, sr, ground_truth_chain, effects=effects_def)

    sanity_loss = spectral_distance(
        render_chain(dry_audio, sr, ground_truth_chain, effects=effects_def),
        target_audio,
    )
    print(f"[CHECK] 정답 체인에서의 loss = {sanity_loss:.6e} (0에 가까워야 정상)")

    ranked_results, dropped = search_and_rank_chains(
        dry_audio=dry_audio,
        sr=sr,
        target_audio=target_audio,
        candidate_chains=DEFAULT_CANDIDATE_CHAINS,
        owned_effect_types=owned_effects,
        effects_def=effects_def,
        budget=budget,
        n_restarts=n_restarts,
        n_seeds=n_seeds,
    )

    print_ranking_report(
        ranked_results=ranked_results,
        dropped=dropped,
        owned_effect_types=owned_effects,
        ground_truth_chain=ground_truth_chain,
    )

    if ranked_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = OUTPUT_DIR / "best_chain_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ranked_results[0]["chain"], f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 1위 최적 체인 JSON 저장 완료: {out_path}")

    return ranked_results


def parse_args():
    p = argparse.ArgumentParser(description="Step 1: Schema-driven multi-chain CMA-ES search & ranking")
    p.add_argument("--budget", type=int, default=200, help="후보 체인당 평가 횟수 예산 (기본값: 200)")
    p.add_argument("--restarts", type=int, default=4, help="CMA-ES 분할 재시작 횟수 (기본값: 4)")
    p.add_argument("--seeds", type=int, default=1, help="후보 체인당 반복 시드 수 (기본값: 1)")
    p.add_argument("--inventory", type=str, nargs="*", default=None,
                   help="보유 이펙터 목록 직접 지정 (미지정 시 schemas/user_inventory.data.json 사용)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search_pipeline(
        budget=args.budget,
        n_restarts=args.restarts,
        n_seeds=args.seeds,
        inventory_override=args.inventory,
    )