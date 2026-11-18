"""
search/pipeline.py

Step 1: 스키마 기반 다중 후보 체인 탐색 엔진
- 사용자 보유 이펙터(user_inventory) 기반 후보 체인 필터링
- 후보 체인별 동적 파라미터 공간 구성 (schemas/effects.data.json 연동)
- 후보 체인별 Restart CMA-ES 실행 및 유사도(Loss 오름차순) 정렬
- 5번 지도학습 모델 초기 추정값(to_search_inputs / initial_guesses) Warm-start 완벽 호환
- 학습·탐색 입력을 생성 폴더의 dry/로 사용 및 VST 탐색 소요 시간 측정
"""
import os
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
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic" / "step1_vst_v0_임시"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "search"
PENALTY = 1e3


# ----------------------------------------------------------------------
# [1] 스키마 기반 체인 파라미터 스펙(ChainParameterSpec) 동적 빌더
# ----------------------------------------------------------------------
class ChainParameterSpec:
    """
    주어진 체인 구조(예: ["compressor", "808_scream"])에 대해
    schemas/effects.data.json을 읽어 연속형 노브들의 탐색 범위와 스케일(linear/log)을 동적으로 구성합니다.
    """
    def __init__(self, effect_types, effects_def=None, amp_id="fixed_amp_v1", round_digits=4):
        self.effects_def = effects_def or load_effects()
        self.effect_types = [e for e in effect_types if e != "bypass"]
        self.amp_id = amp_id
        self.round_digits = round_digits

        self.param_meta = []  # (effect_idx, effect_type, param_name, low, high, is_log)
        self.fixed_values = [{} for _ in self.effect_types]

        for idx, etype in enumerate(self.effect_types):
            if etype not in self.effects_def:
                raise ValueError(f"스키마에 정의되지 않은 effect_type: {etype}")
            edef = self.effects_def[etype]
            param_list = edef.get("active_params", edef["params"])
            active_names = {p["name"] for p in param_list}
            for p in edef["params"]:
                pname = p["name"]
                ptype = p["type"]
                if pname in active_names and ptype == "continuous":
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
        x = np.asarray(x_norm, dtype=float)
        if x.shape != (self.dim,) or not np.isfinite(x).all():
            raise ValueError(f"Expected finite normalized vector of shape ({self.dim},), got {x.shape}")
        x = np.clip(x, 0.0, 1.0)
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
            p = np.asarray(real_params, dtype=float)
            if p.size != 0:
                raise ValueError("Expected empty parameter vector for 0-dim spec")
            return np.array([], dtype=float)
        p = np.asarray(real_params, dtype=float)
        if p.shape != (self.dim,) or not np.isfinite(p).all():
            raise ValueError(f"Expected finite parameter vector of shape ({self.dim},), got {p.shape}")
        if np.any(p < self.lows - 1e-9) or np.any(p > self.highs + 1e-9):
            raise ValueError("Parameter values out of schema range")
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
            fval = float(val)
            chain_list[idx]["values"][pname] = (
                round(fval, self.round_digits) if self.round_digits is not None else fval
            )
        return {"amp_id": self.amp_id, "chain": chain_list}

    def chain_to_vector(self, chain_data):
        items = chain_data["chain"] if isinstance(chain_data, dict) else chain_data
        vals = []
        for idx, _, pname, _, _, _ in self.param_meta:
            vals.append(float(items[idx]["values"][pname]))
        return np.array(vals, dtype=float)


# ----------------------------------------------------------------------
# [2] 사용자 보유 이펙터(user_inventory) 기반 후보 체인 필터링
# ----------------------------------------------------------------------
def load_user_inventory(inventory_path=None):
    path = Path(inventory_path) if inventory_path else (SCHEMAS_DIR / "user_inventory.data.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("owned_effect_types", [])


def extract_effect_types(candidate):
    if isinstance(candidate, dict):
        if "chain" in candidate and isinstance(candidate["chain"], list):
            return [item["effect_type"] for item in candidate["chain"] if item["effect_type"] != "bypass"]
        if "chain" in candidate and isinstance(candidate["chain"], dict):
            return [item["effect_type"] for item in candidate["chain"].get("chain", []) if item["effect_type"] != "bypass"]
        if "effect_types" in candidate:
            return [e for e in candidate["effect_types"] if e != "bypass"]
        if "effect_type" in candidate:
            et = candidate["effect_type"]
            return [] if et == "bypass" else [et]
    if isinstance(candidate, (list, tuple)):
        if not candidate:
            return []
        if isinstance(candidate[0], dict):
            return [item["effect_type"] for item in candidate if item["effect_type"] != "bypass"]
        return [str(x) for x in candidate if str(x) != "bypass"]
    raise ValueError(f"지원하지 않는 후보 체인 형식: {candidate}")


def filter_candidates_by_inventory(candidate_chains, owned_effect_types):
    owned_set = set(owned_effect_types)
    kept, dropped = [], []
    for cand in candidate_chains:
        etypes = extract_effect_types(cand)
        missing = [e for e in etypes if e not in owned_set]
        if not missing:
            kept.append(cand)
        else:
            dropped.append((etypes, missing))
    return kept, dropped


# ----------------------------------------------------------------------
# [3] 단일 체인 최적화기 (Tracker + Restart CMA-ES)
# ----------------------------------------------------------------------
class Tracker:
    def __init__(self, objective_fn, spec: ChainParameterSpec, budget: int, strict: bool = False):
        self.objective_fn = objective_fn
        self.spec = spec
        self.budget = budget
        self.strict = strict
        self.n = 0
        self.best_loss = float("inf")
        self.best_norm = np.full(spec.dim, 0.5, dtype=float)
        self.best_real = self.spec.denormalize(self.best_norm)
        self.best_chain = self.spec.vector_to_chain(self.best_real)
        self.history = []

    @property
    def exhausted(self):
        return self.n >= self.budget

    def __call__(self, x_norm):
        if self.exhausted:
            return PENALTY
        x_norm = np.clip(np.asarray(x_norm, dtype=float), 0.0, 1.0)
        try:
            real_params = self.spec.denormalize(x_norm)
            chain_dict = self.spec.vector_to_chain(real_params)
            loss = float(self.objective_fn(chain_dict))
            if not np.isfinite(loss):
                if self.strict:
                    raise ValueError(f"Objective returned non-finite loss: {loss}")
                loss = PENALTY
        except Exception as e:
            if self.strict:
                raise
            print(f"[WARN] 렌더링/평가 실패 -> penalty: {e}")
            loss = PENALTY

        self.n += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_norm = x_norm.copy()
            self.best_real = real_params
            self.best_chain = chain_dict
        self.history.append(self.best_loss)
        return loss

    def best_result(self):
        h = list(self.history)
        if not h:
            h = [PENALTY] * self.budget
        elif len(h) < self.budget:
            h += [h[-1]] * (self.budget - len(h))
        return self.best_chain, self.best_real, self.best_loss, np.array(h[:self.budget])


def run_cma_es_for_chain(
    objective_fn,
    spec: ChainParameterSpec,
    budget=40,
    seed=0,
    n_runs=2,
    sigma0=0.25,
    popsize=None,
    initial_values=None,
    evaluate_initial=True,
    strict=False,
):
    if not isinstance(budget, int) or budget < 1 or not isinstance(n_runs, int) or n_runs < 1:
        raise ValueError("budget and n_runs must be positive integers")

    init_norm = None
    if initial_values is not None:
        init_norm = spec.normalize(initial_values)

    rng = np.random.default_rng(seed)
    tr = Tracker(objective_fn, spec, budget, strict=strict)

    if spec.dim == 0:
        tr(np.array([]))
        return tr.best_result()

    if spec.dim == 1:
        if evaluate_initial and init_norm is not None:
            tr(init_norm)
        rem_grid = budget - tr.n
        if rem_grid > 0:
            for g in np.linspace(0.0, 1.0, rem_grid):
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

        if run_idx == 0 and init_norm is not None:
            x0 = init_norm
        elif run_idx == 0:
            x0 = np.full(spec.dim, 0.5)
        else:
            x0 = rng.uniform(0.1, 0.9, spec.dim)

        evals_in_run = 0
        if run_idx == 0 and evaluate_initial:
            tr(x0)
            evals_in_run += 1
            if tr.exhausted or evals_in_run >= sub_budget:
                continue

        x0_cma = np.clip(x0, 1e-6, 1.0 - 1e-6).tolist()
        opts = {
            "bounds": [0.0, 1.0],
            "popsize": pop,
            "maxfevals": max(1, sub_budget - evals_in_run),
            "seed": int(seed + 1000 * run_idx + 1),
            "verbose": -9,
            "verb_log": 0,
            "tolfun": 1e-12,
            "tolx": 1e-12,
        }
        es = cma.CMAEvolutionStrategy(x0_cma, sigma0, opts)

        while not es.stop() and not tr.exhausted and evals_in_run < sub_budget:
            xs = es.ask()
            remaining = sub_budget - evals_in_run
            batch = xs[:remaining]
            losses = [tr(x) for x in batch]
            evals_in_run += len(batch)
            if len(batch) == len(xs):
                es.tell(xs, losses)
            else:
                break

    return tr.best_result()


# ----------------------------------------------------------------------
# [4] 다중 후보 체인 탐색 및 유사도순 정렬 (estimation/refine.py 완벽 호환)
# ----------------------------------------------------------------------
def search_and_rank_chains(
    dry_audio,
    sr,
    target_audio,
    candidate_chains=None,
    owned_effect_types=None,
    effects_def=None,
    budget=40,
    n_restarts=2,
    n_seeds=1,
    metric_fn=spectral_distance,
    initial_guesses=None,
    render_fn=None,
    evaluate_initial=True,
    parameter_round_digits=4,
    strict=False,
    base_seed=0,
    verbose=True,
):
    effects_def = effects_def or load_effects()
    renderer = render_fn if render_fn is not None else (
        lambda audio, rate, ch: render_chain(audio, rate, ch, effects=effects_def)
    )

    if (
        candidate_chains
        and isinstance(candidate_chains[0], dict)
        and "effect_type" in candidate_chains[0]
        and "x0" in candidate_chains[0]
        and initial_guesses is None
    ):
        try:
            from estimation.refine import to_search_inputs
            candidate_chains, initial_guesses = to_search_inputs(candidate_chains, effects_def)
        except Exception:
            initial_guesses = {}
            converted = []
            for item in candidate_chains:
                ch = item.get("chain", {"amp_id": "fixed_amp_v1", "chain": []})
                etypes_key = tuple(extract_effect_types(ch))
                converted.append(ch)
                initial_guesses[etypes_key] = ch
            candidate_chains = converted

    # 1. 인벤토리 필터링
    if owned_effect_types is not None:
        valid_candidates, dropped = filter_candidates_by_inventory(candidate_chains, owned_effect_types)
    else:
        valid_candidates = list(candidate_chains)
        dropped = []

    def objective_fn(chain_dict):
        wet = renderer(dry_audio, sr, chain_dict)
        return metric_fn(wet, target_audio)

    ranked_results = []
    start_total = time.perf_counter()

    for cand in valid_candidates:
        etypes = extract_effect_types(cand)
        etypes_tuple = tuple(etypes)
        amp_id = cand.get("amp_id", "fixed_amp_v1") if isinstance(cand, dict) else "fixed_amp_v1"
        spec = ChainParameterSpec(
            etypes,
            effects_def=effects_def,
            amp_id=amp_id,
            round_digits=parameter_round_digits,
        )

        init_vec = None
        if initial_guesses and etypes_tuple in initial_guesses and spec.dim > 0:
            guess = initial_guesses[etypes_tuple]
            init_vec = (
                spec.chain_to_vector(guess)
                if isinstance(guess, (dict, list))
                else np.asarray(guess, dtype=float)
            )
        elif isinstance(cand, dict) and "chain" in cand and spec.dim > 0:
            try:
                init_vec = spec.chain_to_vector(cand)
            except Exception:
                if strict:
                    raise
                init_vec = None

        label = format_chain_label(etypes)
        if verbose:
            warm_str = " (Warm-start 적용)" if init_vec is not None else ""
            print(f"[Search] 후보 탐색 시작: {label:<30} | 차원: {spec.dim} | 예산: {budget}회{warm_str}")

        start_cand = time.perf_counter()
        best_overall = None
        seed_losses = []

        for s_idx in range(n_seeds):
            seed_val = base_seed + s_idx
            best_chain, best_vec, best_loss, hist = run_cma_es_for_chain(
                objective_fn,
                spec,
                budget=budget,
                seed=seed_val,
                n_runs=n_restarts,
                initial_values=init_vec if s_idx == 0 else None,
                evaluate_initial=evaluate_initial,
                strict=strict,
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

        elapsed_cand = time.perf_counter() - start_cand
        if verbose:
            eval_cnt = max(1, budget * n_seeds if spec.dim > 0 else 1)
            print(f"  -> 완료: Loss={best_overall['loss']:.6f} | 소요 시간: {elapsed_cand:.2f}초 "
                  f"(1회당 평균 {elapsed_cand / eval_cnt:.3f}초)")

        best_overall["loss_mean"] = float(np.mean(seed_losses))
        best_overall["loss_std"] = float(np.std(seed_losses))
        best_overall["elapsed"] = elapsed_cand
        ranked_results.append(best_overall)

    total_elapsed = time.perf_counter() - start_total
    if verbose:
        print(f"\n[INFO] 탐색 총 소요 시간: {total_elapsed:.2f}초 (후보 {len(valid_candidates)}개 완료)")

    ranked_results.sort(key=lambda r: r["loss"])
    for rank_idx, item in enumerate(ranked_results, start=1):
        item["rank"] = rank_idx

    return ranked_results, dropped


# ----------------------------------------------------------------------
# [5] 생성 폴더의 dry/ 로드 및 결과 리포트 출력
# ----------------------------------------------------------------------
def load_generated_dry_sample(dataset_dir=None, duration_sec=1.0):
    ds_path = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
    search_roots = [ds_path]
    synth_root = PROJECT_ROOT / "data" / "synthetic"
    if synth_root.exists() and synth_root not in search_roots:
        search_roots.append(synth_root)

    wav_files = []
    for root in search_roots:
        if not root.exists():
            continue
        for mf in sorted(root.glob("**/manifest.json")):
            try:
                with open(mf, encoding="utf-8") as f:
                    mdata = json.load(f)
                for item in mdata.get("dry_id_map", []):
                    rel = item.get("render_input", "")
                    if not rel:
                        continue
                    for p in (mf.parent / rel, root / rel, PROJECT_ROOT / rel):
                        if p.exists() and p.is_file():
                            wav_files.append(p)
            except Exception:
                pass
        if not wav_files:
            if root.name == "dry":
                wav_files.extend(sorted(root.glob("*.wav")))
            else:
                wav_files.extend(sorted(root.glob("dry/*.wav")))
                wav_files.extend(sorted(root.glob("**/dry/*.wav")))
        if wav_files:
            break

    if not wav_files:
        raise FileNotFoundError(
            f"\n[ERROR] 생성 폴더의 dry/ 음원을 찾을 수 없습니다! (확인 경로: {ds_path / 'dry'})\n"
            "공유 드라이브의 생성 데이터셋(step1_vst_v0_임시 등)을 다운로드하거나 --dataset 경로를 지정해 주세요."
        )

    path = wav_files[0]
    audio, sr = sf.read(str(path), dtype="float32")
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
    return np.ascontiguousarray(audio, dtype=np.float32), sr, path.name


def format_chain_label(effect_types):
    return " -> ".join(effect_types) if effect_types else "bypass"


def print_ranking_report(ranked_results, dropped, owned_effect_types, ground_truth_chain=None):
    width = 104
    print("\n" + "=" * width)
    print("[Step 1 탐색 엔진] 후보 체인별 CMA-ES 탐색 및 유사도순 정렬 리포트")
    print("-" * width)
    print(f"보유 이펙터 (user_inventory): {owned_effect_types}")
    if dropped:
        print("인벤토리 부족으로 제외된 후보 체인:")
        for etypes, missing in dropped:
            print(f"  - [제외] {format_chain_label(etypes)} (미보유: {missing})")
    else:
        print("제외된 후보 체인 없음")

    print("-" * width)
    print(f"{'순위':<6} | {'후보 체인 (Effect Chain)':<34} | {'차원':>4} | {'최종 Loss (유사도)':>18} | {'소요 시간':>10} | {'시드 평균 ± 표준편차':>18}")
    print("-" * width)
    for r in ranked_results:
        label = format_chain_label(r["effect_types"])
        print(f"#{r['rank']:<5} | {label:<34} | {r['spec'].dim:>4} | {r['loss']:>18.6f} | "
              f"{r.get('elapsed', 0.0):>8.2f}s | {r['loss_mean']:>8.4f} ± {r['loss_std']:<6.4f}")
    print("=" * width)

    if not ranked_results:
        return

    best = ranked_results[0]
    print(f"\n[1위 체인 상세 노브 값] 체인: {format_chain_label(best['effect_types'])} (Loss: {best['loss']:.6f})")
    for item in best["chain"]["chain"]:
        print(f"  * {item['effect_type']:<16}: {item['values']}")

    if ground_truth_chain is not None:
        gt_types = extract_effect_types(ground_truth_chain)
        if gt_types == best["effect_types"] and best["spec"].dim > 0:
            spec = best["spec"]
            gt_vec = spec.chain_to_vector(ground_truth_chain)
            found_vec = best["best_vector"]
            norm_err = np.abs(spec.normalize(gt_vec) - spec.normalize(found_vec)) * 100.0
            print("\n" + "-" * width)
            print("[정답 체인 vs 1위 탐색 체인 노브 비교]")
            print(f"{'파라미터 (Knob)':<28} | {'정답(Target)':>12} | {'찾은 값(Found)':>14} | {'절대 오차':>10} | {'정규화 오차(%)':>14}")
            print("-" * width)
            for name, t_val, f_val, ne in zip(spec.names, gt_vec, found_vec, norm_err):
                print(f"{name:<28} | {t_val:>12.2f} | {f_val:>14.2f} | {abs(t_val - f_val):>10.2f} | {ne:>14.1f}")
            print("-" * width)
            print(f"{'평균 정규화 오차(%)':<28}   {'':>12}   {'':>14}   {'':>10}   {norm_err.mean():>14.1f}")
            print("=" * width)


# ----------------------------------------------------------------------
# [6] 메인 파이프라인 실행 (기본 VST 이펙터 808_scream 기준)
# ----------------------------------------------------------------------
DEFAULT_CANDIDATE_CHAINS = [
    ["808_scream"],
    ["b_drive"],
    ["compressor", "808_scream"],
    ["808_scream", "eq_3band"],
    ["compressor", "808_scream", "eq_3band"],
]


def run_search_pipeline(budget=40, n_restarts=2, n_seeds=1, dataset_path=None, inventory_override=None):
    effects_def = load_effects()
    dry_audio, sr, _ = load_generated_dry_sample(dataset_dir=dataset_path)
    owned_effects = inventory_override if inventory_override is not None else load_user_inventory()

    ground_truth_chain = {
        "amp_id": "fixed_amp_v1",
        "chain": [
            {
                "effect_type": "808_scream",
                "values": {"level": 50.0, "overdrive": 70.0, "tone": 40.0},
            }
        ],
    }
    target_audio = render_chain(dry_audio, sr, ground_truth_chain, effects=effects_def)

    sanity_loss = spectral_distance(
        render_chain(dry_audio, sr, ground_truth_chain, effects=effects_def),
        target_audio,
    )
    print(f"[CHECK] 정답 체인(808_scream)에서의 loss = {sanity_loss:.6e} (0에 가까워야 정상)\n")

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

    print_ranking_report(ranked_results, dropped, owned_effects, ground_truth_chain=ground_truth_chain)

    if ranked_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = OUTPUT_DIR / "best_chain_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ranked_results[0]["chain"], f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 1위 최적 체인 JSON 저장 완료: {out_path}")

    return ranked_results


def parse_args():
    p = argparse.ArgumentParser(description="Step 1: Schema-based Multi-Chain CMA-ES Search & Ranking")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_DIR),
                   help="생성 데이터셋 폴더 경로 (내부에 dry/ 포함, 기본값: data/synthetic/step1_vst_v0_임시)")
    p.add_argument("--budget", type=int, default=40, help="후보 체인당 평가 횟수 예산 (기본값: 40)")
    p.add_argument("--restarts", type=int, default=2, help="CMA-ES 분할 재시작 횟수 (기본값: 2)")
    p.add_argument("--seeds", type=int, default=1, help="반복 시드 수")
    p.add_argument("--inventory", type=str, nargs="*", default=None,
                   help="보유 이펙터 목록 직접 지정 (미지정 시 schemas/user_inventory.data.json 사용)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search_pipeline(
        budget=args.budget,
        n_restarts=args.restarts,
        n_seeds=args.seeds,
        dataset_path=args.dataset,
        inventory_override=args.inventory,
    )