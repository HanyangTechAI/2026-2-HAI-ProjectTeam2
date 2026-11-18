"""
search/pipeline.py

Step 1: 스키마 기반 다중 후보 체인 탐색 엔진 (VST 7종 & Estimation 모델 완벽 연동)
- 학습·탐색 입력을 생성 폴더의 dry/ (manifest.json의 render_input 또는 dry/ 폴더)로 우선 사용
- 5번(도영님) 모델의 make_candidates / candidate 객체(x0, vector_spec, chain) 완벽 호환
- VST 이펙터 탐색 실행 및 후보별/전체 연산 소요 시간 측정
- 사용자 보유 이펙터(user_inventory) 필터링 및 Loss 오름차순(유사도순) 정렬
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
from effects.render import find_plugin
from metrics.spectral import spectral_distance

try:
    from estimation.chain import make_candidates as estimation_make_candidates
    HAS_ESTIMATION = True
except ImportError:
    HAS_ESTIMATION = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "search"
PENALTY = 1e3

# Step 0 스크립트 하위 호환용 상수 유지
PARAM_SPECS = [
    ("overdrive.hpf_freq", 150.0, 250.0, False),
    ("overdrive.drive", 25.0, 40.0, False),
    ("overdrive.lpf_freq", 3000.0, 5000.0, False),
    ("overdrive.level", -2.0, 2.0, False),
]
PARAM_NAMES = [s[0] for s in PARAM_SPECS]


# ----------------------------------------------------------------------
# [1] 스키마 기반 체인 파라미터 스펙(ChainParameterSpec) 동적 빌더
# ----------------------------------------------------------------------
class ChainParameterSpec:
    """
    주어진 체인 구조(예: ["compressor", "808_scream"])에 대해
    schemas/effects.data.json을 읽어 연속형 노브들의 탐색 범위와 스케일(linear/log)을 동적으로 구성합니다.
    """
    def __init__(self, effect_types, effects_def=None, amp_id="fixed_amp_v1"):
        self.effects_def = effects_def or load_effects()
        self.effect_types = [e for e in effect_types if e != "bypass"]
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
# [2] 도영님(5번) 모델 연동 어댑터 & 인벤토리 필터링
# ----------------------------------------------------------------------
def is_vst_available(class_name="MultiDrivePro"):
    """현재 PC에 VST3 플러그인 파일이 설치되어 있는지 확인합니다."""
    try:
        find_plugin(class_name)
        return True
    except FileNotFoundError:
        return False


def build_estimation_definitions(effects_def=None, include_vst=True):
    """
    estimation/chain.py의 make_candidates()가 요구하는 'active_params' 필드를 포함하여
    gain_stage 이펙터 정의 딕셔너리를 생성합니다 (VST 7종 및 pedalboard 호환).
    """
    effects_def = effects_def or load_effects()
    result = {}
    for name, definition in effects_def.items():
        if definition.get("category") != "gain_stage":
            continue
        if not include_vst and definition.get("source") == "vst":
            continue
        classes = definition.get("composite_of") or [definition["class_name"]]
        params = [
            p for p in definition["params"]
            if p["maps_to"].partition(".")[0] in classes and "." in p["maps_to"]
        ]
        result[name] = dict(definition, active_params=params)
    return result


def build_candidates_from_cnn_prediction(prediction, owned_effect_types, k=3, effects_def=None, include_vst=True):
    """
    5번 CNN 모델의 예측 결과(probabilities, values)를 받아 상위 K개 후보 리스트를 생성합니다.
    - 확률 총합이 1.0이 되도록 자동 보정하여 estimation.chain.make_candidates의 검증을 통과시킵니다.
    """
    defs = build_estimation_definitions(effects_def, include_vst=include_vst)
    expected_keys = ["bypass"] + list(defs.keys())

    raw_probs = prediction.get("probabilities", {})
    clean_probs = {key: max(0.0, float(raw_probs.get(key, 0.0))) for key in expected_keys}
    total_p = sum(clean_probs.values())
    if total_p <= 0:
        clean_probs = {key: 1.0 / len(expected_keys) for key in expected_keys}
    else:
        clean_probs = {key: val / total_p for key, val in clean_probs.items()}

    raw_values = prediction.get("values", {})
    clean_values = {}
    for etype, edef in defs.items():
        defaults = {p["name"]: p["default"] for p in edef["params"]}
        if etype in raw_values and isinstance(raw_values[etype], dict):
            defaults.update(raw_values[etype])
        clean_values[etype] = defaults

    normalized_pred = {"probabilities": clean_probs, "values": clean_values}

    if HAS_ESTIMATION:
        return estimation_make_candidates(
            normalized_pred,
            owned_effect_types=owned_effect_types,
            k=k,
            definitions=defs,
        )

    # estimation 모듈이 없을 때의 자체 호환 폴백
    allowed = {"bypass"} | (set(owned_effect_types) & set(defs))
    ranked = sorted(allowed, key=lambda t: (-clean_probs[t], t))[:k]
    candidates = []
    for t in ranked:
        if t == "bypass":
            candidates.append({"effect_type": "bypass", "model_score": clean_probs[t], "x0": []})
        else:
            spec = ChainParameterSpec([t], effects_def=effects_def)
            chain_dict = {"amp_id": "fixed_amp_v1", "chain": [{"effect_type": t, "values": clean_values[t]}]}
            vec = spec.chain_to_vector(chain_dict)
            candidates.append({
                "effect_type": t,
                "model_score": clean_probs[t],
                "chain": chain_dict,
                "x0": spec.normalize(vec).tolist(),
            })
    return candidates


def load_user_inventory(inventory_path=None):
    path = Path(inventory_path) if inventory_path else (SCHEMAS_DIR / "user_inventory.data.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("owned_effect_types", [])


def extract_effect_types(candidate):
    """
    모든 형태의 후보 입력에서 이펙터 이름 리스트를 추출합니다:
    - 도영님 모델 candidate: {"effect_type": "808_scream", "x0": [...], ...}
    - 다중 체인 딕셔너리: {"effect_types": ["compressor", "808_scream"], ...}
    - chain.schema.json 형태: {"amp_id": ..., "chain": [{"effect_type": ...}]}
    - 단순 리스트: ["compressor", "808_scream"]
    """
    if isinstance(candidate, dict):
        if "effect_type" in candidate:
            et = candidate["effect_type"]
            return [] if et == "bypass" else [et]
        if "effect_types" in candidate:
            return [e for e in candidate["effect_types"] if e != "bypass"]
        if "chain" in candidate and isinstance(candidate["chain"], list):
            return [item["effect_type"] for item in candidate["chain"] if item["effect_type"] != "bypass"]
        if "chain" in candidate and isinstance(candidate["chain"], dict):
            return [item["effect_type"] for item in candidate["chain"].get("chain", []) if item["effect_type"] != "bypass"]
    if isinstance(candidate, (list, tuple)):
        if not candidate:
            return []
        if isinstance(candidate[0], dict):
            return [item["effect_type"] for item in candidate if item["effect_type"] != "bypass"]
        return [str(x) for x in candidate if str(x) != "bypass"]
    raise ValueError(f"지원하지 않는 후보 체인 형식: {candidate}")


def filter_candidates_by_inventory(candidate_configs, owned_effect_types):
    owned_set = set(owned_effect_types)
    kept, dropped = [], []
    for cand in candidate_configs:
        etypes = extract_effect_types(cand)
        missing = [e for e in etypes if e not in owned_set]
        if not missing:
            kept.append(cand)
        else:
            dropped.append((etypes, missing))
    return kept, dropped


# ----------------------------------------------------------------------
# [3] 평가 추적기 및 탐색 알고리즘 (Restart CMA-ES)
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


def run_cma_es_for_chain(objective_fn, spec: ChainParameterSpec, budget=40, seed=0,
                         n_runs=2, sigma0=0.2, popsize=6, initial_values=None, x0_norm=None):
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

        if run_idx == 0 and x0_norm is not None:
            x0 = np.clip(np.asarray(x0_norm, dtype=float), 1e-6, 1.0 - 1e-6)
        elif run_idx == 0 and initial_values is not None:
            x0 = np.clip(spec.normalize(initial_values), 1e-6, 1.0 - 1e-6)
        elif run_idx == 0:
            x0 = np.full(spec.dim, 0.5)
        else:
            x0 = rng.uniform(0.1, 0.9, spec.dim)

        # 초기점(x0)을 먼저 1회 평가하여 Warm-start 성능 보장
        tr(x0)
        if tr.exhausted or sub_budget <= 1:
            continue

        opts = {
            "bounds": [0.0, 1.0],
            "popsize": pop,
            "maxfevals": sub_budget - 1,
            "seed": int(seed + 1000 * run_idx + 1),
            "verbose": -9,
            "verb_log": 0,
            "tolfun": 1e-12,
            "tolx": 1e-12,
        }
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        evals_in_run = 1

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
# [4] 상위 K개 후보 탐색 및 유사도순 정렬 (VST 시간 측정 + 모델 연동)
# ----------------------------------------------------------------------
def search_and_rank_chains(
    dry_audio,
    sr,
    target_audio,
    candidate_configs=None,
    top_k_candidates=None,
    owned_effect_types=None,
    effects_def=None,
    budget=40,
    n_restarts=2,
    n_seeds=1,
    metric_fn=spectral_distance,
):
    effects_def = effects_def or load_effects()
    candidates = candidate_configs if candidate_configs is not None else top_k_candidates
    if candidates is None:
        raise ValueError("candidate_configs 또는 top_k_candidates를 전달해야 합니다.")

    if owned_effect_types is not None:
        valid_candidates, dropped = filter_candidates_by_inventory(candidates, owned_effect_types)
    else:
        valid_candidates = list(candidates)
        dropped = []

    def objective_fn(chain_dict):
        wet = render_chain(dry_audio, sr, chain_dict, effects=effects_def)
        return metric_fn(wet, target_audio)

    ranked_results = []
    start_total = time.perf_counter()

    for cand in valid_candidates:
        etypes = extract_effect_types(cand)
        label = format_chain_label(etypes)
        spec = ChainParameterSpec(etypes, effects_def=effects_def)

        x0_norm = None
        init_real = None
        model_score = None
        if isinstance(cand, dict):
            model_score = cand.get("model_score", cand.get("score"))
            if cand.get("x0") is not None and len(cand["x0"]) == spec.dim and spec.dim > 0:
                x0_norm = np.array(cand["x0"], dtype=float)
            elif "initial_chain" in cand:
                try:
                    init_real = spec.chain_to_vector(cand["initial_chain"])
                except Exception:
                    pass
            elif "chain" in cand and isinstance(cand["chain"], dict) and spec.dim > 0:
                try:
                    init_real = spec.chain_to_vector(cand["chain"])
                except Exception:
                    pass

        warm_tag = " (Warm-start x0 적용)" if (x0_norm is not None or init_real is not None) else ""
        print(f"[Search] 후보 탐색 시작: {label:<28} | 차원: {spec.dim} | 예산: {budget}회{warm_tag}")
        start_cand = time.perf_counter()

        best_overall = None
        seed_losses = []

        for seed in range(n_seeds):
            best_chain, best_vec, best_loss, hist = run_cma_es_for_chain(
                objective_fn,
                spec,
                budget=budget,
                seed=seed,
                n_runs=n_restarts,
                initial_values=init_real if seed == 0 else None,
                x0_norm=x0_norm if seed == 0 else None,
            )
            seed_losses.append(best_loss)
            if best_overall is None or best_loss < best_overall["loss"]:
                best_overall = {
                    "effect_types": etypes,
                    "chain": best_chain,
                    "best_vector": best_vec,
                    "loss": float(best_loss),
                    "model_score": model_score,
                    "spec": spec,
                    "history": hist,
                }

        elapsed_cand = time.perf_counter() - start_cand
        eval_count = max(1, budget * n_seeds if spec.dim > 0 else 1)
        print(f"  -> 완료: Loss={best_overall['loss']:.6f} | 소요 시간: {elapsed_cand:.2f}초 "
              f"(1회당 평균 {elapsed_cand / eval_count:.3f}초)")

        best_overall["loss_mean"] = float(np.mean(seed_losses))
        best_overall["loss_std"] = float(np.std(seed_losses))
        best_overall["elapsed"] = elapsed_cand
        ranked_results.append(best_overall)

    total_elapsed = time.perf_counter() - start_total
    print(f"\n[INFO] 탐색 총 소요 시간: {total_elapsed:.2f}초 (후보 {len(valid_candidates)}개 완료)")

    ranked_results.sort(key=lambda r: r["loss"])
    for rank_idx, item in enumerate(ranked_results, start=1):
        item["rank"] = rank_idx

    return ranked_results, dropped


# ----------------------------------------------------------------------
# [5] 데이터 로딩 (생성 폴더의 dry/ 우선 사용) 및 리포트 출력
# ----------------------------------------------------------------------
def load_step1_dry_sample(dataset_dir=None, duration_sec=1.0):
    """
    Task 4 요구사항: 원본 dry가 아닌 생성 폴더의 dry/를 최우선으로 로드합니다.
    1순위: 생성 폴더 내 manifest.json의 render_input 경로
    2순위: 생성 폴더 내 **/dry/*.wav
    3순위: 로컬에 생성된 dry/가 아직 없을 경우 data/dry/ 원본에서 로드 후 피크 -12dBFS 정규화 적용
    """
    ds_root = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
    candidates = []

    for mf in ds_root.glob("**/manifest.json"):
        try:
            with open(mf, encoding="utf-8") as f:
                mdata = json.load(f)
            for item in mdata.get("dry_id_map", []):
                rel = item.get("render_input", "")
                if not rel:
                    continue
                for p in (mf.parent / rel, PROJECT_ROOT / rel):
                    if p.exists() and p.is_file():
                        candidates.append(p)
        except Exception:
            pass

    if not candidates:
        candidates = sorted(list(ds_root.glob("**/dry/*.wav")))

    is_fallback = False
    if not candidates:
        fallback_files = sorted(list((PROJECT_ROOT / "data" / "dry").glob("**/*.wav")))
        if not fallback_files:
            raise FileNotFoundError("사용 가능한 Dry .wav 음원을 찾을 수 없습니다.")
        path = fallback_files[0]
        is_fallback = True
        print(f"[INFO] 생성 폴더({ds_root.name}) 내 dry/ 폴더가 없어 원본 Dry를 로드하여 -12dBFS 정규화합니다: {path.name}")
    else:
        path = candidates[0]
        print(f"[INFO] 생성 폴더 Dry 오디오 로드 완료: {path}")

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

    # 생성 폴더의 dry/ 규칙(피크 -12 dBFS 정규화)과 동일하게 맞춤
    if is_fallback:
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-8:
            target_peak = 10.0 ** (-12.0 / 20.0)
            audio = audio * (target_peak / peak)

    return np.ascontiguousarray(audio, dtype=np.float32), sr, path.name


def format_chain_label(effect_types):
    return " -> ".join(effect_types) if effect_types else "bypass"


def print_ranking_report(ranked_results, dropped, owned_effect_types, ground_truth_chain=None):
    width = 104
    print("\n" + "=" * width)
    print("[Step 1 탐색 엔진] 모델 상위 K개 후보 기반 탐색 및 유사도순 정렬 리포트")
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
# [6] 메인 파이프라인 실행
# ----------------------------------------------------------------------
def run_search_pipeline(budget=40, n_restarts=2, n_seeds=1, dataset_path=None, inventory_override=None, top_k=3):
    effects_def = load_effects()
    dry_audio, sr, _ = load_step1_dry_sample(dataset_dir=dataset_path)
    owned_effects = inventory_override if inventory_override is not None else load_user_inventory()

    vst_ready = is_vst_available("MultiDrivePro")

    if vst_ready:
        target_effect = "808_scream"
        gt_values = {"level": 50.0, "overdrive": 70.0, "tone": 40.0}
        # 인벤토리에 VST가 없으면 테스트를 위해VST 7종 추가
        if not any(effects_def.get(e, {}).get("source") == "vst" for e in owned_effects):
            owned_effects = list(owned_effects) + ["808_scream", "b_drive", "o_drive"]
    else:
        print("[INFO] 현재 PC에 MultiDrivePro VST3가 설치되지 않아 pedalboard 이펙터('overdrive', 'distortion', 'fuzz')로 탐색을 진행합니다.")
        target_effect = "overdrive"
        gt_values = {"hpf_freq": 180.0, "drive": 35.0, "lpf_freq": 4200.0, "level": 0.5}
        # 인벤토리에 pedalboard 드라이브가 없으면 추가
        if not any(e in owned_effects for e in ("overdrive", "distortion", "fuzz")):
            owned_effects = list(owned_effects) + ["overdrive", "distortion", "fuzz"]

    ground_truth_chain = {
        "amp_id": "fixed_amp_v1",
        "chain": [{"effect_type": target_effect, "values": gt_values}],
    }
    target_audio = render_chain(dry_audio, sr, ground_truth_chain, effects=effects_def)

    # 5번(도영님) CNN 모델 예측 출력 시뮬레이션 -> make_candidates로 상위 K개 후보 생성
    defs_for_cnn = build_estimation_definitions(effects_def, include_vst=vst_ready)
    raw_probs = {etype: 0.05 for etype in ["bypass"] + list(defs_for_cnn.keys())}
    raw_probs[target_effect] = 0.65
    second_choice = "b_drive" if (vst_ready and "b_drive" in defs_for_cnn) else "distortion"
    if second_choice in raw_probs:
        raw_probs[second_choice] = 0.25

    mock_prediction = {
        "probabilities": raw_probs,
        "values": {
            etype: {p["name"]: p["default"] for p in defs_for_cnn[etype]["params"]}
            for etype in defs_for_cnn
        },
    }

    candidates = build_candidates_from_cnn_prediction(
        mock_prediction,
        owned_effect_types=owned_effects,
        k=top_k,
        effects_def=effects_def,
        include_vst=vst_ready,
    )

    ranked_results, dropped = search_and_rank_chains(
        dry_audio=dry_audio,
        sr=sr,
        target_audio=target_audio,
        candidate_configs=candidates,
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
    p = argparse.ArgumentParser(description="Step 1: Top-K Model Candidates Search & Ranking")
    p.add_argument("--budget", type=int, default=40, help="후보 체인당 평가 횟수 예산 (기본값: 40)")
    p.add_argument("--restarts", type=int, default=2, help="CMA-ES 분할 재시작 횟수 (기본값: 2)")
    p.add_argument("--seeds", type=int, default=1, help="반복 시드 수")
    p.add_argument("--top-k", type=int, default=3, help="모델에서 받아올 상위 K개 후보 수 (기본값: 3)")
    p.add_argument("--dataset", type=str, default=None, help="생성 데이터셋 폴더 경로")
    p.add_argument("--inventory", type=str, nargs="*", default=None, help="보유 이펙터 오버라이드")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search_pipeline(
        budget=args.budget,
        n_restarts=args.restarts,
        n_seeds=args.seeds,
        dataset_path=args.dataset,
        inventory_override=args.inventory,
        top_k=args.top_k,
    )