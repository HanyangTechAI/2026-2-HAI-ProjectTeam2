"""
search/pipeline.py

Step 0 baseline: Gradient-free parameter search (Random Search vs Restart-CMA-ES)
for the Overdrive effect (Compressor bypassed).

Usage (프로젝트 루트에서)
-------------------------
python -m search.pipeline
python -m search.pipeline --budgets 200 --restarts 4 --seeds 3
python -m search.pipeline --no-sensitivity
"""
import os
import glob
import argparse
import numpy as np
import soundfile as sf
import cma

from metrics.spectral import spectral_distance
from scripts.Create_Synthetic_pairs import apply_effects

DRY_AUDIO_DIR = "data/dry/Fender_Strat_Clean/audio"
OUTPUT_DIR = "outputs/search"
PENALTY = 1e3

# ----------------------------------------------------------------------
# 파라미터 정의: 컴프레서 제거 -> Overdrive (drive, level) 2차원 탐색
# ----------------------------------------------------------------------
PARAM_SPECS = [
    ("OD  : drive",   0.0, 15.0, False),
    ("OD  : level", -12.0, 12.0, False),
]
PARAM_NAMES = [s[0] for s in PARAM_SPECS]
LOWS = np.array([s[1] for s in PARAM_SPECS], dtype=float)
HIGHS = np.array([s[2] for s in PARAM_SPECS], dtype=float)
IS_LOG = np.array([s[3] for s in PARAM_SPECS], dtype=bool)
DIM = len(PARAM_SPECS)

# 찾아야 할 Overdrive 정답 노브 값 (drive=10.0, level=0.0)
TARGET_PARAMS = np.array([10.0, 0.0])

# 컴프레서 무력화(Bypass)용 고정 파라미터 (threshold=0dB, ratio=1.0이면 압축 안 함)
BYPASS_COMP_PARAMS = {
    "threshold_db": 0.0,
    "ratio": 1.0,
    "attack_ms": 10.0,
    "release_ms": 100.0,
}


# ----------------------------------------------------------------------
# 정규화 / 역정규화 ([0,1]^2 <-> 실제 노브 값)
# ----------------------------------------------------------------------
def denormalize(x):
    """[0,1] 정규화 벡터 -> 실제 파라미터."""
    x = np.asarray(x, dtype=float)
    lin = LOWS + x * (HIGHS - LOWS)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lo, log_hi = np.log(LOWS), np.log(HIGHS)
        logv = np.exp(log_lo + x * (log_hi - log_lo))
    return np.where(IS_LOG, logv, lin)


def normalize(params):
    """실제 파라미터 -> [0,1] 정규화 벡터."""
    p = np.asarray(params, dtype=float)
    lin = (p - LOWS) / (HIGHS - LOWS)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lo, log_hi = np.log(LOWS), np.log(HIGHS)
        logv = (np.log(p) - log_lo) / (log_hi - log_lo)
    return np.where(IS_LOG, logv, lin)


# ----------------------------------------------------------------------
# 실제 Dry 오디오 데이터 로딩 (가짜 합성음 폴백 제거)
# ----------------------------------------------------------------------
def load_real_dry_sample(duration_sec=1.0):
    """
    data/dry/Fender_Strat_Clean/audio 경로에서 실제 .wav 파일을 로드합니다.
    가장 에너지가 큰 duration_sec(1초) 구간을 선택합니다.
    """
    wav_files = sorted(glob.glob(os.path.join(DRY_AUDIO_DIR, "*.wav")))
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


# ----------------------------------------------------------------------
# 파라미터 변환 / 출력
# ----------------------------------------------------------------------
def array_to_dict(params):
    """탐색된 2차원 배열을 팀원의 apply_effects 규격으로 변환 (컴프레서는 바이패스 고정)."""
    drive_params = {
        "drive": float(params[0]),
        "level": float(params[1]),
    }
    return BYPASS_COMP_PARAMS, drive_params


def print_comparison(method_name, target_params, best_params, best_loss):
    tn, bn = normalize(target_params), normalize(best_params)
    norm_err = np.abs(tn - bn) * 100.0
    width = 92
    print("\n" + "=" * width)
    print(f"[{method_name} 탐색 결과 비교] (최종 Loss: {best_loss:.6f})")
    print("-" * width)
    print(f"{'파라미터 (Knob)':<20} | {'정답(Target)':>12} | {'찾은 값(Found)':>14} | "
          f"{'절대 오차':>10} | {'정규화 오차(%)':>14}")
    print("-" * width)
    for name, t, f, ne in zip(PARAM_NAMES, target_params, best_params, norm_err):
        print(f"{name:<20} | {t:>12.2f} | {f:>14.2f} | {abs(t - f):>10.2f} | {ne:>14.1f}")
    print("-" * width)
    print(f"{'평균 정규화 오차(%)':<20}   {'':>12}   {'':>14}   {'':>10}   {norm_err.mean():>14.1f}")
    print("=" * width)


# ----------------------------------------------------------------------
# 평가 추적기: 예산 관리 + best-so-far 기록
# ----------------------------------------------------------------------
class Tracker:
    def __init__(self, objective_fn, budget):
        self.objective_fn = objective_fn
        self.budget = budget
        self.n = 0
        self.best_loss = float("inf")
        self.best_norm = None
        self.history = []

    @property
    def exhausted(self):
        return self.n >= self.budget

    def __call__(self, x_norm):
        if self.exhausted:
            return PENALTY
        x_norm = np.asarray(x_norm, dtype=float)
        try:
            loss = float(self.objective_fn(denormalize(x_norm)))
            if not np.isfinite(loss):
                loss = PENALTY
        except Exception as e:
            print(f"[WARN] 평가 실패 -> penalty: {e}")
            loss = PENALTY
        self.n += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_norm = x_norm.copy()
        self.history.append(self.best_loss)
        return loss

    def padded_history(self):
        h = list(self.history)
        if len(h) < self.budget:
            h += [h[-1]] * (self.budget - len(h))
        return np.array(h[:self.budget])


# ----------------------------------------------------------------------
# 탐색 알고리즘 (Random Search & 50x4 Restart CMA-ES)
# ----------------------------------------------------------------------
def run_random_search(objective_fn, budget, seed):
    rng = np.random.default_rng(seed)
    tr = Tracker(objective_fn, budget)
    while not tr.exhausted:
        tr(rng.uniform(0.0, 1.0, DIM))
    return denormalize(tr.best_norm), tr.best_loss, tr.padded_history()


def run_cma_es(objective_fn, budget, seed, n_runs=4, sigma0=0.25, popsize=6):
    """
    총 예산(예: 200회)을 n_runs(예: 4번)로 균등 분할하여 (50회 x 4번)
    반드시 4번의 독립적인 재시작(Restart)이 수행되도록 보장합니다.
    """
    rng = np.random.default_rng(seed)
    tr = Tracker(objective_fn, budget)

    # 예산을 n_runs 등분 (예: 200 // 4 = 50회씩 4번 할당)
    base_evals = budget // n_runs
    rem = budget % n_runs
    run_budgets = [base_evals + (1 if i < rem else 0) for i in range(n_runs)]

    for run_idx, sub_budget in enumerate(run_budgets):
        if tr.exhausted or sub_budget <= 0:
            break

        # 첫 시도는 중앙(0.5), 이후 재시작(Restart)은 랜덤 시작점
        x0 = np.full(DIM, 0.5) if run_idx == 0 else rng.uniform(0.1, 0.9, DIM)
        opts = {
            "bounds": [0.0, 1.0],
            "popsize": popsize,
            "maxfevals": sub_budget,
            "seed": seed + 1000 * run_idx + 1,
            "verbose": -9,
            "tolfun": 1e-12,
            "tolx": 1e-12,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        evals_in_run = 0

        # 해당 회차에 할당된 예산(예: 50회)을 채울 때까지만 정확히 구동 후 다음 재시작으로 넘어감
        while not es.stop() and not tr.exhausted and evals_in_run < sub_budget:
            xs = es.ask()
            remaining = sub_budget - evals_in_run
            if len(xs) > remaining:
                xs = xs[:remaining]
            losses = [tr(x) for x in xs]
            evals_in_run += len(xs)
            if len(xs) == popsize:
                es.tell(xs, losses)
            else:
                break

    return denormalize(tr.best_norm), tr.best_loss, tr.padded_history()


# ----------------------------------------------------------------------
# 1D 민감도 분석 및 수렴 곡선 저장
# ----------------------------------------------------------------------
def run_sensitivity(objective_fn, n_points=21):
    target_norm = normalize(TARGET_PARAMS)
    grid = np.linspace(0.0, 1.0, n_points)
    results = {}
    print("\n" + "=" * 70)
    print("[1D 민감도 분석] 정답에서 한 파라미터만 변화 (Overdrive 2D)")
    print("-" * 70)
    print(f"{'파라미터':<20} | {'loss(min)':>10} | {'loss(max)':>10} | {'range':>10} | {'해석':<}")
    print("-" * 70)
    for i, name in enumerate(PARAM_NAMES):
        losses = []
        for g in grid:
            x = target_norm.copy()
            x[i] = g
            try:
                losses.append(float(objective_fn(denormalize(x))))
            except Exception:
                losses.append(np.nan)
        losses = np.array(losses)
        results[name] = losses
        rng_ = np.nanmax(losses) - np.nanmin(losses)
        note = "거의 평평 (식별 어려움)" if rng_ < 0.05 else "민감 (탐색 용이)"
        print(f"{name:<20} | {np.nanmin(losses):>10.4f} | {np.nanmax(losses):>10.4f} | {rng_:>10.4f} | {note}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUTPUT_DIR, "sensitivity.npz"), grid=grid, **{
        k.replace(":", "").replace(" ", "_"): v for k, v in results.items()})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, DIM, figsize=(10, 4))
        if DIM == 1:
            axes = [axes]
        for ax, (name, losses) in zip(axes, results.items()):
            ax.plot(grid, losses, marker="o", ms=3)
            ax.axvline(normalize(TARGET_PARAMS)[PARAM_NAMES.index(name)], color="r", ls="--", lw=1)
            ax.set_title(name.strip())
            ax.set_xlabel("normalized value")
            ax.set_ylabel("loss")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "sensitivity.png"), dpi=120)
        plt.close(fig)
        print(f"[INFO] 민감도 그래프 저장: {OUTPUT_DIR}/sensitivity.png")
    except ImportError:
        pass
    return results


def save_convergence_plot(curves, budget):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, hs in curves.items():
        arr = np.array(hs)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        x = np.arange(1, arr.shape[1] + 1)
        ax.plot(x, mean, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("best-so-far loss")
    ax.set_yscale("log")
    ax.set_title(f"Convergence (budget={budget}, mean ± std)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"convergence_{budget}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[INFO] 수렴 곡선 저장: {path}")


# ----------------------------------------------------------------------
# 메인 파이프라인
# ----------------------------------------------------------------------
def run_search_pipeline(budgets=(200,), n_restarts=4, n_seeds=3, sensitivity=True):
    dry_audio, sr, src = load_real_dry_sample()

    target_comp, target_drive = array_to_dict(TARGET_PARAMS)
    target_audio = apply_effects(dry_audio, sr, target_comp, "overdrive", target_drive)

    def objective_function(params):
        comp_p, drive_p = array_to_dict(params)
        wet = apply_effects(dry_audio, sr, comp_p, "overdrive", drive_p)
        return spectral_distance(wet, target_audio)

    sanity = objective_function(TARGET_PARAMS)
    print(f"[CHECK] 정답 파라미터에서의 loss = {sanity:.6e} (0에 가까워야 정상)")

    if sensitivity:
        run_sensitivity(objective_function)

    cma_label = f"CMA-ES ({n_restarts} restarts)"
    methods = {
        "Random Search": lambda b, s: run_random_search(objective_function, b, s),
        cma_label: lambda b, s: run_cma_es(objective_function, b, s, n_runs=n_restarts),
    }

    for budget in budgets:
        per_run = budget // n_restarts
        print("\n" + "#" * 92)
        print(f"# 총 예산 = {budget}회 (CMA-ES: {per_run}회 x {n_restarts}번 실행), 시드 {n_seeds}개, 오디오: {src}")
        print("#" * 92)
        curves = {m: [] for m in methods}
        summary = {}

        for name, fn in methods.items():
            losses, norm_errs, best_run = [], [], None
            for seed in range(n_seeds):
                params, loss, hist = fn(budget, seed)
                curves[name].append(hist)
                losses.append(loss)
                norm_errs.append(np.abs(normalize(params) - normalize(TARGET_PARAMS)).mean() * 100)
                if best_run is None or loss < best_run[1]:
                    best_run = (params, loss)
            summary[name] = (np.mean(losses), np.std(losses),
                             np.mean(norm_errs), np.std(norm_errs))
            print_comparison(f"{name} / best of {n_seeds} seeds (budget={budget})",
                             TARGET_PARAMS, best_run[0], best_run[1])

        print("\n[요약] 시드 평균 ± 표준편차")
        print(f"{'Method':<22} | {'Loss':>20} | {'평균 정규화 파라미터 오차(%)':>30}")
        print("-" * 80)
        for name, (lm, ls, em, es_) in summary.items():
            print(f"{name:<22} | {lm:>10.4f} ± {ls:<7.4f} | {em:>16.1f} ± {es_:<8.1f}")

        save_convergence_plot(curves, budget)


def parse_args():
    p = argparse.ArgumentParser(description="Gradient-free effect parameter search baseline (Overdrive only)")
    p.add_argument("--budgets", type=int, nargs="+", default=[200],
                   help="총 평가 횟수 예산 목록 (기본값: 200)")
    p.add_argument("--restarts", type=int, default=4,
                   help="CMA-ES 분할 실행 횟수 (기본값: 4 -> 200회일 경우 50회씩 4번 실행)")
    p.add_argument("--seeds", type=int, default=3, help="반복 시드 수")
    p.add_argument("--no-sensitivity", action="store_true", help="1D 민감도 분석 생략")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search_pipeline(
        budgets=tuple(args.budgets),
        n_restarts=args.restarts,
        n_seeds=args.seeds,
        sensitivity=not args.no_sensitivity,
    )