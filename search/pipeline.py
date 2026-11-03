"""
search/pipeline.py

Step 0 baseline: Gradient-free parameter search (Random Search vs CMA-ES)
for the Compressor -> Overdrive effect chain.

Usage (프로젝트 루트에서)
-------------------------
python -m search.pipeline
python -m search.pipeline --budgets 50 200 1000 --seeds 10
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
SYNTH_DIR = "data/synthetic_test"   # 합성 테스트 음원 전용 (실제 데이터 폴더와 분리, .gitignore 권장)
OUTPUT_DIR = "outputs/search"
PENALTY = 1e3

# ----------------------------------------------------------------------
# 파라미터 정의: (이름, 하한, 상한, log 스케일 여부)
# ----------------------------------------------------------------------
PARAM_SPECS = [
    ("Comp: threshold_db", -60.0,    0.0, False),
    ("Comp: ratio",          1.0,   20.0, False),
    ("Comp: attack_ms",      0.1,  100.0, True),
    ("Comp: release_ms",    10.0, 1000.0, True),
    ("OD  : drive",          0.0,   15.0, False),
    ("OD  : level",        -12.0,   12.0, False),
]
PARAM_NAMES = [s[0] for s in PARAM_SPECS]
LOWS = np.array([s[1] for s in PARAM_SPECS], dtype=float)
HIGHS = np.array([s[2] for s in PARAM_SPECS], dtype=float)
IS_LOG = np.array([s[3] for s in PARAM_SPECS], dtype=bool)
DIM = len(PARAM_SPECS)

TARGET_PARAMS = np.array([-20.0, 4.0, 10.0, 100.0, 10.0, 0.0])


# ----------------------------------------------------------------------
# 정규화 / 역정규화 ([0,1]^6 <-> 실제 노브 값)
# ----------------------------------------------------------------------
def denormalize(x):
    """[0,1] 정규화 벡터 -> 실제 파라미터 (시간 상수는 log 스케일)."""
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
# 데이터 로딩
# ----------------------------------------------------------------------
def generate_realistic_guitar_wav(filepath, sr=22050, duration_sec=1.5):
    """
    [테스트 전용] 다중 배음 + 피킹 엔벨로프를 가진 합성 기타음 생성.
    컴프레서/드라이브가 사인파보다 실제 기타에 가깝게 반응하도록 한 것일 뿐,
    실제 기타 녹음을 대체할 수 없다. 베이스라인 결과에는 사용하지 말 것.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = np.zeros_like(t)

    # 3번의 피킹 (점점 강하게 -> 컴프레서 작동 유도)
    notes = [(0.05, 110.0, 0.35), (0.5, 146.83, 0.65), (0.95, 196.0, 0.95)]
    for onset, freq, amp in notes:
        mask = t >= onset
        t_rel = t[mask] - onset
        env = (1.0 - np.exp(-t_rel / 0.005)) * np.exp(-t_rel / 0.25)
        sig = np.zeros_like(t_rel)
        for h in range(1, 10):
            sig += (1.0 / (h ** 1.15)) * np.sin(2 * np.pi * freq * h * t_rel)
        audio[mask] += amp * env * sig

    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.85
    sf.write(filepath, audio.astype(np.float32), sr)
    return filepath


def load_real_dry_sample(duration_sec=1.0):
    """
    실제 Dry wav 로드 (에너지가 가장 큰 duration_sec 구간 사용).
    wav가 없으면 별도 폴더(SYNTH_DIR)의 합성음으로 대체하며, 경고와 [SYNTHETIC] 태그를 남긴다.
    """
    wav_files = sorted(glob.glob(os.path.join(DRY_AUDIO_DIR, "*.wav")))  # makedirs 하지 않음
    is_synthetic = False

    if not wav_files:
        print("=" * 70)
        print(f"[WARNING] '{DRY_AUDIO_DIR}'에서 .wav를 찾지 못했습니다 (경로 확인!).")
        print("[WARNING] 합성 기타음으로 대체합니다. 이 결과는 베이스라인으로 쓰지 마세요.")
        print("=" * 70)
        synth_path = os.path.join(SYNTH_DIR, "sample_guitar_dry.wav")
        if not os.path.exists(synth_path):
            generate_realistic_guitar_wav(synth_path)
        wav_files = [synth_path]
        is_synthetic = True

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

    print(f"[INFO] 사용 오디오: {path} (sr={sr}, 길이={len(audio) / sr:.2f}s)")
    tag = "[SYNTHETIC] " if is_synthetic else ""
    return np.ascontiguousarray(audio, dtype=np.float32), sr, tag + os.path.basename(path)


# ----------------------------------------------------------------------
# 파라미터 변환 / 출력
# ----------------------------------------------------------------------
def array_to_dict(params):
    comp_params = {
        "threshold_db": float(params[0]),
        "ratio": float(params[1]),
        "attack_ms": float(params[2]),
        "release_ms": float(params[3]),
    }
    drive_params = {
        "drive": float(params[4]),
        "level": float(params[5]),
    }
    return comp_params, drive_params


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
        """조기 종료로 길이가 budget보다 짧아도 마지막 best로 채워 길이를 맞춘다."""
        h = list(self.history)
        if len(h) < self.budget:
            h += [h[-1]] * (self.budget - len(h))
        return np.array(h[:self.budget])


# ----------------------------------------------------------------------
# 탐색 알고리즘 (Random Search & IPOP-CMA-ES)
# ----------------------------------------------------------------------
def run_random_search(objective_fn, budget, seed):
    rng = np.random.default_rng(seed)
    tr = Tracker(objective_fn, budget)
    while not tr.exhausted:
        tr(rng.uniform(0.0, 1.0, DIM))
    return denormalize(tr.best_norm), tr.best_loss, tr.padded_history()


def run_cma_es(objective_fn, budget, seed, sigma0=0.3, popsize=None,
               max_restarts=3, incpopsize=2):
    rng = np.random.default_rng(seed)
    tr = Tracker(objective_fn, budget)
    base_pop = popsize or (4 + int(3 * np.log(DIM)))
    restart = 0

    while not tr.exhausted and restart <= max_restarts:
        x0 = np.full(DIM, 0.5) if restart == 0 else rng.uniform(0.1, 0.9, DIM)
        pop = int(base_pop * (incpopsize ** restart))
        opts = {
            "bounds": [0.0, 1.0],
            "popsize": pop,
            "seed": seed + 1000 * restart + 1,
            "verbose": -9,
            "tolfun": 1e-10,
            "tolx": 1e-8,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop() and not tr.exhausted:
            xs = es.ask()
            es.tell(xs, [tr(x) for x in xs])
        restart += 1

    return denormalize(tr.best_norm), tr.best_loss, tr.padded_history()


# ----------------------------------------------------------------------
# 1D 민감도 분석 및 수렴 곡선 저장
# ----------------------------------------------------------------------
def run_sensitivity(objective_fn, n_points=21):
    target_norm = normalize(TARGET_PARAMS)
    grid = np.linspace(0.0, 1.0, n_points)
    results = {}
    print("\n" + "=" * 70)
    print("[1D 민감도 분석] 정답에서 한 파라미터만 변화 (나머지는 정답 고정)")
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
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, (name, losses) in zip(axes.ravel(), results.items()):
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
def run_search_pipeline(budgets=(50, 200), n_seeds=3, sensitivity=True):
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

    methods = {
        "Random Search": lambda b, s: run_random_search(objective_function, b, s),
        "CMA-ES (IPOP)": lambda b, s: run_cma_es(objective_function, b, s),
    }

    for budget in budgets:
        print("\n" + "#" * 92)
        print(f"# 예산(max evals) = {budget}, 시드 {n_seeds}개, 오디오: {src}")
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
        print(f"{'Method':<16} | {'Loss':>20} | {'평균 정규화 파라미터 오차(%)':>30}")
        print("-" * 74)
        for name, (lm, ls, em, es_) in summary.items():
            print(f"{name:<16} | {lm:>10.4f} ± {ls:<7.4f} | {em:>16.1f} ± {es_:<8.1f}")

        save_convergence_plot(curves, budget)


def parse_args():
    p = argparse.ArgumentParser(description="Gradient-free effect parameter search baseline")
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 200],
                   help="평가 횟수 예산 목록 (예: --budgets 50 200 1000)")
    p.add_argument("--seeds", type=int, default=3, help="반복 시드 수")
    p.add_argument("--no-sensitivity", action="store_true", help="1D 민감도 분석 생략")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search_pipeline(
        budgets=tuple(args.budgets),
        n_seeds=args.seeds,
        sensitivity=not args.no_sensitivity,
    )