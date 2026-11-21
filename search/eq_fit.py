"""
search/eq_fit.py

EQ(eq_3band) 노브를 탐색(CMA-ES) 없이 계산으로 구합니다.

원리
----
EQ는 소리를 찌그러뜨리지 않는 선형 필터라서, 목표 = EQ(EQ 없이 렌더링한 소리) 관계가 성립합니다.
(앰프 IR도 선형이라 EQ와 순서를 바꿔도 결과가 같습니다.)
따라서 두 소리의 주파수별 에너지 비율 = EQ의 주파수 응답이고, 이 응답에 가장 잘 맞는
노브 4개(저역·중역 게인, 중역 주파수, 고역 게인)를 최소제곱으로 찾습니다. VST 렌더링이 필요 없습니다.

사용 예
-------
from search.eq_fit import fit_eq, eq_metric
values, info = fit_eq(rendered_without_eq, target, sr)    # {"low_gain_db": .., "mid_freq_hz": .., ...}
search_and_rank_chains(..., metric_fn=eq_metric(sr))      # 탐색 중 매 후보마다 EQ를 맞춘 뒤 거리 계산
"""
from functools import lru_cache

import numpy as np

from effects import build_effect, load_effects
from metrics.spectral import spectral_distance

N_FFT = 4096
FIT_BAND_HZ = (60.0, 12000.0)   # 이 범위의 주파수만 맞춥니다 (아주 낮은·높은 대역은 에너지가 작아 불안정)
FLOOR_DB = -60.0                # 렌더링 소리의 에너지가 최대 대비 이보다 작은 주파수는 제외
OFF_THRESHOLD_DB = 1.0          # 세 게인이 모두 이보다 작으면 "EQ 사용 안 함"으로 봅니다


def _eq_def():
    return load_effects()["eq_3band"]


def _ranges():
    return {p["name"]: tuple(p["range"]) for p in _eq_def()["params"]}


def _power_spectrum(audio):
    """프레임별 파워 스펙트럼의 합 (Hann 창, 50% 겹침)."""
    audio = np.asarray(audio, dtype=np.float64).ravel()
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    hop = N_FFT // 2
    window = np.hanning(N_FFT)
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::hop] * window
    return np.sum(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)


def transfer_db(rendered, target, sample_rate):
    """목표 / 렌더링 소리의 주파수별 에너지 비율(dB)과 맞출 주파수 위치(mask)."""
    p_r, p_t = _power_spectrum(rendered), _power_spectrum(target)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sample_rate)
    eps = 1e-20
    level_r = 10 * np.log10(p_r + eps)
    mask = ((freqs >= FIT_BAND_HZ[0]) & (freqs <= min(FIT_BAND_HZ[1], sample_rate / 2 * 0.95))
            & (level_r > level_r.max() + FLOOR_DB) & (p_t > eps))
    return 10 * np.log10((p_t + eps) / (p_r + eps)), mask


@lru_cache(maxsize=4096)
def _response_cached(low, log_mid, mid, high, sample_rate):
    from pedalboard import Pedalboard

    values = {"low_gain_db": low, "mid_freq_hz": float(np.exp(log_mid)), "mid_gain_db": mid, "high_gain_db": high}
    impulse = np.zeros(N_FFT, dtype=np.float32)
    impulse[0] = 1.0
    h = Pedalboard(build_effect("eq_3band", values))(impulse, sample_rate)
    return 20 * np.log10(np.abs(np.fft.rfft(h, N_FFT)) + 1e-12)


def eq_response_db(values, sample_rate):
    """EQ 노브 값의 주파수 응답(dB, rfft 위치)."""
    return _response_cached(round(values["low_gain_db"], 4), round(float(np.log(values["mid_freq_hz"])), 5),
                            round(values["mid_gain_db"], 4), round(values["high_gain_db"], 4), int(sample_rate))


def fit_eq(rendered, target, sample_rate, starts=(300.0, 800.0, 2000.0, 4000.0)):
    """EQ 없이 렌더링한 소리를 목표에 맞추는 EQ 노브 값을 계산합니다.

    반환: (values, info)
      values: eq_3band 노브 dict (스키마 범위 안)
      info: {"active": EQ를 써야 하는지, "residual_db": 맞춘 뒤 남은 평균 오차(dB), "level_db": 전체 음량 차이}
    """
    from scipy.optimize import least_squares

    target_db, mask = transfer_db(rendered, target, sample_rate)
    if mask.sum() < 8:
        return dict(_defaults()), {"active": False, "residual_db": float("nan"), "level_db": 0.0}
    level = float(np.median(target_db[mask]))  # 참고용: 전체 음량 차이
    goal = target_db[mask]
    # FFT 칸은 고음역에 몰려 있으므로, 옥타브마다 같은 비중이 되도록 1/f 가중치
    freqs = np.fft.rfftfreq(N_FFT, 1 / sample_rate)[mask]
    weight = np.sqrt((1 / freqs) / np.mean(1 / freqs))
    r = _ranges()
    lo = [r["low_gain_db"][0], np.log(r["mid_freq_hz"][0]), r["mid_gain_db"][0], r["high_gain_db"][0]]
    hi = [r["low_gain_db"][1], np.log(r["mid_freq_hz"][1]), r["mid_gain_db"][1], r["high_gain_db"][1]]

    def residual(x):
        values = {"low_gain_db": x[0], "mid_freq_hz": float(np.exp(x[1])), "mid_gain_db": x[2], "high_gain_db": x[3]}
        return weight * (eq_response_db(values, sample_rate)[mask] - goal)

    steps = np.array([0.05, 0.005, 0.05, 0.05])  # 고정 미분 간격 (게인 dB, log 주파수)

    def jacobian(x):
        # scipy 기본 미분은 값이 0일 때 간격이 너무 작아져 필터 응답 변화가 안 잡히므로 직접 계산
        base = residual(x)
        cols = []
        for i, h in enumerate(steps):
            xi = np.array(x, dtype=float)
            xi[i] = xi[i] + h if xi[i] + h <= hi[i] else xi[i] - h
            cols.append((residual(xi) - base) / (xi[i] - x[i]))
        return np.stack(cols, axis=1)

    best = None
    for f0 in starts:
        x0 = np.clip([0.0, np.log(f0), 0.0, 0.0], lo, hi)
        sol = least_squares(residual, x0, jac=jacobian, bounds=(lo, hi), max_nfev=60)
        if best is None or sol.cost < best.cost:
            best = sol
    x = best.x
    values = {"low_gain_db": float(x[0]), "mid_freq_hz": float(np.exp(x[1])),
              "mid_gain_db": float(x[2]), "high_gain_db": float(x[3])}
    active = max(abs(values["low_gain_db"]), abs(values["mid_gain_db"]), abs(values["high_gain_db"])) >= OFF_THRESHOLD_DB
    if not active:
        values = dict(_defaults())
    info = {"active": bool(active), "residual_db": float(np.sqrt(np.mean(best.fun ** 2))), "level_db": level}
    return values, info


def _defaults():
    return {p["name"]: p["default"] for p in _eq_def()["params"]}


def apply_eq(audio, sample_rate, values):
    from pedalboard import Pedalboard

    return Pedalboard(build_effect("eq_3band", values))(np.asarray(audio, dtype=np.float32), sample_rate)


def eq_metric(sample_rate, base_metric=spectral_distance, record=None):
    """탐색용 거리 함수: 후보 소리에 EQ를 계산으로 맞춘 뒤 base_metric으로 비교합니다.

    search_and_rank_chains(metric_fn=eq_metric(sr))처럼 넘기면 기존 탐색 코드 수정 없이 동작합니다.
    record에 list를 넘기면 매 평가의 (EQ 노브, info)를 순서대로 남깁니다.
    """
    def metric(wet, target):
        values, info = fit_eq(wet, target, sample_rate)
        shaped = apply_eq(wet, sample_rate, values) if info["active"] else wet
        if record is not None:
            record.append((values, info))
        return base_metric(shaped, target)

    return metric
