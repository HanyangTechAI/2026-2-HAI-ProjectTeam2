"""
stylematch/spectrum_eq.py

평균 스펙트럼 맞추기 EQ (매칭 EQ): 결과의 긴 구간 평균 스펙트럼 모양을 목표에 맞춤.
- 연주가 달라도 긴 구간 평균 스펙트럼은 비교적 안정적 → 녹음 EQ · 마이크 · 캐비닛 차이 흡수
- 1/3옥타브(80 Hz–10 kHz) 차이, 전체 평균은 빼고(음량 무관) ±12 dB 제한, 대역 밖은 끝 값 유지
- 선형 위상 FIR(4096탭), 적용 후 음량(RMS)은 목표에 맞춤
주의: 평균 스펙트럼 거리(assess.spectrum_shape_distance)를 직접 줄이므로, EQ 적용 결과는 그 지표로 평가하지 않음.
"""
import numpy as np
from scipy.signal import fftconvolve

from stylematch.assess import BANDS, avg_spectrum_db

MAX_DB = 12.0
TAPS = 4096


def eq_curve(audio, target):
    """대역별 보정량(dB)."""
    d = avg_spectrum_db(target) - avg_spectrum_db(audio)
    d = d - d.mean()
    return np.clip(d, -MAX_DB, MAX_DB)


def apply_spectrum_eq(audio, target, sample_rate=44100):
    """반환: (EQ 적용 소리, 대역별 보정량 dB)"""
    x = np.asarray(audio, np.float64).reshape(-1)
    g = eq_curve(x, target)
    f = np.fft.rfftfreq(TAPS, 1 / sample_rate)
    gain_db = np.interp(np.log(np.maximum(f, 1.0)), np.log(BANDS), g)  # 대역 밖은 끝 값 유지
    h = np.fft.fftshift(np.fft.irfft(10 ** (gain_db / 20), TAPS)) * np.hanning(TAPS)
    y = fftconvolve(x, h)[TAPS // 2: TAPS // 2 + len(x)]
    rms = lambda v: np.sqrt(np.mean(np.square(v))) + 1e-12
    y *= rms(np.asarray(target, np.float64)) / rms(y)
    return y.astype(np.float32), g
