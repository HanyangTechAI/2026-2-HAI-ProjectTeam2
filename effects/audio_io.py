"""
effects/audio_io.py

오디오 파일을 팀 공통 형식으로 읽습니다: 모노 + 표준 샘플레이트(44.1 kHz), float32.

- 스테레오(또는 다채널)는 채널 평균으로 모노로 바꿉니다. 기타 DI는 원래 모노이고,
  외부 음원·음원 분리 결과처럼 스테레오로 들어오는 소리도 같은 형식으로 맞춥니다.
- 샘플레이트가 다르면 librosa로 리샘플링합니다. CNN·렌더링·유사도 d가 모두 같은 샘플레이트를 씁니다.

사용 예
-------
from effects.audio_io import read_mono
audio, sr = read_mono("guitar.wav")            # 44.1 kHz 모노
audio, sr = read_mono("guitar.wav", None)      # 원래 샘플레이트 유지, 모노만 변환
"""
import numpy as np

STANDARD_SAMPLE_RATE = 44100


def to_mono(audio):
    """(samples,) 또는 (samples, channels) 배열을 모노 (samples,)로 바꿉니다."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"오디오는 1차원 또는 (samples, channels) 2차원이어야 합니다: {audio.shape}")
    return audio.mean(axis=1, dtype=np.float32)


def read_mono(path, sample_rate=STANDARD_SAMPLE_RATE):
    """파일을 읽어 (모노 float32 배열, 샘플레이트)를 돌려줍니다.

    sample_rate=None이면 원래 샘플레이트를 유지합니다.
    빈 파일이나 NaN/inf가 있으면 ValueError를 냅니다.
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"비어 있거나 유효하지 않은 값이 있는 오디오: {path}")
    audio = to_mono(audio)
    if sample_rate is not None and sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
        if not np.isfinite(audio).all():
            raise ValueError(f"리샘플링 후 유효하지 않은 값: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
