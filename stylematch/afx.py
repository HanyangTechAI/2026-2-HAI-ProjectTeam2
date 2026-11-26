"""
stylematch/afx.py

AFx-Rep(ST-ITO, Steinmetz et al., ISMIR 2024) 이펙트 임베딩과 거리.

연주 내용(음·리듬)이 달라도 "어떤 이펙트가 걸렸는지"를 비교하도록 대량 데이터로 학습된 공개 모델입니다.
같은 연주의 dry가 없는 목표(곡에서 분리한 기타, 외부 기타 톤)를 탐색할 때 거리 d 대신 씁니다.

- 가중치: afx-rep.ckpt + config.yaml (HuggingFace csteinmetz1/afx-rep, 약 1.1GB, git 제외)
  찾는 순서: 환경변수 HAI_AFX_DIR → data/local/afx-rep (공유 드라이브 model/afx-rep 에서 복사)
- 가중치는 torch.load(weights_only=True)로만 읽습니다 (파일 안의 실행 코드를 실행하지 않음).
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AFX_DIRS = (ROOT / "data" / "local" / "afx-rep",)
MODEL_SR = 48000


def find_afx_dir():
    dirs = [Path(os.environ["HAI_AFX_DIR"])] if os.environ.get("HAI_AFX_DIR") else []
    for d in [*dirs, *AFX_DIRS]:
        if (d / "afx-rep.ckpt").is_file() and (d / "config.yaml").is_file():
            return d
    raise FileNotFoundError(
        "AFx-Rep 가중치(afx-rep.ckpt, config.yaml)를 찾을 수 없습니다. 공유 드라이브 model/afx-rep 폴더를 "
        "data/local/afx-rep 에 복사하거나, 폴더를 환경변수 HAI_AFX_DIR로 지정하세요.")


@lru_cache(maxsize=1)
def load_afx():
    import torch
    import yaml
    from stylematch.third_party.panns import Cnn14

    d = find_afx_dir()
    cfg = yaml.safe_load(open(d / "config.yaml", encoding="utf-8"))["model"]["init_args"]["encoder"]["init_args"]
    model = Cnn14(**cfg)
    ckpt = torch.load(d / "afx-rep.ckpt", map_location="cpu", weights_only=True)
    state = {k.replace("encoder.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    model.load_state_dict(state)
    model.eval()
    return model


def embed(audio, sample_rate):
    """모노 오디오 → 512차원 단위 벡터 (피크 정규화 후 48 kHz로 변환해 입력)."""
    import torch
    import torchaudio

    x = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(-1))[None, None]
    if sample_rate != MODEL_SR:
        x = torchaudio.functional.resample(x, int(sample_rate), MODEL_SR)
    x = x / x.abs().max().clamp(min=1e-8)
    with torch.no_grad():
        mid, _ = load_afx()(x)
    v = mid[0].numpy()
    return v / max(np.linalg.norm(v), 1e-12)


def embed_long(audio, sample_rate, seconds=5.5, windows=4):
    """긴 목표 소리: 음량 큰 구간 여러 개(겹치지 않게)의 임베딩 평균 → 단위 벡터."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(seconds * sample_rate)
    if len(audio) <= n:
        return embed(audio, sample_rate)
    starts = sorted(range(0, len(audio) - n + 1, int(sample_rate // 2)), key=lambda i: -np.mean(audio[i:i + n] ** 2))
    chosen = []
    for s in starts:
        if all(abs(s - c) >= n for c in chosen):
            chosen.append(s)
        if len(chosen) == windows:
            break
    v = np.mean([embed(audio[s:s + n], sample_rate) for s in chosen], axis=0)
    return v / max(np.linalg.norm(v), 1e-12)


def afx_distance(a, b):
    """두 임베딩의 코사인 거리 (0 = 같은 이펙트 특성)."""
    return float(1.0 - np.dot(a, b))


def afx_metric(target, sample_rate, target_embedding=None, dropout=0.0, seed=0):
    """탐색용 거리 함수 metric_fn(wet, target). 목표 임베딩은 한 번만 계산해 재사용합니다.

    dropout > 0: 평가마다 임베딩 차원 일부를 무작위로 빼고 거리 계산 (ST-ITO 원 코드와 같은 방식).
      탐색이 거리 숫자 하나만 줄이는 쪽(metric hacking)으로 흐르는 것을 막습니다.
    search_and_rank_chains(..., metric_fn=afx_metric(target, sr)) 처럼 넘깁니다.
    """
    te = target_embedding if target_embedding is not None else embed(target, sample_rate)
    rng = np.random.default_rng(seed)

    def metric(wet, _target=None):
        e = embed(wet, sample_rate)
        if dropout <= 0:
            return afx_distance(e, te)
        keep = rng.random(len(e)) >= dropout
        a, b = e[keep], te[keep]
        return float(1.0 - np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    return metric
