"""
stylematch/head.py

AFx-Rep 보정층: 고정된 AFx-Rep 지문(512) 위에 작은 층을 얹어 우리 7종 드라이브 기준으로 다시 정렬합니다.
- 학습: 드라이브 분류 + 게인·톤 회귀 (녹음 환경 무작위화한 소리의 지문으로)
- 거리: 보정층 출력 특징(128, 단위 벡터)의 코사인 거리 → AFx-Rep 거리 대신 사용 가능

단계
1) embed : wetonly 렌더(앰프 없음)에 IR·EQ·크런치를 무작위로 입혀 AFx-Rep 지문 저장 (npz)
2) train : 보정층 학습 → head.pt
사용 예
-------
python -m stylematch.head embed --data data/local/wetonly_v1 --out data/local/afx_head/embeds.npz --variants 2
python -m stylematch.head train --embeds data/local/afx_head/embeds.npz --out data/local/afx_head/head.pt
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np

from estimation.chain import DRIVES

LEVEL, TONE = {"level", "volume"}, {"tone", "filter"}


def randomize_env(x, sr, irs, rng):
    """녹음 환경 무작위화: 앰프 크런치 → 캐비닛 IR(또는 없음) → 앰프·마이크 EQ."""
    from scipy.signal import fftconvolve
    from pedalboard import Pedalboard, LowShelfFilter, HighShelfFilter, PeakFilter, HighpassFilter, LowpassFilter
    from stylematch.match import apply_crunch

    if rng.random() < 0.7:
        x = apply_crunch(x, rng.uniform(0, 12))
    if irs and rng.random() < 0.85:
        x = fftconvolve(x, rng.choice(irs))[: len(x)].astype(np.float32)
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=rng.uniform(50, 160)),
        LowShelfFilter(cutoff_frequency_hz=rng.uniform(120, 400), gain_db=rng.uniform(-6, 6)),
        PeakFilter(cutoff_frequency_hz=float(np.exp(rng.uniform(np.log(400), np.log(3500)))),
                   gain_db=rng.uniform(-6, 6), q=rng.uniform(0.5, 1.5)),
        HighShelfFilter(cutoff_frequency_hz=rng.uniform(2000, 6000), gain_db=rng.uniform(-8, 5)),
        LowpassFilter(cutoff_frequency_hz=rng.uniform(5000, 14000)),
    ])
    return board(np.asarray(x, np.float32), sr)


def load_irs(sr):
    import soundfile as sf
    from effects.render import IR_DIRS
    irs = []
    for d in IR_DIRS:
        for f in sorted(Path(d).rglob("*.wav")):
            h, r = sf.read(f, dtype="float32", always_2d=True)
            h = h.mean(1)
            if r != sr:
                import librosa
                h = librosa.resample(h, orig_sr=r, target_sr=sr)
            irs.append(h / np.sqrt(np.sum(h ** 2)))
    return irs


def do_embed(data, out, variants):
    import soundfile as sf
    from effects import load_effects
    from stylematch.afx import embed

    data, out = Path(data), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eff = load_effects()
    rng = random.Random(0)
    rows = list(csv.DictReader(open(data / "meta.csv", encoding="utf-8")))
    sr = 44100
    irs = load_irs(sr)
    E, Y, G, T, C = [], [], [], [], []
    for i, r in enumerate(rows):
        d = r["drive"]
        x, _ = sf.read(data / d / f"{r['clip']}.wav", dtype="float32")
        params = eff[d]["params"]
        gain = [p["name"] for p in params if p["name"] not in LEVEL | TONE][0]
        tone = next((p["name"] for p in params if p["name"] in TONE), None)
        for _ in range(variants):
            E.append(embed(randomize_env(x, sr, irs, rng), sr))
            Y.append(DRIVES.index(d)); G.append(float(r[f"k_{gain}"]) / 100)
            T.append(float(r[f"k_{tone}"]) / 100 if tone else np.nan); C.append(r["clip"])
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    np.savez(out, emb=np.array(E, np.float32), drive=np.array(Y), gain=np.array(G, np.float32),
             tone=np.array(T, np.float32), clip=np.array(C))
    print("saved", out, len(E))


class Head:
    """보정층 (torch 모듈 래퍼). features(): 단위 특징 벡터, 거리 계산용."""

    def __init__(self, path=None):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.body = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128))
        self.cls = nn.Linear(128, len(DRIVES))
        self.reg = nn.Linear(128, 2)
        if path:
            s = torch.load(path, map_location="cpu", weights_only=True)
            self.body.load_state_dict(s["body"]); self.cls.load_state_dict(s["cls"]); self.reg.load_state_dict(s["reg"])
        self.eval()

    def modules(self):
        return [self.body, self.cls, self.reg]

    def eval(self):
        for m in self.modules():
            m.eval()

    def train(self):
        for m in self.modules():
            m.train()

    def features(self, emb):
        with self.torch.no_grad():
            f = self.body(self.torch.as_tensor(np.atleast_2d(emb), dtype=self.torch.float32))
            f = self.torch.nn.functional.normalize(f, dim=-1)
        return f.numpy()


def do_train(embeds, out, epochs=60):
    import torch
    import torch.nn.functional as F

    z = np.load(embeds)
    clips = sorted(set(z["clip"].tolist())); random.Random(0).shuffle(clips)
    val_clips = set(clips[: len(clips) // 7])
    tr = np.array([c not in val_clips for c in z["clip"]])
    X = torch.tensor(z["emb"]); y = torch.tensor(z["drive"]); g = torch.tensor(z["gain"]); t = torch.tensor(z["tone"])
    head = Head()
    params = [p for m in head.modules() for p in m.parameters()]
    opt = torch.optim.AdamW(params, 2e-3, weight_decay=1e-3)
    torch.manual_seed(0)
    idx = np.where(tr)[0]; vidx = np.where(~tr)[0]

    def run(ids, train):
        head.train() if train else head.eval()
        f = F.normalize(head.body(X[ids]), dim=-1)
        logits = head.cls(f); reg = torch.sigmoid(head.reg(f))
        tm = ~torch.isnan(t[ids])
        loss = F.cross_entropy(logits, y[ids]) + 2 * F.mse_loss(reg[:, 0], g[ids])
        if tm.any():
            loss = loss + 2 * F.mse_loss(reg[tm, 1], t[ids][tm])
        acc = (logits.argmax(1) == y[ids]).float().mean().item()
        return loss, acc

    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(idx)
        for b in range(0, len(perm), 128):
            loss, _ = run(perm[b:b + 128], True)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                vl, va = run(vidx, False)
            print(f"epoch {ep + 1} val loss {vl.item():.3f} drive acc {va * 100:.1f}%", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"body": head.body.state_dict(), "cls": head.cls.state_dict(), "reg": head.reg.state_dict()}, out)
    print("saved", out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["embed", "train"])
    p.add_argument("--data"); p.add_argument("--embeds"); p.add_argument("--out", required=True)
    p.add_argument("--variants", type=int, default=2); p.add_argument("--epochs", type=int, default=60)
    a = p.parse_args(argv)
    if a.mode == "embed":
        do_embed(a.data, a.out, a.variants)
    else:
        do_train(a.embeds, a.out, a.epochs)


if __name__ == "__main__":
    main()
