"""
stylematch/listening.py

청취 비교 페이지 생성 (목표 vs 우리 결과, 순위대로).

곡마다: 목표(분리 기타) → 유사 설정 1~3위(페달 · 노브 · 크런치 표시) → 대조(1위 페달, 기본 노브)
- --eq: 각 순위 아래에 평균 스펙트럼 맞추기 EQ 적용본(top*_eq.wav)도 함께 (spectrum_eq.py)
- 대조도 목표 음량(RMS)에 맞춤 → 음량 차이 없이 노브 보정 효과만 비교

사용 예
-------
python -m stylematch.listening --dir outputs/stylematch/step3_v1 --kind separated
→ <dir>/listening_<kind>.html 을 브라우저로 열기
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from effects import load_effects, render_chain
from stylematch.evaluate import common_dry
from stylematch.spectrum_eq import apply_spectrum_eq

STYLE = """:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#999;--line:#333}}
body{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px}
h2{font-size:1.05rem;margin:28px 0 8px;border-top:1px solid var(--line);padding-top:16px}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:6px 0}
.lab{width:110px;font-weight:600}.set{color:var(--muted);font-size:.9rem}audio{height:34px;max-width:100%}"""


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64))))) + 1e-12


def knob_text(c):
    knobs = ", ".join(f"{k} {v:.0f}" for k, v in c["knobs"].items())
    return f"{c['pedal'] or c['drive']} – {knobs} · 앰프 크런치 {c['amp_crunch_db']:.1f} dB · AFx 거리 {c['afx_distance']:.2f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True); p.add_argument("--kind", default="separated")
    p.add_argument("--eq", action="store_true", help="평균 스펙트럼 맞추기 EQ 적용본도 함께")
    a = p.parse_args(argv)
    root = Path(a.dir)
    effects = load_effects()
    dry, sr = common_dry(5.5)
    rows = []
    for rj in sorted((root / a.kind).glob("*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        folder = rj.parent
        target, _ = sf.read(folder / "target.wav", dtype="float32")
        top = r["candidates"][0]["drive"]
        base_path = folder / "baseline_default.wav"
        if not base_path.is_file():
            chain = {"amp_id": "marshall_v30_ir_v1", "chain": [
                {"effect_type": top, "values": {q["name"]: q["default"] for q in effects[top]["params"]}}]}
            base = np.asarray(render_chain(dry, sr, chain), np.float32)
            sf.write(base_path, base * (rms(target) / rms(base)), sr, subtype="FLOAT")
        src = lambda f: html.escape(f"{a.kind}/{folder.name}/{f}")
        item = lambda lab, f, text="": (f'<div class="row"><span class="lab">{lab}</span>'
                                        f'<audio controls preload="none" src="{src(f)}"></audio>'
                                        f'<span class="set">{html.escape(text)}</span></div>')
        block = [f"<h2>{html.escape(r['id'])}</h2>", item("목표", "target.wav", "원곡에서 분리한 기타")]
        for c in r["candidates"]:
            block.append(item(f"{c['rank']}위", f"top{c['rank']}.wav", knob_text(c)))
            if a.eq:
                x, _ = sf.read(folder / f"top{c['rank']}.wav", dtype="float32")
                y, g = apply_spectrum_eq(x, target, sr)
                sf.write(folder / f"top{c['rank']}_eq.wav", y, sr, subtype="FLOAT")
                block.append(item(f"{c['rank']}위 + EQ", f"top{c['rank']}_eq.wav",
                                  f"평균 스펙트럼 EQ (최대 +{g.max():.0f} / {g.min():.0f} dB)"))
        ref = effects[top].get("reference_pedal") or {}
        block.append(item("대조", "baseline_default.wav",
                          f"{ref.get('brand', '')} {ref.get('model', '')} 기본 노브 (보정 전)".strip()))
        rows.append("\n".join(block))
    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>청취 비교</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{STYLE}</style></head><body>'
            f'<h1>청취 비교 – {html.escape(a.kind)}</h1>'
            f'<p class="set">dry는 공통 IDMT 스트랫 프레이즈(목표와 연주가 다름). 결과·대조는 목표 음량에 맞춤.'
            + (' "+ EQ"는 목표의 평균 스펙트럼에 맞춘 EQ 적용본.' if a.eq else "") + '</p>'
            + "\n".join(rows) + "</body></html>")
    out = root / f"listening_{a.kind}.html"
    out.write_text(page, encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
