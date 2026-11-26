# demo/app_service.py
import json, subprocess, sys, tempfile
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIRS = [ROOT / "outputs" / "stylematch", ROOT / "demo" / "samples"]

st.set_page_config(page_title="ToneMatch · 내 페달로 찾는 기타 톤", page_icon="🎸", layout="wide")

st.markdown("""<style>
.stApp { background: #0d1017; color: #f1f5f9; }
.block-container { max-width: 1100px; padding-top: 2.2rem; }
.brand { color: #a3e635; font-size: 13px; font-weight: 700; letter-spacing: .12em; }
.subtitle { color: #94a3b8; font-size: 16px; margin-bottom: 1.5rem; }
.signal-flow {
    padding: 16px 20px; background: #161d29; border: 1px solid #2a364b;
    border-radius: 12px; color: #d9f99d; font-size: 16px; font-weight: 600;
}
.pedal-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 700; background: #26331b; color: #bef264;
}
</style>""", unsafe_allow_html=True)

PEDALS = {
    "808_scream": "Ibanez TS808 (오버드라이브)",
    "b_drive": "BOSS BD-2 (블루스 드라이브)",
    "o_drive": "BOSS OD-1 (오버드라이브)",
    "one_distortion": "BOSS DS-1 (디스토션)",
    "bat_distortion": "ProCo RAT (디스토션)",
    "f_fuzz": "Dunlop Fuzz Face (퍼즈)",
    "m_fuzz": "EHX Big Muff (퍼즈)",
    "amp_crunch": "앰프 크런치 (Amp Crunch)",
}

MULTI_FX = {
    "단품 페달 직접 선택": [],
    "BOSS GT-1 (내장 드라이브 6종 자동 선택)": ["808_scream", "b_drive", "o_drive", "one_distortion", "bat_distortion", "m_fuzz"],
    "Line 6 HX Stomp (내장 드라이브 5종 자동 선택)": ["808_scream", "b_drive", "bat_distortion", "f_fuzz", "m_fuzz"],
    "Zoom G1 Four (내장 드라이브 4종 자동 선택)": ["808_scream", "o_drive", "one_distortion", "m_fuzz"],
}

KNOB_LABELS = {
    "level": "볼륨 (Level)", "volume": "볼륨 (Volume)",
    "overdrive": "드라이브 (Overdrive)", "gain": "게인 (Gain)", "drive": "드라이브 (Drive)",
    "distortion": "디스토션 (Distortion)", "sustain": "서스테인 (Sustain)",
    "tone": "톤 (Tone)", "filter": "필터 (Filter)", "fuzz": "퍼즈 (Fuzz)",
    "drive_db": "앰프 게인 (Drive dB)",
}

MATCH_WORKER = """
import json, sys, soundfile as sf
from stylematch.match import match_tone, match_effects, render_match

req = json.loads(open(sys.argv[1], encoding="utf-8").read())
dry, sr = sf.read(req["dry"], dtype="float32")
tgt, _ = sf.read(req["target"], dtype="float32")
if dry.ndim > 1: dry = dry.mean(axis=1)
if tgt.ndim > 1: tgt = tgt.mean(axis=1)

res = match_tone(dry, tgt, sr, k=3, budget=req["budget"], allowed=req["allowed"], room=req["room"])
eff = match_effects()
cands = []
for idx, c in enumerate(res["candidates"], 1):
    wav_path = req["out_dir"] + f"/top{idx}.wav"
    sf.write(wav_path, render_match(dry, sr, c["chain"], eff), sr)
    cands.append({"rank": idx, "effect_type": c["effect_type"], "afx_distance": float(c["afx_distance"]),
                  "chain": c["chain"], "wav": wav_path})
open(req["out_json"], "w", encoding="utf-8").write(json.dumps({"candidates": cands}, indent=2, ensure_ascii=False))
"""

RENDER_WORKER = """
import json, sys, soundfile as sf
from stylematch.match import match_effects, render_match

req = json.loads(open(sys.argv[1], encoding="utf-8").read())
dry, sr = sf.read(req["dry"], dtype="float32")
if dry.ndim > 1: dry = dry.mean(axis=1)
eff = match_effects()
sf.write(req["out_wav"], render_match(dry, sr, req["chain"], eff), sr)
"""


def scan_presets():
    items = {}
    for base in SAMPLE_DIRS:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*result*.json")):
            name = p.parent.name if p.stem == "result" else p.stem.replace("_result", "")
            items[name] = p
    return items


def render_interactive_knobs(chain_obj, dry_path: Path | None, key_prefix: str):
    items = chain_obj.get("chain", chain_obj) if isinstance(chain_obj, dict) else chain_obj
    if not items:
        return

    names = [PEDALS.get(x["effect_type"], x["effect_type"]) for x in items]
    st.markdown(f'<div class="signal-flow">🎸 기타 원음  →  {"  →  ".join(names)}  →  🔊 앰프 (캐비닛 IR)</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown("#### 🎛️ 추천 세팅 확인 및 미세조정")
    st.caption("AI가 찾은 노브 값을 그대로 쓰거나, 슬라이더를 움직여 내 취향대로 소리를 다시 들어볼 수 있습니다.")

    edited_items = []
    for idx, it in enumerate(items):
        with st.container(border=True):
            ename = PEDALS.get(it["effect_type"], it["effect_type"])
            st.markdown(f'<span class="pedal-badge">이펙터 {idx+1}</span> &nbsp; **{ename}**', unsafe_allow_html=True)
            st.write("")
            vals = it.get("values", {})
            cols = st.columns(max(len(vals), 1))
            new_vals = {}
            for j, (k, v) in enumerate(vals.items()):
                fv = float(v)
                hi = 1.0 if (0 <= fv <= 1.0 and k not in KNOB_LABELS) else 100.0
                label = KNOB_LABELS.get(k, k)
                new_vals[k] = cols[j % len(cols)].slider(label, 0.0, hi, min(max(fv, 0.0), hi), key=f"{key_prefix}_{idx}_{k}")
            edited_items.append({"effect_type": it["effect_type"], "values": new_vals})

    if dry_path and dry_path.exists():
        if st.button("🔊 조정한 노브로 소리 다시 들어보기", key=f"btn_{key_prefix}", use_container_width=True):
            with tempfile.TemporaryDirectory() as td:
                out_w, req_p = Path(td) / "custom.wav", Path(td) / "req.json"
                req_p.write_text(json.dumps({"dry": str(dry_path), "chain": edited_items, "out_wav": str(out_w)}), encoding="utf-8")
                with st.spinner("노브 설정을 적용해 소리를 만드는 중..."):
                    proc = subprocess.run([sys.executable, "-c", RENDER_WORKER, str(req_p)], cwd=str(ROOT), capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("적용 완료! 아래에서 바뀐 소리를 들어보세요.")
                    st.audio(out_w.read_bytes(), format="audio/wav")
                else:
                    st.error("소리 생성 중 오류가 발생했습니다.")
                    st.code(proc.stderr)


# --- 헤더 ---
st.markdown('<div class="brand">TONEMATCH AI</div>', unsafe_allow_html=True)
st.title("내 이펙터로 찾는 목표 기타 톤")
st.markdown('<div class="subtitle">따라 하고 싶은 기타 소리를 들려주면, 내가 가진 페달로 가장 비슷한 노브 세팅을 찾아드립니다.</div>', unsafe_allow_html=True)

tab_demo, tab_live = st.tabs(["🎧 샘플 곡으로 바로 체험하기", "🎸 내 음원으로 직접 찾기"])

# ======================================================================
# 탭 1: 샘플 곡으로 바로 체험하기 (사전 분석 결과 + 즉시 노브 미세조정)
# ======================================================================
with tab_demo:
    presets = scan_presets()
    if not presets:
        st.info("아직 준비된 샘플 곡이 없습니다. 오른쪽 [내 음원으로 직접 찾기] 탭을 이용해 보세요.")
    else:
        sel_name = st.selectbox("체험할 곡 / 기타 톤 선택", list(presets.keys()))
        rpath = presets[sel_name]
        data = json.loads(rpath.read_text(encoding="utf-8"))
        prefix = rpath.stem.replace("_result", "")
        cands = data.get("candidates")

        if cands:
            rank = st.radio(
                "AI 추천 순위 (가장 비슷한 3가지 조합)",
                range(len(cands)),
                format_func=lambda i: f"추천 {i+1}위 · {PEDALS.get(cands[i]['effect_type'], cands[i]['effect_type'])}",
                horizontal=True,
            )
            chosen = cands[rank]
            dry_f, tgt_f, res_f = rpath.parent / "dry.wav", rpath.parent / "target.wav", rpath.parent / f"top{rank+1}.wav"
        else:
            chosen = data
            dry_f, tgt_f, res_f = rpath.parent / f"{prefix}_dry.wav", rpath.parent / f"{prefix}_target.wav", rpath.parent / f"{prefix}_found.wav"

        st.markdown("#### 🔊 소리 비교해보기")
        c1, c2, c3 = st.columns(3)
        for col, title, desc, fp in [
            (c1, "① 기타 원음", "이펙터를 끄고 친 생소리", dry_f),
            (c2, "② 목표 톤", "따라 하고 싶은 기준 소리", tgt_f),
            (c3, "③ AI 추천 결과", "추천 세팅을 적용한 소리", res_f),
        ]:
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(desc)
                    if fp.exists():
                        st.audio(str(fp))
                    else:
                        st.caption("오디오 파일 없음")

        st.write("")
        render_interactive_knobs(chosen.get("chain", []), dry_f, key_prefix=f"preset_{sel_name}")

# ======================================================================
# 탭 2: 내 음원으로 직접 찾기 (장비 등록 → 업로드 → 탐색 → 미세조정)
# ======================================================================
with tab_live:
    st.markdown("#### 1단계 · 내가 가진 이펙터 선택")
    col_eq, col_opt = st.columns([2, 1])
    with col_eq:
        mfx = st.selectbox("멀티이펙터를 쓰신다면 기종을 골라주세요", list(MULTI_FX.keys()))
        preset_set = set(MULTI_FX[mfx])
        owned = []
        chk_cols = st.columns(2)
        for i, (k, name) in enumerate(PEDALS.items()):
            default_on = (k in preset_set) if preset_set else (i < 3)
            if chk_cols[i % 2].checkbox(name, value=default_on, key=f"srv_{mfx}_{k}"):
                owned.append(k)
    with col_opt:
        with st.container(border=True):
            st.markdown("**탐색 옵션**")
            unlock = st.toggle("내 페달 외에 더 비슷한 페달도 추천받기", value=False)
            room = st.toggle("공간감(리버브)이 큰 곡 보정", value=False)
            quality = st.select_slider("탐색 속도 / 정밀도", options=["빠르게", "보통 (추천)", "정밀하게"], value="보통 (추천)")
            budget_map = {"빠르게": 20, "보통 (추천)": 30, "정밀하게": 50}

    st.markdown("#### 2단계 · 오디오 파일 올리기")
    u1, u2 = st.columns(2)
    dry_up = u1.file_uploader("내 기타 연주 파일 (이펙터 없는 원음, .wav)", type=["wav"])
    tgt_up = u2.file_uploader("따라 하고 싶은 목표 기타 소리 (.wav)", type=["wav"])

    if st.button("✨ 내 페달로 최적의 세팅 찾기", type="primary", use_container_width=True):
        if not dry_up or not tgt_up:
            st.error("내 기타 연주 파일과 목표 기타 소리 파일을 모두 올려주세요.")
        elif not unlock and not owned:
            st.error("보유한 이펙터를 최소 1개 이상 선택하거나, '더 비슷한 페달도 추천받기'를 켜주세요.")
        else:
            td = tempfile.mkdtemp()
            dp, tp = Path(td) / "dry.wav", Path(td) / "target.wav"
            dp.write_bytes(dry_up.getvalue())
            tp.write_bytes(tgt_up.getvalue())
            req_p, out_p = Path(td) / "req.json", Path(td) / "out.json"
            req_p.write_text(json.dumps({
                "dry": str(dp), "target": str(tp), "budget": budget_map[quality],
                "room": room, "allowed": None if unlock else owned,
                "out_dir": td, "out_json": str(out_p),
            }), encoding="utf-8")
            with st.spinner("AI가 목표 톤을 분석하고 최적의 노브 조합을 찾는 중입니다... (약 20~40초)"):
                proc = subprocess.run([sys.executable, "-c", MATCH_WORKER, str(req_p)], cwd=str(ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                st.error("분석 중 오류가 발생했습니다. (AFx-Rep 모델 가중치 및 패키지 설치 상태를 확인해 주세요.)")
                st.code(proc.stderr or proc.stdout)
            else:
                st.session_state["srv_res"] = json.loads(out_p.read_text(encoding="utf-8"))
                st.session_state["srv_dry"] = dp
                st.session_state["srv_tgt"] = tp

    if "srv_res" in st.session_state:
        st.divider()
        st.markdown("#### 3단계 · AI 추천 세팅 결과")
        cands = st.session_state["srv_res"]["candidates"]
        pick = st.radio(
            "추천 순위 선택",
            range(len(cands)),
            format_func=lambda i: f"추천 {i+1}위 · {PEDALS.get(cands[i]['effect_type'], cands[i]['effect_type'])}",
            horizontal=True,
        )
        chosen = cands[pick]
        a1, a2, a3 = st.columns(3)
        a1.markdown("**① 내 기타 원음**"); a1.audio(str(st.session_state["srv_dry"]))
        a2.markdown("**② 목표 톤**"); a2.audio(str(st.session_state["srv_tgt"]))
        a3.markdown("**③ AI 추천 결과**"); a3.audio(chosen["wav"])
        render_interactive_knobs(chosen["chain"], st.session_state["srv_dry"], key_prefix=f"live_{pick}")