import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "demo" / "samples"

st.set_page_config(page_title="Tone Match · 기타 톤 데모", page_icon="🎸", layout="wide")

st.markdown("""<style>
.stApp { background: #0e1118; }
.block-container { max-width: 1180px; padding-top: 3rem; }
h1, h2, h3 { letter-spacing: -0.035em; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
.eyebrow { color: #b6f36a; letter-spacing: .18em; font-size: 12px; font-weight: 700; }
.chain { padding: 22px; background: #18221b; border: 1px solid #38513b;
         border-radius: 12px; color: #c7f6a0; font-size: 18px; line-height: 1.8; }
</style>
""", unsafe_allow_html=True)

AVAILABLE_DRIVES = {
    "808_scream": "808 Scream (TS808)",
    "b_drive": "B-Drive (BD-2)",
    "o_drive": "O-Drive (OD-1)",
    "one_distortion": "One Distortion (DS-1)",
    "bat_distortion": "BAT Distortion (RAT)",
    "f_fuzz": "F-Fuzz (Fuzz Face)",
    "m_fuzz": "M-Fuzz (Big Muff)",
}

# VST 플러그인은 메인 스레드에서만 동작하므로 별도 프로세스로 실행
WORKER_CODE = """
import json, sys
import soundfile as sf
from effects import load_effects, render_chain
from search.pipeline import search_and_rank_chains

with open(sys.argv[1], "r", encoding="utf-8") as f:
    req = json.load(f)

dry_audio, sr = sf.read(req["dry_path"], dtype="float32")
target_audio, _ = sf.read(req["target_path"], dtype="float32")
if dry_audio.ndim > 1:
    dry_audio = dry_audio.mean(axis=1)
if target_audio.ndim > 1:
    target_audio = target_audio.mean(axis=1)

effects_def = load_effects()
candidate_chains = [[d] for d in req["selected_inventory"]]

res = search_and_rank_chains(
    dry_audio=dry_audio,
    sr=sr,
    target_audio=target_audio,
    candidate_chains=candidate_chains,
    owned_effect_types=req["selected_inventory"],
    effects_def=effects_def,
    budget=req["budget"],
    n_restarts=2,
)
ranked = res[0] if isinstance(res, tuple) else res
best = ranked[0]

try:
    found_wet = render_chain(dry_audio, sr, best["chain"], effects=effects_def)
except TypeError:
    found_wet = render_chain(dry_audio, sr, best["chain"])

sf.write(req["out_wav"], found_wet, sr)

chain_data = best.get("chain", [])
if isinstance(chain_data, dict):
    chain_data = chain_data.get("chain", [])

out = {
    "effect_types": best["effect_types"],
    "chain": chain_data,
    "loss": float(best["loss"]),
}
with open(req["out_json"], "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
"""


def render_chain_cards(chain_items):
    if not chain_items:
        st.info("표시할 체인 정보가 없습니다.")
        return
    cols = st.columns(min(len(chain_items), 3))
    for i, item in enumerate(chain_items):
        with cols[i % len(cols)]:
            with st.container(border=True):
                ename = item.get("effect_type", "unknown")
                st.markdown(f"**{i + 1:02d} · {AVAILABLE_DRIVES.get(ename, ename)}**")
                for k, v in item.get("values", {}).items():
                    fval = float(v)
                    st.write(f"{k} · **{fval:.1f}**")
                    pct = int(min(max(fval * 100 if fval <= 1.0 else fval, 0), 100))
                    st.progress(pct)


st.markdown('<div class="eyebrow">GUITAR EFFECT CHAIN LAB / TONE MATCH</div>', unsafe_allow_html=True)
st.title("내 이펙터로 찾는 목표 톤")
st.write("레퍼런스 기타 톤을 분석해 최적의 이펙터 조합과 노브 값을 역추정합니다.")

tab1, tab2 = st.tabs(["📊 빠른 청취 비교 (사전 분석)", "🔍 실시간 톤 탐색 (Local)"])

# ----------------------------------------------------------------------
# [탭 1] 빠른 청취 비교 (로컬 & Streamlit Cloud 모두 동작)
# ----------------------------------------------------------------------
with tab1:
    sample_jsons = sorted(SAMPLE_DIR.glob("*_result.json")) if SAMPLE_DIR.exists() else []
    if not sample_jsons:
        st.warning(
            f"시연용 샘플이 아직 없습니다. (`{SAMPLE_DIR}`)\n\n"
            "터미널에서 아래 명령어를 실행해 샘플을 먼저 생성해 주세요:\n"
            "`python -m search.evaluate_dataset --limit 3 --budget 60 --save_audio`"
        )
    else:
        sample_ids = [f.stem.replace("_result", "") for f in sample_jsons]
        selected_id = st.selectbox("시연 샘플 선택", sample_ids)

        with open(SAMPLE_DIR / f"{selected_id}_result.json", "r", encoding="utf-8") as f:
            result_data = json.load(f)

        st.subheader("01 · 톤 비교")
        tracks = [
            ("Dry · 기타 원음", "이펙터 적용 전 원본 기타", f"{selected_id}_dry.wav"),
            ("정답 · 목표 톤", "재현하고 싶은 기준 사운드", f"{selected_id}_target.wav"),
            ("결과 · AI 재현 톤", "AI 추천 체인으로 렌더링한 결과", f"{selected_id}_found.wav"),
        ]
        for col, (title, desc, fname) in zip(st.columns(3), tracks):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {title}")
                    st.caption(desc)
                    fpath = SAMPLE_DIR / fname
                    if fpath.exists():
                        st.audio(str(fpath), format="audio/wav")
                    else:
                        st.info(f"{fname} 없음")

        st.subheader("02 · AI 추정 체인과 노브 값")
        chain_items = result_data.get("chain", [])
        if isinstance(chain_items, dict):
            chain_items = chain_items.get("chain", [])
        etypes = [AVAILABLE_DRIVES.get(item["effect_type"], item["effect_type"]) for item in chain_items]
        chain_str = " → ".join(["기타 (Dry)"] + etypes + ["앰프 / 출력"])
        st.markdown(f'<div class="chain">{chain_str}</div>', unsafe_allow_html=True)
        st.write("")

        m1, m2, m3 = st.columns(3)
        if "gt_chain" in result_data:
            m1.metric("정답 이펙터", " → ".join(result_data["gt_chain"]))
        if "loss" in result_data:
            m2.metric("유사도 거리 (Loss)", f"{result_data['loss']:.4f}")
        if "elapsed_sec" in result_data:
            m3.metric("탐색 소요 시간", f"{result_data['elapsed_sec']:.1f}초")

        render_chain_cards(chain_items)

# ----------------------------------------------------------------------
# [탭 2] 실시간 톤 탐색 (로컬 전용)
# ----------------------------------------------------------------------
with tab2:
    st.subheader("01 · 보유 이펙터 및 탐색 설정")
    col_inv, col_opt = st.columns([2, 1])
    with col_inv:
        st.write("**보유 이펙터 선택**")
        selected_inventory = []
        chk_cols = st.columns(4)
        for i, (key, label) in enumerate(AVAILABLE_DRIVES.items()):
            if chk_cols[i % 4].checkbox(label, value=(i < 3), key=f"inv_{key}"):
                selected_inventory.append(key)
    with col_opt:
        st.info("💡 **Local 전용 기능**\n\nVST 플러그인과 PyCMA가 설치된 로컬 PC에서만 동작합니다.")
        budget = st.slider("탐색 정밀도 (Budget)", 20, 120, 40)

    st.subheader("02 · 오디오 업로드 및 실시간 매칭")
    c_up1, c_up2 = st.columns(2)
    with c_up1:
        dry_file = st.file_uploader("원본(Dry) 오디오 업로드 (.wav)", type=["wav"])
    with c_up2:
        target_file = st.file_uploader("목표(Target Wet) 오디오 업로드 (.wav)", type=["wav"])

    if st.button("🚀 AI 톤 매칭 시작 (Search & Refine)", use_container_width=True):
        if not dry_file or not target_file:
            st.error("원본(Dry)과 목표(Target) 오디오 파일을 모두 업로드해 주세요.")
        elif not selected_inventory:
            st.error("최소 1개 이상의 보유 이펙터를 선택해 주세요.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                dry_path = tmp_path / "dry.wav"
                tgt_path = tmp_path / "target.wav"
                out_wav = tmp_path / "found.wav"
                out_json = tmp_path / "result.json"
                req_json = tmp_path / "req.json"

                dry_path.write_bytes(dry_file.getvalue())
                tgt_path.write_bytes(target_file.getvalue())
                req_json.write_text(
                    json.dumps({
                        "dry_path": str(dry_path),
                        "target_path": str(tgt_path),
                        "selected_inventory": selected_inventory,
                        "budget": budget,
                        "out_wav": str(out_wav),
                        "out_json": str(out_json),
                    }),
                    encoding="utf-8",
                )

                with st.spinner("AI가 보유 이펙터 중에서 최적의 노브를 탐색 중입니다... (약 20~40초 소요)"):
                    proc = subprocess.run(
                        [sys.executable, "-c", WORKER_CODE, str(req_json)],
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                    )

                if proc.returncode != 0:
                    st.error("실행 중 오류가 발생했습니다.")
                    st.code(proc.stderr or proc.stdout)
                else:
                    best = json.loads(out_json.read_text(encoding="utf-8"))
                    st.success(
                        f"탐색 완료! 최적 체인: {' → '.join(best['effect_types'])} "
                        f"(Loss: {best['loss']:.4f})"
                    )
                    a1, a2, a3 = st.columns(3)
                    with a1:
                        st.write("🎸 Dry 원음")
                        st.audio(dry_file.getvalue(), format="audio/wav")
                    with a2:
                        st.write("🎯 목표 톤")
                        st.audio(target_file.getvalue(), format="audio/wav")
                    with a3:
                        st.write("✨ AI 재현 결과")
                        st.audio(out_wav.read_bytes(), format="audio/wav")

                    render_chain_cards(best.get("chain", []))

st.divider()
st.caption("HAI 2026-2 Project Team 2 · CNN-guided CMA-ES Guitar Effect Chain Estimation")