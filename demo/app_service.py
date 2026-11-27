# demo/app_service.py
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = Path(os.environ["HAI_DEMO_DIR"]) if os.environ.get("HAI_DEMO_DIR") else (ROOT / "results" / "demo_bundle_v1")
STEP2_DIR = ROOT / "demo" / "samples"

st.set_page_config(page_title="ToneMatch · 내 페달로 찾는 기타 톤", page_icon="🎸", layout="wide")

st.markdown("""<style>
.stApp { background: #0d1017; color: #f1f5f9; }
.block-container { max-width: 1180px; padding-top: 2rem; }
.brand { color: #a3e635; font-size: 13px; font-weight: 700; letter-spacing: .12em; }
.subtitle { color: #94a3b8; font-size: 15px; margin-bottom: 1.2rem; }
.signal-flow {
    padding: 14px 18px; background: #161d29; border: 1px solid #2a364b;
    border-radius: 12px; color: #d9f99d; font-size: 15px; font-weight: 600;
}
.pedal-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 700; background: #26331b; color: #bef264;
}
.clock-tag {
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 12px; font-weight: 600; background: #1e293b; color: #e2e8f0;
}
.crunch-note { color: #cbd5e1; font-size: 13px; margin-top: -6px; }
</style>""", unsafe_allow_html=True)

PEDALS = {
    "808_scream": "Ibanez TS808 (오버드라이브)",
    "b_drive": "BOSS BD-2 (블루스 드라이브)",
    "o_drive": "BOSS OD-1 (오버드라이브)",
    "one_distortion": "BOSS DS-1 (디스토션)",
    "bat_distortion": "ProCo RAT (디스토션)",
    "f_fuzz": "Dallas Arbiter Fuzz Face (퍼즈)",
    "m_fuzz": "EHX Big Muff (퍼즈)",
    "amp_crunch": "앰프 크런치 (Amp Crunch)",
}

DRIVE_KEYS = [
    "808_scream", "b_drive", "o_drive",
    "one_distortion", "bat_distortion", "f_fuzz", "m_fuzz",
]

KNOB_LABELS = {
    "level": "Level", "volume": "Volume",
    "overdrive": "Overdrive", "gain": "Gain", "drive": "Drive",
    "distortion": "Distortion", "sustain": "Sustain",
    "tone": "Tone", "filter": "Filter", "fuzz": "Fuzz",
    "drive_db": "앰프 게인 (Drive dB)",
}

ROOM_OPTIONS = {
    "없음": {"suffix": "", "mix": 0.0, "size": 0.0},
    "약": {"suffix": "_room1", "mix": 0.20, "size": 0.50},
    "강": {"suffix": "_room2", "mix": 0.35, "size": 0.75},
}

MATCH_WORKER = """
import json, sys, soundfile as sf
from stylematch.match import match_tone, match_effects, render_match, apply_room

req = json.loads(open(sys.argv[1], encoding="utf-8").read())
dry, sr = sf.read(req["dry"], dtype="float32")
tgt, _ = sf.read(req["target"], dtype="float32")
if dry.ndim > 1: dry = dry.mean(axis=1)
if tgt.ndim > 1: tgt = tgt.mean(axis=1)

res = match_tone(dry, tgt, sr, k=3, budget=req["budget"], allowed=req["allowed"], room=False)
eff = match_effects()
cands = []
for idx, c in enumerate(res["candidates"], 1):
    wet = render_match(dry, sr, c["chain"], eff)
    base = req["out_dir"] + f"/top{idx}"
    sf.write(base + ".wav", wet, sr)
    sf.write(base + "_room1.wav", apply_room(wet, sr, 0.20, 0.50), sr)
    sf.write(base + "_room2.wav", apply_room(wet, sr, 0.35, 0.75), sr)
    cands.append({
        "rank": idx,
        "effect_type": c["effect_type"],
        "afx_distance": float(c["afx_distance"]),
        "chain": c["chain"],
        "wav": base + ".wav",
        "wav_room1": base + "_room1.wav",
        "wav_room2": base + "_room2.wav",
    })
open(req["out_json"], "w", encoding="utf-8").write(json.dumps({"candidates": cands}, indent=2, ensure_ascii=False))
"""

RENDER_WORKER = """
import json, sys, soundfile as sf
from stylematch.match import match_effects, render_match, apply_room

req = json.loads(open(sys.argv[1], encoding="utf-8").read())
dry, sr = sf.read(req["dry"], dtype="float32")
if dry.ndim > 1: dry = dry.mean(axis=1)
eff = match_effects()
wet = render_match(dry, sr, req["chain"], eff)
mix = float(req.get("room_mix", 0.0))
size = float(req.get("room_size", 0.0))
if mix > 0.0:
    wet = apply_room(wet, sr, mix, size)
sf.write(req["out_wav"], wet, sr)
"""


def pct_to_clock(pct: float) -> str:
    """0%=7시, 50%=12시, 100%=5시 (30분 단위 반올림)."""
    pct = max(0.0, min(100.0, float(pct)))
    hours = 7.0 + 10.0 * (pct / 100.0)
    half_steps = round(hours * 2)
    h = half_steps // 2
    is_half = (half_steps % 2) == 1
    if h > 12:
        h -= 12
    return f"{h}시 반" if is_half else f"{h}시"


def amp_crunch_desc(db: float) -> str:
    if db < 3.0:
        return "앰프는 클린 (0–3 dB)"
    if db < 8.0:
        return "앰프 게인을 살짝 올린 크런치 (3–8 dB)"
    return "앰프 게인을 꽤 올린 크런치 (8–15 dB)"


def scan_grouped_samples() -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {
        "실제 곡 (음원 분리 기타)": {},
        "실물 페달 데모": {},
        "같은 연주 조건": {},
    }
    sep_dir = BUNDLE_DIR / "separated"
    if sep_dir.exists():
        for p in sorted(sep_dir.glob("*/result.json")):
            groups["실제 곡 (음원 분리 기타)"][p.parent.name] = p

    iso_dir = BUNDLE_DIR / "isolated"
    if iso_dir.exists():
        for p in sorted(iso_dir.glob("*/result.json")):
            groups["실물 페달 데모"][p.parent.name] = p

    if STEP2_DIR.exists():
        for p in sorted(STEP2_DIR.glob("*_result.json")):
            groups["같은 연주 조건"][p.stem.replace("_result", "")] = p

    return groups


def render_interactive_knobs(chain_obj, dry_path: Path | None, room_choice: str, key_prefix: str):
    is_dict = isinstance(chain_obj, dict)
    items = chain_obj.get("chain", []) if is_dict else (chain_obj or [])
    if not items:
        return

    names = [PEDALS.get(x["effect_type"], x["effect_type"]) for x in items]
    st.markdown(
        f'<div class="signal-flow">🎸 기타 원음  →  {"  →  ".join(names)}  →  🔊 캐비닛 IR (고정)</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown("#### 🎛️ 설정 확인 및 노브 미세 조정")
    st.caption(
        "슬라이더를 움직여 내 취향대로 소리를 다시 들어볼 수 있습니다. "
        "시계 위치는 VST 노브 기준이라 실물 페달과 조금 다를 수 있으며, 미세 조정 결과는 음량이 달라질 수 있습니다."
    )

    edited_items = []
    for idx, it in enumerate(items):
        etype = it["effect_type"]
        with st.container(border=True):
            ename = PEDALS.get(etype, etype)
            badge = "앰프 게인 안내" if etype == "amp_crunch" else f"이펙터 {idx + 1}"
            st.markdown(f'<span class="pedal-badge">{badge}</span> &nbsp; **{ename}**', unsafe_allow_html=True)
            st.write("")

            vals = it.get("values", {})
            cols = st.columns(max(len(vals), 1))
            new_vals = {}
            for j, (k, v) in enumerate(vals.items()):
                fv = float(v)
                with cols[j % len(cols)]:
                    if etype == "amp_crunch" or k == "drive_db":
                        val = st.slider(
                            KNOB_LABELS.get(k, k), 0.0, 15.0, min(max(fv, 0.0), 15.0),
                            step=0.1, key=f"{key_prefix}_{idx}_{k}",
                        )
                        st.markdown(f'<div class="crunch-note">→ {amp_crunch_desc(val)}</div>', unsafe_allow_html=True)
                        new_vals[k] = val
                    else:
                        label = KNOB_LABELS.get(k, k)
                        val = st.slider(
                            label, 0.0, 100.0, min(max(fv, 0.0), 100.0),
                            step=0.5, key=f"{key_prefix}_{idx}_{k}",
                        )
                        st.markdown(
                            f'<span class="clock-tag">{label} {pct_to_clock(val)} ({val:.0f}%)</span>',
                            unsafe_allow_html=True,
                        )
                        new_vals[k] = val
            edited_items.append({"effect_type": etype, "values": new_vals})

    if dry_path and dry_path.exists():
        room_cfg = ROOM_OPTIONS[room_choice]
        btn_label = f"🔊 조정한 노브로 다시 듣기 (리버브: {room_choice})"
        if st.button(btn_label, key=f"btn_{key_prefix}", use_container_width=True):
            payload_chain = {**chain_obj, "chain": edited_items} if is_dict else edited_items
            with tempfile.TemporaryDirectory() as td:
                out_w, req_p = Path(td) / "custom.wav", Path(td) / "req.json"
                req_p.write_text(json.dumps({
                    "dry": str(dry_path),
                    "chain": payload_chain,
                    "out_wav": str(out_w),
                    "room_mix": room_cfg["mix"],
                    "room_size": room_cfg["size"],
                }), encoding="utf-8")
                with st.spinner("조정한 노브로 소리를 렌더링하는 중..."):
                    proc = subprocess.run(
                        [sys.executable, "-c", RENDER_WORKER, str(req_p)],
                        cwd=str(ROOT), capture_output=True, text=True,
                    )
                if proc.returncode == 0:
                    st.success(f"렌더링 완료 (리버브: {room_choice})")
                    st.audio(out_w.read_bytes(), format="audio/wav")
                else:
                    st.error("소리 생성 중 오류가 발생했습니다.")
                    st.code(proc.stderr or proc.stdout)


# --- 헤더 ---
st.markdown('<div class="brand">TONEMATCH · GUITAR TONE EXPLORER</div>', unsafe_allow_html=True)
st.title("내 페달로 찾는 비슷한 기타 톤")
st.markdown(
    '<div class="subtitle">목표 기타 톤을 들려주면, 보유한 페달로 비슷한 설정 3개를 찾아 시작점으로 제시합니다.</div>',
    unsafe_allow_html=True,
)

tab_demo, tab_live = st.tabs(["🎧 샘플 곡으로 바로 듣기", "🎸 내 음원으로 직접 찾기"])

# ======================================================================
# 탭 1: 샘플 곡으로 바로 듣기 (demo_bundle_v1 + Step 2)
# ======================================================================
with tab_demo:
    grouped = scan_grouped_samples()
    non_empty_groups = [g for g, items in grouped.items() if items]

    if not non_empty_groups:
        st.warning(
            f"데모 샘플 폴더를 찾지 못했습니다.\n\n"
            f"공유 드라이브의 `results/demo_bundle_v1`을 `{BUNDLE_DIR}` 경로에 복사하거나, "
            "환경변수 `HAI_DEMO_DIR`로 지정해 주세요."
        )
    else:
        g_col, s_col = st.columns([1, 2])
        with g_col:
            selected_group = st.radio(
                "샘플 그룹",
                non_empty_groups,
                format_func=lambda g: f"{g} ({len(grouped[g])})",
            )
        with s_col:
            sample_map = grouped[selected_group]
            sel_name = st.selectbox("목표 곡 / 페달 샘플 선택", list(sample_map.keys()))

        rpath = sample_map[sel_name]
        data = json.loads(rpath.read_text(encoding="utf-8"))
        folder = rpath.parent
        prefix = rpath.stem.replace("_result", "")
        cands = data.get("candidates")
        is_bundle = bool(cands)

        if is_bundle:
            st.caption("이번 시연은 공통 기타 연주(Fender Strat)를 사용했습니다.")
            top_col, rev_col = st.columns([2, 1])
            with top_col:
                rank_idx = st.radio(
                    "비슷한 설정 3개 선택",
                    range(len(cands)),
                    format_func=lambda i: (
                        f"비슷한 설정 {i + 1}위 · "
                        f"{cands[i].get('pedal') or PEDALS.get(cands[i]['effect_type'], cands[i]['effect_type'])} "
                        f"(톤 거리 {cands[i]['afx_distance']:.3f})"
                    ),
                    horizontal=True,
                )
            with rev_col:
                room_choice = st.radio(
                    "리버브 듣기 (직접 고르기)",
                    list(ROOM_OPTIONS.keys()),
                    horizontal=True,
                )
                st.caption("리버브는 추정하지 않습니다. 원곡의 공간감에 맞춰 원하는 양을 골라 들어 보세요.")

            chosen = cands[rank_idx]
            rank_num = rank_idx + 1
            suffix = ROOM_OPTIONS[room_choice]["suffix"]

            dry_f = folder / "dry.wav"
            tgt_f = folder / "target.wav"
            res_f = folder / f"top{rank_num}{suffix}.wav"
            eq_f = folder / f"top{rank_num}_eq{suffix}.wav"
            base_f = folder / "baseline_default.wav"

            # 핵심 청취 비교 (4열: 원음 / 목표 / 비슷한 설정 / +EQ)
            st.markdown("#### 🔊 소리 비교")
            c1, c2, c3, c4 = st.columns(4)
            tracks = [
                (c1, "① 기타 원음 (공통 Strat)", "이펙터 없는 클린 연주", dry_f),
                (c2, "② 목표 톤", "따라 하고 싶은 기준 소리", tgt_f),
                (c3, f"③ 비슷한 설정 {rank_num}위", f"리버브: {room_choice} · 음량 맞춤", res_f),
                (c4, "④ +EQ (녹음 톤 보정)", f"리버브: {room_choice} · 원곡 밝기 보정", eq_f),
            ]
            for col, title, desc, fp in tracks:
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(desc)
                        if fp.exists():
                            st.audio(str(fp))
                        else:
                            st.caption(f"파일 없음 ({fp.name})")

            st.caption("+EQ는 원곡 녹음의 밝기에 맞춘 참고 소리입니다. 실제 장비에서는 앰프 EQ로 비슷하게 맞출 수 있습니다.")

            # 대조 재생 (1위 페달 기본 노브) + 정답 보기 (실물 페달 데모 전용)
            extra_c1, extra_c2 = st.columns(2)
            with extra_c1:
                if base_f.exists():
                    with st.container(border=True):
                        st.markdown("**대조 · 1위 페달 기본 노브 (보정 전)**")
                        st.caption("노브 보정 전 기본값 소리와 비교해 보세요.")
                        st.audio(str(base_f))

            with extra_c2:
                if selected_group == "실물 페달 데모":
                    ref_pedal = data.get("reference_pedal_label") or data.get("reference_pedal")
                    knobs_clock = data.get("knobs_clock")
                    if ref_pedal or knobs_clock:
                        with st.container(border=True):
                            show_ans = st.toggle("🔍 정답 보기 (실물 페달 데모 전용)", value=False, key=f"ans_{sel_name}")
                            if show_ans:
                                top1_pedal = cands[0].get("pedal") or PEDALS.get(cands[0]["effect_type"], "")
                                st.markdown(f"**실물 페달:** {ref_pedal or '-'}")
                                if knobs_clock:
                                    if isinstance(knobs_clock, dict):
                                        k_str = " · ".join(f"{k} {v}" for k, v in knobs_clock.items())
                                    else:
                                        k_str = str(knobs_clock)
                                    st.markdown(f"**실물 노브 위치:** {k_str}")
                                if ref_pedal and (
                                    ref_pedal.lower() in top1_pedal.lower()
                                    or top1_pedal.lower() in ref_pedal.lower()
                                ):
                                    st.success("정답 페달이 1위에 나왔습니다.")

            # 지표 요약 (톤 거리 / 앰프 크런치 안내)
            m1, m2 = st.columns(2)
            m1.metric("톤 거리 (작을수록 비슷)", f"{chosen['afx_distance']:.4f}")
            crunch_db = chosen.get("amp_crunch_db")
            if crunch_db is not None:
                m2.metric("앰프 게인 안내", f"{float(crunch_db):.1f} dB", amp_crunch_desc(float(crunch_db)), delta_color="off")

            st.write("")
            render_interactive_knobs(
                chosen.get("chain", []),
                dry_f,
                room_choice=room_choice,
                key_prefix=f"bundle_{sel_name}_{rank_num}",
            )

        else:
            # Step 2 (같은 연주 조건)
            room_choice = "없음"
            dry_f = folder / f"{prefix}_dry.wav"
            tgt_f = folder / f"{prefix}_target.wav"
            res_f = folder / f"{prefix}_found.wav"

            st.markdown("#### 🔊 소리 비교 (같은 연주 조건)")
            c1, c2, c3 = st.columns(3)
            for col, title, desc, fp in [
                (c1, "① 기타 원음", "이펙터 없는 원본 연주", dry_f),
                (c2, "② 목표 톤", "따라 하고 싶은 기준 소리", tgt_f),
                (c3, "③ 비슷한 설정 1위", "탐색된 세팅 적용 결과", res_f),
            ]:
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(desc)
                        if fp.exists():
                            st.audio(str(fp))
                        else:
                            st.caption("파일 없음")

            st.write("")
            render_interactive_knobs(
                data.get("chain", []),
                dry_f,
                room_choice=room_choice,
                key_prefix=f"step2_{sel_name}",
            )

# ======================================================================
# 탭 2: 내 음원으로 직접 찾기 (실시간 AFx-Rep 탐색)
# ======================================================================
with tab_live:
    st.markdown("#### 1단계 · 보유한 페달 선택")
    col_eq, col_opt = st.columns([2, 1])
    with col_eq:
        owned = []
        chk_cols = st.columns(2)
        for i, k in enumerate(DRIVE_KEYS):
            if chk_cols[i % 2].checkbox(PEDALS[k], value=(i < 3), key=f"live_inv_{k}"):
                owned.append(k)
    with col_opt:
        with st.container(border=True):
            st.markdown("**탐색 옵션**")
            unlock = st.toggle("보유 페달 외 7종 전체에서 찾기", value=False)
            quality = st.select_slider(
                "탐색 속도 / 정밀도",
                options=["빠르게", "보통 (권장)", "정밀하게"],
                value="보통 (권장)",
            )
            budget_map = {"빠르게": 20, "보통 (권장)": 30, "정밀하게": 50}

    st.markdown("#### 2단계 · 오디오 파일 올리기")
    u1, u2 = st.columns(2)
    dry_up = u1.file_uploader("내 기타 클린 연주 파일 (.wav)", type=["wav"])
    tgt_up = u2.file_uploader("따라 하고 싶은 목표 기타 소리 (.wav)", type=["wav"])

    if st.button("✨ 비슷한 설정 3개 찾기", type="primary", use_container_width=True):
        if not dry_up or not tgt_up:
            st.error("내 기타 클린 연주 파일과 목표 기타 소리 파일을 모두 올려주세요.")
        elif not unlock and not owned:
            st.error("보유한 페달을 최소 1개 이상 선택하거나, '7종 전체에서 찾기'를 켜주세요.")
        else:
            td = tempfile.mkdtemp()
            dp, tp = Path(td) / "dry.wav", Path(td) / "target.wav"
            dp.write_bytes(dry_up.getvalue())
            tp.write_bytes(tgt_up.getvalue())
            req_p, out_p = Path(td) / "req.json", Path(td) / "out.json"
            req_p.write_text(json.dumps({
                "dry": str(dp),
                "target": str(tp),
                "budget": budget_map[quality],
                "allowed": None if unlock else owned,
                "out_dir": td,
                "out_json": str(out_p),
            }), encoding="utf-8")
            with st.spinner("목표 톤을 분석해 비슷한 설정 3개를 찾는 중입니다... (약 1~2분 소요)"):
                proc = subprocess.run(
                    [sys.executable, "-c", MATCH_WORKER, str(req_p)],
                    cwd=str(ROOT), capture_output=True, text=True,
                )
            if proc.returncode != 0:
                st.error("분석 중 오류가 발생했습니다. (AFx-Rep 가중치 및 패키지 설치 상태를 확인해 주세요.)")
                st.code(proc.stderr or proc.stdout)
            else:
                st.session_state["live_res"] = json.loads(out_p.read_text(encoding="utf-8"))
                st.session_state["live_dry"] = dp
                st.session_state["live_tgt"] = tp

    if "live_res" in st.session_state:
        st.divider()
        st.markdown("#### 3단계 · 비슷한 설정 3개 결과")
        cands = st.session_state["live_res"]["candidates"]

        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            pick = st.radio(
                "순위 선택",
                range(len(cands)),
                format_func=lambda i: (
                    f"비슷한 설정 {i + 1}위 · {PEDALS.get(cands[i]['effect_type'], cands[i]['effect_type'])} "
                    f"(톤 거리 {cands[i]['afx_distance']:.3f})"
                ),
                horizontal=True,
                key="live_rank",
            )
        with r_col2:
            live_room = st.radio("리버브 듣기 (직접 고르기)", list(ROOM_OPTIONS.keys()), horizontal=True, key="live_room")
            st.caption("리버브는 추정하지 않습니다. 원하는 양을 골라 들어 보세요.")

        chosen = cands[pick]
        wav_key = {"없음": "wav", "약": "wav_room1", "강": "wav_room2"}[live_room]

        a1, a2, a3 = st.columns(3)
        a1.markdown("**① 내 기타 원음**"); a1.audio(str(st.session_state["live_dry"]))
        a2.markdown("**② 목표 톤**"); a2.audio(str(st.session_state["live_tgt"]))
        a3.markdown(f"**③ 비슷한 설정 {pick + 1}위 (리버브: {live_room})**"); a3.audio(chosen[wav_key])

        render_interactive_knobs(
            chosen["chain"],
            st.session_state["live_dry"],
            room_choice=live_room,
            key_prefix=f"live_{pick}",
        )
