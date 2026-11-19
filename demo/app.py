import streamlit as st


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

EFFECTS = {
    "오버드라이브": {"Drive": 35, "Tone": 55, "Level": 65},
    "디스토션": {"Gain": 55, "Tone": 50, "Level": 60},
    "퍼즈": {"Fuzz": 60, "Tone": 45, "Level": 55},
}
AMP = {"Gain": 40, "Bass": 50, "Middle": 60, "Treble": 55, "Presence": 45, "Master": 65}

st.markdown('<div class="eyebrow">GUITAR EFFECT CHAIN LAB / UI DEMO</div>', unsafe_allow_html=True)
st.title("내 이펙터로 찾는 목표 톤")
st.write("기타 리프와 보유 이펙터를 선택하고, 원음부터 재현 톤까지 한눈에 비교하세요.")
st.caption("화면 프로토타입 · 오디오 재생과 톤 추정 기능은 추후 연결됩니다.")

with st.container(border=True):
    st.subheader("01 · 입력 설정")
    st.selectbox("기타 리프", ["클린 아르페지오", "블루스 리드", "록 파워 코드", "펑크 리듬"], key="riff")
    st.write("보유 이펙터")
    selected = []
    for column, name in zip(st.columns(len(EFFECTS)), EFFECTS):
        with column:
            if st.checkbox(name, value=name == "오버드라이브", key=f"effect_{name}"):
                selected.append(name)

st.subheader("02 · 톤 비교")
st.caption(f"선택한 리프: {st.session_state.riff} · 각 버튼은 연결 안내만 표시합니다.")
tracks = [
    ("Dry · 기타 원음", "이펙터가 적용되기 전의 기타 신호", "dry"),
    ("정답 · 목표 톤", "재현하고 싶은 기준 사운드", "target"),
    ("결과 · 재현 톤", "추천 체인을 적용할 결과 사운드", "result"),
]
for column, (title, description, key) in zip(st.columns(3), tracks):
    with column:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.caption(description)
            if st.button("▶ 재생", key=f"play_{key}", use_container_width=True):
                st.info("오디오 연결 예정")

st.subheader("03 · 추천 체인과 노브 값")
st.info("예시 세팅 — 선택한 보유 이펙터를 고정된 순서로 나열한 화면 데모입니다. 리프나 오디오를 분석한 추정 결과가 아닙니다.")
chain = " → ".join(["기타", *selected, "앰프 / 출력"])
st.markdown(f'<div class="chain">{chain}</div>', unsafe_allow_html=True)
st.write("")
if not selected:
    st.info("선택한 이펙터가 없어 기타를 앰프로 바로 연결합니다. 앰프 예시 세팅은 아래에 표시됩니다.")
st.caption("모든 노브 값은 데모용 0–100 상대값입니다. 실제 장비의 단위나 범위와 다를 수 있습니다.")
settings = [(name, EFFECTS[name]) for name in selected] + [("앰프", AMP)]
columns = st.columns(min(len(settings), 3))
for index, (name, knobs) in enumerate(settings):
    with columns[index % len(columns)]:
        with st.container(border=True):
            st.markdown(f"**{index + 1:02d} · {name}**")
            st.caption("예시 세팅")
            for knob, value in knobs.items():
                st.write(f"{knob} · **{value} / 100**")
                st.progress(value)

st.divider()
st.caption("TONE MATCH / LOCAL DEMO · 실제 오디오 파일이나 추정 모델을 사용하지 않습니다.")
