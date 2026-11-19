# Tone Match 화면 데모

기타 이펙터 체인 추정 프로젝트의 로컬 Streamlit 첫 화면입니다.
오디오 재생과 추정 로직 없이 동작하며, 모든 노브 값은 고정된 예시입니다.
보유 이펙터를 선택하면 해당 이펙터만 예시 체인에 표시됩니다.

## Windows / VS Code PowerShell

Python 3.10 이상과 pip가 필요합니다. DemoUI 폴더에서 다음 명령을 실행하세요.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

브라우저에서 http://localhost:8501 에 접속합니다. 종료는 터미널에서 Ctrl+C입니다.
가상 환경 활성화가 필요 없어 PowerShell 실행 정책 변경 없이 실행할 수 있습니다.

`py` 명령이 없다면 pip와 venv를 포함하는 Windows Python을 설치하고 VS Code 터미널을 다시 여세요.
일반 Python 설치 환경에서는 `py` 대신 `python`을 사용할 수도 있습니다.

## 화면 구성

- 상단: 기타 리프 선택, 오버드라이브·디스토션·퍼즈 보유 체크박스
- 가운데: Dry · 기타 원음 / 정답 · 목표 톤 / 결과 · 재현 톤 재생 버튼
- 버튼 클릭: “오디오 연결 예정” 안내
- 하단: 선택한 이펙터의 예시 체인과 0–100 상대 노브 값
- 앰프: 이펙터 선택 여부와 관계없이 Gain, Bass, Middle, Treble, Presence, Master 예시 값 표시

리프 선택은 화면의 선택 상태만 변경합니다. 실제 분석은 수행하지 않습니다.
