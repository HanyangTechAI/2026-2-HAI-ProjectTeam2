# 합성 데이터

합성 Dry/Wet과 정답 라벨을 보관하는 폴더입니다. WAV와 대용량 결과는 Git에서 제외합니다.

## 현재 CNN 데이터 — VST 드라이브 전용

- `vst7_drive_only/`: MultiDrive PRO 7종만 사용하는 로컬 새 생성본입니다. 컴프·EQ는 없습니다.
- `dry/` 78개, `audio/` 2,730개, `labels/` 2,730개와 manifest·생성/전수 검사 기록이 있습니다.
- 모두 44.1 kHz이며, 학습·탐색 입력은 이 생성본의 peak −12 dBFS `dry/`입니다.
- 이전 컴프·EQ 혼합 데이터의 로컬 연결은 제거했습니다. 공유 드라이브 원본은 보존합니다.
- 현재 학습·평가 결과는 `outputs/vst7_drive_only/`에 있습니다.

팀 공용 `search.pipeline`과 `search.evaluate_dataset`을 직접 실행할 때는
`--dataset data/synthetic/vst7_drive_only`를 지정하세요. 공용 코드의 기본 경로는 이전
공유 데이터 이름이므로, 로컬 연결을 제거한 현재 환경에서는 명시적인 경로가 필요합니다.

자세한 설정과 결과는 [estimation/README.md](../../estimation/README.md)와
[실험 결과](../../estimation/EXPERIMENTS.md)를 참고하세요.
