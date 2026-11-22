# 합성 데이터와 현재 Step 2 데이터

현재 CNN 학습·평가는 공유 드라이브에서 복사한 `data/local/`의 Step 2 데이터를 사용한다.
이 `data/synthetic/` 폴더의 예전 로컬 데이터는 과거 실험용이며 현재 기본 입력이 아니다.

| 데이터 | 현재 로컬 경로 | 용도 |
|---|---|---|
| step2_ir_v1 | `data/local/step2_ir_v1/` | VST 7종 + 고정 캐비닛 IR, 2,730쌍. 학습·validation·test |
| step2_real_v1 | `data/local/step2_real_v1/` | 새로운 기타 Dry의 VST+IR 변형 280쌍, 전체 외부 평가 |

- 학습·탐색 Dry는 각 데이터셋 `manifest.json`의 `render_input`을 사용한다. 생성 폴더의 peak −12 dBFS `dry/`다.
- 입력은 VST 드라이브 한 개와 고정 IR 조건이며 컴프·EQ는 사용하지 않는다.
- IR 파일은 `data/IR/`에 준비하며 checkpoint에 저장된 해시와 일치해야 한다.
- 학습 결과·모델은 `outputs/step2_ir_v1/`, 외부 평가·탐색 결과는 `outputs/step2_real_v1/`이다.
- 이전 `vst7_drive_only/`, `vst7_amp_ir/` 데이터·모델·결과는 이번 코드 정리에서 삭제하지 않았다.
- 공유 원본·WAV·모델·캐시는 Git에 올리지 않는다.

생성이 필요하면 팀 공용 `scripts/generate_step1_dataset.py`와 `scripts/apply_amp_dataset.py`를 사용한다.
옵션 설명은 [생성기 문서](../../scripts/STEP1_DATA_GENERATION.md)를 참고하되,
현재 CNN용으로는 컴프·EQ를 끄고 드라이브 한 개만 사용한다.
공유 Step 2 데이터가 있으면 로컬에서 재생성할 필요가 없다.

학습·평가 명령은 [estimation/README.md](../../estimation/README.md),
현재 결과와 이전 실험 기록은 [EXPERIMENTS.md](../../estimation/EXPERIMENTS.md)를 참고한다.
