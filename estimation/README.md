# MultiDrive PRO 7종 전용 CNN

생성 Dry/목표 Wet으로 드라이브 7종과 해당 드라이브의 노브 초기값을 예측한다.
컴프레서·EQ는 데이터 생성, 학습 출력, 추론 후보, 렌더링에서 사용하지 않는다.

- 모델: 808_scream, b_drive, o_drive, one_distortion, bat_distortion, f_fuzz, m_fuzz.
- 체인: MultiDrive PRO 드라이브 한 개. Top-K는 서로 다른 드라이브 후보다.
- 입력: 생성 폴더의 peak −12 dBFS `dry/`만 사용한다.
- CNN/VST/거리 계산: 모두 44,100 Hz, 2.5초. CNN은 64-band log-mel이다.
- 전처리: n_fft=2048, hop=512. 이전 22.05 kHz 설정과 프레임 시간 간격을 유지한다.
- 원본 Dry ID·내용 해시 단위로 train/validation/test 누출을 검사한다.
- 이전 컴프·EQ 혼합 캐시/모델은 format_version=3에서 거부한다.
- 생성 WAV 전부에 무음/Dry pass-through 검사를 적용하고 검사 기록 hash를 모델에 보존한다.

## 파일 구성

문서는 이 사용 안내와 [EXPERIMENTS.md](EXPERIMENTS.md)의 실험 결과로 구성한다.

| 파일 | 역할 |
|---|---|
| audio.py | 생성 Dry/Wet 읽기, 구간·log-mel 전처리, 무음/미처리 출력 검사 |
| chain.py | VST 7종 정의, 노브 정규화, Top-K 체인 후보 구성 |
| cnn.py | 공유 CNN encoder/fusion과 분류·노브·거리 보조 head |
| data.py | 라벨 검증, 원본별 분할, 특징/오디오 캐시와 provenance 검사 |
| distance.py | train-only scale 보정과 스펙트럼·다이내믹스 복합 d |
| train.py | CNN 학습과 validation loss 기준 checkpoint 선택 |
| evaluate.py | 분류·노브 오차와 실제 VST 렌더링 d 평가 |
| experiments.py | 캐시 준비 → 학습 → validation/test 평가 실행 |
| infer.py | Dry/목표 Wet으로 Top-K 후보·초기 노브 추론 |
| renderer.py | main 공용 VST 렌더러 연결과 활성·재현 검사 |
| refine.py | CNN 후보를 main CMA-ES API에 전달하고 예산 내 최선 후보 선택 |
| cma_experiment.py | 동일 후보에서 CNN 시작·기본값 시작 CMA-ES 비교 |
| generate_drive_only.py | native Dry 캐시에서 현재 VST 전용 데이터 생성·전수 검사 보조 |

팀 공용 생성기·렌더러·스키마·거리·탐색 구현은 각각 `scripts/`, `effects/`,
`schemas/`, `metrics/`, `search/`에 유지한다.

## 유지한 CNN 구조

Dry와 Wet에 가중치를 공유하는 Conv(8/16/32/64) → GroupNorm → ReLU → MaxPool
블록 4개를 적용한다. AdaptiveAvgPool((4,1)) 후 Dry, Wet, 차이, 절댓값 차이를
연결하고 256차원으로 변환하는 구조를 유지했다.

7-class 분류, 7종별 노브, 거리 성분 보조 head를 학습한다.
사용하지 않는 컴프·EQ 노브와 사용 여부 head만 제거했다.

`loss = CE(drive) + Huber(active drive knobs) + 0.1 × Huber(distance components)`.

거리 보조 head의 정답은 d(Dry, target)의 두 성분이다.
VST를 미분하거나 예측 체인의 렌더링 d를 직접 역전파하지 않는다.

## 유사도 d

`d = 0.5 × spectral / spectral_scale + 0.5 × dynamics / dynamics_scale`.
scale은 train Dry/target 거리의 중앙값으로 정하고 validation/test에서는 고정한다.
낮을수록 가깝다. 종류 확률이나 청감 확률은 아니다.

학습 중 고정 validation 14개를 주기적으로 렌더링한다.
best 모델은 전체 validation loss로 선택하고 test로 선택하지 않는다.
분류·노브는 split 전체, 실제 렌더링 d는 종류별 균형 표본 70개/split에서 평가한다.
`--render-samples -1`은 전체 렌더링이다.

## 렌더러

main의 `effects.render.render_chain`을 그대로 사용한다.
모델·표본율별 플러그인을 로드하고, buffer=2048, 무음 warmup=0.5초 후 렌더링한다.
입력 게인은 0 dB 고정이며 앰프는 처리하지 않는다.
활성 상태와 정답 노브 재현 오차를 별도로 검사한다. UI/체험판을 자동으로 열지 않는다.
평가 중 일시적 무음/Dry 통과가 발생하면 최대 3회 안에서 재시도하고 횟수를 기록한다.
CMA-ES adapter에서는 잘못된 출력을 거리로 사용하지 않고 오류로 전파한다.

## 실행

프로젝트 루트의 기존 Conda 환경을 사용한다.

```powershell
# 생성된 VST 전용 데이터로 캐시 → 30epoch 학습 → validation/test 평가
.\.conda-cnn\python.exe -u -m estimation.experiments `
  --dataset data/synthetic/vst7_drive_only --output outputs/vst7_drive_only --epochs 30

# best 모델의 평가만 다시 수행
.\.conda-cnn\python.exe -m estimation.experiments `
  --dataset data/synthetic/vst7_drive_only --output outputs/vst7_drive_only `
  --evaluate-only --render-samples 70

# 단일 파일 Top-3 후보 출력
.\.conda-cnn\python.exe -m estimation.infer `
  --checkpoint outputs/vst7_drive_only/model/best.pt `
  --dry '<생성 폴더>/dry/<파일>.wav' --reference '<목표>.wav' `
  --k 3 --output outputs/vst7_drive_only/prediction.json
```

데이터 재생성은 팀 `scripts/generate_step1_dataset.py`를 사용한다.
`--effects`에 위 VST 7종만 지정하고, `--per-drive --variations 5 --dry-peak-db -12`로 실행한다.
컴프·EQ 포함 확률은 0으로 지정한다. 팀 생성기의 출력 제한에 따라 새 로컬 임시 폴더에
생성한 뒤 검증된 로컬 데이터 폴더로 옮긴다. 공유 원본에는 쓰지 않는다.

## CNN → CMA-ES 연결

CNN은 드라이브별 확률과 노브 초기값을 예측한다. 후보 하나는 VST 드라이브 한 개이며,
`model_score`는 종류 확률이다. 음향 유사도로 사용하지 않는다.

후보에는 `effect_type`, `model_score`, 공통 체인 형식의 `chain`, 0~1 정규화 노브 `x0`,
좌표 순서·범위·정규화 방식이 담긴 `vector_spec`이 포함된다.
`refine.to_search_inputs`가 후보와 좌표를 검사하고 main의
`search_and_rank_chains(candidate_configs=...)`가 받을 수 있는 형식으로 반환한다.
평가 adapter는 `run_cma_es_for_chain(..., x0_norm=...)`을 호출한다.
노브 정밀도를 보존하는 spec은 estimation 안에서 확장하며 main 탐색 파일은 수정하지 않는다.

같은 Top-K 드라이브 구조에서 CNN 노브 시작과 스키마 기본 노브 시작을 비교한다.
seed·예산·실행 수·목적함수 d는 동일하다. 종류 후보는 고정하고 노브만 탐색한다.
정확한 초기 후보도 평가하고 보존하며, 그 평가를 포함한 모든 렌더를 예산에 넣는다.

```powershell
# 플러그인 없이 후보와 실행 계획 준비
.\.conda-cnn\python.exe -m estimation.cma_experiment --prepare-only `
  --output outputs/vst7_drive_only/cma_validation_new

# validation 종류별 1개, CNN Top-3 후보로 비교
.\.conda-cnn\python.exe -u -m estimation.cma_experiment `
  --split validation --samples 7 --k 3 --budget 180 --restarts 2 --seed 42 `
  --output outputs/vst7_drive_only/cma_validation_new
```

`--split test`는 test 분할을 사용한다. 실행마다 새 `--output` 경로를 사용한다.
기본 cache/checkpoint는 `outputs/vst7_drive_only/` 아래다.
`--budget`은 표본·방법당 K개 후보가 나누는 총 렌더 예산이다.
`--restarts 2`는 main의 `n_runs=2`, 즉 초기 실행과 재시작을 합해 2회다.
표본 7개·방법 2개·예산 180이면 탐색 상한은 2,520회이며 활성/정답 진단 14회가 별도다.
정답 노브는 탐색에 전달하지 않고 종료 후 재현 검사용으로만 사용한다.

## 검사와 결과

`tests/test_estimation.py`는 단일 VST 라벨·원본별 분할·CNN 구조·과거 checkpoint 거부를,
`tests/test_cnn_cma.py`는 main CMA API·좌표·예산·초기값 보존·재현성을 검사한다.

```powershell
.\.conda-cnn\python.exe -m pytest tests/test_estimation.py tests/test_cnn_cma.py metrics/test_combined.py metrics/test_dynamics.py -k 'not CompressorIntegration' -q
```

공용 CompressorIntegration 검사는 이번 CNN 범위 밖이며 기존 의존성 문제로 제외한다.
matplotlib 부재 경고는 선택적 `cma.plot()`에 관한 것으로 탐색에는 영향을 주지 않는다.

현재 로컬 산출물은 다음 명령으로 다시 검사할 수 있다.

```powershell
.\.conda-cnn\python.exe outputs/vst7_drive_only/audit_results.py
.\.conda-cnn\python.exe outputs/vst7_drive_only/audit_cma_results.py
```

첫 명령은 데이터·분할·후보·평균 거리·모델/캐시/소스 hash를 검사한다.
두 번째는 CMA 예산·탐색 이력·노브 오차·평균값을 검사하고 결과 문서의 CMA 표를 갱신한다.
JSON·로그는 `outputs/vst7_drive_only/`에 재검증 원자료로 보존한다.
모든 실험 설정·수치·해석은 [EXPERIMENTS.md](EXPERIMENTS.md)에서만 관리한다.
