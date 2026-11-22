# MultiDrive PRO 7종 CNN · Step 2

생성 폴더의 Dry와 목표 Wet을 함께 입력해 VST 드라이브 종류와 노브 초기값을 예측한다.
Top-K 후보를 팀 공용 CMA-ES에 전달하고 스펙트럼·다이내믹스 복합 거리 d로 비교한다.
컴프·EQ는 사용하지 않는다. 캐비닛은 알려진 고정 렌더링 조건으로 전달한다.

## 데이터와 생성 코드

현재 데이터는 공유 드라이브에서 복사한 `data/local/`의 Step 2 데이터다.
학습·탐색 입력은 각 `manifest.json`의 `render_input`, 즉 생성 폴더의 peak −12 dBFS `dry/`를 사용한다.

| 경로 | 역할 |
|---|---|
| `data/local/step2_ir_v1/` | 2,730쌍, Dry 78개. train/validation/test = 1,890/420/420 |
| `data/local/step2_real_v1/` | 새로운 기타 Dry 40개의 VST 7종 변형, 280쌍 전체 외부 평가 |
| `data/IR/` | checkpoint에 기록된 고정 캐비닛 IR. 파일 해시도 일치해야 함 |
| `outputs/step2_ir_v1/` | 학습 캐시·checkpoint·validation/test 결과 |
| `outputs/step2_real_v1/` | 외부 평가·CMA-ES 결과·WAV·검사 기록 |

`step2_real_v1`의 Wet도 VST+IR 합성이다. 새로운 기타·연주에 대한 평가이며,
실제 페달·앰프로 녹음한 Wet의 성능 검증과는 구분한다.

데이터 생성은 팀 공용 [generate_step1_dataset.py](../scripts/generate_step1_dataset.py),
저장된 드라이브 Wet에 고정 IR을 적용하는 작업은 [apply_amp_dataset.py](../scripts/apply_amp_dataset.py)를 사용한다.
옵션 설명은 [STEP1_DATA_GENERATION.md](../scripts/STEP1_DATA_GENERATION.md)를 참고하되,
이 CNN용 데이터에서는 **컴프·EQ를 끄고 VST 드라이브 한 개만** 사용해야 한다.
공유 Step 2 데이터가 있으면 로컬에서 재생성할 필요가 없다.

2026-11-23에 Step 2 이전의 로컬 드라이브 생성기, IR 데이터 복제 생성기,
정답 없는 Demucs 비교 실행기를 제거했다. 과거 실험 수치는 [EXPERIMENTS.md](EXPERIMENTS.md)에 보존한다.
기존 데이터·모델·결과와 공유 드라이브 원본은 삭제하지 않았다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| audio.py | Dry/Wet 읽기, 구간·log-mel 전처리, 무음·미처리 출력 검사 |
| chain.py | VST 7종 정의, 노브 정규화, Top-K 후보 구성 |
| cnn.py | 공유 CNN encoder/fusion, 분류·노브·거리 보조 head |
| data.py | 라벨 검사, Dry 원본별 분할, 캐시·출처 검사 |
| distance.py | train-only scale과 스펙트럼·다이내믹스 복합 d |
| train.py | 학습과 validation loss 기준 checkpoint 선택 |
| evaluate.py | 분류·노브 오차 및 실제 VST 렌더링 d 평가 |
| experiments.py | 캐시 준비 → 학습 → validation/test 평가 |
| infer.py | Dry/목표 Wet에서 Top-K 후보·초기 노브 추론 |
| renderer.py | 공용 VST 렌더러 연결, 활성 상태·재현 검사 |
| amp.py | 고정 캐비닛 정의·IR 해시 검사와 후보 문맥 전달 |
| refine.py | CNN 후보·초기값을 공용 CMA-ES API 형식으로 변환 |
| cma_experiment.py | 동일 후보에서 CNN 노브 시작과 기본 노브 시작 비교 |
| step2_experiment.py | Step 2 외부 전체 평가, 병렬 CMA-ES, WAV·통계·보고서 저장 |

팀 공용 생성기·렌더러·스키마·거리·탐색 구현은 각각 `scripts/`, `effects/`,
`schemas/`, `metrics/`, `search/`에 둔다.

## 학습·평가·탐색

저장소 루트에서 기존 Conda 환경을 사용한다. WAV 기반 학습에는 플러그인이 필요 없고,
렌더링 평가·탐색에는 활성화한 MultiDrive PRO와 checkpoint의 IR 파일이 필요하다.

```powershell
# 공유 Step 2 데이터로 30 epoch 학습 및 validation/test 평가
.\.conda-cnn\python.exe -m estimation.experiments --dataset data/local/step2_ir_v1 --output outputs/step2_ir_v1 --epochs 30

# 같은 모델로 합성 test 420쌍과 외부 기타 280쌍 평가
.\.conda-cnn\python.exe -m estimation.step2_experiment evaluate

# 외부 기타 280쌍 전체 CMA-ES 탐색
.\.conda-cnn\python.exe -m estimation.step2_experiment parallel-search --workers 4 --budget 40 --k 3

# 기존 결과를 읽어 EXPERIMENTS.md의 Step 2 구간 갱신
.\.conda-cnn\python.exe -m estimation.step2_experiment report
```

- 모델은 `outputs/step2_ir_v1/model/best.pt`다. 외부 데이터로 학습하거나 모델을 선택하지 않는다.
- 외부 평가는 자동 분할의 test 42쌍 대신 전체 280쌍을 사용하고, 학습 Dry ID·내용 해시와의 중복을 검사한다.
- 데이터 스키마·입력 설정·고정 앰프를 검사하며, d scale은 학습 checkpoint 값으로 고정한다.
- Step 2 탐색의 `--budget 40`은 **후보당** 예산이다. K=3이면 샘플당 최대 120회다.
- `search.evaluate_dataset`의 후보 API와 `search.pipeline.search_and_rank_chains`를 호출한다.
  공개 `metric_fn`/`render_fn`으로 combined d와 검사 렌더러를 전달한다.
- 후보 amp_id는 정답의 알려진 고정 앰프다. 정답 드라이브는 후보에 주입하지 않는다.
- 샘플별 WAV와 JSON은 `outputs/step2_real_v1/search/shards/*/samples/*/`에 저장한다.
- 탐색 결과는 입력·코드·조건·WAV 해시가 같은 경우에만 재사용한다. 변경 후에는 새 `--output`을 사용한다.
- 학습을 다시 시작할 때도 기존 모델 보존이 필요하면 새 output을 지정한다.

팀 CLI 자체의 checkpoint 연결 확인:

```powershell
.\.conda-cnn\python.exe -m search.evaluate_dataset --dataset data/local/step2_real_v1 --checkpoint outputs/step2_ir_v1/model/best.pt --limit 1 --top-k 3 --budget 6
```

팀 CLI의 기본 loss와 Step 2 실험의 combined d는 구분한다.
단일 페어의 후보만 출력하려면:

```powershell
.\.conda-cnn\python.exe -m estimation.infer --checkpoint outputs/step2_ir_v1/model/best.pt --dry '<생성 폴더>/dry/<파일>.wav' --reference '<목표>.wav' --k 3 --output outputs/step2_ir_v1/prediction.json
```

## 모델과 거리

VST 종류는 `808_scream`, `b_drive`, `o_drive`, `one_distortion`,
`bat_distortion`, `f_fuzz`, `m_fuzz`의 7종이다. 후보 하나는 드라이브 한 개다.

Dry와 Wet에 가중치를 공유하는 Conv(8/16/32/64) → GroupNorm → ReLU → MaxPool
4블록을 적용한다. AdaptiveAvgPool((4,1)) 뒤 Dry, Wet, 차이, 절댓값 차이를 연결해
256차원 fusion으로 변환하고 종류·종류별 노브·거리 성분을 예측한다.

`loss = CE(drive) + Huber(active drive knobs) + 0.1 × Huber(distance components)`

거리 보조 head의 정답은 d(Dry, target)의 두 성분이다.
예측 체인의 VST 렌더링을 미분해 역전파하는 방식은 아니다.

`d = 0.5 × spectral / spectral_scale + 0.5 × dynamics / dynamics_scale`

scale은 train Dry/target 거리의 중앙값으로 정하고 validation/test/외부 평가에서 고정한다.
d는 낮을수록 가깝다. 분류 정확도와 별도 지표다.
CNN·VST·거리 계산은 44,100 Hz, 최대 2.5초이며 log-mel은 64 bands, n_fft=2048, hop=512다.
best 모델은 전체 validation loss로 선택한다. 렌더링 d는 기본적으로 split별 균형 표본 70쌍이며,
분류·노브 오차는 split 전체를 사용한다. `--render-samples -1`이면 전체 렌더링한다.

렌더링은 `effects.render.render_chain`을 사용한다.
buffer=2048, 무음 warmup=0.5초, 입력 게인=0 dB이며, 캐비닛은 저장된 정의·게인·IR 해시로 처리한다.
VST 출력은 캐비닛 적용 전에 검사해 IR이 bypass를 가리지 않도록 한다.

## CNN 초기값의 효과만 비교하는 실험

`cma_experiment.py`는 같은 Top-K 후보에서 CNN 노브 시작과 스키마 기본값 시작을 비교한다.
공유 Step 2 학습 캐시와 checkpoint가 기본값이다. 외부 280쌍 평가에는 위의 `step2_experiment.py`를 사용한다.

```powershell
.\.conda-cnn\python.exe -m estimation.cma_experiment --prepare-only --output outputs/step2_ir_v1/cma_validation_new
.\.conda-cnn\python.exe -m estimation.cma_experiment --split validation --samples 7 --k 3 --budget 180 --restarts 2 --seed 42 --output outputs/step2_ir_v1/cma_validation_new
```

이 비교기의 `--budget`은 **샘플·방법별로 K개 후보가 나누는 총예산**이다.
Step 2 실행기의 후보당 예산과 다르다. 정답 노브는 종료 후 재현 검사에만 사용한다.
후보는 `effect_type`, 종류 확률 `model_score`, 공통 형식 `chain`,
0~1 초기 좌표 `x0`, 노브 순서·범위·정규화 방식 `vector_spec`을 담는다.
`refine.to_search_inputs`가 이를 검증해 공용 탐색 API로 전달한다.

## 검사와 결과 기록

```powershell
.\.conda-cnn\python.exe -m unittest discover -s tests -v
```

검사 범위는 라벨·분할·모델 구조·노브 좌표·CMA 예산·초기값 보존·IR 문맥·
bypass 거부·외부 평가의 명시적 허용과 중복 방지·고정 d scale이다.
모든 실험 수치와 해석은 [EXPERIMENTS.md](EXPERIMENTS.md)에 기록한다.
현재 Step 2 결과를 문서 앞에, 이전 실험을 접힌 과거 기록에 둔다.
최종 결과의 독립 검사 기록은 `outputs/step2_real_v1/artifact_audit.json`이다.
