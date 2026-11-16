# CNN 초기 추정 모델 구현 및 변경 사항

공동작업자가 기존 코드에 미치는 영향과 이번에 추가한 기능을 확인할 수 있도록 정리한 문서다.

- 작업 브랜치: `feat/cnn-initial-estimator`
- 비교 기준: 기존 `bfcbd4e` 커밋 (`feat: Step 1 effect rendering`)
- 작성일: 2026-11-09
- CNN 단일 모델 구성 정리: 2026-11-10
- 실행 방법: [estimation README](README.md)
- 실제 검증 환경과 결과: [VALIDATION.md](VALIDATION.md)

## 1. 기존 파일에서 변경한 사항

기존 추적 파일을 직접 수정한 범위는 아래 세 파일이다.

| 파일 | 변경 내용 | 목적과 기존 기능에 미치는 영향 |
|---|---|---|
| [`.gitignore`](../.gitignore) | `.venv-cnn/`, `.conda-cnn/` 제외 규칙 추가 | 로컬 검증 환경이 Git에 포함되지 않게 한다. 기존 오디오·모델·`outputs/` 제외 규칙은 그대로 사용한다. |
| [프로젝트 `README.md`](../README.md) | 현재 렌더러·탐색·CNN 구현 상태 설명 및 estimation 문서 링크 추가 | 과거의 “렌더링·탐색은 이후 구현” 안내를 현재 상태로 수정한다. CNN의 실제 범위를 드라이브 최대 한 개와 bypass로 명시한다. |
| [`effects/render.py`](../effects/render.py) | 모듈 상단의 `pedalboard` import를 `build_effect()`와 `render_chain()` 내부로 이동 | `load_effects()`를 통해 스키마를 조회할 때 DSP 패키지까지 즉시 import하지 않게 한다. 실제 이펙터 구성과 렌더링 알고리즘, 함수 인자 및 반환값은 유지한다. |

`effects/render.py`의 렌더러 자체는 기존 구현을 재사용한다.
매 호출마다 새로운 Pedalboard를 만드는 기존 동작을 이용하여 샘플 사이에 상태를 공유하지 않는다.
드라이브별 클래스 구성이나 고정 필터 주파수를 이번 작업에서 새로 정의하지 않았다.

다음 기존 파일은 수정하지 않고 재사용하거나 참고했다.

| 기존 코드 | 사용 방식 |
|---|---|
| [`schemas/effects.data.json`](../schemas/effects.data.json) | 이펙터 종류, 노브 이름·타입·범위·기본값·처리 매핑의 원본으로 사용 |
| [`schemas/chain.schema.json`](../schemas/chain.schema.json) | 출력 체인을 기존 `ChainInstance` 형태로 구성 |
| [`metrics/spectral.py`](../metrics/spectral.py) | 기존 `spectral_distance()`와 `METRIC_VERSION`을 평가·최적화에 사용 |
| [`scripts/Create_Synthetic_pairs.py`](../scripts/Create_Synthetic_pairs.py) | 기존 생성 방식과 label 형식을 확인하고, 기존 label 설정을 가져오는 경로 제공 |
| [`search/pipeline.py`](../search/pipeline.py) | 기존 overdrive 2차원 CMA 실험을 유지하고, 별도 초기값 연결 어댑터 추가 |

기존 스키마, 기본 `requirements.txt`, UI 및 기존 Step 0 탐색 실험은 변경하지 않았다.

## 2. estimation 폴더에 새로 구현한 부분

| 파일 | 구현 내용 |
|---|---|
| [`__init__.py`](__init__.py) | estimation 패키지 진입점 |
| [`audio.py`](audio.py) | 공통 오디오 설정, mono 변환·resample·crop·padding, 고정 기준 log-mel 전처리, 비정상 입력 검사 |
| [`data.py`](data.py) | 원본 녹음 ID 기준 split, 합성 Dry/Wet 생성, 기존 label 설정 재사용, manifest 저장·로딩·검증 |
| [`chain.py`](chain.py) | 렌더링에 반영되는 활성 노브 선별, 정규화·역변환, ChainInstance 구성, inventory 기반 후보 생성, CMA 벡터 변환 |
| [`cnn.py`](cnn.py) | 공유 CNN Encoder, 종류 분류 및 종류별 노브 헤드, 활성 정답 노브만 사용하는 손실 함수 |
| [`train.py`](train.py) | PyTorch Dataset/DataLoader, 학습·검증, seed·학습 옵션, best/last checkpoint 및 학습 이력 저장 |
| [`infer.py`](infer.py) | checkpoint 로드, Dry/reference 추론, 종류 확률과 top-K 후보 출력, 선택적 CMA 보정 및 최종 WAV 저장 |
| [`refine.py`](refine.py) | CNN 후보별 초기값을 받는 CMA 보정, 활성 연속 노브 차원 처리 및 정확한 렌더 예산 관리 |
| [`evaluate.py`](evaluate.py) | CNN의 정답 종류를 제공한 노브 평가와 전체 파이프라인 평가, 지표 및 JSON/JSONL 결과 저장 |
| [`experiments.py`](experiments.py) | 3종 분류와 overdrive 전용 노브 회귀의 데이터 준비·학습·평가를 분리 실행 |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | 분리 실험의 데이터 조건, 실행 명령, 결과 및 해석 |
| [`smoke.py`](smoke.py) | 임시 harmonic 신호로 데이터 준비부터 실제 렌더링·CMA까지 소량 기능 검증 |
| [`README.md`](README.md) | CNN 설치·데이터 준비·학습·추론·평가 명령, 폴더 구성 및 가정 |
| [`VALIDATION.md`](VALIDATION.md) | 실제 Conda 환경에서 실행한 검증과 한계 기록 |
| [`CHANGE_SUMMARY.md`](CHANGE_SUMMARY.md) | 기존 파일 변경과 신규 구현을 구분한 팀 공유 문서 |

### CNN 구조와 학습

Dry와 reference를 각각 log-mel로 변환한 뒤 **가중치를 공유하는 같은 Encoder**에 전달한다.

```text
zd = Encoder(dry)
zr = Encoder(reference)
h  = concat(zd, zr, zr - zd, abs(zr - zd))
h  → 종류 분류 / 종류별 노브 예측
```

Encoder는 Conv2D 블록 4개(16 → 32 → 64 → 128), GroupNorm, ReLU,
MaxPool과 Global Average Pooling으로 구성했다. 이번 프로젝트의 기준 모델 설계이며
특정 논문을 그대로 재현한 모델이라는 의미는 아니다.

종류 분류에는 Cross-Entropy, 활성 연속 노브에는 Huber Loss를 사용한다.
정답 종류가 아닌 노브와 bypass의 노브에는 회귀 손실을 적용하지 않는다.
배치 전체가 bypass인 경우도 처리한다. 활성 범주형·boolean 노브가 있는 정의에는
각각 CE·BCE 헤드를 사용하도록 구현했지만 현재 드라이브의 활성 노브는 모두 연속형이다.

### 데이터와 재현성

원본 녹음을 먼저 train/validation/test로 나눈 뒤 crop과 Wet 변형을 생성한다.
동일 원본 ID의 split 누출과 같은 원본 파일이 다른 ID로 split을 넘는 경우를 검사한다.
기본 전처리는 **mono / 22050 Hz / 4초 / STFT 1024·256 / mel 64**다.
짧은 구간은 우측 padding하며, Dry와 Wet을 각자 peak normalization하지 않는다.

manifest에는 원본 ID·경로·파일 hash, split, crop 위치, 유효 길이, Dry/Wet 경로와
정답 체인을 기록한다. 설정·렌더러 정보·제외된 입력의 사유는 별도 metadata에 남긴다.
학습용 manifest는 기존 스키마 문서와 별도로 관리하며 기존 JSON Schema에 필드를 추가하지 않는다.

기존 합성 label은 `--legacy-labels`로 설정을 가져올 수 있다.
이때 기존 Wet 파일을 그대로 복사하지 않고 **준비한 Dry 구간에 정답 설정을 다시 렌더링**한다.
전체 파일 처리 후 crop했을 때의 필터 상태와 구간 단독 렌더의 상태 차이를 피하기 위한 방식이다.

checkpoint에는 모델 가중치뿐 아니라 모델·전처리 설정, 종류 및 파라미터 순서,
정규화 규칙, 스키마 snapshot/hash, 렌더러 설정, 학습 옵션과 manifest hash를 함께 저장한다.

### CNN 추론과 평가

추론 결과에는 전체 종류별 확률과 보유 이펙터 제약을 적용한 top-K 후보가 포함된다.
후보는 종류별 하나씩, 기본 최대 3개이며 드라이브를 최대 한 개 사용한다.
bypass는 보유 목록이 비어 있어도 허용한다.

각 후보는 초기 `ChainInstance`, `model_score`, 정규화된 연속 벡터 `x0`,
벡터 순서·범위·변환 명세 `vector_spec`를 제공한다.
`model_score`는 종류의 softmax 확률이다. 오디오 유사도는 렌더 후 별도로 계산한다.
범주형·boolean 값은 후보 단계에서 결정하고 연속 탐색 벡터에서 제외한다.

평가는 다음 조건을 구분한다.

1. **정답 종류 제공:** 종류만 정답으로 지정하고 모델이 예측한 노브 초기값을 평가한다.
2. **전체 파이프라인:** 모델이 선택한 후보에 대해 기본값+CMA, CNN-only, CNN+CMA를 비교한다.

두 CMA 방식에는 같은 후보 종류와 같은 렌더 예산을 제공한다.
후보 사전 평가와 bypass 평가도 횟수에 포함한다.
CNN-only는 top-K 초기 후보를 각각 한 번 렌더하여 최소 거리 후보를 선택한다.
보유 목록에 정답 종류가 없는 사례는 별도로 집계한다.

accuracy, macro F1, top-K 포함률, 활성 노브 정규화 MAE, 초기·보정 후 거리,
추론·처리 시간, 렌더 횟수 및 최종 세팅을 저장한다.
정답 정보는 oracle 노브 평가와 지표 계산에만 사용하고, 일반 추론·후보 선택에는 전달하지 않는다.

## 3. estimation 바깥에 새로 추가한 파일

| 파일 | 추가 이유 |
|---|---|
| [`configs/cnn_audio.json`](../configs/cnn_audio.json) | 오디오와 STFT·mel 설정을 파일로 지정하기 위한 기본 설정 |
| [`configs/cnn-requirements-win-py311.lock.txt`](../configs/cnn-requirements-win-py311.lock.txt) | 검증에 실제 사용한 Windows/Python 3.11 pip 패키지 버전 |
| [`requirements-cnn.txt`](../requirements-cnn.txt) | 기존 requirements를 포함하고 PyTorch와 pycma를 추가하는 CNN용 설치 목록 |
| [`tests/test_estimation.py`](../tests/test_estimation.py) | split, 노브 변환, inventory, 마스킹된 손실, 렌더 재현성 및 CMA 예산 관련 테스트 9개 |

## 4. 지원 범위와 공동작업 시 알아둘 제약

| 종류 | 현재 예측·탐색하는 노브 |
|---|---|
| bypass | 없음, 빈 체인 |
| overdrive | `drive`, `level` |
| distortion | `dist`, `tone`, `level` |
| fuzz | `fuzz`, `tone`, `level` |

`fuzz.mode`는 기존 스키마에 있지만 렌더링에는 반영되지 않는다.
따라서 `Vintage`로 고정하며 학습 정답·탐색 변수에서 제외한다.
`amp_id="fixed_amp_v1"`은 기존 체인 형식에 맞춰 유지하지만 현재 앰프 처리는 적용되지 않는다.
컴프레서·EQ·앰프를 이번 모델의 출력이나 학습 정답에 추가하지 않았다.

세 드라이브가 기존 Distortion 알고리즘을 공유하므로 중립 tone과 겹치는 drive 범위에서는
서로 다른 종류가 같은 오디오를 낼 수 있다. 실제 검증에서도 해당 설정의 출력 차이는 0이었다.
종류 복원 정확도와 목표 소리 재현 거리를 함께 해석해야 한다.

## 5. CNN 단일 모델 구성으로 정리한 사항

여러 모델을 비교하던 계획을 종료하고 CNN의 학습·추론·평가에 집중하도록 정리했다.

| 항목 | 정리 내용 |
|---|---|
| 파일 역할 명확화 | `common.py`를 `chain.py`로 바꿔 스키마·노브 변환·체인 후보를 다루는 파일임을 명시했다. |
| CNN 코드 집중 | `search/refine.py`를 `estimation/refine.py`로 옮겼다. 기존 `search/pipeline.py` 등 Step 0 실험 코드는 유지했다. |
| 평가 입력 단순화 | `--checkpoint`를 필수 입력으로 지정하고 외부 예측 JSONL의 `--predictions` 경로를 제거했다. |
| 직접 CNN 추론 | 평가기는 예측 함수를 주입받던 방식 대신 로드된 CNN 모델로 직접 추론한다. smoke와 모든 import를 새 경로로 수정했다. |
| 문서 정리 | 후속 모델 교체와 다중 모델 비교 안내를 제거하고 CNN 실험·평가 순서로 안내한다. |
| 형식과 실험 결과 유지 | CNN 구조, 학습 손실, 파라미터 정규화, manifest와 checkpoint 형식, 결과의 세 평가 방법 키를 유지한다. 기존 데이터와 학습 산출물은 이동하거나 덮어쓰지 않는다. |

데이터 로딩, 오디오 전처리와 체인 생성은 학습·추론·평가에서 계속 사용하므로 역할별 모듈로 유지했다.
기본값+CMA / CNN-only / CNN+CMA는 CNN 초기값과 최적화의 효과를 확인하는 평가 조건으로 유지한다.
코드를 직접 사용하는 작업자는 `estimation.common` → `estimation.chain`,
`search.refine` → `estimation.refine`로 import를 바꿔야 한다.
기존 `python -m estimation.data/train/infer/evaluate/smoke` 진입점과 CNN 체크포인트 명령은 유지한다.

## 6. 검증 결과 및 Git 공유 범위

Windows의 독립 Conda 환경 `.conda-cnn`에서 검증했다.

- 기존 테스트와 신규 테스트를 합쳐 **88 passed, 111 subtests passed**.
- 임시 harmonic 신호 6개에서 24개 pair를 생성하고 1 epoch 학습 후 checkpoint 저장·로드 확인.
- 실제 Pedalboard 렌더와 실제 pycma 실행, 기본 4초 CLI 및 최종 WAV 저장 확인.
- 저장된 Dry와 정답 체인으로 Wet을 재생성한 최대 절대 오차 **0.0**.
- 두 CMA 방식 모두 샘플당 **15회** 렌더. 빈 inventory에서는 bypass 한 번만 평가.
- 초기 버전에서는 외부 예측 JSONL을 다시 읽었을 때 거리·평가 횟수 일치도 확인했다. 해당 입력 경로는 CNN 단일 모델 정리 과정에서 제거했다.

위 결과는 2026-11-09에 실행한 **임시 신호를 이용한 기능 검증**이다.
이후 생성된 실제 데이터의 `cnn_real_model/best.pt`도 이번 코드 정리의 호환성 확인에 사용한다.
자세한 환경·결과·명령은 [검증 기록](VALIDATION.md)에 남겼다.

코드·설정·테스트·문서는 Git으로 공유한다. 로컬 가상환경, 생성 오디오,
checkpoint 및 `outputs/` 실행 산출물은 기존/추가 ignore 규칙에 따라 Git에서 제외된다.
다른 작업자는 문서의 설치 및 실행 명령으로 환경과 산출물을 재생성할 수 있다.

## 7. 분류 / overdrive 노브 회귀 분리

- 기존 outputs 결과 6개 폴더는 `outputs/archive/legacy_before_task_split`에 보관했다.
- 활성 데이터는 `outputs/datasets/drive_classification`과 `overdrive_regression`으로 나눴다.
- `data.prepare`에 효과 종류·bypass 포함 여부·학습 task 설정을 추가했다.
- `PairCNN`은 기존 joint 외에 classification(3종 CE만)과 regression(알려진 OD Huber만)을 지원한다.
  이전 joint checkpoint의 클래스·가중치 구조는 유지했다.
- 회귀에서는 분류 헤드·accuracy를 사용하지 않고 노브별 MAE/RMSE와 기본값 대비 렌더 거리를 평가한다.
- 학습 이력에 분류 loss, 노브 Huber와 normalized MAE를 구분해 저장한다.
- 반복 전처리를 줄이는 선택적 RAM feature cache를 추가했다.
- 분리 모델은 `estimation.experiments evaluate`로 평가하며 기존 joint용 평가와 구분한다.

분리 실험의 구체적인 데이터 구성과 결과는 [EXPERIMENTS.md](EXPERIMENTS.md)를 참고한다.
