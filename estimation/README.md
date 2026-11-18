# VST 7종 CNN 초기 추정

생성된 Dry/Wet 쌍으로 드라이브 종류, 노브 초기값, 컴프·EQ 사용 여부와 노브를 함께 학습한다.
공유 데이터·팀 스키마·공용 렌더러는 이 모듈에서 변경하지 않는다.

- 드라이브: `808_scream`, `b_drive`, `o_drive`, `one_distortion`, `bat_distortion`, `f_fuzz`, `m_fuzz`.
- 체인 순서: 선택적 Compressor → VST 드라이브 1개 → 선택적 EQ.
- 입력: 생성 폴더 `manifest.json`의 `dry_id_map[].render_input`만 사용한다.
  이미 peak −12 dBFS인 `dry/`를 추가 정규화하지 않는다.
- 라벨의 생성자 PC 절대경로 대신 현재 `audio/<target_id>.wav`를 사용한다.
- 원본 Dry ID·내용 해시를 기준으로 train/validation/test 누출을 검사한다.
- 44.1 kHz 원본 렌더링 입력을 따로 캐시한다. CNN 입력은 22.05 kHz, 2.5초, 64-band log-mel이다.

## 모델과 loss

가중치를 공유하는 CNN으로 Dry와 Wet 특징을 각각 추출한다.
CPU 학습용 Conv(8/16/32/64) + GroupNorm + ReLU + MaxPool 블록 4개 후
주파수 영역 4개를 보존하는 평균 풀링을 한다.
Dry, Wet, Wet−Dry, 차이의 절댓값을 연결하고 256차원 특징으로 변환한다.
7-class softmax, 컴프·EQ sigmoid, 효과별 노브 sigmoid, 음향 거리 보조 head가 출력된다.

`loss = CE(drive) + BCE(stage presence) + Huber(active knobs) + 0.1 × Huber(distance components)`.
노브 loss는 정답 체인에 실제 있는 효과에만 적용한다. 스키마의 선형/로그 범위로 정규화한다.
생성 시 샘플링되지 않은 `input_gain_db`는 학습하지 않고 렌더링에서 0으로 고정한다.

## 유사도 d의 두 가지 용도

`d = 0.5 × spectral / spectral_scale + 0.5 × dynamics / dynamics_scale`.
공통 `metrics.combined`를 사용하며 scale은 **train 원본의 Dry/target 거리 중앙값**으로 고정한다.
validation/test로 scale을 다시 맞추지 않는다. 낮을수록 가깝고 지각적 확률이 아니다.

1. **보조 학습**: 이미 있는 Dry/Wet에서 측정한 두 거리 성분을 예측한다.
   이 MAE는 `d(Dry, target)` 추정 오차이며, 예측 노브의 렌더링 품질과 다르다.
2. **실제 출력 비교**: CNN의 전체 예측 체인을 원래 표본율로 렌더링한 뒤 CNN 평가 표본율로 변환한다.
   Top-1, Top-3 중 d 최소 후보, 같은 예측 구조의 기본 노브, 정답 전체 체인을 비교한다.
   VST 활성 상태와 7종 정답 재현 오차를 먼저 검사한다.
   정답 재현 오차가 남으면 그 사실과 oracle d를 함께 기록하며, CNN 오차로만 해석하지 않는다.

VST는 비미분 렌더러다. 렌더링 d로 직접 역전파하지 않으며, 학습 중에는 정해진 validation
샘플 14개를 주기적으로 렌더링한다. 모델 선택은 전체 validation supervised loss로만 한다.
test는 선택된 best checkpoint로 마지막에 평가한다.
종류·노브·사용 여부는 split 전체를 평가한다. CPU 비용이 큰 실제 렌더링 d는
각 split에서 사전에 정한 70개(종류별 10개)를 균형 추출하여 비교한다.
`estimation.evaluate --render-samples -1`로 전체 렌더링 평가도 가능하다.

## 실행

```powershell
# 기존 공유 드라이브 연결 폴더를 읽어 로컬 캐시 생성
.\.conda-cnn\python.exe -m estimation.data `
  --dataset data/synthetic/step1_vst_v0_임시 --output outputs/vst7/dataset

# 캐시가 있으면 재사용, 30epoch 학습 후 validation/test 평가
.\.conda-cnn\python.exe -m estimation.experiments `
  --dataset data/synthetic/step1_vst_v0_임시 --output outputs/vst7 --epochs 30

# 학습 완료 후 같은 best checkpoint로 평가만 다시 실행
.\.conda-cnn\python.exe -m estimation.experiments `
  --dataset data/synthetic/step1_vst_v0_임시 --output outputs/vst7 `
  --evaluate-only --render-samples 70

# 생성 폴더 dry/와 목표음을 입력으로 후보 JSON 출력
.\.conda-cnn\python.exe -m estimation.infer `
  --checkpoint outputs/vst7/model/best.pt `
  --dry '<생성 폴더>/dry/<파일>.wav' --reference '<목표>.wav' `
  --k 3 --output outputs/vst7/prediction.json
```

학습·추론은 WAV/캐시만 필요하다. 실제 렌더링 평가·CMA-ES는 활성화된 MultiDrive PRO가 필요하다.
`HAI_VST_DIR`에 설치 폴더, VST 번들 또는 내부 바이너리 경로를 지정할 수 있다.
라이선스 접근이 제한되는 샌드박스에서는 출력이 Dry 그대로일 수 있다. preflight 실패를 무시하지 않는다.
플러그인 설치나 체험판 시작을 이 코드가 자동 수행하지 않는다.

## 탐색으로 넘기는 후보 형식

각 후보에는 `effect_type`, `model_score`, `chain`, `x0`, `vector_spec`가 있다.
`chain`은 공통 `amp_id/chain/effect_type/values` 형식이며 앰프는 실제 렌더링하지 않는다.
`vector_spec[].effect_index`는 전체 체인에서 노브가 속한 효과 인덱스다.
CMA-ES는 같은 순서의 0~1 좌표 `x0`로 시작하고 같은 `d`를 최소화한다.
`model_score`는 종류 확률이므로 오디오 거리로 해석하지 않는다.

현재 `estimation/refine.py`는 팀 `search/pipeline.py`의 Restart CMA-ES를 호출한다.
후보 변환, 초기값 보존, 동일 거리·예산 비교와 실행 명령은
[CNN_CMA_EXPERIMENT.md](CNN_CMA_EXPERIMENT.md)에 정리했다.
`python -m estimation.cma_experiment --prepare-only`로 VST 없이 실험 입력을 생성할 수 있다.

상세 실행 결과는 [EXPERIMENTS.md](EXPERIMENTS.md), 검증은 [VALIDATION.md](VALIDATION.md),
공동 작업 변경점은 [CHANGE_SUMMARY.md](CHANGE_SUMMARY.md)에 기록한다.
