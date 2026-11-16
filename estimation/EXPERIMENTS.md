# 드라이브 분류와 overdrive 노브 예측 분리 실험

기존 결합 모델에서 종류 분류와 노브 예측을 분리해 각각 검증한다.
입력은 기존과 같은 Dry/reference 쌍과 공유 CNN Encoder다.

| 실험 | 학습 데이터 | 출력 | 학습 손실 | 주요 검증 지표 |
|---|---|---|---|---|
| `drive_classification` | overdrive / distortion / fuzz, bypass 제외 | 3종 확률 | Cross-Entropy만 사용 | accuracy, macro F1, 종류별 recall, confusion matrix |
| `overdrive_regression` | overdrive만 포함, bypass 제외 | 정규화된 drive / level | Huber만 사용 | 노브별 normalized MAE·RMSE, dB MAE, 기본값 대비 오디오 거리 |

회귀 모델은 종류를 overdrive로 고정한다. 분류 헤드와 분류 손실을 만들지 않으며,
한 종류만 있다는 이유로 accuracy=100%를 보고하지 않는다. 해당 값은 `null`로 기록한다.
분류 모델에는 노브 회귀 헤드와 회귀 손실을 만들지 않는다.
기존 결합 모델의 checkpoint는 기본 `joint` 설정으로 계속 로드할 수 있다.

## 결과 폴더

```text
outputs/
├── archive/
│   ├── legacy_before_task_split/    # 이전 검증·데이터·학습 결과 보관
│   └── task_split_smoke/            # 분리 모델의 소량 기능 검증
├── datasets/
│   ├── drive_classification/        # 3종 분류용 데이터
│   └── overdrive_regression/        # overdrive 전용 데이터
└── experiments/
    ├── source_index.json            # 기존 원본 녹음 ID와 파일 위치
    ├── plan.json                    # 설정과 데이터 구성
    ├── drive_classification/
    │   ├── model/                   # best.pt, last.pt, history.json
    │   ├── validation/              # summary.json, samples.jsonl
    │   └── test/
    └── overdrive_regression/
        ├── model/
        ├── validation/
        └── test/
```

이전 결과 6개 폴더는 삭제하지 않고 `archive/legacy_before_task_split`에 모았다.
원래 위치와 보관 위치는 그 안의 `moves.json`에 기록되어 있다.
원본 공유 드라이브 오디오는 수정하지 않는다. `outputs`의 산출물은 Git에서 제외된다.

## 데이터 조건

- 기존 원본 녹음 78개를 같은 seed=42로 분리: train 54 / validation 12 / test 12.
- 원본마다 각 드라이브에 대해 노브 설정 5개를 스키마 범위에서 생성.
- 분류 데이터: 총 1,170쌍. train 810 / validation 180 / test 180, 종류별 균등.
- 회귀 데이터: 총 390쌍. train 270 / validation 60 / test 60.
- 회귀 폴더는 분류 데이터의 overdrive 쌍을 파일 그대로 복사해 구성한다.
  두 실험의 원본별 split과 overdrive의 Dry/Wet·정답 값은 동일하다.
- mono / 22050 Hz / 기본 4초, STFT 1024·256, mel 64.
  원본 2.5초 뒤의 나머지는 우측 zero padding하며 상대 음량을 유지한다.
- `fuzz.mode`는 현재 렌더링에 반영되지 않아 Vintage로 고정.

정답은 기존 Pedalboard 렌더러를 사용한 소프트웨어 처리의 종류와 노브다.
중립 tone과 겹치는 drive 설정에서는 서로 다른 종류가 같은 소리를 낼 수 있으므로
분류 성능의 식별 한계는 남아 있다. 이번 분리 실험에서 렌더러 알고리즘은 바꾸지 않는다.

## 실행

저장소 루트에서 Conda 환경을 활성화한다.

```powershell
conda activate .\.conda-cnn
```

새 output 경로로 두 실험 전체를 실행하려면:

```powershell
python -m estimation.experiments run --dry-dir data/dry/Fender_Strat_Clean/audio --output-root outputs/new_run --variations 5 --epochs 30 --seed 42 --batch-size 16 --threads 4
```

원본 ID를 이미 정의한 경우 `--dry-dir` 대신 `--source-index`를 사용한다.
아래는 현재 실험에 사용한 원본 인덱스로 데이터 준비와 학습을 나눠 실행한 명령이다.
이미 존재하는 데이터나 실험 결과는 덮어쓰지 않으므로 재실행에는 새 output 경로를 사용한다.

```powershell
python -m estimation.experiments prepare --source-index outputs/experiments/source_index.json --output-root outputs --audio-config configs/cnn_audio.json --variations 5 --seed 42
python -m estimation.experiments train --output-root outputs --epochs 30 --seed 42 --batch-size 16 --threads 4
```

학습 중 log-mel을 RAM에 캐시하여 매 epoch의 반복 추출을 줄인다.
모델은 동일한 CNN 블록(16/32/64/128)을 사용하며 Adam, learning rate=0.001,
batch size=16, 최대 30 epoch로 학습한다. validation loss 최저 checkpoint를 선택한 뒤
validation과 test를 평가한다. test 결과로 checkpoint를 선택하거나 재학습하지 않는다.

분리 모델의 평가를 다시 실행하려면:

```powershell
python -m estimation.experiments evaluate --manifest outputs/datasets/drive_classification/manifest.jsonl --checkpoint outputs/experiments/drive_classification/model/best.pt --split validation --output outputs/experiments/drive_classification/recheck_validation
python -m estimation.experiments evaluate --manifest outputs/datasets/overdrive_regression/manifest.jsonl --checkpoint outputs/experiments/overdrive_regression/model/best.pt --split validation --output outputs/experiments/overdrive_regression/recheck_validation
```

결합 모델용 `estimation.infer` / `estimation.evaluate`와 구분해 분리 모델에는 위 명령을 사용한다.

## 지표 해석

3종 분류는 균등 데이터이므로 무작위 선택의 기대 정확도는 약 33.3%다.
accuracy와 macro F1 및 혼동행렬을 함께 확인한다.

overdrive 노브는 drive [0,15] dB, level [-12,12] dB를 각각 [0,1]로 변환해 예측한다.
normalized MAE 0.1은 해당 노브 전체 범위의 10%만큼 평균 오차가 난다는 의미다.
`mae_in_original_units`는 실제 dB 단위 오차다.
`all_knobs_within_0_1_rate`는 두 노브의 정규화 오차가 모두 0.1 이하인 샘플 비율이다.

회귀 평가에는 같은 데이터에서 스키마 기본값(drive=6, level=0)의 노브 오차와
렌더링 거리도 함께 저장한다. 종류를 알려준 노브 초기값 검증이므로 CMA-ES는 적용하지 않는다.
오디오 거리는 기존 프로젝트 지표를 사용하며 작을수록 좋다.

## 실행 결과 — 2026-11-10

### 3종 드라이브 분류

validation CE가 가장 낮은 **30 epoch** checkpoint를 선택했다.

| Split | 샘플 수 | Accuracy | Macro F1 |
|---|---:|---:|---:|
| validation | 180 | 60.56% (109/180) | 0.5622 |
| test | 180 | 57.78% (104/180) | 0.5406 |

| 정답 종류 | Validation recall | Test recall |
|---|---:|---:|
| overdrive | 90.00% | 85.00% |
| distortion | 18.33% | 18.33% |
| fuzz | 73.33% | 70.00% |

Test 혼동행렬은 행이 정답, 열이 예측이다.

| 정답 / 예측 | overdrive | distortion | fuzz |
|---|---:|---:|---:|
| overdrive | 51 | 9 | 0 |
| distortion | 25 | 11 | 24 |
| fuzz | 7 | 11 | 42 |

무작위 3종 선택의 기대 정확도보다 높지만 **distortion 구분력은 부족하다**.
현재 렌더러가 같은 Distortion 처리와 겹치는 drive 범위를 공유한다는 조건을 함께 고려해야 한다.
이번 실험으로 실물 페달 3종을 구분한다고 해석하지 않는다.

상세 결과:

- [분류 validation 요약](../outputs/experiments/drive_classification/validation/summary.json)
- [분류 test 요약](../outputs/experiments/drive_classification/test/summary.json)
- [분류 학습 이력](../outputs/experiments/drive_classification/model/history.json)

### overdrive 노브 예측

30 epoch 학습 후 validation Huber가 가장 낮은 **18 epoch** checkpoint를 선택했다.
종류는 overdrive로 고정하고 drive/level 두 노브만 예측했다. CMA-ES는 사용하지 않았다.

| Split | 샘플 수 | 예측 normalized MAE | 기본값 normalized MAE | 두 노브 모두 정규화 오차 0.1 이하 |
|---|---:|---:|---:|---:|
| validation | 60 | 0.1679 | 0.2477 | 31.67% |
| test | 60 | 0.1579 | 0.2564 | 36.67% |

| Split | 노브 | 예측 MAE (dB) | 기본값 MAE (dB) | 예측 normalized MAE |
|---|---|---:|---:|---:|
| validation | drive | 3.098 | 3.847 | 0.2065 |
| validation | level | 3.102 | 5.734 | 0.1293 |
| test | drive | 2.959 | 3.901 | 0.1972 |
| test | level | 2.845 | 6.067 | 0.1185 |

| Split | 예측 노브의 평균 오디오 거리 | 기본값의 평균 오디오 거리 |
|---|---:|---:|
| validation | 0.019153 | 0.104144 |
| test | 0.011235 | 0.098105 |

지표는 기존 `log_stft_joint_active_energy_v2_floor_m80db`이며 낮을수록 좋다.
test의 평균 normalized MAE는 기본값 대비 약 **38.4% 감소**했다.
두 노브 모두 기본값보다 평균 오차가 작지만, dB MAE가 약 3 dB 남아 있어 정답 노브를
정확하게 복원하는 단계라고 보기는 어렵다. 낮은 오디오 거리와 정확한 노브 복원은 별도로 해석한다.

상세 결과:

- [노브 validation 요약](../outputs/experiments/overdrive_regression/validation/summary.json)
- [노브 test 요약](../outputs/experiments/overdrive_regression/test/summary.json)
- [노브 학습 이력](../outputs/experiments/overdrive_regression/model/history.json)
- [test 청취 예시 인덱스](../outputs/experiments/overdrive_regression/audio_examples/index.json)
- [두 실험 통합 요약](../outputs/experiments/summary.json)

청취 예시는 test 샘플의 순서상 처음 3개를 사용했으며 오차가 작은 사례를 골라내지 않았다.
인덱스에는 정답/예측 노브, 데이터 폴더의 Dry/target 위치 및 예측/기본값 렌더 WAV가 있다.

### 검증 범위와 한계

- 원본 녹음 기준 split을 유지했고, 두 데이터 폴더의 OD 390쌍은 오디오와 정답이 모두 동일하다.
- 종류·split별 총 9개 사례에서 정답 세팅으로 Wet을 재렌더한 최대 절대 오차는 0.0이었다.
- 자동 테스트: **90 passed, 111 subtests passed**. 분리 head/loss와 OD 쌍·split 동일성 테스트 포함.
- 실제 기타 Dry를 사용한 합성 Wet 실험이며 단일 기타·단음 데이터와 seed=42 한 번의 결과다.
  다른 기타/연주자/실물 페달 및 다른 연주의 reference 일반화는 검증하지 않았다.
- 모델 선택은 validation loss로만 수행했고 선택된 모델의 test 결과를 위에 기록했다.
- 패키지 추가 다운로드 없이 기존 Conda 환경에서 수행했다. 원음 읽기 이후 학습·평가는 로컬 데이터만 사용했다.
