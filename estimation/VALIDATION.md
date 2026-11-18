# Conda 검증 기록 — 2026-11-09

아래는 최초 기능 검증 기록이다. 2026-11-10부터 평가 입력은 CNN checkpoint로 통일했으며,
당시 확인했던 외부 예측 JSONL 입력 옵션은 제거했다. 최신 사용법은 [README](README.md)를 참고한다.

브랜치: `feat/cnn-initial-estimator`. 프로젝트 내부 `.conda-cnn`은 새로 만든 독립 Conda 환경이다.
기존 `.venv-cnn`이나 다른 Conda 환경의 패키지를 공유하지 않는다.

| 항목 | 검증 환경 |
|---|---|
| Python | 3.11.18, conda-forge, Windows x64 |
| PyTorch | 2.15.0, CPU |
| Pedalboard | 0.9.26 |
| pycma | 4.5.0 |
| librosa | 0.11.0 |
| NumPy | 2.4.6 |
| SoundFile | 0.14.0 |

전체 pip 버전은 `configs/cnn-requirements-win-py311.lock.txt`에 기록했다.
`python -m pip check` 결과: `No broken requirements found.`

## 자동 테스트

```powershell
python -m pytest -q tests effects metrics schemas
```

**88 passed, 111 subtests passed.** CNN 관련 새 테스트 9개와 기존 테스트를 함께 실행했다.
검증한 조건은 원본별 split 분리/누출 거부, 정규화·역변환/벡터 순서,
inventory 제한, 비활성 fuzz.mode 제외, 무음/NaN/Inf 거부, 상대 음량 보존,
기본 4초 log-mel과 forward, bypass 전체 배치, 비활성 헤드 손실 마스킹,
새 렌더러의 재현성, CMA 예산/부분 population/bypass 빠른 경로다.

pycma가 선택적 matplotlib 부재를 알리는 경고 1개를 출력했다.
이번 실행에서는 그래프 기능을 사용하지 않았고 최적화는 정상 완료했다.

## 소량 전체 흐름

```powershell
python -m estimation.smoke --output outputs/cnn_smoke_conda
```

6개 임시 harmonic 신호에서 24개 pair를 만들었다. 원본 기준으로 train 4개,
validation 1개, test 1개를 먼저 나누고 원본마다 bypass/overdrive/distortion/fuzz를
하나씩 렌더했다. 16 kHz/0.6초 설정으로 1 epoch 학습했다.

데이터 준비 → 로딩 → 학습 → best/last checkpoint 저장 → weights_only 로드 →
추론 → inventory 제한 → ChainInstance/정규화 벡터 변환 → 실제 Pedalboard 렌더 →
실제 pycma 최적화 → JSONL/JSON/WAV 저장을 완료했다.

- 24개 Wet 모두 저장된 Dry와 정답 체인으로 재렌더했을 때 최대 절대 오차 **0.0**.
- 두 CMA 방식 모두 test 4개 각각 **15회** 렌더. 후보 사전 평가 포함.
- CNN-only는 각 샘플 top-3 초기 후보를 각각 한 번 평가하여 **3회** 렌더.
- 빈 inventory에서는 각 방법이 bypass만 **1회** 평가하고 CMA를 실행하지 않음.
- 빈 inventory 평가에서 정답 드라이브 미보유 샘플 **3개**, 정답 bypass **1개**를 별도 집계.
- oracle 노브 평가와 전체 파이프라인 평가는 별도 결과에 기록.

아래 수치는 **임시 신호의 기능 검증 결과이며 실제 기타 데이터의 모델 성능 비교가 아니다**.
지표는 기존 `log_stft_joint_active_energy_v2_floor_m80db`이며 낮을수록 좋다.

| 방법 | 평균 초기 거리 | 평균 최종 거리 | 샘플당 렌더 횟수 |
|---|---:|---:|---:|
| 기본 연속값 + CMA | 0.384531 | 0.305612 | 15 |
| CNN 초기값만 | 0.291267 | 0.291267 | 3 |
| CNN 초기값 + CMA | 0.291267 | 0.178993 | 15 |

실행 산출물은 Git에서 제외된 다음 경로에 있다.

- `outputs/cnn_smoke_conda/report.json`: 패키지 버전과 재현 오차, 전체 평가 요약.
- `outputs/cnn_smoke_conda/prepared/manifest.jsonl`: 분할과 추적 가능한 dry/wet pair.
- `outputs/cnn_smoke_conda/model/best.pt`: 1 epoch 기능 검증용 checkpoint.
- `outputs/cnn_smoke_conda/evaluation/`: 샘플별/집계 결과와 최종 렌더 WAV.
- `outputs/cnn_smoke_conda/no_inventory/`: 미보유 종류를 분리한 평가 결과.
- `outputs/cnn_smoke_conda/inference.json`: 종류 확률, 후보 체인, x0, 변환 명세.

## 클래스 식별 한계 확인

같은 dry에 drive/dist/fuzz=12 dB, level=0 dB, tone=0 dB를 적용했을 때
overdrive 대비 distortion 및 fuzz의 최대 오디오 절대 차이가 모두 **0.0**이었다.
즉, 범위가 겹치고 tone이 중립인 설정에서는 종류 이름을 소리만으로 구분할 수 없다.
fuzz의 Vintage/Modern도 현재 렌더링에 차이가 없어 mode를 학습하지 않는다.
따라서 모든 세팅에서 분류 정답을 유일하게 복원할 수 있다고 가정하면 안 된다.

## 기본 4초 설정의 CLI 검증

별도 `outputs/cnn_cli_conda`에서 README의 실제 CLI 진입점도 실행했다.
동일한 임시 원본 신호를 **mono → 22050 Hz resample → 4초 우측 padding**으로 준비했다.
짧은 신호를 채운 기능 검증이며 4초짜리 실제 기타 녹음을 쓴 것은 아니다.

```powershell
python -m estimation.data --source-index outputs/cnn_cli_conda/sources.json --output outputs/cnn_cli_conda/prepared --audio-config configs/cnn_audio.json --variations 1
python -m estimation.train --manifest outputs/cnn_cli_conda/prepared/manifest.jsonl --output outputs/cnn_cli_conda/model --epochs 1 --batch-size 4 --threads 2
python -m estimation.evaluate --manifest outputs/cnn_cli_conda/prepared/manifest.jsonl --checkpoint outputs/cnn_cli_conda/model/best.pt --owned overdrive distortion fuzz --k 3 --budget 15 --output outputs/cnn_cli_conda/evaluation --save-audio
python schemas/validate.py
```

`estimation.infer` CLI에도 해당 manifest의 dry/reference를 전달하여 `--refine-budget 15`와
`--rendered-output outputs/cnn_cli_conda/final.wav`를 실행했다.
최종 WAV는 22050 Hz / 88200 samples / 4초이며 JSON에 최종 체인과 거리를 저장했다.

- 24개 pair 모두 정답 재렌더 오차 **0.0**.
- 4초 입력의 1 epoch 학습, checkpoint 로드, 후보/최종 WAV 출력 완료.
- 기본값+CMA, CNN-only, CNN+CMA의 평균 최종 거리는 각각 **0.278726 / 0.251825 / 0.208310**.
- 두 CMA 방식 모두 샘플당 **15회**, CNN-only는 **3회** 렌더.
- 초기 버전의 외부 prediction JSONL 입력에서도 각 방법의 거리/횟수 일치를 확인했다. 이 입력 옵션은 현재 제거되어 당시 산출물만 보관한다.
- 기존 스키마 6종과 예시 참조 검증도 통과.

checkpoint: `outputs/cnn_cli_conda/model/best.pt`.
추론 결과: `outputs/cnn_cli_conda/inference.json`, 최종 오디오: `outputs/cnn_cli_conda/final.wav`.
집계 결과: `outputs/cnn_cli_conda/evaluation/summary.json`.

## 최초 기능 검증 당시의 한계

2026-11-09 기능 검증 당시에는 지정된 `data/dry/Fender_Strat_Clean/audio`에 실제 clean dry 녹음이 없어
실제 기타 데이터의 정확도/일반화 성능이나 장시간 학습은 검증하지 않았다.
GPU, 외부 VST, 실물 페달, 다른 연주의 reference도 이번 검증 범위가 아니다.
실제 데이터를 준비한 뒤 README의 같은 manifest/평가 명령으로 측정해야 한다.

## CNN 단일 모델 구성 정리 후 재검증 — 2026-11-10

`common.py`를 `chain.py`로, `search/refine.py`를 `estimation/refine.py`로 옮기고
평가기를 CNN checkpoint 직접 추론으로 단순화했다. 외부 예측 JSONL 입력은 제거했다.
CNN 구조·학습 손실·데이터 전처리·checkpoint 형식·CMA 알고리즘은 유지했다.

- `python -m pytest -q tests effects metrics schemas`: **88 passed, 111 subtests passed**.
- 기존 실제 데이터 모델 `outputs/cnn_real_model/best.pt`(19 epoch) 로드 성공.
- 실제 데이터 validation 샘플 4개에서 정리 전후의 종류 확률, 노브 값,
  top-K 후보, CMA 최종 체인·거리·평가 횟수와 렌더 오디오 hash가 모두 정확히 일치.
- 비교에 사용한 기존 checkpoint와 manifest의 파일 hash도 변경되지 않음.
- `estimation.evaluate` CLI가 `--checkpoint`를 필수로 받고 외부 예측 입력 옵션을 노출하지 않음을 확인.
- 기존 실제 데이터 manifest/checkpoint로 validation 4개를 평가하고 최종 WAV 저장 성공.
  해당 확인에서는 두 CMA 방식에 각각 9회, CNN-only에 3회 렌더 예산 사용.
- `python -m estimation.smoke --output outputs/cnn_smoke_cnn_only`: 24개 임시 신호 pair,
  1 epoch 학습부터 checkpoint 저장·로드·추론·렌더·CMA·빈 inventory 평가까지 완료.
  Wet 재현 오차 0.0, 두 CMA 방식의 샘플당 15회 렌더 조건 유지.

이는 코드 정리에 따른 동작 유지 확인이며 추가 성능 개선 실험은 아니다.
기존 실제 데이터의 20 epoch 학습을 다시 실행하거나 기존 학습 결과를 덮어쓰지 않았다.
동일한 선택적 matplotlib 경고 1개가 있었으며 최적화는 정상 완료했다.

로컬 산출물:

- `outputs/cnn_cleanup_check/before.json`, `after.json`: 기존 모델의 정리 전후 결과 비교.
- `outputs/cnn_cleanup_check/evaluation/`: 기존 checkpoint를 사용한 새 평가 CLI 결과.
- `outputs/cnn_smoke_cnn_only/report.json`: 정리 후 소량 전체 흐름 검증 기록.

## 분류와 overdrive 노브 회귀 분리 실험 — 2026-11-10

기존 outputs 폴더 6개는 `outputs/archive/legacy_before_task_split`에 보관했다.
기존 실제 기타 원음 78개로 두 데이터 폴더를 새로 구성하고 각 CNN을 30 epoch 학습했다.

- 3종 분류: 1,170쌍, bypass 제외, classification head와 CE만 학습.
  선택된 30 epoch 모델의 validation/test accuracy는 60.56% / 57.78%.
- overdrive 노브 회귀: 분류 데이터와 동일한 OD 390쌍만 사용.
  알려진 OD의 drive/level head와 Huber만 학습하며 분류 accuracy는 계산하지 않음.
  선택된 18 epoch 모델의 validation/test normalized MAE는 0.1679 / 0.1579.
  test 기본값 MAE 0.2564 대비 평균 정규화 오차 약 38.4% 감소.
- 자동 테스트 **90 passed, 111 subtests passed**.
- 기존 joint checkpoint 로드 성공. 두 데이터의 기존 원본 split과 OD 오디오/정답 일치 확인.
- 이 결과는 실제 기타 Dry와 기존 소프트웨어 렌더러로 생성한 Wet의 검증이다.
  distortion의 test recall 18.33%, OD 노브의 약 3 dB MAE 등 한계가 남아 있다.

조건·실행 명령·혼동행렬·노브별 오차와 청취 예시는 [EXPERIMENTS.md](EXPERIMENTS.md)를 참고한다.
