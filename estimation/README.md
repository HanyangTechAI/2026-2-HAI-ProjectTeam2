# CNN 초기 추정 기준 모델

이 기준 모델은 같은 연주의 dry/reference log-mel을 공유 CNN으로 인코딩하고
`concat(zd, zr, zr-zd, abs(zr-zd))`로 종류와 종류별 노브 초기값을 예측한다.
4개 Conv2D 블록(16/32/64/128), GroupNorm, ReLU, MaxPool, global average pooling을 쓴다.
특정 논문의 재현 구현이 아니다. UI는 변경하지 않는다.

현재 실험 모델은 CNN 하나다. 데이터 준비부터 학습·추론·평가·CMA 보정까지
`estimation` 패키지에서 실행한다. 기존 `search` 폴더는 Step 0 탐색 실험을 보관한다.

**현재 진행하는 3종 드라이브 분류 / overdrive 노브 예측 분리 실험은
[EXPERIMENTS.md](EXPERIMENTS.md)에 정리했다.** 아래 명령들은 기존 결합 모델도 계속 지원한다.

```text
estimation/
├── audio.py       # 오디오 구간과 log-mel 전처리
├── data.py        # 원본별 split, Dry/Wet 생성과 manifest
├── chain.py       # 스키마, 노브 변환, inventory와 후보 체인
├── cnn.py         # 공유 CNN과 학습 손실
├── train.py       # Dataset, 학습·검증과 checkpoint 저장
├── infer.py       # checkpoint 추론과 top-K 후보 출력
├── refine.py      # CNN 후보의 CMA-ES 보정
├── evaluate.py    # CNN 종류·노브·오디오 재현 평가
├── experiments.py # 3종 분류 / overdrive 노브 예측 분리 실험
├── smoke.py       # 임시 신호로 전체 흐름 검증
└── *.md           # 사용법, 변경 사항과 검증 기록
```

| 파일 | 역할 |
|---|---|
| `audio.py`, `configs/cnn_audio.json` | waveform 구간과 CNN log-mel 설정 |
| `data.py` | 원본 단위 split, 기존 label 재사용, 합성 pair/manifest 준비와 검증 |
| `chain.py` | 스키마 기반 활성 노브, 정규화, ChainInstance, inventory 후보 |
| `cnn.py`, `train.py`, `infer.py` | 공유 CNN, 마스킹된 손실, 학습/checkpoint/추론 |
| `refine.py` | CNN 후보의 CMA warm-start와 정확한 렌더 예산 |
| `evaluate.py` | CNN의 oracle/전체 평가와 결과 저장 |
| `smoke.py`, `tests/test_estimation.py` | 소량 전체 흐름과 주요 불변 조건 검증 |
| `VALIDATION.md`, `configs/cnn-requirements-win-py311.lock.txt` | 실제 Conda 검증 기록과 패키지 버전 |

기존 `effects/render.py`는 패키지를 사용할 때 import하도록 조정하여
노브 정의 조회를 DSP 패키지 설치와 독립적으로 사용할 수 있게 했다. 오디오 처리 구성은 같다.

## Windows Anaconda 설치

저장소 루트에서 Anaconda Prompt를 사용한다.

```powershell
conda create --prefix .\.conda-cnn python=3.11 pip --solver classic --override-channels --channel conda-forge --yes
conda activate .\.conda-cnn
python -m pip install -r requirements-cnn.txt
```

검증한 Windows/Python 3.11 패키지 버전을 그대로 사용하려면 설치 명령을
`python -m pip install -r configs/cnn-requirements-win-py311.lock.txt`로 대체한다.

PowerShell에서 활성화하지 않고 실행하려면 `python` 대신 `& '.\.conda-cnn\python.exe'`를 사용한다.
환경은 프로젝트 내부에만 만들며 Git에서 제외한다. GPU 없이 CPU로 검증할 수 있다.

## 실제 지원 범위와 가정

`schemas/effects.data.json`이 종류, 범위, 기본값과 파라미터 순서의 원본이다.

| 분류 | 활성 연속 노브 (스키마 순서, dB) | 처리 |
|---|---|---|
| bypass | 없음 | 빈 chain |
| overdrive | drive [0,15], level [-12,12] | Distortion → Gain |
| distortion | dist [0,40], tone [-9,9], level [-12,12] | Distortion → PeakFilter(1000 Hz) → Gain |
| fuzz | fuzz [10,50], tone [-9,9], level [-12,12] | Distortion → PeakFilter(800 Hz) → Gain |

`fuzz.mode`는 스키마에 있으나 렌더러에서 구현되지 않았다. 항상 `Vintage`로 고정하고
학습/평가/탐색에서 제외한다. 현재 활성 boolean/categorical 노브는 없다.
실제로 렌더링되는 해당 타입이 추가되면 종류별 CE/BCE 헤드와 손실이 적용된다.
비활성 노브에 임의의 분류 정답을 만들지 않는다.

기존 `amp_id="fixed_amp_v1"` 문자열을 유지하지만 렌더러는 앰프를 적용하지 않는다.
컴프레서/EQ/앰프를 추가하지 않는다. 각 렌더는 기존 `render_chain`의 새 Pedalboard를
사용하여 상태를 공유하지 않는다. reference_pedal은 메타데이터이며 실물 페달 재현을 뜻하지 않는다.

세 종류가 같은 Distortion 알고리즘을 공유한다. **tone=0이고 drive가 겹치는 경우
overdrive/distortion/fuzz가 같은 소리를 낼 수 있다.** 서로 다른 필터 주파수도
gain=0이면 종류를 구분하는 정보를 주지 못한다. 소리만으로 원래 종류를 완전히 복원할 수 있다는
가정은 하지 않는다. 분류 지표와 재현 오디오 거리를 함께 해석한다.

## 데이터 준비

기본 4초, 22050 Hz, mono, STFT 1024/256, mel 64의 설정은 `configs/cnn_audio.json`에 있다.
설정 전체가 manifest와 checkpoint에 저장된다. 짧은 입력은 우측 zero padding하며,
독립 peak normalization이나 증강은 하지 않는다. stereo는 mono 평균으로 변환한다.
원본을 resample/segment/pad한 뒤 wet을 렌더하므로 저장된 dry와 정답 체인으로 wet을 재현할 수 있다.
dry와 wet의 crop은 동일하고 자동 시간 이동 보정은 하지 않는다.
실제 reference가 다른 연주라면 이 합성 paired 학습 조건과 다르다.

각 파일이 서로 다른 원본 녹음인 경우:

```powershell
python -m estimation.data --dry-dir data/dry/Fender_Strat_Clean/audio --output outputs/cnn_data --audio-config configs/cnn_audio.json --variations 3 --seed 42
```

이미 crop된 파일이나 같은 take의 여러 파일이 있으면 파일명만으로 원본을 추측하지 않는다.
사용자가 **같은 원본에는 같은 recording_id**를 지정한 JSON source index를 제공해야 한다.
경로는 index 파일 기준이며, `starts_sec`는 resample한 오디오의 crop 위치다.

```json
[
  {"recording_id": "take-01", "file_path": "audio/one.wav", "starts_sec": [0, 4]},
  {"recording_id": "take-02", "file_path": "audio/two.wav"},
  {"recording_id": "take-03", "file_path": "audio/three.wav"}
]
```

```powershell
python -m estimation.data --source-index data/sources.json --output outputs/cnn_data
```

원본 ID를 먼저 70/15/15로 분할한다. 작은 데이터에서는 validation/test에 각각 최소 원본 1개를
할당하므로 비율이 달라진다. 최소 3개 원본이 필요하다. 각 crop마다 bypass와 드라이브 변형을 만든다.
`--validation`, `--test`로 비율을 바꿀 수 있다. 손상/무음/비유한 입력은 제외 사유를
`manifest.meta.json`의 rejected에 남기고, 사용 가능한 split이 빠지면 실패한다.

기존 생성기 `scripts/Create_Synthetic_pairs.py`는 bypass/split/crop manifest를 만들지 않는다.
기존 `TargetSample` label이 있다면 `--legacy-labels data/synthetic/Fender_Strat_Clean/labels`로
설정을 재사용한다. source의 `sample_id` 또는 파일 stem을 기존 `dry_sample_id`와 연결한다.
**기존 wet을 그대로 복사하지 않고** 해당 label 설정으로 준비된 dry segment를 다시 렌더한다.
전체 파일을 처리한 뒤 crop하면 필터의 이전 상태가 섞일 수 있기 때문이다.
기존 wet과 동일하다고 주장하지 않으며 원 label 경로를 기록한다.

`manifest.jsonl`에는 sample_id, 원본 recording_id/path/hash, split, crop 위치/유효 길이,
dry/wet 경로, 기존 ChainInstance 형식의 ground_truth_chain이 있다.
이는 학습용 외부 manifest이며 기존 JSON Schema 문서에 임의 필드를 추가하지 않는다.
실제 오디오 포맷과 split 누출, 정답 범위를 읽을 때 검증한다.

## 학습

```powershell
python -m estimation.train --manifest outputs/cnn_data/manifest.jsonl --output outputs/cnn_model --epochs 20 --batch-size 16 --learning-rate 0.001 --class-weight 1 --knob-weight 1 --seed 42
```

분류 CE + 활성 연속 노브 Huber + 활성 범주형 CE/boolean BCE다.
정답 종류의 노브만 손실에 포함하며 전부 bypass인 배치도 안전하다.
연속 노브는 스키마 range에 따라 [0,1]로 선형 변환한다. 현재 드라이브에는 로그 스케일 노브가 없다.
`scale=log`인 활성 노브에는 `(ln(value)-ln(lo))/(ln(hi)-ln(lo))`와 그 역변환을 공통 적용한다.
log-mel은 자연로그 power/10이며 고정 floor만 사용하여 상대 음량을 유지한다.

validation의 가중 합성 loss가 가장 낮은 `best.pt`, 마지막 `last.pt`, `history.json`을 저장한다.
checkpoint에는 weights, optimizer, epoch, model/preprocessing 설정, 종류/파라미터 순서,
정규화 설명, catalog 전체/hash, renderer 설정, seed, manifest hash가 포함된다.
test는 모델 선택에 사용하지 않는다. `--device`, `--threads`, `--num-workers` 설정이 가능하다.

## 추론과 최종 오디오

```powershell
python -m estimation.infer --checkpoint outputs/cnn_model/best.pt --dry path/to/dry.wav --reference path/to/reference.wav --owned overdrive distortion fuzz --k 3 --output outputs/prediction.json
python -m estimation.infer --checkpoint outputs/cnn_model/best.pt --dry path/to/dry.wav --reference path/to/reference.wav --owned overdrive distortion fuzz --k 3 --refine-budget 60 --rendered-output outputs/final.wav --output outputs/final.json
```

`--start-sec`는 두 입력에 공통 적용한다. default top-K=3이다. 보유 목록을 생략하거나
`--owned` 뒤를 비우면 bypass만 가능하다. unknown/미지원 종류는 후보에서 제외한다.
전체 종류의 softmax 확률을 내보내고 보유 종류+bypass를 확률 내림차순으로 정렬한다.
허용 후보가 K보다 적으면 그만큼 반환한다. 각 후보는 ChainInstance, model_score, x0,
파라미터 순서/range/transform/normalized_bounds를 포함한다.
**model_score는 종류의 softmax 확률이며 오디오 유사도가 아니다.**
범주형/boolean은 argmax/threshold로 하나를 선택하고 연속 벡터에서 제외한다.
종류당 후보 하나, 전체 최대 K개이며 조합 열거는 하지 않는다.
bypass는 CMA 없이 한 번 렌더/거리 평가한다.

기존 `search.pipeline.run_cma_es`는 overdrive 2차원/중앙 초기값에 고정되어 있다.
`estimation/refine.py`는 같은 pycma bounded ask/tell을 사용하는 동적 차원 warm-start 어댑터다.
기존 Step 0 실험의 상수나 초기화 정책은 변경하지 않는다.

## CNN 평가

```powershell
python -m estimation.evaluate --manifest outputs/cnn_data/manifest.jsonl --checkpoint outputs/cnn_model/best.pt --owned overdrive distortion fuzz --k 3 --budget 60 --output outputs/cnn_eval --save-audio
```

평가는 학습한 CNN의 `--checkpoint`를 필수 입력으로 받는다.

각 샘플에서 다음을 분리한다.

1. oracle-knobs: 정답 종류만 제공하고 모델 노브를 렌더한다. inventory를 무시하며 노브 초기값만 평가한다.
2. default+CMA: 모델이 선택한 동일 top-K 종류에 기본 연속값과 예측 discrete값을 주고 CMA를 실행한다.
3. CNN-only: top-K 예측 초기값을 각각 한 번 렌더하여 최소 거리 후보를 선택한다. CMA는 실행하지 않는다.
4. CNN+CMA: 같은 top-K 종류/예측 discrete값에 예측 연속 초기값으로 CMA를 실행한다.

두 CMA 방식은 **같은 후보와 총 렌더 예산**을 사용한다. 후보 사전 평가와 bypass 평가도 포함한다.
사전 평가 후 남은 예산을 연속 후보별로 균등 분배한다. partial population도 평가 횟수에 포함한다.
연속 후보가 전혀 없으면 bypass 한 번만 평가하고 unused_budget을 기록한다.
예산이 후보 수보다 작으면 실패한다. 기존 스펙트럼 지표를 재사용하며 방향은 작을수록 좋다.

`summary.json` 및 `samples.jsonl`에 accuracy/macro F1(전체 고정 클래스, zero_division=0),
inventory 제약 분류, top-K 포함률, 활성 노브 normalized MAE, discrete accuracy,
초기/보정 후 거리, 추론/처리 시간, 렌더 횟수, 최종 체인과 파라미터를 저장한다.
`--save-audio`는 이미 평가한 최적 오디오를 저장하므로 렌더 횟수를 늘리지 않는다.
정답 종류가 없는 inventory의 샘플은 truth_unavailable로 별도 집계하며 오디오 재현 거리도 따로 낸다.
oracle 외에는 정답을 후보 생성/추론에 전달하지 않는다.

추론 시간에는 log-mel/device 동기화가 포함된다. 렌더/CMA/거리 계산 시간은 별도다.
checkpoint/오디오 읽기와 결과 저장은 방법별 시간에서 제외하며 실제 wall time도 샘플에 기록한다.
첫 실행은 librosa/numba 초기화로 느릴 수 있다.

## 검증

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s effects -p "test_*.py" -v
python -m estimation.smoke --output outputs/cnn_smoke
```

smoke는 6개 임시 harmonic signal, 24개 dry/wet segment, 1 epoch로 데이터 준비부터
checkpoint 저장/로드, 추론, 렌더, 동일 예산 CMA 비교, 빈 inventory까지 실행한다.
**이 결과는 기능 검증이며 실제 기타 데이터의 정확도나 모델 비교 성능을 뜻하지 않는다.**
이미 존재하는 output을 덮어쓰지 않으므로 다시 실행할 때 새 output 경로를 지정한다.

## CNN 실험 시 유지할 기준

CNN 구조나 학습 옵션을 바꿀 때는 같은 manifest와 원본별 split으로 비교한다.
모델 선택에는 validation을 사용하고 test는 최종 평가에 사용한다.
`default_cma`, `cnn_only`, `cnn_cma`는 같은 CNN 파이프라인에서 초기값과 보정의 효과를
확인하기 위한 평가 조건이다. 기본값+CMA도 CNN과 같은 후보 종류를 사용한다.

정리 전 생성한 manifest와 checkpoint 형식은 유지한다. 기존 `best.pt`를 그대로
`estimation.infer`와 `estimation.evaluate --checkpoint ...`에 전달할 수 있다.
코드에서 직접 import하는 경우 `estimation.common`은 `estimation.chain`으로,
`search.refine`은 `estimation.refine`으로 변경한다.
외부 예측 파일을 받던 `--predictions` 옵션은 제거했다.
