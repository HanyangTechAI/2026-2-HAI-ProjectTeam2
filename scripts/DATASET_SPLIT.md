# Step 1: dry 원본 기준 Train / Val / Test 분리

하나의 dry 원본에서 만든 모든 변형을 한 그룹에 배정합니다.
train은 학습용, val은 학습 중 설정을 고르는 검증용, test는 마지막 평가용입니다.
같은 dry의 다른 이펙터 변형이 test와 train에 나뉘는 문제를 막습니다.

이번 기능은 정답 라벨 목록을 새로 저장합니다. WAV를 열거나 복사·이동·삭제하지 않고,
원본 라벨 파일도 수정하지 않습니다. 기존 데이터 생성기를 import하거나 변경하지 않습니다.
추가 라이브러리 설치 없이 Python 3.11 이상에서 실행할 수 있습니다.

## 실제 데이터에 적용

`feat/spectral-distance`의 작업 폴더인
`C:\hai_TeamProject-spectral-submission`에서 실행합니다.
아래 입력 폴더는 현재 생성기가 라벨을 저장하도록 설정된 위치입니다.
실제 라벨 파일을 받은 후 자신의 위치에 맞게 바꾸세요.

```powershell
python -m scripts.split_dataset --input data/synthetic/Fender_Strat_Clean/labels --output data/splits/step1_seed42 --ratios 80 10 10 --seed 42
```

- `--input`: 라벨 JSON 폴더 또는 JSON 파일. 폴더는 하위 폴더까지 읽습니다.
- `--output`: **새로 만들 결과 폴더**. 이미 있는 폴더에는 저장하지 않습니다.
- `--ratios`: train, val, test 순서의 비율. `80 10 10`과 `0.8 0.1 0.1`은 같습니다.
- `--seed`: 섞는 기준 숫자. 기본값은 42이며 사용자가 바꿀 수 있습니다.

다른 비율 예시:

```powershell
python -m scripts.split_dataset --input data/synthetic/Fender_Strat_Clean/labels --output data/splits/step1_seed7_60_20_20 --ratios 60 20 20 --seed 7
```

**80:10:10은 잠정 기본값이며, 팀이 확정한 실험 기준이 아닙니다.**
비율은 음원 파일/변형 개수가 아닌 **서로 다른 dry 원본 개수**에 적용됩니다.
원본마다 변형 수가 다르면 실제 라벨 비율은 80:10:10과 다를 수 있습니다.

현재 이 작업 사본에는 실제 합성 라벨과 WAV가 없습니다.
WAV 없이도 저장소의 예제 라벨 하나로 실행을 확인할 수 있습니다.

```powershell
python -m scripts.split_dataset --input schemas/target_sample.data.json --output data/splits/schema_example_seed42
```

예제는 dry가 1개뿐이므로 train 1개, val/test 0개와 경고가 나오는 것이 정상입니다.
이 명령은 실제 파일 유무나 소리 품질을 검증하지 않습니다. 예제 오디오 경로는 자리표시자입니다.

## 생성되는 파일

| 파일 | 내용 |
|---|---|
| `train.json` | train에 배정된 원본 정답 라벨의 JSON 배열 |
| `val.json` | val에 배정된 원본 정답 라벨의 JSON 배열 |
| `test.json` | test에 배정된 원본 정답 라벨의 JSON 배열 |
| `summary.json` | 시드, 알고리즘 버전, 요청/실제 비율, 원본/라벨 개수, dry ID 목록, 검증 결과와 경고 |

각 라벨의 효과 종류·파라미터·추가 메타데이터는 그대로 보존합니다.
오디오 경로를 결과 폴더 기준으로 고치지 않으므로, 원래 경로 해석 기준을 유지해야 합니다.
현재 생성기는 wet 경로 대신 `synthetic_id`를 기록합니다. wet 파일 연결 규칙은
기존 생성기의 규칙을 따르거나 팀에서 별도로 정해야 합니다.

출력 폴더가 이미 있으면 빈 폴더여도 실패합니다. 덮어쓰기 옵션은 제공하지 않습니다.
입력이 라벨 폴더일 때는 그 안에 결과를 저장할 수도 없습니다.
이렇게 해야 다음 실행에서 결과 JSON을 원본 라벨로 다시 읽는 일을 피할 수 있습니다.
잘못된 입력은 출력 폴더를 만들기 전에 검사합니다. 저장 중 파일시스템 오류가 나면
명령은 실패하며 부분 출력이 남을 수 있으니 성공한 실행의 결과를 사용하세요.

## 지원하는 라벨과 ID 검사

공통 스키마 형식:

```json
{"target_id": "wet_001", "dry_sample_id": "dry_001", "ground_truth_chain": {}}
```

현재 `Create_Synthetic_pairs.py`가 만드는 형식:

```json
{"synthetic_id": "wet_002", "dry_sample": {"sample_id": "dry_001"}, "applied_effects": {}}
```

위 두 라벨은 dry ID가 같아서 반드시 같은 그룹에 들어갑니다.
한 JSON 파일에 객체 하나 또는 객체 배열을 저장할 수 있습니다.
폴더 입력은 JSON 파일만 읽으며 JSON 내용이 잘못된 파일을 몰래 건너뛰지 않습니다.

다음은 오류로 거부합니다.

- dry ID나 개별 변형 ID가 없거나 비어 있는 경우
- 같은 라벨에 두 형식의 ID가 함께 있는데 값이 다른 경우
- 원본 ID 앞뒤에 공백이 있는 경우
- 변형 ID가 중복된 경우: 내용이 같아도 중복을 자동으로 제거하지 않음
- NaN/Inf, 중복된 JSON 키, 깨진 JSON

파일명에서 dry ID를 추측하지 않습니다. 이 기능은 **ID와 분리 결과 검사**이며
이펙터 파라미터와 전체 JSON Schema의 유효성 검사기를 대체하지 않습니다.
ID 문자열은 대소문자를 구별합니다. 서로 다른 ID가 사실 같은 녹음인지 WAV 없이 알아낼 수는 없습니다.

## 분리 방식과 재현성

1. 두 형식의 라벨에서 같은 의미의 dry ID를 읽습니다.
2. 동일 dry ID의 모든 변형을 한 묶음으로 만듭니다.
3. `seed`와 dry ID를 SHA-256으로 계산해 순서를 정합니다.
4. 요청 비율대로 dry 묶음을 train/val/test에 배정합니다.
5. 각 그룹 안에서는 변형 ID로 정렬하고 라벨 원문 내용을 보존합니다.

다른 코드의 전역 난수 상태를 바꾸지 않습니다. 알고리즘 식별자는
`sha256-dry-group-largest-remainder-v1`입니다. 입력 목록 순서나 파일 순서가 바뀌어도
같은 dry ID 집합·비율·시드이면 배정은 같습니다. 같은 입력과 설정으로 다른 새 폴더에
실행하면 네 출력 파일의 내용도 같습니다. 생성 시간이나 출력 폴더 경로를 결과 내용에 넣지 않습니다.

그룹 개수는 정수가 되어야 하므로 각 비율에 해당하는 개수를 내림한 뒤,
남은 묶음을 소수 부분이 큰 순서로 나눕니다. 동률이면 train → val → test 순서입니다.
작은 데이터에서는 비율이 양수여도 어떤 그룹이 비어 있을 수 있으며 `warnings`에 표시합니다.
비율을 바꿔서라도 val/test를 억지로 1개씩 만들지는 않습니다.

가상 예시: dry 78개, dry당 변형 9개, 기본 설정.

| 그룹 | dry 원본 | 정답 라벨 |
|---|---:|---:|
| train | 62 | 558 |
| val | 8 | 72 |
| test | 8 | 72 |
| 합계 | 78 | 702 |

이는 **가상 라벨 테스트 결과**입니다. 실제 데이터 702개가 저장소에 있다는 뜻은 아닙니다.

동일 dry ID에 새 이펙터 변형을 추가하는 경우 기존 배정은 유지됩니다.
이펙터가 3개인지 10개인지, 체인 순서가 무엇인지는 분리 로직이 읽지 않습니다.
반면 **새 dry ID를 추가하거나 비율·시드를 바꾸면 기존 dry의 배정도 달라질 수 있습니다.**
한 실험을 시작했다면 그 데이터 버전의 manifest를 고정해서 학습·평가에 사용해야 합니다.
test를 보고 설정을 고른 뒤 같은 test를 최종 평가라고 부르는 문제까지 자동으로 막지는 않습니다.

## Python 코드에서 사용

```python
from scripts.split_dataset import load_labels, split_labels, validate_splits, write_splits

labels = load_labels("data/synthetic/Fender_Strat_Clean/labels")
parts = split_labels(labels, ratios=(80, 10, 10), seed=42)
report = validate_splits(labels, parts, ratios=(80, 10, 10))
print(report["group_counts"])
print(report["validation"])

# 새 폴더에 목록을 저장하려는 경우
# write_splits(labels, "data/splits/run42", ratios=(80, 10, 10), seed=42,
#              input_source="data/synthetic/Fender_Strat_Clean/labels")
```

`validate_splits`는 dry의 그룹 간 중복, 라벨 중복, 누락, 알 수 없는 라벨,
내용이 바뀐 라벨, 비율에 맞지 않는 dry 개수를 검사합니다.
문제가 있으면 성공 결과 대신 `ValueError`를 냅니다.

## 실제 실행한 검증

```powershell
python -m unittest discover -s scripts -p "test_split_dataset.py" -v
C:\hai_TeamProject\.venv\Scripts\python.exe -m unittest discover -s metrics -p "test_*.py"
```

2026-11-09 작업 중 신규 테스트 **31/31**, 기존 스펙트럼/평가 테스트 **66/66**이 통과했습니다.
신규 31개는 기본 Python 3.12.11과 프로젝트 가상환경 Python 3.11.0에서 각각 통과했습니다.
두 환경에서 가상 78원본/702라벨을 CLI로 저장한 네 파일의 내용도 바이트 단위로 같았습니다.

- 같은 dry 변형의 그룹 누출 없음, 중복/누락 없음
- 동일 시드·입력 재정렬의 재현성, 다른 시드와 사용자 비율
- 정수 반올림·동률·0 비율·매우 작은 데이터·불균등 변형 수
- 두 라벨 형식과 잘못된/서로 충돌하는 ID
- 10개 이펙터 메타데이터 보존, 변형 추가 시 그룹 유지
- 원본 라벨/가상 오디오 파일 내용·수정 시각 보존
- 실제 CLI 실행, 덮어쓰기 거부, 잘못된 입력에서 출력 생성 안 함

실제 WAV와 합성 라벨 묶음이 없어 실제 데이터의 분포·ID 품질은 아직 검증하지 못했습니다.

## 팀에서 결정할 사항

- 최종 train/val/test 비율과 사용할 시드, 데이터 버전/manifest 보관 위치
- 모든 데이터셋에 걸친 dry ID의 유일성·이름 규칙과 최종 라벨 형식
- dry 파일 단위 외에 원래 녹음/연주자/곡/세션까지 묶어야 하는지
- 그룹별 이펙터 조합·파라미터·음역 분포를 추가로 맞출 필요가 있는지
- 실제 라벨/오디오 경로와 wet 파일 연결 규칙

현재는 dry 단위 무작위 배정이며 이펙터별 비율을 강제로 맞추지 않습니다.
10개 이펙터의 종류·구현·조합 규칙이 확정되기 전에도 이 기능을 사용할 수 있습니다.
이번 작업에서는 커밋·푸시를 수행하지 않습니다.
