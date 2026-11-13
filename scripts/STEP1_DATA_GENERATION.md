# Step 1: 무작위 이펙터 조합 데이터 생성

이번 업무인 **조합 생성 기능·테스트 구현은 완료**했습니다. 실제 기타 데이터 대량 생성은 아직 하지 않았습니다.
팀의 최종 조합 정책, 추가 기타 위치와 생성량도 결정하지 않았습니다.

작업 폴더: `C:\hai_TeamProject-step1-synthetic-data`
브랜치: `feat/step1-synthetic-data` (기준: `origin/main`, `bfcbd4e`)
기존 `feat/spectral-distance`와 PR #12, 다른 폴더의 미커밋 작업은 유지했습니다.
commit·push·PR 생성은 하지 않았습니다.

## 1. 먼저 미리보기

PowerShell에서 아래처럼 실행합니다. `--execute`가 없으면 **WAV·JSON·출력 폴더를 만들지 않습니다.**
입력 WAV의 이름과 위치만 확인하며 오디오를 읽거나 렌더링하지 않습니다.
Python의 캐시 파일 생성까지 피하려면 예시처럼 `-B`를 붙입니다.

```powershell
cd C:\hai_TeamProject-step1-synthetic-data
$python = 'C:\hai_TeamProject\.venv\Scripts\python.exe'

# 입력 경로는 실제 사용할 폴더로 바꾸세요. 이 명령은 미리보기만 합니다.
& $python -B -m scripts.generate_step1_dataset `
  --dry-input Fender_Strat_Clean 'C:\hai_TeamProject-spectral-submission\data\dry\Fender_Strat_Clean\audio' `
  --variations 3 --seed 42
```

화면에 원본 수, 생성할 WAV·라벨 수, manifest 1개를 포함한 총 파일 수, 출력 위치,
설정, 경고와 정답 라벨 예시 최대 3개가 JSON으로 나옵니다.
출력 위치를 생략하면 `<OUTPUT>`이라고 표시합니다. `--dry-run`을 붙여도 같은 미리보기입니다.
위 명령은 사용 예시이며 이번 개발 중 실제 공유 폴더를 대상으로 실행하지 않았습니다.

## 2. 지원하는 조합

임시 기본 순서는 **Compressor → Drive → EQ**입니다.

| 단계 | 기본 동작 |
|---|---|
| Compressor | 50% 확률로 적용, 나머지는 생략 |
| Drive | overdrive / distortion / fuzz 중 하나를 같은 확률로 선택 |
| EQ | eq_3band를 50% 확률로 적용, 나머지는 생략 |

Drive 단독, Compressor + Drive, Drive + EQ, Compressor + Drive + EQ의 네 형태를 지원합니다.
Drive가 3종이므로 기본 조합은 총 12가지입니다. 매번 모든 조합을 일정 개수씩 만드는 방식은 아닙니다.
**이 순서·확률은 팀에서 확정한 최종 정책이 아닙니다.**

각 이펙터의 모든 파라미터 이름·범위·기본값은 `schemas/effects.data.json`에서 읽습니다.

- 일반 연속형: 최솟값과 최댓값 사이에서 균등 추출합니다.
- `scale: log`: 로그 값 사이에서 균등 추출한 뒤 원래 단위로 돌립니다.
  compressor의 ratio·attack·release와 EQ의 mid frequency가 해당합니다.
- 범주형: 지정된 `options` 중 하나를 같은 확률로 뽑습니다. boolean도 지원합니다.
- **fuzz.mode는 현재 렌더러가 무시합니다.** 실제 소리 차이가 있다고 기록하지 않도록
  기존 생성기처럼 `Vintage`로 고정하고 미리보기·manifest에 경고합니다.
- 렌더러가 처리하지 않는 다른 노브가 추가돼도 기본값으로 고정하고 경고합니다.
  외부 VST나 팀 렌더러가 생성할 수 없는 이펙터는 거부합니다.

오디오 처리는 기존 `effects.render_chain`만 사용합니다.
`amp_id`는 기존 `schemas/amp.data.json`의 `fixed_amp_v1`을 유지합니다.
앰프는 현재 팀 렌더러에서 실제 처리되지 않습니다.
음량 기준 재추출·자동 정규화·클리핑 보정은 추가하지 않았습니다. 매우 작은 음량과 1을 넘는
진폭이 나올 수 있으며 FLOAT WAV로 저장합니다. 실제 음질·음량 기준은 이후 팀이 결정해야 합니다.

## 3. 개수·시드·정책 바꾸기

`--variations 3`은 **원본 하나당 총 3개**입니다. 기존 생성기의 드라이브별 개수 설정과 다릅니다.
원본이 N개이면 WAV는 `N × variations`개, 라벨도 같은 수이며 manifest가 1개 추가됩니다.

```powershell
# 원본당 10개, seed 123: 미리보기
& $python -B -m scripts.generate_step1_dataset `
  --dry-input Fender_Strat_Clean '<입력 WAV 폴더>' --variations 10 --seed 123

# 컴프·EQ를 항상 포함하고 드라이브는 OD/DS만 사용: 미리보기
& $python -B -m scripts.generate_step1_dataset `
  --dry-input Fender_Strat_Clean '<입력 WAV 폴더>' `
  --effects compressor overdrive distortion eq_3band `
  --compressor-probability 1 --eq-probability 1

# 컴프·EQ를 생략하고 fuzz만 사용: 미리보기
& $python -B -m scripts.generate_step1_dataset `
  --dry-input Fender_Strat_Clean '<입력 WAV 폴더>' --effects fuzz
```

`--compressor-probability`, `--eq-probability`는 0~1 사이의 값입니다.
`--chain-order eq drive compressor`처럼 세 단계의 순서를 바꿀 수 있습니다.
기본 정책에서는 드라이브가 하나 포함됩니다. `--effects`에서 드라이브를 모두 제외하면
남은 Compressor 또는 EQ 중 하나의 포함 확률을 1로 지정해야 합니다. 빈 체인은 거부합니다.
`--source-dataset`으로 다른 데이터셋 이름을 명시할 수 있습니다(기본 `IDMT-SMT-Guitar V2`).
서로 다른 데이터셋 이름을 사용할 경우 별도 실행으로 구분하세요.

다른 기타는 `--dry-input 이름 폴더`를 반복해 선택합니다. 추가 기타의 실제 위치는 팀이
확정한 뒤 넣어야 합니다. 이름에는 영문·숫자·`_`·`-`를 쓰며 앞으로도 동일한 이름을 유지하세요.
입력 폴더 안의 하위 폴더 WAV도 포함하며 `.WAV`도 인식합니다.

```powershell
& $python -B -m scripts.generate_step1_dataset `
  --dry-input guitar_a '<팀이 정한 첫 번째 폴더>' `
  --dry-input guitar_b '<팀이 정한 두 번째 폴더>' --variations 3 --seed 42
```

같은 seed·입력 이름·상대 경로·설정·스키마·출력 경로이면 같은 체인과 라벨이 생성됩니다.
원본별·변형 번호별 독립 난수를 사용하므로 입력 순서나 변형 개수를 늘려도 기존 번호의
체인은 바뀌지 않습니다. 출력 경로가 다르면 라벨의 `file_path`만 달라집니다.
실제 오디오의 비트 단위 재현은 같은 렌더러·라이브러리 환경에서 확인했습니다.
전역 random 상태는 변경하지 않습니다.

향후 이펙터 확장 시 먼저 팀 스키마와 렌더러가 해당 이펙터를 지원해야 합니다.
코드의 `Stage(effect_types, probability)`와 `Config.policy`에 단계를 추가해 순서·선택지를
정의할 수 있습니다. 기본 CLI의 세 단계도 그 객체로 구성되며, CLI 단계 정의만 확장하면
샘플링·라벨·저장 로직을 재작성할 필요가 없습니다. 스키마에 새 이름이 생겼다는 이유만으로
확인 없이 자동 포함하지 않습니다.

## 4. ID와 정답 라벨

라벨은 기존 `target_sample.schema.json`을 따릅니다.

```json
{
  "target_id": "wet_<dry 해시>_<설정 해시>_000001",
  "is_synthetic": true,
  "dry_sample_id": "dry_<입력 이름과 상대 파일 경로의 SHA-256>",
  "ground_truth_chain": {
    "amp_id": "fixed_amp_v1",
    "chain": [
      {"effect_type": "overdrive", "values": {"drive": 6.0, "level": 0.0}}
    ]
  },
  "source_dataset": "IDMT-SMT-Guitar V2",
  "file_path": "<출력 폴더>/audio/<target_id>.wav"
}
```

위 값은 구조 설명용이며 실제 난수 결과가 아닙니다. 실제 라벨에는 렌더러에 전달한
체인의 순서와 모든 파라미터를 그대로 기록합니다. 각 라벨을 JSON Schema와 팀의
`validate_chain`으로 검증합니다.

- Dry ID는 입력 이름과 폴더 내 상대 파일 경로를 함께 해시합니다.
  다른 기타의 동명 WAV, 서로 다른 하위 폴더의 동명 WAV는 구분됩니다.
- 같은 원본의 모든 변형은 같은 Dry ID를 사용합니다.
- Target ID는 Dry ID·시드 등 설정 해시·변형 번호를 사용합니다.
- 입력 이름 중복, 겹치는 입력 폴더와 같은 실제 파일의 링크·하드링크 중복 선택을 거부합니다.
- `manifest.json`에는 원래 입력 이름·상대 경로·실제 경로와 Dry ID의 대응표를 남깁니다.
  동일 녹음의 복사본 여부는 파일 내용만으로 자동 판별하지 않습니다.

PR #12의 기존 분할기는 `target_id`/`dry_sample_id`를 그대로 사용할 수 있습니다.
분할할 때는 **출력 폴더의 `labels` 하위 폴더만** 지정하세요. `manifest.json`은 라벨이 아닙니다.
기존 생성기는 basename을 Dry ID로 사용하므로, 과거 라벨과 새 라벨을 섞는 경우에는
팀이 원본 대응표로 Dry ID를 통일한 뒤 분할해야 합니다. 이전 라벨은 이번 작업에서 변경하지 않았습니다.
입력 이름이나 상대 경로를 바꾸면 Dry ID가 달라지므로 운영 단계에서 원본 목록을 고정해야 합니다.

## 5. 로컬 임시 폴더에서만 실제 실행

이번 개발에서는 공유 드라이브 출력을 허용하지 않습니다. 링크된 실제 데이터를 보호하고
팀이 아직 정하지 않은 대량 생성량·폴더·정책으로 파일이 생기는 것을 막기 위한 제한입니다.

- 기본값은 미리보기입니다. `--execute`와 `--output`을 둘 다 명시해야 저장합니다.
- 출력은 **시스템 임시 폴더 아래의 새 물리 폴더**만 허용합니다.
- Windows에서는 고정 로컬 디스크만 허용하며 UNC·네트워크 드라이브를 거부합니다.
- 부모 경로를 포함해 심볼릭 링크·junction·mount/cloud reparse point를 거부합니다.
  경로를 해석한 후 시스템 임시 폴더 안인지도 확인합니다.
- Google Drive·공유 드라이브 이름, 프로젝트 `data/raw`, `data/dry`, `data/synthetic` 형태도 거부합니다.
- 기존 출력 폴더는 비어 있어도 거부합니다. WAV·JSON을 배타적 생성 모드로 열어 덮어쓰기를 막습니다.

아래는 **로컬 테스트용 WAV만** 있는 경우의 예시입니다. 입력도 임시 폴더에 준비하세요.
먼저 동일한 옵션으로 미리보기의 파일 수와 경로를 확인하고, 다음 명령에서만 저장합니다.

```powershell
$localInput = Join-Path $env:TEMP 'step1-local-input'
$localOutput = Join-Path $env:TEMP ('step1-local-output-' + [guid]::NewGuid().ToString('N'))

# 1. 미리보기: 출력 폴더를 만들지 않음
& $python -B -m scripts.generate_step1_dataset `
  --dry-input local_test $localInput --variations 2 --seed 42 --output $localOutput

# 2. 실제 렌더링: 이 명령에서만 로컬 임시 폴더에 저장
& $python -B -m scripts.generate_step1_dataset `
  --dry-input local_test $localInput --variations 2 --seed 42 --output $localOutput --execute
```

결과는 `audio/*.wav`, `labels/*.json`, `manifest.json`입니다. 원본 WAV는 변경하지 않습니다.
오류가 나면 일부 파일이 남을 수 있으며 완료 manifest는 작성되지 않습니다.
같은 폴더에 이어 쓰거나 자동 삭제하지 않습니다. 로컬 결과를 확인한 뒤 새 출력 폴더로 재시도하세요.
현재는 파일 개수와 경로를 미리 보여주며, 용량 추정·병렬 처리·중단 후 재개 기능은 제공하지 않습니다.

## 6. 검증 방법과 결과

```powershell
& $python -B -m unittest discover -s scripts -p 'test_*.py' -v
& $python -B -m unittest discover -s metrics -p 'test_*.py'
& $python -B -m unittest discover -s effects -p 'test_*.py'
& $python -B -m unittest discover -s schemas -p 'test_*.py'
& $python -B schemas/validate.py
```

신규 테스트는 로컬 `TemporaryDirectory`에서 만든 0.1초 합성 파형만 사용합니다.
공유 드라이브의 WAV를 읽거나 쓰지 않습니다. 단일/복합 체인, 3종 Drive, 로그/선형/범주/boolean
추출, 범위·옵션·스키마, 시드 재현성·다양성, ID와 그룹 유지, 오류 설정, 미리보기 무쓰기,
덮어쓰기 거부, 실제 Windows junction 거부, 모노·스테레오 저장과 라벨/렌더링 일치를 검사합니다.

2026-11-10, 기존 프로젝트 가상환경 Python 3.11.0에서 확인한 결과입니다.

| 검사 | 결과 |
|---|---:|
| 신규 Step 1 scripts 테스트 | 49/49 통과 |
| 기존 main metrics 테스트 | 66/66 통과 |
| 기존 main effects 테스트 | 5/5 통과 |
| 기존 main schemas 테스트 | 8/8 통과 |
| main 기준 전체 + 신규 | **128/128 통과, 실패·skip 없음** |
| PR #12 작업 폴더의 기존 분할 테스트 (읽기 전용 추가 검증) | 31/31 통과 |
| 스키마 예시 검증 | 6개와 참조 관계 모두 통과 |

추가로 로컬 합성 파형 3개에서 변형 12개를 생성해 PR #12의 기존 분할기로 읽고
분할했습니다. Dry 그룹 3개가 유지되며 누출·중복·누락·라벨 변경이 없음을 확인했습니다.
실제로 연결된 공유 Drive junction 3곳도 출력 경로 검사만 호출하여 모두 거부됨을 확인했습니다.
기존 렌더러·스키마·분할기 및 기존 테스트는 수정하지 않았습니다.

## 7. 다음 업무: 데이터 규모 확대

대량 생성 전에 팀이 아래를 확정해야 합니다.

1. 추가 기타·데이터셋의 실제 입력 위치, 고정 입력 이름과 원본 목록.
2. 원본당 변형 수·전체 목표량·시드·조합별 분포.
3. 최종 이펙터 목록과 순서·포함 확률·파라미터 분포.
4. fuzz.mode 등 미구현 노브와 앰프 처리 방침.
5. 무음·낮은 음량·클리핑·길이·표본율 등 품질 기준.
6. 공유 드라이브의 출력 위치·생성 담당자·용량·라이선스 관리.
7. 과거/신규 Dry ID 통합과 원본별 Train/Val/Test 분리 적용·누출 검증.

그 결정 후 별도 작업에서 규모·용량을 산정하고 작은 파일럿을 청취·검증한 뒤
공유 드라이브 출력 제한을 팀 검토 하에 변경해야 합니다. 이 제한을 우회해 생성하지 마세요.
이번에는 기존 생성기·렌더러·스키마·지표·검색·분할 코드를 수정하지 않았습니다.
