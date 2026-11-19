# 이펙터 및 렌더링

`schemas/effects.data.json`의 이펙터 정의를 읽어 Spotify `pedalboard` 체인을 만들고 렌더링합니다.
노브 이름·범위·연결 방식은 스키마에서만 관리하며, 렌더링 코드에는 적지 않습니다.

## 사용법

```python
from effects import render_chain

wet = render_chain(dry, sr, [
    {"effect_type": "compressor", "values": {"threshold_db": -18.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 100.0}},
    {"effect_type": "808_scream", "values": {"level": 50.0, "overdrive": 70.0, "tone": 50.0}},
])
```

- `chain`은 `chain.schema.json`의 `chain` 배열 형식이며, 배열 순서대로 처리합니다. `{"amp_id": ..., "chain": [...]}` 형태도 받습니다.
- `values`에는 그 이펙터의 모든 노브를 적어야 합니다. 빠지거나 남는 노브가 있으면 오류가 납니다.

## 스키마를 읽는 방식

| 스키마 필드 | 렌더링에서의 의미 |
|---|---|
| `source` | `pedalboard`(내장 클래스 조합) 또는 `vst`(외부 플러그인) |
| `class_name` / `composite_of` | 만들 pedalboard 클래스와 처리 순서. VST는 플러그인 이름 (`MultiDrivePro`) |
| `fixed_params` | 탐색하지 않는 고정값 (`"클래스.속성": 값`). VST는 페달 모델·입력 게인 |
| `params[].maps_to` | 노브 값을 넣을 위치 (`"클래스.속성"`) |

새 pedalboard 이펙터는 `effects.data.json`에 정의만 추가하면 코드 수정 없이 렌더링됩니다.

## 외부 플러그인 (MultiDrive PRO)

Step 1 드라이브 7종(`808_scream`, `b_drive`, `o_drive`, `one_distortion`, `bat_distortion`, `f_fuzz`, `m_fuzz`)은
Audified MultiDrive PRO 플러그인으로 렌더링합니다.

- **설치**: 플러그인은 라이선스 때문에 저장소에 올리지 않습니다. 각자 설치하고 체험판을 활성화해야 합니다.
  활성화하지 않으면 dry 소리가 그대로 나옵니다.
- **위치**: 환경변수 `HAI_VST_DIR` 폴더 → 저장소 루트 → `effects/VST_files/` →
  `C:\Program Files\Common Files\VST3\Audified\` 순서로 찾습니다.
- **렌더러가 자동으로 지키는 규칙** (직접 플러그인을 다룰 때도 똑같이 지켜야 함)
  - 페달 모델은 **불러올 때** 지정합니다. 불러온 뒤 바꾸면 노브가 이전 모델 기준으로 동작합니다.
  - 모델·샘플레이트마다 플러그인을 한 번만 불러와 재사용합니다 (불러오기 약 0.6초).
  - 처리 단위(buffer)는 2048입니다. 기본값 8192에서는 첫 0.19초가 무음이 되고 소리도 달라집니다.
  - 렌더링 전에 무음 0.5초를 먼저 통과시킵니다. 직전에 처리한 다른 모델의 영향이 남는 문제를 막기 위함입니다.
- **알려진 한계**: 같은 설정이라도 렌더링 순서에 따라 일부 모델(주로 `f_fuzz`)에서 최대 1~2% 차이가 날 수 있습니다.
- 한 체인에 같은 VST 이펙터를 두 번 넣을 수 없습니다.

## 아직 지원하지 않는 것

- **앰프(`amp_id`)**: Step 1 "앰프 정하기" 결정 전까지 렌더링하지 않습니다.
- **`maps_to`가 `composite_of`에 없는 노브**: 렌더링에 반영되지 않습니다.
- **`composite_of`에 같은 클래스가 두 번**: `"클래스.속성"` 키로 구분할 수 없어 오류가 납니다.

## 테스트

```bash
python -m unittest effects.test_render
```

플러그인이 없는 PC에서는 VST 테스트를 건너뜁니다.
