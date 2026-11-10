# 이펙터 및 렌더링

`schemas/effects.data.json`의 이펙터 정의를 읽어 Spotify `pedalboard` 체인을 만들고 렌더링합니다.
노브 이름·범위·연결 방식은 스키마에서만 관리하며, 렌더링 코드에는 적지 않습니다.

## 사용법

```python
from effects import render_chain

wet = render_chain(dry, sr, [
    {"effect_type": "compressor", "values": {"threshold_db": -18.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 100.0}},
    {"effect_type": "overdrive",  "values": {"drive": 6.0, "level": 0.0}},
])
```

- `chain`은 `chain.schema.json`의 `chain` 배열 형식이며, 배열 순서대로 처리합니다. `{"amp_id": ..., "chain": [...]}` 형태도 받습니다.
- `values`에는 그 이펙터의 모든 노브를 적어야 합니다. 빠지거나 남는 노브가 있으면 오류가 납니다.
- `scripts/Create_Synthetic_pairs.py`의 `apply_effects`도 내부적으로 이 함수를 사용합니다.

## 스키마를 읽는 방식

| 스키마 필드 | 렌더링에서의 의미 |
|---|---|
| `class_name` / `composite_of` | 만들 pedalboard 클래스와 처리 순서 |
| `fixed_params` | 탐색하지 않는 고정값 (`"클래스.속성": 값`) |
| `params[].maps_to` | 노브 값을 넣을 위치 (`"클래스.속성"`) |

새 pedalboard 이펙터는 `effects.data.json`에 정의만 추가하면 코드 수정 없이 렌더링됩니다.

## 아직 지원하지 않는 것

- **외부 플러그인(`source: "vst"`)**: Step 1 드라이브 차별화 결정 후 추가합니다. 지금은 `NotImplementedError`가 납니다.
- **앰프(`amp_id`)**: Step 1 "앰프 정하기" 결정 전까지 렌더링하지 않습니다.
- **`maps_to`가 `composite_of`에 없는 노브**: 렌더링에 반영되지 않습니다. 현재 fuzz의 `mode`(`custom_clipping_curve`)가 해당합니다.
- **`composite_of`에 같은 클래스가 두 번**: `"클래스.속성"` 키로 구분할 수 없어 오류가 납니다.

## 테스트

```bash
python -m unittest discover -s effects -p "test_*.py"
```
