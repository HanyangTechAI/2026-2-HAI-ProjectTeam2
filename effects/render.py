"""
effects/render.py

schemas/effects.data.json의 이펙터 정의를 읽어 pedalboard 체인을 만들고 렌더링합니다.
노브 이름·범위·연결 방식은 스키마에서만 관리하고, 이 파일에는 적지 않습니다.

사용 예
-------
from effects import render_chain
wet = render_chain(dry, sr, [
    {"effect_type": "overdrive", "values": {"drive": 6.0, "level": 0.0}},
])
"""
import json
from functools import lru_cache
from pathlib import Path

EFFECTS_PATH = Path(__file__).resolve().parent.parent / "schemas" / "effects.data.json"


@lru_cache(maxsize=None)
def load_effects(path=EFFECTS_PATH):
    """이펙터 정의(effects.data.json)를 읽습니다. 한 번 읽은 뒤에는 캐시를 사용합니다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _classes_of(definition):
    """이펙터를 구성하는 pedalboard 클래스 이름 목록 (처리 순서)"""
    if definition["class_name"] == "composite":
        return list(definition["composite_of"])
    return [definition["class_name"]]


def build_effect(effect_type, values, effects=None):
    """이펙터 하나를 pedalboard 플러그인 목록으로 만듭니다.

    - composite 이펙터는 composite_of 순서대로 클래스를 만듭니다.
    - 각 클래스의 인자는 fixed_params와 params의 maps_to("클래스.속성")에서 가져옵니다.
    - maps_to가 composite_of에 없는 클래스를 가리키면 아직 렌더링에 구현되지 않은 노브로 보고 건너뜁니다.
      (예: fuzz의 mode → custom_clipping_curve)
    """
    import pedalboard

    effects = effects or load_effects()
    if effect_type not in effects:
        raise ValueError(f"알 수 없는 effect_type: {effect_type}")
    definition = effects[effect_type]

    if definition["source"] != "pedalboard":
        # 외부 플러그인(VST) 지원은 Step 1 드라이브 차별화 결정 후 추가합니다.
        raise NotImplementedError(f"{effect_type}: source '{definition['source']}'는 아직 지원하지 않습니다.")

    classes = _classes_of(definition)
    if len(classes) != len(set(classes)):
        # "클래스.속성" 키로는 같은 클래스 두 개를 구분할 수 없습니다.
        raise ValueError(f"{effect_type}: composite_of에 같은 클래스가 두 번 이상 있습니다: {classes}")

    names = {p["name"] for p in definition["params"]}
    if set(values) != names:
        raise ValueError(f"{effect_type}: values에는 정확히 {sorted(names)}가 있어야 합니다. (받은 값: {sorted(values)})")

    kwargs = {cls: {} for cls in classes}
    for key, value in (definition.get("fixed_params") or {}).items():
        cls, attr = key.split(".", 1)
        kwargs[cls][attr] = value
    for param in definition["params"]:
        cls, _, attr = param["maps_to"].partition(".")
        if cls in kwargs and attr:
            kwargs[cls][attr] = values[param["name"]]

    return [getattr(pedalboard, cls)(**kwargs[cls]) for cls in classes]


def render_chain(audio, sample_rate, chain, effects=None):
    """체인(이펙터 목록)을 순서대로 적용해 렌더링합니다.

    chain: [{"effect_type": ..., "values": {...}}, ...]  (chain.schema.json의 "chain" 배열)
           또는 {"amp_id": ..., "chain": [...]} 형태도 받습니다.
    앰프(amp_id)는 Step 1 "앰프 정하기" 결정 전까지 렌더링하지 않습니다.
    """
    from pedalboard import Pedalboard

    items = chain["chain"] if isinstance(chain, dict) else chain
    board = Pedalboard()
    for item in items:
        for plugin in build_effect(item["effect_type"], item["values"], effects):
            board.append(plugin)
    return board(audio, sample_rate)
