"""
effects/render.py

schemas/effects.data.json의 이펙터 정의를 읽어 pedalboard 체인을 만들고 렌더링합니다.
노브 이름·범위·연결 방식은 스키마에서만 관리하고, 이 파일에는 적지 않습니다.

사용 예
-------
from effects import render_chain
wet = render_chain(dry, sr, [
    {"effect_type": "808_scream", "values": {"level": 50.0, "overdrive": 70.0, "tone": 40.0}},
])
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EFFECTS_PATH = ROOT / "schemas" / "effects.data.json"

# 외부 플러그인 파일 이름 (스키마 class_name → 설치 파일 이름 후보)
PLUGIN_FILES = {"MultiDrivePro": ("MultiDrive PRO.vst3", "MultiDrivePRO.vst3")}
# 플러그인을 찾는 폴더. 환경변수 HAI_VST_DIR가 있으면 가장 먼저 봅니다.
PLUGIN_DIRS = (
    ROOT,
    ROOT / "effects" / "VST_files",
    Path(r"C:\Program Files\Common Files\VST3\Audified"),
    Path(r"C:\Program Files\Common Files\VST3"),
)
# 외부 플러그인이 들어간 체인의 처리 단위 (512~4096에서 결과가 같고, 8192에서는 깨짐)
VST_BUFFER_SIZE = 2048
# 외부 플러그인이 들어간 체인은 렌더링 전에 이 길이의 무음을 먼저 통과시킵니다 (render_chain 참고)
VST_WARMUP_SECONDS = 0.5


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


def _check_values(effect_type, definition, values):
    names = {p["name"] for p in definition["params"]}
    if set(values) != names:
        raise ValueError(f"{effect_type}: values에는 정확히 {sorted(names)}가 있어야 합니다. (받은 값: {sorted(values)})")


def find_plugin(class_name):
    """플러그인 파일 경로를 찾습니다.

    순서: 환경변수 HAI_VST_DIR 폴더 → 저장소 루트 → effects/VST_files → 기본 설치 폴더.
    .vst3가 폴더(번들)이면 그 안의 Windows 실행 파일을 사용합니다.
    """
    dirs = [Path(os.environ["HAI_VST_DIR"])] if os.environ.get("HAI_VST_DIR") else []
    for directory in [*dirs, *PLUGIN_DIRS]:
        for name in PLUGIN_FILES.get(class_name, (f"{class_name}.vst3",)):
            path = directory / name
            if path.is_dir():
                inner = path / "Contents" / "x86_64-win" / name
                return str(inner if inner.is_file() else path)
            if path.is_file():
                return str(path)
    raise FileNotFoundError(
        f"{class_name} 플러그인을 찾을 수 없습니다. 플러그인을 설치하거나, "
        f"플러그인이 있는 폴더를 환경변수 HAI_VST_DIR로 지정하세요.")


@lru_cache(maxsize=None)
def _load_vst(class_name, load_values, sample_rate):
    """플러그인을 불러옵니다. (페달 모델 등 고정값, 샘플레이트)마다 한 번만 불러옵니다.

    - pedal_model은 불러온 뒤에 바꾸면 노브 연결이 이전 모델 기준으로 남으므로,
      반드시 불러올 때(parameter_values) 지정합니다.
    - 같은 플러그인을 다른 샘플레이트로 쓰면 무음이 나오거나 멈추므로 샘플레이트별로 따로 둡니다.
    """
    from pedalboard import load_plugin

    plugin = load_plugin(find_plugin(class_name), parameter_values=dict(load_values))
    if "bypass" in plugin.parameters:
        plugin.bypass = False
    return plugin


def _load_values(definition):
    return tuple(sorted((key.split(".", 1)[1], value)
                        for key, value in (definition.get("fixed_params") or {}).items()))


def build_vst(effect_type, definition, values, sample_rate=None):
    """외부 플러그인(source: vst) 이펙터 하나를 만듭니다.

    - fixed_params(페달 모델, 입력 게인)는 불러올 때 지정합니다.
    - 노브 값(params)은 매번 새로 설정합니다.
    - 같은 모델의 플러그인은 재사용하므로, 한 체인에 같은 이펙터를 두 번 넣을 수 없습니다.
    """
    _check_values(effect_type, definition, values)
    plugin = _load_vst(definition["class_name"], _load_values(definition), sample_rate)
    for param in definition["params"]:
        setattr(plugin, param["maps_to"].split(".", 1)[1], values[param["name"]])
    return [plugin]


def build_effect(effect_type, values, effects=None, sample_rate=None):
    """이펙터 하나를 pedalboard 플러그인 목록으로 만듭니다.

    - composite 이펙터는 composite_of 순서대로 클래스를 만듭니다.
    - 각 클래스의 인자는 fixed_params와 params의 maps_to("클래스.속성")에서 가져옵니다.
    - maps_to가 composite_of에 없는 클래스를 가리키면 아직 렌더링에 구현되지 않은 노브로 보고 건너뜁니다.
    - source가 "vst"이면 외부 플러그인을 불러옵니다 (build_vst). sample_rate는 VST에만 쓰입니다.
    """
    import pedalboard

    effects = effects or load_effects()
    if effect_type not in effects:
        raise ValueError(f"알 수 없는 effect_type: {effect_type}")
    definition = effects[effect_type]

    if definition["source"] == "vst":
        return build_vst(effect_type, definition, values, sample_rate)
    if definition["source"] != "pedalboard":
        raise NotImplementedError(f"{effect_type}: source '{definition['source']}'는 아직 지원하지 않습니다.")

    classes = _classes_of(definition)
    if len(classes) != len(set(classes)):
        # "클래스.속성" 키로는 같은 클래스 두 개를 구분할 수 없습니다.
        raise ValueError(f"{effect_type}: composite_of에 같은 클래스가 두 번 이상 있습니다: {classes}")
    _check_values(effect_type, definition, values)

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
    effects = effects or load_effects()
    vst_types = [item["effect_type"] for item in items
                 if effects.get(item["effect_type"], {}).get("source") == "vst"]
    if len(vst_types) != len(set(vst_types)):
        raise ValueError(f"같은 외부 플러그인 이펙터를 한 체인에 두 번 넣을 수 없습니다: {vst_types}")
    board = Pedalboard()
    for item in items:
        for plugin in build_effect(item["effect_type"], item["values"], effects, sample_rate):
            board.append(plugin)
    if not vst_types:
        return board(audio, sample_rate)
    # MultiDrive PRO 렌더링 규칙
    # - 기본 버퍼(8192)에서는 첫 블록이 무음이 되고 소리도 달라지므로 VST_BUFFER_SIZE로 처리합니다.
    # - 직전에 처리한 다른 모델의 상태가 남아 앞부분 소리가 달라지므로 (b_drive 최대 50%),
    #   무음으로 먼저 통과시킨 뒤 상태를 초기화하지 않고(reset=False) 이어서 렌더링합니다.
    audio = np.asarray(audio, dtype=np.float32)
    silence = np.zeros((*audio.shape[:-1], int(VST_WARMUP_SECONDS * sample_rate)), dtype=np.float32)
    board(silence, sample_rate, buffer_size=VST_BUFFER_SIZE, reset=True)
    return board(audio, sample_rate, buffer_size=VST_BUFFER_SIZE, reset=False)
