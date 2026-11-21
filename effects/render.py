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
AMP_PATH = ROOT / "schemas" / "amp.data.json"

# Step 1 데이터의 amp_id. 앰프 없이 렌더링합니다.
NO_AMP_IDS = ("fixed_amp_v1",)
# IR 파일을 찾는 폴더. 환경변수 HAI_IR_DIR가 있으면 가장 먼저 봅니다.
# data/IR은 공유 드라이브 data/IR 폴더를 연결해서 사용합니다 (DRIVE_SETUP.md 참고).
IR_DIRS = (ROOT / "data" / "IR",)

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
# 외부 플러그인이 이펙트 없이 입력을 그대로 내보냈다고 보는 기준(최대 차이)과 다시 렌더링하는 횟수
VST_PASSTHROUGH_TOLERANCE = 1e-6
VST_PASSTHROUGH_RETRIES = 2


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


@lru_cache(maxsize=None)
def load_amp(path=AMP_PATH):
    """앰프 정의(amp.data.json)를 읽습니다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_ir(file_path):
    """IR 파일 경로를 찾습니다. 순서: 절대 경로 → 환경변수 HAI_IR_DIR 폴더 → data/IR."""
    if Path(file_path).is_absolute() and Path(file_path).is_file():
        return str(file_path)
    dirs = [Path(os.environ["HAI_IR_DIR"])] if os.environ.get("HAI_IR_DIR") else []
    for directory in [*dirs, *IR_DIRS]:
        path = directory / file_path
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        f"IR 파일({file_path})을 찾을 수 없습니다. 공유 드라이브 data/IR 폴더를 data/IR에 연결하거나, "
        f"IR 폴더를 환경변수 HAI_IR_DIR로 지정하세요.")


@lru_cache(maxsize=None)
def _ir_board(ir_path, gain_db, sample_rate):
    """IR은 처리 샘플레이트에 맞춰 내부에서 변환되므로, VST처럼 샘플레이트마다 따로 둡니다.
    (하나를 여러 샘플레이트에 번갈아 쓰면 결과가 미세하게 달라짐)"""
    from pedalboard import Convolution, Gain, Pedalboard

    return Pedalboard([Convolution(impulse_response_filename=ir_path), Gain(gain_db=gain_db)])


def apply_amp(audio, sample_rate, amp_id, amp=None):
    """드라이브까지 렌더링한 소리에 고정 앰프(캐비닛 IR + 음량 보정)를 적용합니다.

    - amp_id가 없거나 Step 1 값(fixed_amp_v1)이면 그대로 돌려줍니다.
    - IR은 매번 처음 상태(reset=True)에서 적용합니다. scripts/apply_cabinet_ir.py와 같은 결과입니다.
    """
    if amp_id is None or amp_id in NO_AMP_IDS:
        return audio
    amp = amp or load_amp()
    if amp_id != amp["amp_id"]:
        raise ValueError(f"알 수 없는 amp_id: {amp_id} (현재 앰프: {amp['amp_id']})")
    if amp["render_method"] != "impulse_response":
        raise NotImplementedError(f"앰프 방식 '{amp['render_method']}'는 아직 지원하지 않습니다.")
    board = _ir_board(find_ir(amp["file_path"]), float(amp.get("gain_db", 0.0)), int(sample_rate))
    return board(np.asarray(audio, dtype=np.float32), sample_rate, reset=True)


def render_chain(audio, sample_rate, chain, effects=None, amp=None):
    """체인(이펙터 목록)을 순서대로 적용하고, 마지막에 앰프를 적용해 렌더링합니다.

    chain: [{"effect_type": ..., "values": {...}}, ...]  (chain.schema.json의 "chain" 배열)
           또는 {"amp_id": ..., "chain": [...]} 형태도 받습니다.
    앰프는 {"amp_id": ...} 형태일 때만 적용합니다 (apply_amp 참고). 배열만 주면 앰프 없이 렌더링합니다.
    """
    amp_id = chain.get("amp_id") if isinstance(chain, dict) else None
    return apply_amp(_render_effects(audio, sample_rate, chain, effects), sample_rate, amp_id, amp)


def _render_effects(audio, sample_rate, chain, effects=None):
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
    # - 드물게 플러그인이 이펙트 없이 입력을 그대로 내보냅니다 (2,730개 중 5개, 280개 중 1개 관측).
    #   뒤에 앰프(IR)가 붙으면 겉보기로는 구분되지 않으므로, 앰프 전에 확인하고 다시 렌더링합니다.
    audio = np.asarray(audio, dtype=np.float32)
    silence = np.zeros((*audio.shape[:-1], int(VST_WARMUP_SECONDS * sample_rate)), dtype=np.float32)
    for _ in range(VST_PASSTHROUGH_RETRIES + 1):
        board(silence, sample_rate, buffer_size=VST_BUFFER_SIZE, reset=True)
        out = board(audio, sample_rate, buffer_size=VST_BUFFER_SIZE, reset=False)
        if out.shape != audio.shape or np.max(np.abs(out - audio)) > VST_PASSTHROUGH_TOLERANCE:
            return out
    raise RuntimeError(
        f"외부 플러그인이 이펙트 없이 소리를 통과시켰습니다 ({vst_types}, {VST_PASSTHROUGH_RETRIES + 1}회 시도). "
        "체험판·라이선스 활성화 상태를 확인하세요.")
