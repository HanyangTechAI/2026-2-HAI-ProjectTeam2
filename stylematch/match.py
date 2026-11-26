"""
stylematch/match.py

같은 연주의 dry가 없는 목표 톤 맞추기:  AFx-Rep 격자로 상위 K개 드라이브 → CMA-ES(AFx-Rep 거리)로 노브 보정.

체인: 드라이브(VST) → 앰프 크런치(tanh 포화) → 캐비닛 IR → (선택) 공간 리버브
- 앰프 크런치: 실제 녹음의 앰프 찌그러짐을 흡수 (없으면 실제 오버드라이브를 디스토션으로 착각)
- 공간 리버브(room=True): 목표 녹음의 리버브·공간감을 흡수 (추천 대상 아님, 결과 설명에서 따로 표시)
- 드라이브 게인·톤은 0–100% 전체에서 찾음 (실제 데모의 저게인 설정 포함)
- level/volume은 고정 (AFx-Rep은 음량을 정규화해서 보므로 탐색할 필요 없음)

사용 예
-------
from stylematch.match import match_tone
result = match_tone(dry, target, sr, k=3, budget=30)
result["best"]["chain"]  # 최종 체인 (드라이브 + 크런치 노브)
"""
import copy
import hashlib

import numpy as np

from effects import load_effects, render_chain
from effects.render import apply_amp
from estimation.chain import DRIVES
from search.pipeline import search_and_rank_chains
from stylematch.afx import afx_distance, afx_metric, embed

CRUNCH = "amp_crunch"
CRUNCH_RANGE = (0.0, 15.0)
LEVEL_NAMES = {"level", "volume"}
TONE_NAMES = {"tone", "filter"}
GRID_GAIN = (0, 15, 35, 50, 65, 85)
GRID_TONE = (20, 50, 80)
GRID_CRUNCH = (0.0, 6.0, 12.0)
DEFAULT_AMP = "marshall_v30_ir_v1"
ROOM = "room"
ROOM_MIX_RANGE = (0.0, 0.5)
ROOM_SIZE_RANGE = (0.2, 0.9)
GRID_ROOM = (0.0, 0.2, 0.4)
ROOM_SIZE_DEFAULT = 0.6
# 리버브 켜기 게이트 (합성 검증: 리버브 없음 잘못 켬 18%, 강한 리버브 89%, 약한 리버브 32%)
ROOM_GATE_MIX = 0.4        # AFx 격자가 고른 리버브 양이 이 이상이고
ROOM_GATE_DECAY = -60.0    # 목표의 가장 빠른 감쇠 속도(dB/s)가 이보다 느릴 때만 켬


def match_effects(effects=None):
    """탐색용 이펙트 정의: 드라이브 게인·톤 0–100%, level 고정, 앰프 크런치 · 공간 리버브 추가."""
    eff = copy.deepcopy(effects or load_effects())
    for drive in DRIVES:
        params = []
        for p in eff[drive]["params"]:
            p = dict(p)
            if p["name"] in LEVEL_NAMES:
                p["range"] = [p["default"], p["default"]]
            elif p.get("type") == "continuous":
                p["range"] = [0.0, 100.0]
            params.append(p)
        eff[drive]["params"] = params
    # 탐색 변수 정의용(노브 범위). 실제 소리는 apply_crunch로 만든다 (render_match 참고).
    eff[CRUNCH] = {
        "category": "amp_stage", "reference_pedal": None, "source": "pedalboard", "class_name": "Distortion",
        "composite_of": None, "fixed_params": None,
        "params": [{"name": "drive_db", "maps_to": "Distortion.drive_db", "type": "continuous", "unit": "dB",
                    "range": list(CRUNCH_RANGE), "options": None, "default": 0.0}],
    }
    # 공간 리버브 (흡수용). 실제 소리는 apply_room.
    eff[ROOM] = {
        "category": "room", "reference_pedal": None, "source": "pedalboard", "class_name": "Reverb",
        "composite_of": None, "fixed_params": None,
        "params": [{"name": "mix", "maps_to": "Reverb.wet_level", "type": "continuous", "unit": "ratio",
                    "range": list(ROOM_MIX_RANGE), "options": None, "default": 0.0},
                   {"name": "size", "maps_to": "Reverb.room_size", "type": "continuous", "unit": "ratio",
                    "range": list(ROOM_SIZE_RANGE), "options": None, "default": ROOM_SIZE_DEFAULT}],
    }
    return eff


def apply_crunch(audio, drive_db):
    """앰프 크런치: 최대 음량 기준으로 맞춘 tanh 포화 (입력 음량과 무관하게 drive_db만큼 찌그러짐).
    pedalboard Distortion은 절대 음량에 반응해 드라이브 뒤 작은 신호에서는 거의 찌그러지지 않으므로 쓰지 않는다."""
    x = np.asarray(audio, dtype=np.float32)
    if drive_db <= 0:
        return x
    peak = max(float(np.max(np.abs(x))), 1e-9)
    k = 10 ** (drive_db / 20)
    return (np.tanh(k * x / peak) / np.tanh(k) * peak).astype(np.float32)


def apply_room(audio, sample_rate, mix, size=ROOM_SIZE_DEFAULT):
    """공간 리버브: 목표 녹음의 리버브·공간감 흡수용 (pedalboard Reverb, 결정적)."""
    x = np.asarray(audio, dtype=np.float32)
    if mix <= 0:
        return x
    from pedalboard import Reverb
    rv = Reverb(room_size=float(size), damping=0.5, wet_level=float(mix), dry_level=float(1 - mix), width=0.0)
    return np.asarray(rv(x.reshape(1, -1), sample_rate), np.float32).reshape(-1)


def _stage(items, name, key, default):
    return next((float(i["values"][key]) for i in items if i["effect_type"] == name), default)


def render_match(audio, sample_rate, chain, effects=None):
    """탐색용 렌더러: 드라이브(VST, 앰프 없이) → 앰프 크런치 → 캐비닛 IR → (있으면) 공간 리버브."""
    effects = effects or match_effects()
    items = chain["chain"] if isinstance(chain, dict) else chain
    drives = [i for i in items if i["effect_type"] not in (CRUNCH, ROOM)]
    crunch = _stage(items, CRUNCH, "drive_db", 0.0)
    bare = render_chain(audio, sample_rate, {"amp_id": "fixed_amp_v1", "chain": drives}, effects=effects)
    amp_id = chain.get("amp_id", DEFAULT_AMP) if isinstance(chain, dict) else DEFAULT_AMP
    wet = apply_amp(apply_crunch(bare, crunch), sample_rate, amp_id)
    return apply_room(wet, sample_rate, _stage(items, ROOM, "mix", 0.0), _stage(items, ROOM, "size", ROOM_SIZE_DEFAULT))


def _gain_tone(eff, drive):
    params = eff[drive]["params"]
    gain = [p["name"] for p in params if p["name"] not in LEVEL_NAMES | TONE_NAMES][0]
    tone = next((p["name"] for p in params if p["name"] in TONE_NAMES), None)
    return gain, tone


_SCREEN_CACHE = {}


def screen(dry, sample_rate, target_embedding, eff=None, amp_id=DEFAULT_AMP, k=3, allowed=None, room=False):
    """7종 × 게인 × 톤 × 크런치 격자를 AFx-Rep 거리로 훑어 드라이브별 최선 → 상위 k개 초기 체인.
    allowed: 사용자가 가진 드라이브만 후보로 (None이면 7종 전체). 격자 렌더는 dry별로 캐시.
    room: 공간 리버브 양도 격자에 포함 (GRID_ROOM)."""
    eff = eff or match_effects()
    key = (hashlib.sha1(np.asarray(dry, np.float32).tobytes()).hexdigest(), int(sample_rate), amp_id, bool(room))
    if key not in _SCREEN_CACHE:
        grid = []
        for drive in DRIVES:
            gain, tone = _gain_tone(eff, drive)
            for g in GRID_GAIN:
                for t in (GRID_TONE if tone else (None,)):
                    values = {p["name"]: p["default"] for p in eff[drive]["params"]}
                    values[gain] = float(g)
                    if tone:
                        values[tone] = float(t)
                    bare = render_chain(dry, sample_rate, {"amp_id": "fixed_amp_v1",
                                                           "chain": [{"effect_type": drive, "values": values}]},
                                        effects=eff)
                    for c in GRID_CRUNCH:
                        wet = apply_amp(apply_crunch(bare, c), sample_rate, amp_id)  # 크런치 + IR만 (VST 재렌더 없음)
                        for m in (GRID_ROOM if room else (None,)):
                            items = [{"effect_type": drive, "values": dict(values)},
                                     {"effect_type": CRUNCH, "values": {"drive_db": c}}]
                            w = wet
                            if m is not None:
                                items.append({"effect_type": ROOM, "values": {"mix": m, "size": ROOM_SIZE_DEFAULT}})
                                w = apply_room(wet, sample_rate, m)
                            grid.append((drive, {"amp_id": amp_id, "chain": items}, embed(w, sample_rate)))
        _SCREEN_CACHE[key] = grid
    best = {}
    for drive, chain, e in _SCREEN_CACHE[key]:
        if allowed is not None and drive not in allowed:
            continue
        dist = afx_distance(e, target_embedding)
        if drive not in best or dist < best[drive][0]:
            best[drive] = (dist, chain)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])
    return [{"effect_type": d, "screen_distance": v[0], "chain": copy.deepcopy(v[1])} for d, v in ranked[:k]], \
        [d for d, _ in ranked]


def room_gate(dry, sample_rate, target, target_embedding, eff=None, amp_id=DEFAULT_AMP, allowed=None):
    """리버브 단계를 켤지 결정. AFx 거리만으로 정하면 리버브가 연주 차이를 덮으므로(해킹)
    감쇠 속도(거리와 무관한 지표)도 함께 넘을 때만 켬."""
    from stylematch.reverb_est import decay_rate
    starts, _ = screen(dry, sample_rate, target_embedding, eff, amp_id, 1, allowed, room=True)
    mix = next((i["values"]["mix"] for i in starts[0]["chain"]["chain"] if i["effect_type"] == ROOM), 0.0)
    decay = decay_rate(target, sample_rate)
    on = bool(mix >= ROOM_GATE_MIX and decay >= ROOM_GATE_DECAY)
    return {"on": on, "grid_mix": float(mix), "decay_db_per_s": float(decay)}


def _rms_db(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64))))), 1e-12))


def match_volume(dry, chain, target_rms_db, sample_rate, eff, steps=6):
    """level/volume 노브만 이분 탐색해 결과 음량(RMS)을 목표와 맞춤 (AFx-Rep은 음량을 보지 않으므로 따로).
    level 범위는 0–100%. 반환: (조정된 체인, 결과 RMS dB)."""
    chain = copy.deepcopy(chain)
    item = next(i for i in chain["chain"] if i["effect_type"] in DRIVES)
    name = next(p["name"] for p in eff[item["effect_type"]]["params"] if p["name"] in LEVEL_NAMES)
    wide = copy.deepcopy(eff)
    for p in wide[item["effect_type"]]["params"]:
        if p["name"] == name:
            p["range"] = [0.0, 100.0]
    lo, hi = 0.0, 100.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        item["values"][name] = mid
        level = _rms_db(render_match(dry, sample_rate, chain, wide))
        lo, hi = (mid, hi) if level < target_rms_db else (lo, mid)
    item["values"][name] = (lo + hi) / 2
    return chain, _rms_db(render_match(dry, sample_rate, chain, wide))


def match_tone(dry, target, sample_rate, k=3, budget=30, restarts=1, target_embedding=None,
               amp_id=DEFAULT_AMP, seed=0, volume=True, allowed=None, dropout=0.1, room=False):
    """목표 톤 맞추기 (같은 연주 dry 불필요).

    1) AFx-Rep 격자로 드라이브 순위 → 상위 k개
    2) 각 후보 페달 안에서만 CMA-ES로 게인·톤·크런치 보정 (탐색이 페달을 바꾸지 않음:
       AFx-Rep 거리로 페달까지 바꾸게 하면 퍼즈가 거리 숫자만 줄여 실제 판별이 나빠짐)
    3) level/volume은 음량 맞추기로 따로 설정
    room=True: 리버브 게이트(room_gate)를 넘는 목표만 공간 리버브 단계로 흡수 (격자 + CMA-ES 모두).
               게이트를 못 넘으면 room=False와 같은 결과
    반환: candidates(격자 순위대로, 노브 보정·음량 맞춤 완료), best(=candidates[0]), screen_order
    """
    eff = match_effects()
    te = target_embedding if target_embedding is not None else embed(target, sample_rate)
    if allowed is not None:
        allowed = [d for d in allowed if d in DRIVES]
        if not allowed:
            raise ValueError("보유한 드라이브가 없습니다 (VST 7종 중 하나 이상 필요).")
    gate = None
    if room:
        gate = room_gate(dry, sample_rate, target, te, eff, amp_id, allowed)
        room = gate["on"]
    starts, screen_order = screen(dry, sample_rate, te, eff, amp_id, k, allowed, room)
    target_rms = _rms_db(target)
    candidates = []
    for start in starts:
        ranked, _ = search_and_rank_chains(
            dry_audio=np.asarray(dry, np.float32), sr=sample_rate, target_audio=np.asarray(target, np.float32),
            candidate_chains=[start["chain"]], effects_def=eff, budget=budget, n_restarts=restarts,
            metric_fn=afx_metric(target, sample_rate, te, dropout=dropout, seed=seed), base_seed=seed, verbose=False,
            render_fn=lambda audio, rate, ch: render_match(audio, rate, ch, eff))
        refined = ranked[0]
        chain = refined["chain"]
        out_rms = None
        if volume:
            chain, out_rms = match_volume(dry, chain, target_rms, sample_rate, eff)
        candidates.append({"effect_type": start["effect_type"], "screen_distance": start["screen_distance"],
                           "afx_distance": refined["loss"], "chain": chain,
                           "target_rms_db": target_rms, "output_rms_db": out_rms})
    return {"best": candidates[0], "candidates": candidates, "screen_order": screen_order, "room_gate": gate}
