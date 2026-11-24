"""
search/comp_candidates.py

컴프레서 추정: CNN은 그대로 두고, 탐색 후보를 "드라이브 × 컴프 있음/없음"으로 늘립니다.

- expand_with_compressor: 후보마다 컴프(기본 노브)를 앞에 붙인 후보를 하나씩 추가합니다.
  컴프 노브 4개는 CMA-ES가 드라이브 노브와 함께 맞춥니다. (체인 순서: 컴프 → 드라이브 → EQ → 앰프)
- choose_compressor: 컴프 있는 후보가 같은 드라이브의 컴프 없는 후보보다 거리가 충분히(margin) 낮을 때만
  "컴프 사용"으로 판정합니다. 노브가 많은 후보는 컴프가 없어도 조금은 더 맞출 수 있기 때문입니다.

Step 2 점검(search/eq_fit.py와 같은 날)에서 약한 컴프는 노브 5% 오차 수준의 흔적만 남겨 구분이 어렵고,
강한 컴프(threshold −25 dB 이하, ratio 6 이상)는 항상 구분되었습니다 → 약한 컴프는 "없음"으로 처리됩니다.

사용 예
-------
cands = expand_with_compressor(make_candidates(prediction, ...))
ranked, _ = search_and_rank_chains(..., candidate_chains=cands, budget=...)
best = choose_compressor(ranked)
"""
import copy

from effects import load_effects

COMPRESSOR = "compressor"
DEFAULT_MARGIN = 0.10  # 컴프 후보의 거리가 10% 이상 낮아야 "컴프 사용"


def compressor_defaults(effects=None):
    effects = effects or load_effects()
    return {p["name"]: p["default"] for p in effects[COMPRESSOR]["params"]}


def _chain_of(candidate):
    """CNN 후보({"chain": {...}, "x0": ...}) 또는 체인 dict에서 {"amp_id", "chain"}을 꺼냅니다."""
    if isinstance(candidate, dict) and isinstance(candidate.get("chain"), dict):
        return copy.deepcopy(candidate["chain"])
    if isinstance(candidate, dict) and isinstance(candidate.get("chain"), list):
        return copy.deepcopy(candidate)
    raise ValueError(f"지원하지 않는 후보 형식: {candidate}")


def expand_with_compressor(candidates, effects=None):
    """[컴프 없는 후보들] + [같은 후보 앞에 컴프(기본 노브)를 붙인 후보들] (체인 dict 목록)."""
    comp = {"effect_type": COMPRESSOR, "values": compressor_defaults(effects)}
    plain = [_chain_of(c) for c in candidates]
    with_comp = []
    for chain in plain:
        if any(item["effect_type"] == COMPRESSOR for item in chain["chain"]):
            continue
        with_comp.append(dict(copy.deepcopy(chain), chain=[copy.deepcopy(comp)] + copy.deepcopy(chain["chain"])))
    return plain + with_comp


def _drive_key(effect_types):
    return tuple(e for e in effect_types if e != COMPRESSOR)


def choose_compressor(ranked, margin=DEFAULT_MARGIN):
    """탐색 결과(search_and_rank_chains의 ranked)에서 최종 후보를 고릅니다.

    같은 드라이브끼리 비교해 컴프 후보가 (1 - margin)배보다 낮을 때만 컴프 후보를 남기고,
    그 뒤 거리순으로 1위를 돌려줍니다. 반환값에 "compressor_used"를 추가합니다.
    """
    by_drive = {}
    for item in ranked:
        key = _drive_key(item["effect_types"])
        slot = by_drive.setdefault(key, {})
        slot["comp" if COMPRESSOR in item["effect_types"] else "plain"] = item
    finalists = []
    for slot in by_drive.values():
        plain, comp = slot.get("plain"), slot.get("comp")
        if comp is not None and (plain is None or comp["loss"] < plain["loss"] * (1 - margin)):
            finalists.append(dict(comp, compressor_used=True))
        elif plain is not None:
            finalists.append(dict(plain, compressor_used=False))
    return min(finalists, key=lambda r: r["loss"])
