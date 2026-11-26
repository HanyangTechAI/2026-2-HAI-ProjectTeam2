"""
scripts/apply_eq_dataset.py

드라이브(+앰프 IR) 데이터셋에 무작위 EQ(eq_3band)를 입혀 EQ 검증셋을 만듭니다. VST가 필요 없습니다.

- EQ와 캐비닛 IR은 둘 다 선형 필터라 순서를 바꿔도 결과가 같습니다.
  → 저장된 wet(드라이브 → IR)에 EQ를 적용 = 렌더러의 드라이브 → EQ → IR 결과와 같습니다.
- --probability 비율의 쌍에만 EQ를 넣고 나머지는 그대로 둡니다 ("EQ 없음" 판정 확인용).
- EQ 노브는 스키마 범위에서 무작위로 뽑습니다 (게인: 균등, 중역 주파수: log 균등).
- labels: ground_truth_chain에 eq_3band를 드라이브 뒤에 추가하고 file_path를 새 폴더 기준으로 바꿉니다.

사용 예 (저장소 루트에서)
-------------------------
python -m scripts.apply_eq_dataset --input data/local/step2_real_v1 --output "%TEMP%\\step3_eq_v1" --probability 0.5 --seed 7
"""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from effects import load_effects
from search.eq_fit import apply_eq

EQ = "eq_3band"


def sample_eq(rng, effects=None):
    values = {}
    for p in (effects or load_effects())[EQ]["params"]:
        low, high = p["range"]
        values[p["name"]] = (math.exp(rng.uniform(math.log(low), math.log(high))) if p.get("scale") == "log"
                             else rng.uniform(low, high))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="드라이브(+IR) 데이터셋 폴더 (dry/, audio/, labels/, manifest.json)")
    parser.add_argument("--output", required=True, help="새로 만들 폴더 (이미 있으면 중단)")
    parser.add_argument("--probability", type=float, default=0.5, help="EQ를 넣을 쌍의 비율")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    src, dst = Path(args.input), Path(args.output)
    if dst.exists():
        raise FileExistsError(f"이미 있는 폴더입니다. 새 경로를 지정하세요: {dst}")
    if not 0 <= args.probability <= 1:
        raise ValueError("--probability는 0~1이어야 합니다.")
    rng = random.Random(args.seed)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    labels = sorted(p for p in (src / "labels").glob("*.json") if not p.name.startswith("._"))

    (dst / "audio").mkdir(parents=True)
    (dst / "labels").mkdir()
    shutil.copytree(src / "dry", dst / "dry", ignore=shutil.ignore_patterns("._*"))

    with_eq = 0
    for path in labels:
        label = json.loads(path.read_text(encoding="utf-8"))
        chain = label["ground_truth_chain"]
        if any(item["effect_type"] == EQ for item in chain["chain"]):
            raise ValueError(f"이미 EQ가 있는 라벨입니다: {path.name}")
        wet, sr = sf.read(src / "audio" / f"{path.stem}.wav", dtype="float32", always_2d=True)
        if rng.random() < args.probability:
            values = sample_eq(rng)
            wet = apply_eq(wet.T, sr, values).T
            chain = dict(chain, chain=chain["chain"] + [{"effect_type": EQ, "values": values}])
            with_eq += 1
        if not np.isfinite(wet).all():
            raise ValueError(f"EQ 적용 결과가 유효하지 않습니다: {path.name}")
        audio_path = dst / "audio" / f"{path.stem}.wav"
        sf.write(audio_path, wet, sr, subtype="FLOAT")
        label["ground_truth_chain"] = chain
        label["file_path"] = str(audio_path)
        (dst / "labels" / path.name).write_text(json.dumps(label, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest["eq_probability"] = args.probability
    manifest["eq_seed"] = args.seed
    manifest["source_dataset_dir"] = src.name
    manifest["output"] = str(dst)
    manifest["warnings"] = list(manifest.get("warnings", [])) + [
        f"eq_3band applied to {with_eq}/{len(labels)} saved wet files by scripts/apply_eq_dataset.py (EQ and IR commute)"]
    (dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(labels), "with_eq": with_eq, "output": str(dst)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
