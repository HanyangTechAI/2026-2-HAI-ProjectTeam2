"""
scripts/apply_amp_dataset.py

드라이브만 있는 생성 데이터셋(예: step1_vst_v1)에 현재 고정 앰프(schemas/amp.data.json, 캐비닛 IR)를
입혀 새 데이터셋 폴더를 만듭니다. VST는 필요 없습니다 (앰프는 드라이브 뒤에 붙으므로 저장된 wet에 바로 적용).

- wet: effects.render.apply_amp로 앰프 적용 → 렌더러(render_chain)로 dry부터 다시 만든 결과와 같습니다.
- labels: ground_truth_chain.amp_id를 현재 앰프 ID로, file_path를 새 폴더 기준으로 바꿉니다.
- dry/, manifest.json: 그대로 복사하고 manifest에 앰프 정보를 기록합니다.

사용 예 (저장소 루트에서)
-------------------------
python -m scripts.apply_amp_dataset --input data/synthetic/step1_vst_v1 --output "%TEMP%\\step2_ir_v1"
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from effects.render import apply_amp, load_amp, NO_AMP_IDS
from scripts.generate_step1_dataset import amp_context_fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="드라이브만 있는 데이터셋 폴더 (dry/, audio/, labels/, manifest.json)")
    parser.add_argument("--output", required=True, help="새로 만들 폴더 (이미 있으면 중단)")
    args = parser.parse_args(argv)

    src, dst = Path(args.input), Path(args.output)
    if dst.exists():
        raise FileExistsError(f"이미 있는 폴더입니다. 새 경로를 지정하세요: {dst}")
    amp = load_amp()
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    labels = sorted((src / "labels").glob("*.json"))
    if len(labels) != manifest["completed_pairs"]:
        raise ValueError("manifest와 라벨 수가 다릅니다: 완성된 데이터셋인지 확인하세요.")

    (dst / "audio").mkdir(parents=True)
    (dst / "labels").mkdir()
    shutil.copytree(src / "dry", dst / "dry")

    clipped = 0
    for i, path in enumerate(labels, 1):
        label = json.loads(path.read_text(encoding="utf-8"))
        chain = label["ground_truth_chain"]
        if chain["amp_id"] not in NO_AMP_IDS:
            raise ValueError(f"이미 앰프가 적용된 라벨입니다: {path.name} ({chain['amp_id']})")
        wet, sr = sf.read(src / "audio" / f"{path.stem}.wav", dtype="float32", always_2d=True)
        out = apply_amp(wet.T, sr, amp["amp_id"], amp).T
        if not np.isfinite(out).all():
            raise ValueError(f"앰프 적용 결과가 유효하지 않습니다: {path.name}")
        clipped += int(np.max(np.abs(out)) > 1.0)
        audio_path = dst / "audio" / f"{path.stem}.wav"
        sf.write(audio_path, out, sr, subtype="FLOAT")
        label["ground_truth_chain"] = dict(chain, amp_id=amp["amp_id"])
        label["file_path"] = str(audio_path)
        (dst / "labels" / path.name).write_text(json.dumps(label, ensure_ascii=False) + "\n", encoding="utf-8")
        if i % 300 == 0 or i == len(labels):
            print(json.dumps({"done": i, "total": len(labels)}), flush=True)

    manifest["amp_id"] = amp["amp_id"]
    manifest["amp"] = amp
    manifest.update(amp_context_fields(amp["amp_id"]))  # CNN(estimation) 호환
    manifest["source_dataset_dir"] = src.name
    manifest["output"] = str(dst)
    manifest["warnings"] = [w for w in manifest.get("warnings", []) if "amp" not in w] + [
        f"amp '{amp['amp_id']}' applied to saved drive wet by scripts/apply_amp_dataset.py"]
    (dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(labels), "clipped_files": clipped, "output": str(dst)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
