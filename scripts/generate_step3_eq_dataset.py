"""기존 이펙터+IR 데이터셋에서 드라이브당 N개를 파일명 기준으로 먼저 무작위 선별한 뒤,
선택된 파일만 읽어 eq_3band를 100% 적용합니다.

실행 예 (저장소 루트):
    python -m scripts.generate_step3_eq_dataset --dry-run
    python -m scripts.generate_step3_eq_dataset
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import shutil
import sys

import numpy as np
import soundfile as sf
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from effects import render_chain
from effects.audio_io import read_mono
from schemas.validate import load_schemas, validate_chain, validate_effects
from scripts.generate_step1_dataset import (
    DRIVES,
    _json,
    amp_context_fields,
    sample_parameter,
)

# 기존 파일명 규칙: wet_{dry}_{drive}_{번호}.json
DRIVE_PATTERN = re.compile(rf"_({'|'.join(DRIVES)})_\d+$")


def select_candidates_by_filename(input_dir: Path, per_drive_count: int, seed: int) -> list:
    """JSON 파일을 열지 않고 파일명만으로 드라이브별 per_drive_count개씩 먼저 추립니다."""
    audio_dir = input_dir / "audio"
    labels_dir = input_dir / "labels"
    if not audio_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"audio/ 또는 labels/ 폴더를 찾을 수 없습니다: {input_dir}")

    by_drive = defaultdict(list)
    for label_path in sorted(labels_dir.glob("*.json")):
        match = DRIVE_PATTERN.search(label_path.stem)
        if not match:
            continue
        by_drive[match.group(1)].append(label_path)

    rng = random.Random(seed)
    selected_paths = []
    for drive in DRIVES:
        candidates = sorted(by_drive[drive])
        if len(candidates) < per_drive_count:
            raise ValueError(
                f"드라이브 '{drive}'의 후보가 부족합니다 (보유: {len(candidates)}개, 필요: {per_drive_count}개)."
            )
        selected_paths.extend(rng.sample(candidates, per_drive_count))

    return selected_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "synthetic" / "step2_ir_v1",
        help="기존 이펙터+IR 데이터셋 경로 (기본값: data/synthetic/step2_ir_v1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop" / "Step3_eq_applied",
        help="출력 폴더 경로 (기본값: ~/Desktop/Step3_eq_applied)",
    )
    parser.add_argument("--per-drive-count", type=int, default=40, help="드라이브당 추출할 음원 개수 (기본값: 40)")
    parser.add_argument("--seed", type=int, default=7, help="난수 시드 (기본값: 7)")
    parser.add_argument("--dry-run", action="store_true", help="파일을 생성하지 않고 선별 결과만 미리 확인")
    args = parser.parse_args(argv)

    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not args.dry_run and output_dir.exists():
        raise FileExistsError(f"출력 폴더가 이미 존재합니다 (덮어쓰기 방지): {output_dir}")

    # 1. 스키마 및 앰프 정의 로드·검증
    schemas, registry = load_schemas(ROOT / "schemas")
    effects = json.loads((ROOT / "schemas" / "effects.data.json").read_text(encoding="utf-8"))
    amp = json.loads((ROOT / "schemas" / "amp.data.json").read_text(encoding="utf-8"))
    Draft202012Validator(schemas["effects"]).validate(effects)
    validate_effects(effects)
    target_validator = Draft202012Validator(schemas["target_sample"], registry=registry)

    # 2. 파일명만으로 드라이브별 무작위 40개씩(총 280개) 즉시 선별
    selected_label_paths = select_candidates_by_filename(input_dir, args.per_drive_count, args.seed)

    print(f"[준비 완료] 입력: {input_dir}", flush=True)
    print(f"[준비 완료] 출력: {output_dir}", flush=True)
    print(
        f"[선별 개수] 드라이브 {len(DRIVES)}종 × {args.per_drive_count}개 = 총 {len(selected_label_paths)}쌍 (seed={args.seed})",
        flush=True,
    )

    if args.dry_run:
        print("[Dry-Run] 실제 파일은 생성하지 않았습니다. 샘플 5건 미리보기:", flush=True)
        for label_path in selected_label_paths[:5]:
            print(f"  - {label_path.stem}", flush=True)
        return

    # 3. 출력 디렉터리 생성
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "audio").mkdir()
    (output_dir / "labels").mkdir()
    (output_dir / "dry").mkdir()

    # 4. 선택된 280개만 읽어 eq_3band 단독 렌더링(IR 미중복) 및 라벨 저장
    eq_def = effects["eq_3band"]
    eq_rng = random.Random(args.seed)
    used_dry_ids = set()
    total = len(selected_label_paths)

    for idx, label_path in enumerate(selected_label_paths, start=1):
        label = json.loads(label_path.read_text(encoding="utf-8"))
        target_id = label["target_id"]
        audio_path = input_dir / "audio" / f"{target_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(f"라벨과 대응하는 음원 파일이 없습니다: {audio_path}")

        effect_types = [item["effect_type"] for item in label["ground_truth_chain"]["chain"]]
        if "eq_3band" in effect_types or "compressor" in effect_types:
            raise ValueError(f"이미 eq_3band 또는 compressor가 포함된 라벨입니다: {target_id}")

        used_dry_ids.add(label["dry_sample_id"])

        eq_values = {param["name"]: sample_parameter(param, eq_rng) for param in eq_def["params"]}
        eq_item = {"effect_type": "eq_3band", "values": eq_values}

        # 모노 44.1 kHz 로드 후 (channels, samples) 형태로 변환
        audio, sr = read_mono(audio_path)
        wav_2d = audio[np.newaxis, :].astype(np.float32)

        # chain을 dict가 아닌 list로 넘겨 amp_id(IR) 중복 적용 없이 eq_3band만 통과
        eq_wet = np.asarray(render_chain(wav_2d, sr, [deepcopy(eq_item)], effects))
        if eq_wet.shape != wav_2d.shape or not np.isfinite(eq_wet).all():
            raise ValueError(f"비정상 렌더링 결과가 발생했습니다: {target_id}")

        out_audio_path = output_dir / "audio" / f"{target_id}.wav"
        with out_audio_path.open("xb") as stream:
            sf.write(stream, eq_wet.T, sr, format="WAV", subtype="FLOAT")

        new_label = deepcopy(label)
        new_label["ground_truth_chain"]["amp_id"] = amp["amp_id"]
        new_label["ground_truth_chain"]["chain"].append(eq_item)
        new_label["file_path"] = out_audio_path.as_posix()

        validate_chain(new_label["ground_truth_chain"], effects, amp)
        target_validator.validate(new_label)

        out_label_path = output_dir / "labels" / f"{target_id}.json"
        with out_label_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(new_label) + "\n")

        if idx % 10 == 0 or idx == total:
            print(f"  진행률: {idx}/{total} 완료 ({target_id})", flush=True)

    # 5. 사용된 dry 음원 복사 및 manifest.json 작성
    src_manifest_path = input_dir / "manifest.json"
    src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8")) if src_manifest_path.is_file() else {}
    used_dry_map = [
        entry for entry in src_manifest.get("dry_id_map", [])
        if entry.get("dry_sample_id") in used_dry_ids
    ]

    copied_dry = 0
    for entry in used_dry_map:
        rel_dry = entry.get("render_input")
        if not rel_dry:
            continue
        src_dry = input_dir / rel_dry
        dst_dry = output_dir / rel_dry
        if src_dry.is_file():
            dst_dry.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dry, dst_dry)
            copied_dry += 1

    manifest = {
        "algorithm": "step3-eq-applied-from-ir-v1",
        "seed": args.seed,
        "source_ir_dataset": str(input_dir),
        "source_dataset": src_manifest.get("source_dataset", "IDMT-SMT-Guitar V2"),
        "drives": list(DRIVES),
        "per_drive_count": args.per_drive_count,
        "compressor_probability": 0.0,
        "eq_probability": 1.0,
        "pairs": total,
        "completed_pairs": total,
        "wav_files": total,
        "label_files": total,
        "dry_files": copied_dry,
        "dry_id_map": used_dry_map,
    }
    # 평가·탐색 모듈이 IR 데이터로 인식할 수 있도록 amp_context 및 해시 기록
    manifest.update(amp_context_fields(amp["amp_id"]))

    with (output_dir / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_json(manifest) + "\n")

    print(f"[완료] {output_dir} 생성 완료 (audio/labels 각 {total}개, dry {copied_dry}개)", flush=True)


if __name__ == "__main__":
    main()