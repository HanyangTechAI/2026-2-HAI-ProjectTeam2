"""
scripts/prepare_step3_targets.py

Step 3 목표 소리(외부 기타 톤) 전처리: 원본 wav → 탐색에 넣을 2.5초 구간.

- 모노 44.1 kHz (effects.audio_io.read_mono)
- 음량이 큰 2.5초 구간 후보를 겹치지 않게 최대 3개 고르고, 기본 선택은 가장 큰 구간
- 음량(RMS)을 학습 데이터(step2_ir_v1 wet) 수준으로 맞춤 (외부 음원은 보통 10~15 dB 큼)
- 출력: <out>/<kind>/<id>.wav (선택 구간), <out>/candidates/<id>_c1~3.wav (청취 확인용), <out>/targets.csv

kind: separated = 음원에서 분리한 기타 / isolated = 기타 단독 톤(페달 데모 등)
isolated 파일명이 "브랜드_모델_노브_값_..." 형식이면 페달과 노브(시계 위치)를 targets.csv에 기록합니다.

사용 예 (저장소 루트에서)
-------------------------
python -m scripts.prepare_step3_targets --separated <드라이브>/data/step3_real_data
    --isolated <드라이브>/data/step3_val_data --output data/local/step3_targets_v1
선택 구간 바꾸기: targets.csv의 chosen 칸을 c2/c3으로 고친 뒤 --rechoose로 다시 실행
--intro: 분리 기타는 가장 큰 구간 대신 "기타가 처음 들어오는 지점"부터 2.5초 (가장 큰 구간은 잔여음·깨짐이 많음)
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from effects.audio_io import STANDARD_SAMPLE_RATE, read_mono

SEGMENT_SECONDS = 2.5
TARGET_RMS_DB = -33.8   # step2_ir_v1 wet 평균 RMS (11/24 측정, 273개)
HOP_SECONDS = 0.25
CANDIDATES = 3

# 페달 데모 파일명 → 우리 7종 (없으면 빈 값: 7종 밖 페달)
PEDAL_MAP = {"tubescreamer": "808_scream", "ts808": "808_scream", "ts9": "808_scream", "bd2": "b_drive",
             "od1": "o_drive", "ds1": "one_distortion", "rat": "bat_distortion", "fuzzface": "f_fuzz",
             "bigmuff": "m_fuzz"}
KNOB_WORDS = {"O", "OD", "D", "G", "T", "L", "V", "OUT", "TREBLE", "TONE", "LEVEL", "GAIN", "DRIVE", "VOL", "F"}


def slug(name):
    name = re.sub(r"\(mp3cut\.net\)|_guitar|\[.*?\]|\(.*?\)|official.*$", "", name, flags=re.I)
    name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return name[:48] or "target"


def parse_isolated(stem):
    """'Boss_BD2_L_12_T_11_G_9' → pedal 'Boss BD2', knobs 'L=12 T=11 G=9', drive 'b_drive'"""
    tokens = stem.split("_")
    i = next((k for k, t in enumerate(tokens) if t.upper() in KNOB_WORDS and k + 1 < len(tokens)), len(tokens))
    pedal = " ".join(tokens[:i])
    knobs = " ".join(f"{tokens[k]}={tokens[k + 1]}" for k in range(i, len(tokens) - 1, 2))
    key = re.sub(r"[^a-z0-9]", "", "".join(tokens[1:i]).lower())
    drive = next((v for k, v in PEDAL_MAP.items() if k in key), "")
    return pedal, knobs, drive


def candidates(audio, sr):
    win, hop = int(SEGMENT_SECONDS * sr), int(HOP_SECONDS * sr)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    starts = np.arange(0, len(audio) - win + 1, hop)
    energy = np.array([np.mean(audio[s:s + win] ** 2) for s in starts])
    chosen = []
    for idx in np.argsort(energy)[::-1]:
        s = int(starts[idx])
        if all(abs(s - c) >= win for c in chosen):
            chosen.append(s)
        if len(chosen) == CANDIDATES:
            break
    return [(s, audio[s:s + win]) for s in chosen]


def intro_start(audio, sr, rel_db=-15.0, active_share=0.7):
    """기타가 처음 들어오는 지점(초). 100 ms 창 음량이 곡의 큰 부분(95번째 백분위) −15 dB를 넘고,
    이후 2.5초 중 70% 이상이 그 수준이면 시작으로 봄 (분리 잔여음의 작은 소리는 무시)."""
    win = int(0.1 * sr)
    lv = 10 * np.log10(np.array([np.mean(audio[i:i + win] ** 2) for i in range(0, len(audio) - win, win)]) + 1e-12)
    active = lv > np.percentile(lv, 95) + rel_db
    need = int(SEGMENT_SECONDS / 0.1)
    for i in range(len(active) - need):
        if active[i] and active[i:i + need].mean() >= active_share:
            return i * win
    return candidates(audio, sr)[0][0]


def normalize(clip):
    rms = np.sqrt(np.mean(clip ** 2))
    return (clip * (10 ** (TARGET_RMS_DB / 20) / max(rms, 1e-9))).astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--separated", help="음원에서 분리한 기타 wav 폴더")
    parser.add_argument("--isolated", help="기타 단독 톤 wav 폴더")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rechoose", action="store_true", help="targets.csv의 chosen 칸대로 선택 구간만 다시 저장")
    parser.add_argument("--intro", action="store_true", help="분리 기타는 기타가 처음 들어오는 지점부터 (chosen=intro)")
    args = parser.parse_args(argv)
    out = Path(args.output)

    if args.rechoose:
        rows = list(csv.DictReader(open(out / "targets.csv", encoding="utf-8-sig")))
        for r in rows:
            src = out / "candidates" / f"{r['id']}_{r['chosen']}.wav"
            x, sr = sf.read(src, dtype="float32")
            sf.write(out / r["kind"] / f"{r['id']}.wav", x, sr, subtype="FLOAT")
            r["start_sec"] = r[f"start_{r['chosen']}"]
        _write_csv(out / "targets.csv", rows)
        print(f"{len(rows)}개 선택 구간 갱신")
        return

    if out.exists():
        raise FileExistsError(f"이미 있는 폴더입니다. 새 경로를 지정하거나 --rechoose를 쓰세요: {out}")
    (out / "candidates").mkdir(parents=True)
    rows, used = [], set()
    for kind, folder in (("separated", args.separated), ("isolated", args.isolated)):
        if not folder:
            continue
        (out / kind).mkdir()
        for path in sorted(p for p in Path(folder).glob("*.wav") if not p.name.startswith("._")):
            audio, sr = read_mono(path)
            base = slug(path.stem)
            rid, n = base, 2
            while rid in used:
                rid, n = f"{base}_{n}", n + 1
            used.add(rid)
            cands = candidates(audio, sr)
            row = {"id": rid, "kind": kind, "source_file": path.name, "chosen": "c1",
                   "start_sec": round(cands[0][0] / sr, 2)}
            chosen_clip = cands[0][1]
            if args.intro and kind == "separated":
                s0 = intro_start(audio, sr)
                chosen_clip = audio[s0:s0 + int(SEGMENT_SECONDS * sr)]
                row.update({"chosen": "intro", "start_sec": round(s0 / sr, 2)})
            for k, (start, clip) in enumerate(cands, 1):
                sf.write(out / "candidates" / f"{rid}_c{k}.wav", normalize(clip), sr, subtype="FLOAT")
                row[f"start_c{k}"] = round(start / sr, 2)
            sf.write(out / kind / f"{rid}.wav", normalize(chosen_clip), sr, subtype="FLOAT")
            pedal, knobs, drive = parse_isolated(path.stem) if kind == "isolated" else ("", "", "")
            row.update({"pedal": pedal, "knobs_clock": knobs, "drive_in_catalog": drive,
                        "source_rms_db": round(float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)), 1),
                        "note": ""})
            rows.append(row)
    _write_csv(out / "targets.csv", rows)
    print(f"{len(rows)}개 → {out} (sr={STANDARD_SAMPLE_RATE}, {SEGMENT_SECONDS}s, RMS {TARGET_RMS_DB} dBFS)")


def _write_csv(path, rows):
    fields = ["id", "kind", "source_file", "chosen", "start_sec", "start_c1", "start_c2", "start_c3",
              "pedal", "knobs_clock", "drive_in_catalog", "source_rms_db", "note"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)


if __name__ == "__main__":
    main()
