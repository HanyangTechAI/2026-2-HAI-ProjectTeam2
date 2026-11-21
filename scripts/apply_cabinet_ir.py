import time
from pathlib import Path
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Convolution, Gain

# 1. 경로 및 고정 게인(Gain) 설정
BASE_DIR = Path(r"C:\Users\jinwo\AppData\Local\Temp\step1_vst_v1")
IR_PATH = BASE_DIR / "EV MIX B.wav"
INPUT_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / f"audio_ir_{IR_PATH.stem}"  # -> audio_ir_EV MIX B

# 입력 평균 RMS(-33.86 dBFS)와 출력 평균 RMS(-46.32 dBFS) 차이(+12.46 dB) 보정
MAKEUP_GAIN_DB = 12.5


def calc_rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, eps))


def main() -> None:
    if not IR_PATH.exists():
        raise FileNotFoundError(f"IR 파일을 찾을 수 없습니다: {IR_PATH}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"입력 오디오 폴더를 찾을 수 없습니다: {INPUT_DIR}")

    ir_info = sf.info(str(IR_PATH))
    print(f"[IR 정보] 파일: {IR_PATH.name}")
    print(
        f"  - 표본화율(Sample Rate): {ir_info.samplerate} Hz | "
        f"채널: {ir_info.channels} | 길이: {ir_info.duration:.4f} 초"
    )
    print(f"  - 적용 메이크업 게인(Make-up Gain): +{MAKEUP_GAIN_DB:.2f} dB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(INPUT_DIR.rglob("*.wav"))
    total = len(wav_files)
    if total == 0:
        raise RuntimeError(f"처리할 .wav 파일이 없습니다: {INPUT_DIR}")

    print(f"\n[처리 시작] 총 {total}개 파일 -> {OUTPUT_DIR}")

    # 컨볼루션(Convolution) + 고정 메이크업 게인(Make-up Gain) 체인
    cab_board = Pedalboard([
        Convolution(impulse_response_filename=str(IR_PATH)),
        Gain(gain_db=MAKEUP_GAIN_DB),
    ])

    start_time = time.time()
    in_rms_list = []
    out_rms_list = []
    max_peak = 0.0
    clipped_count = 0

    for idx, wav_path in enumerate(wav_files, start=1):
        info = sf.info(str(wav_path))
        audio, sr = sf.read(str(wav_path), dtype="float32")

        in_rms_list.append(calc_rms_db(audio))

        if audio.ndim == 2:
            processed = cab_board(audio.T, sr, reset=True).T
        else:
            processed = cab_board(audio, sr, reset=True)

        peak = float(np.max(np.abs(processed)))
        max_peak = max(max_peak, peak)
        if peak > 1.0:
            clipped_count += 1
        out_rms_list.append(calc_rms_db(processed))

        # "ir파일이름_기존음원이름.wav" 형태로 저장
        rel_path = wav_path.relative_to(INPUT_DIR)
        new_filename = f"{IR_PATH.stem}_{rel_path.name}"
        out_path = OUTPUT_DIR / rel_path.parent / new_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), processed, sr, subtype=info.subtype)

        if idx % 200 == 0 or idx == total:
            elapsed = time.time() - start_time
            print(f"  진행률: {idx}/{total} ({idx / total * 100:.1f}%) - {elapsed:.1f}초 경과")

    max_peak_db = 20.0 * np.log10(max(max_peak, 1e-12))
    print("\n[처리 완료]")
    print(f"  - 저장 경로: {OUTPUT_DIR}")
    print(f"  - 입력 평균 실효값(RMS): {np.mean(in_rms_list):.2f} dBFS")
    print(f"  - 출력 평균 실효값(RMS): {np.mean(out_rms_list):.2f} dBFS")
    print(f"  - 출력 최대 피크(Peak): {max_peak:.4f} ({max_peak_db:.2f} dBFS)")
    print(f"  - 클리핑(Peak > 1.0) 발생 파일 수: {clipped_count}개")


if __name__ == "__main__":
    main()