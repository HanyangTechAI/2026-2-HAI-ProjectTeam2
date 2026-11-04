
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

# ======================================
# 1. 기본 설정
# ======================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_NAME = "htdemucs_6s"

SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a"
}

OVERWRITE = False


# ======================================
# 2. 기타 추출 함수
# ======================================

def extract_guitar(input_file, output_file):

    with tempfile.TemporaryDirectory() as temp_dir:

        command = [
            sys.executable,
            "-m", "demucs",
            "-n", MODEL_NAME,
            "--two-stems=guitar",
            "-o", temp_dir,
            str(input_file)
        ]

        # Demucs 실행
        subprocess.run(command, check=True)

        # 추출된 기타 파일 경로
        guitar_file = (
            Path(temp_dir)
            / MODEL_NAME
            / input_file.stem
            / "guitar.wav"
        )

        if not guitar_file.exists():
            raise FileNotFoundError(
                f"기타 파일을 찾을 수 없습니다: {guitar_file}"
            )

        # output 폴더에 복사
        shutil.copy2(guitar_file, output_file)


# ======================================
# 3. 폴더 단위 일괄 추출
# ======================================

def main():

    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    audio_files = sorted(
        [
            file for file in INPUT_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_FORMATS
        ],
        key=lambda file: file.name.lower()
    )

    if not audio_files:
        print("input 폴더에 음원이 없습니다.")
        return

    print(f"총 {len(audio_files)}개의 음원을 발견했습니다.")

    success = 0
    skipped = 0
    failed = 0
    used_names = set()

    for i, audio_file in enumerate(audio_files, 1):

        output_name = f"{audio_file.stem}_guitar.wav"

        # 확장자만 다른 동명 파일 구분
        if output_name.lower() in used_names:
            output_name = (
                f"{audio_file.stem}_"
                f"{audio_file.suffix[1:].lower()}_guitar.wav"
            )

        base_name = Path(output_name).stem
        number = 2

        while output_name.lower() in used_names:
            output_name = f"{base_name}_{number}.wav"
            number += 1

        used_names.add(output_name.lower())
        output_file = OUTPUT_DIR / output_name

        print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")

        if output_file.exists() and not OVERWRITE:
            print("이미 추출된 파일 - 건너뛰기")
            skipped += 1
            continue

        try:
            extract_guitar(audio_file, output_file)

            print(f"추출 완료: {output_name}")
            success += 1

        except (subprocess.CalledProcessError, OSError) as e:
            print(f"추출 실패: {e}")
            failed += 1

    print("\n========== 결과 ==========")
    print(f"전체 음원: {len(audio_files)}개")
    print(f"성공: {success}개")
    print(f"건너뛰기: {skipped}개")
    print(f"실패: {failed}개")
    print(f"저장 폴더: {OUTPUT_DIR}")
    print("==========================")


if __name__ == "__main__":
    main()
