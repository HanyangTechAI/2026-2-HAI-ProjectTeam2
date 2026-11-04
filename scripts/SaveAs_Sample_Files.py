#python 3.11.17
#이진원

import os
import zipfile

# 원본 데이터셋(Dataset) zip 경로: 공유 드라이브의 data/raw (DRIVE_SETUP.md 참고)
# https://zenodo.org/records/7544110 의 IDMT-SMT-GUITAR_V2.zip 을 압축 해제 없이 그대로 사용합니다.
SOURCE_ZIP = "data/raw/IDMT-SMT-GUITAR_V2.zip"
# zip 내부에서 가져올 폴더. 이름이 비슷한 "... Neck SC Chords" 폴더와 섞이지 않도록 끝에 "/"까지 비교합니다.
SOURCE_SUBSET = "IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/"

# 변환된 파일을 저장할 목적지 경로(Path)
DEST_BASE_DIR = "data/dry/Fender_Strat_Clean"
DEST_AUDIO_DIR = os.path.join(DEST_BASE_DIR, "audio")
DEST_ANNO_DIR = os.path.join(DEST_BASE_DIR, "annotation")

# 목적지 디렉터리(Directory) 생성
os.makedirs(DEST_AUDIO_DIR, exist_ok=True)
os.makedirs(DEST_ANNO_DIR, exist_ok=True)

with zipfile.ZipFile(SOURCE_ZIP) as zf:
    members = set(zf.namelist())

    # 원본 오디오 파일 목록 가져오기 (.wav 확장자 필터링)
    audio_files = sorted(
        os.path.basename(name) for name in members
        if name.startswith(SOURCE_SUBSET + "audio/") and name.endswith(".wav")
    )

    for filename in audio_files:
        # 파일명 분리 (확장자 제외)
        base_name = os.path.splitext(filename)[0]

        # 주석(Annotation) 파일명 추론 (.xml 가정)
        anno_filename = base_name + ".xml"

        # 원본 경로(Path): zip 내부 경로
        src_audio_path = SOURCE_SUBSET + "audio/" + filename
        src_anno_path = SOURCE_SUBSET + "annotation/" + anno_filename

        # 새 파일명 생성
        new_audio_filename = f"sample_{filename}"
        new_anno_filename = f"sample_{anno_filename}"

        # 목적지 경로(Path)
        dest_audio_path = os.path.join(DEST_AUDIO_DIR, new_audio_filename)
        dest_anno_path = os.path.join(DEST_ANNO_DIR, new_anno_filename)

        # 오디오(Audio) 파일 복사
        with open(dest_audio_path, "wb") as f:
            f.write(zf.read(src_audio_path))

        # 주석(Annotation) 파일이 존재하면 동일하게 접두사(Prefix)를 붙여 복사
        # (이미 GitHub에 있으므로, 받은 파일이 있으면 덮어쓰지 않습니다)
        if os.path.exists(dest_anno_path):
            continue
        if src_anno_path in members:
            with open(dest_anno_path, "wb") as f:
                f.write(zf.read(src_anno_path))
        else:
            print(f"주석 파일 없음: {anno_filename}")

print(f"총 {len(audio_files)}개 파일의 이름 변환 및 복사가 완료되었습니다.")
