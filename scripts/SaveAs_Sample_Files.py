import os
import shutil

# 경로(Path) 설정: 다운로드한 실제 원본 데이터셋(Dataset)의 폴더 경로
SOURCE_AUDIO_DIR = "/Users/jinwon/Desktop/HAI_2026/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/audio"
SOURCE_ANNO_DIR = "/Users/jinwon/Desktop/HAI_2026/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/annotation"

# 변환된 파일을 저장할 목적지 경로(Path)
DEST_BASE_DIR = "data/dry/Fender_Strat_Clean"
DEST_AUDIO_DIR = os.path.join(DEST_BASE_DIR, "audio")
DEST_ANNO_DIR = os.path.join(DEST_BASE_DIR, "annotation")

# 목적지 디렉터리(Directory) 생성
os.makedirs(DEST_AUDIO_DIR, exist_ok=True)
os.makedirs(DEST_ANNO_DIR, exist_ok=True)

# 원본 오디오 파일 목록 가져오기 (.wav 확장자 필터링)
audio_files = [f for f in os.listdir(SOURCE_AUDIO_DIR) if f.endswith(".wav")]

for filename in audio_files:
    # 파일명 분리 (확장자 제외)
    base_name = os.path.splitext(filename)[0]
    
    # 주석(Annotation) 파일명 추론 (.xml 가정)
    anno_filename = base_name + ".xml"
    
    # 원본 경로(Path)
    src_audio_path = os.path.join(SOURCE_AUDIO_DIR, filename)
    src_anno_path = os.path.join(SOURCE_ANNO_DIR, anno_filename)
    
    # 새 파일명 생성
    new_audio_filename = f"sample_{filename}"
    new_anno_filename = f"sample_{anno_filename}"
    
    # 목적지 경로(Path)
    dest_audio_path = os.path.join(DEST_AUDIO_DIR, new_audio_filename)
    dest_anno_path = os.path.join(DEST_ANNO_DIR, new_anno_filename)
    
    # 오디오(Audio) 파일 복사
    shutil.copy2(src_audio_path, dest_audio_path)
    
    # 주석(Annotation) 파일이 존재하면 동일하게 접두사(Prefix)를 붙여 복사
    if os.path.exists(src_anno_path):
        shutil.copy2(src_anno_path, dest_anno_path)

print(f"총 {len(audio_files)}개 파일의 이름 변환 및 복사가 완료되었습니다.")