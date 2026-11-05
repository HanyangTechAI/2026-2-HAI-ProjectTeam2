# 임시;  pedalboard 내장 디스토션으로만 파일 생성하는 코드.
import os
import json
import random
import soundfile as sf
from pedalboard import Pedalboard, Distortion, Limiter

# =====================================================================
# 1. 초기 설정 및 난수 시드(Random Seed) 고정
# =====================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 현재 스크립트 실행 위치 기준 동적 상대경로 세팅
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DRY_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "dry","Fender_Strat_Clean", "audio")
SYNTH_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic", "audio")
SYNTH_LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic", "labels")

os.makedirs(SYNTH_AUDIO_DIR, exist_ok=True)
os.makedirs(SYNTH_LABEL_DIR, exist_ok=True)

# =====================================================================
# 2. 매개변수(Parameter) 무작위 생성 (내장 Distortion 기준)
# =====================================================================
def get_builtin_params(effect_type):
    """
    pedalboard 내장 Distortion의 drive_db 수치를 
    오버드라이브, 디스토션, 퍼즈 질감에 맞춰 3단계로 분리합니다.
    """
    if effect_type == "overdrive":
        # 가벼운 드라이브 질감 (5dB ~ 15dB)
        return {"drive_db": round(random.uniform(5.0, 15.0), 2)}
    elif effect_type == "distortion":
        # 강렬한 하드 클리핑 질감 (20dB ~ 35dB)
        return {"drive_db": round(random.uniform(20.0, 35.0), 2)}
    elif effect_type == "fuzz":
        # 찌그러지고 먹먹한 극단적 게인 질감 (40dB ~ 60dB)
        return {"drive_db": round(random.uniform(40.0, 60.0), 2)}
    return {}

# =====================================================================
# 3. 메인 파이프라인(Main Pipeline) 로직
# =====================================================================
def generate_dataset():
    # dry 폴더에서 wav 파일 목록 불러오기
    if not os.path.exists(DRY_AUDIO_DIR):
        print(f"원본 폴더를 찾을 수 없습니다: {DRY_AUDIO_DIR}")
        return
        
    dry_files = [f for f in os.listdir(DRY_AUDIO_DIR) if f.endswith(".wav")]
    effect_types = ["overdrive", "distortion", "fuzz"]
    
    # 디지털 클리핑 방지용 리미터
    safety_limiter = Limiter(threshold_db=-0.5, release_ms=100.0)
    
    for filename in dry_files:
        dry_path = os.path.join(DRY_AUDIO_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        # 오디오 데이터 읽기
        audio_data, sample_rate = sf.read(dry_path)
        if audio_data.ndim == 2:
            audio_data = audio_data.T 
            
        for e_type in effect_types:
            params = get_builtin_params(e_type)
            
            # VST 로드 대신 내장 Distortion 객체 즉시 생성
            builtin_dist = Distortion(drive_db=params["drive_db"])
            
            # 페달보드 체인 조립 및 렌더링
            board = Pedalboard([builtin_dist, safety_limiter])
            wet_audio = board(audio_data, sample_rate)
            
            # 파일명 및 저장 경로 지정
            wet_filename = f"wet_{base_name}_{e_type}.wav"
            label_filename = f"params_{base_name}_{e_type}.json"
            
            wet_path = os.path.join(SYNTH_AUDIO_DIR, wet_filename)
            label_path = os.path.join(SYNTH_LABEL_DIR, label_filename)
            
            # JSON 정답지 조립
            label_data = {
                "target_id": os.path.splitext(wet_filename)[0],
                "is_synthetic": True,
                "dry_sample_id": base_name,
                "source_dataset": "IDMT-SMT-Guitar V2",
                "file_path": wet_path,
                "ground_truth_chain": {
                    "amp_id": "bypass",
                    "chain": [
                        {
                            "effect_type": e_type,
                            "plugin_type": "pedalboard_builtin",
                            "values": params
                        }
                    ]
                }
            }
            
            # 결과물 디스크에 저장
            sf.write(wet_path, wet_audio.T if wet_audio.ndim == 2 else wet_audio, sample_rate, subtype='PCM_16')
            
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, indent=4, ensure_ascii=False)
                
        print(f"생성 완료: {filename} 기반 3종 데이터")

if __name__ == "__main__":
    generate_dataset()