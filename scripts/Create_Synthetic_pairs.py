import os
import json
import random
import soundfile as sf
from pedalboard import Pedalboard, Limiter, load_plugin

# =====================================================================
# 1. 초기 설정 및 난수 시드 고정
# =====================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 데이터셋 입출력 경로 설정
DRY_AUDIO_DIR = "../data/dry/IDMT-SMT-Guitar_V2/audio"
SYNTH_AUDIO_DIR = "../data/synthetic/IDMT-SMT-Guitar_V2/audio"
SYNTH_LABEL_DIR = "../data/synthetic/IDMT-SMT-Guitar_V2/labels"

os.makedirs(SYNTH_AUDIO_DIR, exist_ok=True)
os.makedirs(SYNTH_LABEL_DIR, exist_ok=True)

# =====================================================================
# 2. VST3 플러그인 로드
# =====================================================================
try:
    # 직접 수정하실 경로입니다. 해당 폴더에 .vst3 파일이 위치해야 합니다.
    overdrive_vst = load_plugin("./data/synthetic/VST_Plugin/plugins/OverDiver.vst3")
    distortion_vst = load_plugin("./data/synthetic/VST_Plugin/plugins/ProFET.vst3")
    fuzz_vst = load_plugin("./data/synthetic/VST_Plugin/plugins/PigPie.vst3")
except Exception as e:
    print(f"VST 플러그인 로드 실패: {e}")
    # 플러그인이 로드되지 않으면 실행을 중단합니다.
    exit(1)

# =====================================================================
# 3. 매개변수 무작위 생성
# =====================================================================
def get_vst_params(effect_type):
    """
    확정된 3종 VST의 실제 노브(Knob) 구성에 맞춘 파라미터를 반환합니다.
    """
    if effect_type == "overdrive": 
        # OverDiver
        return {
            "drive": round(random.uniform(0.1, 0.8), 2),
            "tone": round(random.uniform(0.2, 0.8), 2),
            "level": round(random.uniform(0.2, 0.6), 2)
        }
    elif effect_type == "distortion": 
        # ProF.E.T.
        return {
            "gain": round(random.uniform(0.3, 1.0), 2),
            "bass": round(random.uniform(0.3, 0.7), 2),
            "middle": round(random.uniform(0.2, 0.8), 2),
            "treble": round(random.uniform(0.3, 0.8), 2),
            "volume": round(random.uniform(0.2, 0.6), 2)
        }
    elif effect_type == "fuzz": 
        # Pig Pie 
        return {
            "sustain": round(random.uniform(0.4, 1.0), 2), 
            "tone": round(random.uniform(0.2, 0.8), 2),
            "volume": round(random.uniform(0.2, 0.6), 2)
        }
    return {}

# =====================================================================
# 4. 메인 파이프라인(Main Pipeline) 로직
# =====================================================================
def generate_dataset():
    dry_files = [f for f in os.listdir(DRY_AUDIO_DIR) if f.endswith(".wav")]
    effect_types = ["overdrive", "distortion", "fuzz"]
    
    #Digital Clipping 방지용 리미터
    safety_limiter = Limiter(threshold_db=-0.5, release_ms=100.0)
    
    for filename in dry_files:
        dry_path = os.path.join(DRY_AUDIO_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        # 오디오 데이터 읽기
        audio_data, sample_rate = sf.read(dry_path)
        if audio_data.ndim == 2:
            audio_data = audio_data.T # (samples, channels) -> (channels, samples)
            
        for e_type in effect_types:
            params = get_vst_params(e_type)
            board_plugins = []
            
            # 1. 플러그인에 매개변수 할당
            if e_type == "overdrive":
                # OverDiver 파라미터 적용
                overdrive_vst.drive = params["drive"]
                overdrive_vst.tone = params["tone"]
                overdrive_vst.level = params["level"]
                board_plugins.append(overdrive_vst)
                
            elif e_type == "distortion":
                # ProFET 파라미터 적용
                distortion_vst.gain = params["gain"]
                distortion_vst.bass = params["bass"]
                distortion_vst.middle = params["middle"]
                distortion_vst.treble = params["treble"]
                distortion_vst.volume = params["volume"]
                board_plugins.append(distortion_vst)
                
            elif e_type == "fuzz":
                # PigPie 파라미터 적용
                fuzz_vst.sustain = params["sustain"]
                fuzz_vst.tone = params["tone"]
                fuzz_vst.volume = params["volume"]
                board_plugins.append(fuzz_vst)

            # 2. 신호 사슬 구성 및 렌더링
            board_plugins.append(safety_limiter)
            board = Pedalboard(board_plugins)
            wet_audio = board(audio_data, sample_rate)
            
            # 3. 파일명 및 경로 지정
            wet_filename = f"wet_{base_name}_{e_type}.wav"
            label_filename = f"params_{base_name}_{e_type}.json"
            
            wet_path = os.path.join(SYNTH_AUDIO_DIR, wet_filename)
            label_path = os.path.join(SYNTH_LABEL_DIR, label_filename)
            
            # 4. JSON 정답지
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
                            "values": params
                        }
                    ]
                }
            }
            
            # 5. 디스크에 저장
            sf.write(wet_path, wet_audio.T if wet_audio.ndim == 2 else wet_audio, sample_rate, subtype='PCM_16')
            
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, indent=4, ensure_ascii=False)
                
        print(f"생성 완료: {filename} 기반 데이터")

if __name__ == "__main__":
    generate_dataset()