import os
import json
import random
import numpy as np
import soundfile as sf
import sys
import pyloudnorm as pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from effects import render_chain

# =====================================================================
# configuration 변수
# =====================================================================
VARIATIONS_PER_DRIVE = 3
ENABLE_NORMALIZATION = False
TARGET_LUFS = -24.0

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MIN_PEAK_DB = -20.0
MAX_RETRIES = 20

SOURCE_DATASET = "IDMT-SMT-Guitar V2"
AMP_ID = "fixed_amp_v1"

DRY_AUDIO_DIR = "data/dry/Fender_Strat_Clean/audio"
SYNTH_AUDIO_DIR = "data/synthetic/Fender_Strat_Clean/audio"
SYNTH_LABEL_DIR = "data/synthetic/Fender_Strat_Clean/labels"

# =====================================================================
# directory 생성
# =====================================================================
os.makedirs(SYNTH_AUDIO_DIR, exist_ok=True)
os.makedirs(SYNTH_LABEL_DIR, exist_ok=True)

# =====================================================================
# parameter 추출
# =====================================================================
def get_random_drive(drive_type):
    if drive_type == "overdrive":
        return {
            "hpf_freq": round(random.uniform(150.0, 250.0), 2),
            "drive": round(random.uniform(20.0, 35.0), 2),
            "lpf_freq": round(random.uniform(3000.0, 5000.0), 2),
            "level": round(random.uniform(0.0, 8.0), 2)
        }
    elif drive_type == "distortion":
        return {
            "hpf_freq": round(random.uniform(50.0, 70.0), 2),
            "dist": round(random.uniform(20.0, 35.0), 2),
            "lpf_freq": round(random.uniform(3000.0, 8000.0), 2),
            "level": round(random.uniform(0.0, 8.0), 2)
        }
    elif drive_type == "fuzz":
        return {
            "fuzz": round(random.uniform(20.0, 35.0), 2),
            "notch_freq": round(random.uniform(850.0, 1150.0), 2),
            "notch_gain": round(random.uniform(-9.0, -5.0), 2),
            "lpf_freq": round(random.uniform(2000.0, 3000.0), 2),
            "level": round(random.uniform(0.0, 8.0), 2)
        }

# =====================================================================
# audio 처리
# =====================================================================
def apply_effects(audio_data, sample_rate, comp_params, drive_type, drive_params):
    chain = []
    if comp_params is not None:
        chain.append({"effect_type": "compressor", "values": comp_params})
    chain.append({"effect_type": drive_type, "values": drive_params})
    return render_chain(audio_data, sample_rate, chain)

# =====================================================================
# pipeline 실행
# =====================================================================
def generate_dataset():
    dry_files = sorted(f for f in os.listdir(DRY_AUDIO_DIR) if f.endswith(".wav"))
    drive_types = ["overdrive", "distortion", "fuzz"]
    total_generated = 0
    
    for filename in dry_files:
        dry_path = os.path.join(DRY_AUDIO_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            audio_data, sample_rate = sf.read(dry_path)
        except Exception as e:
            print(f"audio 로드 실패: {filename} - {e}")
            continue
            
        if ENABLE_NORMALIZATION:
            meter = pyln.Meter(sample_rate)
            
        variation_counter = 1
        
        for d_type in drive_types:
            for _ in range(VARIATIONS_PER_DRIVE):
                comp_params = None

                for _ in range(MAX_RETRIES):
                    drive_params = get_random_drive(d_type)
                    wet_audio = apply_effects(audio_data, sample_rate, comp_params, d_type, drive_params)
                    
                    # normalization 적용.
                    if ENABLE_NORMALIZATION:
                        try:
                            audio_for_meter = wet_audio.T if wet_audio.ndim == 2 else wet_audio
                            current_lufs = meter.integrated_loudness(audio_for_meter)
                            wet_audio = pyln.normalize.loudness(audio_for_meter, current_lufs, TARGET_LUFS)
                            if wet_audio.ndim == 2:
                                wet_audio = wet_audio.T
                        except ValueError:
                            pass

                    # peak 검사.
                    peak_db = 20 * np.log10(np.max(np.abs(wet_audio)) + 1e-12)
                    if peak_db >= MIN_PEAK_DB:
                        break

                wet_filename = f"wet_{base_name}_{d_type}_{variation_counter:03d}.wav"
                label_filename = f"params_{base_name}_{d_type}_{variation_counter:03d}.json"

                wet_path = f"{SYNTH_AUDIO_DIR}/{wet_filename}"
                label_path = os.path.join(SYNTH_LABEL_DIR, label_filename)

                # json 구성.
                label_data = {
                    "target_id": os.path.splitext(wet_filename)[0],
                    "is_synthetic": True,
                    "dry_sample_id": base_name,
                    "ground_truth_chain": {
                        "amp_id": AMP_ID,
                        "chain": [
                            {
                                "effect_type": d_type,
                                "values": drive_params
                            }
                        ]
                    },
                    "source_dataset": SOURCE_DATASET,
                    "file_path": wet_path
                }
                
                # 결과물 저장.
                sf.write(wet_path, wet_audio, sample_rate, subtype="FLOAT")
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, indent=4, ensure_ascii=False)
                    
                total_generated += 1
                variation_counter += 1
                
        print(f"처리 완료: {filename} -> {variation_counter - 1}개 생성")
        
    print(f"\ndataset 생성 완료. {len(dry_files)}개 파일 기준 총 {total_generated}개 생성.")

if __name__ == "__main__":
    generate_dataset()