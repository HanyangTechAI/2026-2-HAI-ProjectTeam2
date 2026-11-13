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
VARIATIONS_PER_DRIVE = 1 ####### 1개의 이펙터로 1개의 dry 음원 당 생성할 wet 음원 개수
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

DEFAULT_INPUT_GAIN_DB = 0.0

# =====================================================================
# directory variables
# =====================================================================
os.makedirs(SYNTH_AUDIO_DIR, exist_ok=True)
os.makedirs(SYNTH_LABEL_DIR, exist_ok=True)

# =====================================================================
# parameter 추출
# =====================================================================
def get_random_drive(drive_type):
    if drive_type == "overdrive": #Pedalboard 
        return {
            "hpf_freq": round(random.uniform(150.0, 250.0), 2),
            "drive": round(random.uniform(25.0, 40.0), 2),
            "lpf_freq": round(random.uniform(3000.0, 5000.0), 2),
            "level": round(random.uniform(-2.0, 2.0), 2)
        }
    elif drive_type == "distortion": #Pedalboard
        return {
            "hpf_freq": round(random.uniform(32.0, 48.0), 2),
            "dist": round(random.uniform(25.0, 40.0), 2),
            "lpf_freq": round(random.uniform(3000.0, 8000.0), 2),
            "level": round(random.uniform(-2.0, 2.0), 2)
        }
    elif drive_type == "fuzz": #pedalboard
        return {
            "fuzz": round(random.uniform(25.0, 40.0), 2),
            "notch_freq": round(random.uniform(850.0, 1150.0), 2),
            "notch_gain": round(random.uniform(-9.0, -5.0), 2),
            "lpf_freq": round(random.uniform(2000.0, 3000.0), 2),
            "level": round(random.uniform(-2.0, 2.0), 2)
        }
    elif drive_type in [
        "808-Scream", "9-Scream", "B-Drive", "O-Drive", "S-Drive", 
        "One-Distortion", "BAT-Distortion", "XXL-Distortion", 
        "M-Distortion", "F-Fuzz", "M-Fuzz", "T-Boost"
    ]:
        params = {"input_gain_db": DEFAULT_INPUT_GAIN_DB}
        
        if drive_type == "808-Scream":
            params.update({"level": round(random.uniform(45.0, 55.0), 2), "overdrive": round(random.uniform(30.0, 100.0), 2), "tone": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "9-Scream":
            params.update({"level": round(random.uniform(45.0, 55.0), 2), "drive": round(random.uniform(30.0, 100.0), 2), "tone": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "B-Drive":
            params.update({"level": round(random.uniform(45.0, 55.0), 2), "gain": round(random.uniform(30.0, 100.0), 2), "tone": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "O-Drive":
            params.update({"level": round(random.uniform(45.0, 55.0), 2), "overdrive": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "S-Drive":
            params.update({"level": round(random.uniform(45.0, 55.0), 2), "overdrive": round(random.uniform(30.0, 100.0), 2), "tone": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "One-Distortion":
            params.update({"level": round(random.uniform(45.0, 55.0), 2), "distortion": round(random.uniform(30.0, 100.0), 2), "tone": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "BAT-Distortion":
            params.update({"volume": round(random.uniform(45.0, 55.0), 2), "distortion": round(random.uniform(30.0, 100.0), 2), "filter": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "XXL-Distortion":
            params.update({"output": round(random.uniform(45.0, 55.0), 2), "distortion": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "M-Distortion":
            params.update({"level": round(random.uniform(45.0, 55.0), 2), "distortion": round(random.uniform(30.0, 100.0), 2), "high": round(random.uniform(30.0, 100.0), 2), "low": round(random.uniform(30.0, 100.0), 2), "mid": round(random.uniform(30.0, 100.0), 2), "mid_frequency": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "F-Fuzz":
            params.update({"volume": round(random.uniform(45.0, 55.0), 2), "fuzz": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "M-Fuzz":
            params.update({"volume": round(random.uniform(45.0, 55.0), 2), "sustain": round(random.uniform(30.0, 100.0), 2), "tone": round(random.uniform(30.0, 100.0), 2)})
        elif drive_type == "T-Boost":
            pass
            
        return params

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
# pipeline
# =====================================================================
def generate_dataset():
    dry_files = sorted(f for f in os.listdir(DRY_AUDIO_DIR) if f.endswith(".wav"))
    ##################################################################################
    ###################################### 적용하고 싶은 이펙터만 작성하기.
    drive_types = [
       # "overdrive", "distortion", "fuzz",  페달보드 기본 함수로 만든 애들 
        "808-Scream", "9-Scream", "B-Drive", "O-Drive", "S-Drive", 
        "One-Distortion", "BAT-Distortion", "XXL-Distortion", 
        "M-Distortion", "F-Fuzz", "M-Fuzz", "T-Boost"
    ]
    ######################################################################################
    total_generated = 0
    dryfile_order = 1 # 현재 작업중인 dryfile의 순서(작업 진행도 측정용)
    for filename in dry_files:
        dry_path = os.path.join(DRY_AUDIO_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            audio_data, sample_rate = sf.read(dry_path)
        except Exception as e:
            print(f"audio load failed: {filename} - {e}")
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
                    
                    # Applying normalization.
                    if ENABLE_NORMALIZATION:
                        try:
                            audio_for_meter = wet_audio.T if wet_audio.ndim == 2 else wet_audio
                            current_lufs = meter.integrated_loudness(audio_for_meter)
                            wet_audio = pyln.normalize.loudness(audio_for_meter, current_lufs, TARGET_LUFS)
                            if wet_audio.ndim == 2:
                                wet_audio = wet_audio.T
                        except ValueError:
                            pass

                    # peak test 및 바이패스(원음 통과) 여부 검증
                    peak_db = 20 * np.log10(np.max(np.abs(wet_audio)) + 1e-12)
                    max_diff = np.max(np.abs(audio_data - wet_audio))

                    if peak_db >= MIN_PEAK_DB and max_diff > 1e-4:
                        break
                    elif max_diff <= 1e-4:
                        print(f"[경고] 바이패스 감지됨 ({filename} - {d_type}). 플러그인 캐시 초기화 후 재시도합니다.")
                        from effects.render import _VST_CACHE
                        _VST_CACHE.clear()

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
                
                # Store results.
                sf.write(wet_path, wet_audio, sample_rate, subtype="FLOAT")
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, indent=4, ensure_ascii=False)
                    
                total_generated += 1
                variation_counter += 1
                
        print(f"Work Complete: {filename} -> {variation_counter - 1}개 렌더링, 작업: {dryfile_order}/{len(dry_files)}")
        dryfile_order += 1
        
    print(f"\ndataset Created. from {len(dry_files)} dry files ->  to {total_generated} wet files.")

if __name__ == "__main__":
    generate_dataset()