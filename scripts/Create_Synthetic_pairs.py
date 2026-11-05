# 이진원
# python 3.11.17
# 너무 작은 소리는 MIN_PEAK_DB 기준으로 다시 생성합니다.

import os
import json
import random
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Compressor, Distortion, Gain, PeakFilter

# =====================================================================
# 사용자 설정 변수 (Configuration Variables)
# =====================================================================
# 각 원본 파일에 대해 각 드라이브 타입(OD, DS, FZ)을 몇 번씩 적용할지 설정합니다.
# 예: 3으로 설정 시 -> (OD 3번 + DS 3번 + FZ 3번) = 파일당 총 9개의 변형 생성
# 78개 원본 * 9 = 702개의 합성 데이터(Synthetic Data) 쌍(Pair) 생성
VARIATIONS_PER_DRIVE = 3

# 난수 시드(Random Seed) 고정: 다시 실행해도 같은 데이터가 생성됩니다.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 결과 최대 음량(Peak)이 이 값(dBFS)보다 작으면 너무 작은 소리로 보고 매개변수를 다시 뽑습니다.
MIN_PEAK_DB = -20.0
MAX_RETRIES = 20

# 정답(Label) 메타데이터: schemas/target_sample.schema.json, schemas/amp.data.json 기준
SOURCE_DATASET = "IDMT-SMT-Guitar V2"
AMP_ID = "fixed_amp_v1"

DRY_AUDIO_DIR = "data/dry/Fender_Strat_Clean/audio"
SYNTH_AUDIO_DIR = "data/synthetic/Fender_Strat_Clean/audio"
SYNTH_LABEL_DIR = "data/synthetic/Fender_Strat_Clean/labels"

# =====================================================================
# 디렉터리(Directory) 초기화
# =====================================================================
os.makedirs(SYNTH_AUDIO_DIR, exist_ok=True)
os.makedirs(SYNTH_LABEL_DIR, exist_ok=True)

# =====================================================================
# 매개변수(Parameter) 무작위 추출 함수
# =====================================================================
def get_random_compressor():
    """스키마(Schema)에 정의된 연속형(Continuous) 범위를 바탕으로 압축기(Compressor) 값을 무작위(Random) 추출합니다."""
    return {
        "threshold_db": round(random.uniform(-60.0, 0.0), 2),
        "ratio": round(random.uniform(1.0, 20.0), 2),
        "attack_ms": round(random.uniform(0.1, 100.0), 2),
        "release_ms": round(random.uniform(10.0, 1000.0), 2)
    }

def get_random_drive(drive_type):
    """선택된 증폭 단(Gain Stage) 이펙터(Effector)의 매개변수를 추출합니다."""
    if drive_type == "overdrive":
        return {
            "drive": round(random.uniform(0.0, 15.0), 2),
            "level": round(random.uniform(-12.0, 12.0), 2)
        }
    elif drive_type == "distortion":
        return {
            "dist": round(random.uniform(0.0, 40.0), 2),
            "tone": round(random.uniform(-9.0, 9.0), 2),
            "level": round(random.uniform(-12.0, 12.0), 2)
        }
    elif drive_type == "fuzz":
        return {
            "fuzz": round(random.uniform(10.0, 50.0), 2),
            "tone": round(random.uniform(-9.0, 9.0), 2),
            "level": round(random.uniform(-12.0, 12.0), 2),
            # mode는 렌더링에 아직 구현되지 않아 소리에 영향이 없으므로 기본값으로 고정합니다.
            "mode": "Vintage"
        }

# =====================================================================
# 오디오 신호 처리(Digital Signal Processing) 함수
# =====================================================================
def apply_effects(audio_data, sample_rate, comp_params, drive_type, drive_params):
    """페달보드(Pedalboard) 라이브러리를 활용하여 직렬 신호 사슬(Signal Chain)을 구성하고 렌더링(Rendering)합니다."""
    board = Pedalboard()
    
    # 1. 압축기(Compressor) 추가 (comp_params가 None이면 생략: Step 0은 컴프 제외)
    if comp_params is not None:
        board.append(Compressor(
            threshold_db=comp_params["threshold_db"],
            ratio=comp_params["ratio"],
            attack_ms=comp_params["attack_ms"],
            release_ms=comp_params["release_ms"]
        ))
    
    # 2. 증폭 단(Gain Stage) 이펙터 추가 (복합 클래스 매핑)
    if drive_type == "overdrive":
        board.append(Distortion(drive_db=drive_params["drive"]))
        board.append(Gain(gain_db=drive_params["level"]))
        
    elif drive_type == "distortion":
        board.append(Distortion(drive_db=drive_params["dist"]))
        # Tone 매핑을 위한 피크 필터(Peak Filter) 적용
        board.append(PeakFilter(cutoff_frequency_hz=1000.0, gain_db=drive_params["tone"])) 
        board.append(Gain(gain_db=drive_params["level"]))
        
    elif drive_type == "fuzz":
        # 퍼즈(Fuzz)는 더 강한 왜곡(Distortion)을 위해 드라이브 값을 기본적으로 증폭시킵니다.
        board.append(Distortion(drive_db=drive_params["fuzz"]))
        board.append(PeakFilter(cutoff_frequency_hz=800.0, gain_db=drive_params["tone"]))
        board.append(Gain(gain_db=drive_params["level"]))
        
    # 오디오 배열(Audio Array) 연산 수행
    wet_audio = board(audio_data, sample_rate)
    return wet_audio

# =====================================================================
# 메인 파이프라인(Main Pipeline)
# =====================================================================
def generate_dataset():
    # 파일 목록 정렬: OS마다 순서가 달라 시드를 고정해도 결과가 바뀌는 것을 방지
    dry_files = sorted(f for f in os.listdir(DRY_AUDIO_DIR) if f.endswith(".wav"))
    drive_types = ["overdrive", "distortion", "fuzz"]
    total_generated = 0
    
    for filename in dry_files:
        dry_path = os.path.join(DRY_AUDIO_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        # 오디오 불러오기
        try:
            audio_data, sample_rate = sf.read(dry_path)
        except Exception as e:
            print(f"오디오(Audio) 로드 실패: {filename} - {e}")
            continue
            
        variation_counter = 1
        
        # 3가지 드라이브(Drive) 타입을 모두 순회합니다.
        for d_type in drive_types:
            # 사용자가 설정한 배수만큼 반복하여 무작위(Random) 합성 수행
            for _ in range(VARIATIONS_PER_DRIVE):
                comp_params = None  # Step 0: 컴프 제외

                # 렌더링(Rendering): 결과가 너무 작으면 매개변수를 다시 뽑습니다.
                for _ in range(MAX_RETRIES):
                    drive_params = get_random_drive(d_type)
                    wet_audio = apply_effects(audio_data, sample_rate, comp_params, d_type, drive_params)
                    peak_db = 20 * np.log10(np.max(np.abs(wet_audio)) + 1e-12)
                    if peak_db >= MIN_PEAK_DB:
                        break

                # 고유 파일명 식별자(Identifier) 생성
                wet_filename = f"wet_{base_name}_{d_type}_{variation_counter:03d}.wav"
                label_filename = f"params_{base_name}_{d_type}_{variation_counter:03d}.json"

                # 파일 경로(Path): 정답 json에는 OS와 무관하게 "/"로 기록
                wet_path = f"{SYNTH_AUDIO_DIR}/{wet_filename}"
                label_path = os.path.join(SYNTH_LABEL_DIR, label_filename)

                # 규격(Schema)에 맞춘 JSON 구조체(Structure) 조립: schemas/target_sample.schema.json
                # 라이선스(License)는 dry 원본 기준이며 data/dry/README.md에 기록되어 있습니다.
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
                
                # float 형식으로 저장: 16비트 저장 시 최대 음량을 넘는 부분이 잘리는 문제 방지
                sf.write(wet_path, wet_audio, sample_rate, subtype="FLOAT")
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, indent=4, ensure_ascii=False)
                    
                total_generated += 1
                variation_counter += 1
                
        print(f"처리 완료(Processed): {filename} -> {variation_counter - 1}개의 변형 생성됨")
        
    print(f"\n데이터 생성 완료! 총 {len(dry_files)}개의 원본에서 {total_generated}개의 데이터 쌍(Data Pair)이 생성되었습니다.")

if __name__ == "__main__":
    generate_dataset()