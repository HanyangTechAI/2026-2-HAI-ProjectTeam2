import os
import random
import soundfile as sf
from pedalboard import Pedalboard, load_plugin

# ==========================================
# 1. 설정 (경로를 사용자 환경에 맞게 수정하세요)
# ==========================================
DRY_AUDIO_PATH = "data/dry/Fender_Strat_Clean/audio/sample_G53-42102-1111-00003.wav" # 테스트할 원본 음원 1개 지정
OUTPUT_DIR = "./testWet"                                      # 결과물이 저장될 폴더
VST_PATH = "./effects/VST_files/MultiDrivePro.vst3"                # VST 파일 경로

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 12개의 드라이브 모델 리스트
DRIVE_MODELS = [
    "808-Scream", "9-Scream", "B-Drive", "O-Drive", "S-Drive", 
    "One-Distortion", "BAT-Distortion", "XXL-Distortion", 
    "M-Distortion", "F-Fuzz", "M-Fuzz", "T-Boost"
]

def generate_12_wet_files():
    # 1. 원본 오디오 로드
    if not os.path.exists(DRY_AUDIO_PATH):
        print(f"에러: 원본 음원을 찾을 수 없습니다 -> {DRY_AUDIO_PATH}")
        return
        
    audio, sample_rate = sf.read(DRY_AUDIO_PATH)
    
    # 2. VST 플러그인 로드
    plugin = load_plugin(VST_PATH)
    
    # [중요] 헤드리스 환경 강제 바이패스 해제 (오디오 처리 활성화)
    if hasattr(plugin, 'bypass'):
        plugin.bypass = False

    print(f"'{os.path.basename(DRY_AUDIO_PATH)}' 소스에 12개의 이펙터를 적용합니다...\n")

    # 3. 12개 모델을 순회하며 음원 생성
    for model in DRIVE_MODELS:
        # A. 내부 이펙터 모드 스위칭
        plugin.pedal_model = model
        
        # B. 공통 파라미터 초기화
        plugin.input_gain_db = 0.0
        
        # C. 모델별 노브 조절 (볼륨 45~55%, 강도/톤 30~100% 랜덤 적용)
        if model == "808-Scream":
            plugin.level = random.uniform(45.0, 55.0)
            plugin.overdrive = random.uniform(30.0, 100.0)
            plugin.tone = random.uniform(30.0, 100.0)
        elif model == "9-Scream":
            plugin.level = random.uniform(45.0, 55.0)
            plugin.drive = random.uniform(30.0, 100.0)
            plugin.tone = random.uniform(30.0, 100.0)
        elif model == "B-Drive":
            plugin.level = random.uniform(45.0, 55.0)
            plugin.gain = random.uniform(30.0, 100.0)
            plugin.tone = random.uniform(30.0, 100.0)
        elif model == "O-Drive":
            plugin.level = random.uniform(45.0, 55.0)
            plugin.overdrive = random.uniform(30.0, 100.0)
        elif model == "S-Drive":
            plugin.level = random.uniform(45.0, 55.0)
            plugin.overdrive = random.uniform(30.0, 100.0)
            plugin.tone = random.uniform(30.0, 100.0)
        elif model == "One-Distortion":
            plugin.level = random.uniform(45.0, 55.0)
            plugin.distortion = random.uniform(30.0, 100.0)
            plugin.tone = random.uniform(30.0, 100.0)
        elif model == "BAT-Distortion":
            plugin.volume = random.uniform(45.0, 55.0)
            plugin.distortion = random.uniform(30.0, 100.0)
            plugin.filter = random.uniform(30.0, 100.0)
        elif model == "XXL-Distortion":
            plugin.output = random.uniform(45.0, 55.0)
            plugin.distortion = random.uniform(30.0, 100.0)
        elif model == "M-Distortion":
            plugin.level = random.uniform(45.0, 55.0)
            plugin.distortion = random.uniform(30.0, 100.0)
            plugin.high = random.uniform(30.0, 100.0)
            plugin.low = random.uniform(30.0, 100.0)
            plugin.mid = random.uniform(30.0, 100.0)
            plugin.mid_frequency = random.uniform(30.0, 100.0)
        elif model == "F-Fuzz":
            plugin.volume = random.uniform(45.0, 55.0)
            plugin.fuzz = random.uniform(30.0, 100.0)
        elif model == "M-Fuzz":
            plugin.volume = random.uniform(45.0, 55.0)
            plugin.sustain = random.uniform(30.0, 100.0)
            plugin.tone = random.uniform(30.0, 100.0)
        # T-Boost는 노브가 없으므로 공통 input_gain_db만 적용됨

        # D. 페달보드 체인 적용 및 오디오 렌더링
        board = Pedalboard([plugin])
        wet_audio = board(audio, sample_rate)

        # E. 결과물 저장
        output_filename = f"wet_{model}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        sf.write(output_path, wet_audio, sample_rate, subtype="FLOAT")
        
        print(f"저장 완료: {output_filename}")

if __name__ == "__main__":
    generate_12_wet_files()