# 이 코드는 12 in 1 vst인 MultiDrive Pro.vst3의 이펙터 별 파라미터 정보를 확인하기 위해서 만든 코드입니다.
import os
from pedalboard import load_plugin

# 1. 대상 VST 지정
vst_filename = "MultiDrivePro.vst3"
vst_path = os.path.join("./effects/VST_files", vst_filename)

try:
    # 2. VST 플러그인 로드
    plugin = load_plugin(vst_path)
    
    # 3. 모든 모드 이름 리스트 가져오기
    available_models = plugin.parameters['pedal_model'].valid_values
    
    # 4. 모드를 순회하며 터미널에 전부 출력
    for model_name in available_models:
        # VST 내부 스위치 변경
        plugin.pedal_model = model_name
        
        print(f"========== [ MODEL: {model_name} ] ==========")
        
        for name, param in plugin.parameters.items():
            if name == "bypass":
                continue
            print(f"- Parameter: {name}")
            print(f"  Detail: {param}")
            
        print("\n") # 모델 구분을 위한 줄바꿈
            
    print("출력 완료. 터미널 내용 확인.")

except Exception as e:
    print(f"에러 발생: {e}")