# vst 로드 되는지 테스트용 코드
import os
from pedalboard import load_plugin

# 1. VST3 파일명 입력
vst_filename = "MultiDrive Pro.vst3" 
vst_path = os.path.join("./effects/VST_files", vst_filename)

try:
    # 2. VST 플러그인 로드
    plugin = load_plugin(vst_path)
    
    # 3. 파라미터 정보 전체 출력
    print("=== [ VST3 파라미터 스키마 정보 ] ===")
    for name, param in plugin.parameters.items():
        print(f"- 식별자(Key): {name}")
        print(f"  세부 정보: {param}")
    print("=====================================")
    
except Exception as e:
    print(f"VST 로드 중 에러 발생: {e}")