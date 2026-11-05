import os
from pedalboard import load_plugin

# 현재 실행 중인 파이썬 스크립트 파일(scripts 폴더 안)의 절대 경로를 가져옴
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 스크립트 위치를 기준으로 한 칸 위(..)로 간 뒤 VST 파일 경로를 조립
VST_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "synthetic", "VST_Plugin", "OverDiver.vst3"))

def run_test():
    print(f"플러그인 로드 시도: {VST_PATH}")
    # 격리 해제가 완료되었다면 프로젝트 폴더 내에서도 정상 로드됨
    vst = load_plugin(VST_PATH)
    print("💡 사용 가능한 파라미터 키 이름:", vst.parameters.keys())

if __name__ == "__main__":
    run_test()