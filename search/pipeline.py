import numpy as np
from scipy.optimize import minimize

# =======================================================
# [1] 팀원들이 짠 실제 평가 함수 불러오기 (Import)
# =======================================================
# metrics 폴더의 spectral.py 파일에서 spectral_distance 함수를 가져옵니다.
from metrics.spectral import spectral_distance 

def dummy_render_effect(dry_audio, params):
    """가짜 렌더링: 파라미터 값에 따라 오디오 배열을 단순 계산"""
    wet_audio = dry_audio * params[0] + params[1]
    return wet_audio

# =======================================================
# [2] 최적화 탐색 파이프라인
# =======================================================
def run_search_pipeline():
    # 1. 가짜 오디오 데이터 준비
    np.random.seed(42)
    # ⚠️ 중요: 팀원의 librosa.stft 함수가 작동하려면 오디오 길이가 충분히 길어야 합니다.
    # 기존 100에서 4096으로 배열 길이를 늘려줍니다.
    dry_audio = np.random.rand(4096) 
    target_audio = dry_audio * 0.7 + 0.3  # 찾아야 할 정답: 드라이브 0.7, 톤 0.3
    
    # 2. 목적 함수 정의
    def objective_function(params):
        # 렌더링 모듈은 아직 없으니 가짜 렌더링 함수 유지
        wet = dummy_render_effect(dry_audio, params)
        
        # ⭐ 가짜 평가 함수를 지우고, 팀원이 만든 진짜 함수(spectral_distance)로 교체!
        loss = spectral_distance(wet, target_audio)
        
        return loss

    # 3. 탐색 시작
    initial_params = [0.5, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    print("탐색 엔진 가동 시작...")
    
    result = minimize(
        objective_function, 
        initial_params, 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    
    print("-" * 30)
    print("탐색 완료!")
    print(f"알고리즘이 찾아낸 노브 정답: {result.x}")
    print(f"최종 오차 점수: {result.fun:.6f}")

if __name__ == "__main__":
    run_search_pipeline()