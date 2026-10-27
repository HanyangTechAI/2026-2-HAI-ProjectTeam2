import numpy as np
from scipy.optimize import minimize
from metrics.spectral import spectral_distance

# 팀원이 작성한 scripts 폴더의 파일에서 렌더링 함수를 불러옵니다.
from scripts.Create_Synthetic_pairs import apply_effects

def run_search_pipeline():
    np.random.seed(42)
    sample_rate = 22050
    dry_audio = np.random.rand(4096) 
    
    # 임의의 정답(Target) 파라미터 세팅
    target_comp = {"threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}
    target_drive = {"drive": 10.0, "level": 0.0}
    
    # 타겟 오디오 생성 (팀원의 진짜 렌더링 함수 사용)
    target_audio = apply_effects(dry_audio, sample_rate, target_comp, "overdrive", target_drive)
    
    def objective_function(params):
        # 탐색 알고리즘이 던져주는 1차원 숫자 배열(params)을 팀원 함수 규격(딕셔너리)에 맞게 변환
        comp_params = {
            "threshold_db": params[0],
            "ratio": params[1],
            "attack_ms": params[2],
            "release_ms": params[3]
        }
        drive_params = {
            "drive": params[4],
            "level": params[5]
        }
        
        # 가짜 렌더링을 지우고 팀원 함수로 교체!
        wet = apply_effects(dry_audio, sample_rate, comp_params, "overdrive", drive_params)
        loss = spectral_distance(wet, target_audio)
        
        return loss

    # 탐색 시작점 및 각 노브별 허용 범위 (팀원 코드의 제한 범위 반영)
    initial_params = [-30.0, 10.0, 50.0, 500.0, 5.0, 0.0]
    bounds = [
        (-60.0, 0.0), (1.0, 20.0), (0.1, 100.0), (10.0, 1000.0), # Compressor
        (0.0, 15.0), (-12.0, 12.0)                               # Overdrive
    ]
    
    print("탐색 엔진 가동 시작...")
    
    result = minimize(
        objective_function, 
        initial_params, 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    
    print("-" * 30)
    print("탐색 완료!")
    print(f"최종 오차 점수: {result.fun:.6f}")

if __name__ == "__main__":
    run_search_pipeline()