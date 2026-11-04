# Search Pipeline (탐색 엔진 베이스라인)

Compressor -> Overdrive 직렬 이펙터 체인의 파라미터 역추정을 위한 Gradient-free 탐색 파이프라인입니다.

## 주요 기능
- **탐색 알고리즘**: Random Search 및 IPOP-CMA-ES 지원 (시간 상수 파라미터 로그 스케일 정규화 적용)
- **결과 비교**: 정답(Target) 노브 값과 탐색된(Found) 노브 값의 절대 오차 및 정규화 오차(%) 비교 출력
- **검증 도구**: 1D 파라미터 민감도 분석(`outputs/search/sensitivity.png`) 및 예산별 수렴 곡선(`outputs/search/convergence_*.png`) 자동 저장

## 사전 패키지 설치
```bash
pip install numpy scipy librosa soundfile pedalboard cma matplotlib

실행 방법 (프로젝트 루트 경로에서 실행)
1. 기본 실행 (예산 50/200회, 시드 3개, 1D 민감도 분석 포함)
python -m search.pipeline

2. 탐색 예산(평가 횟수) 및 반복 시드 수 지정
python -m search.pipeline --budgets 50 200 1000 --seeds 10

3. 1D 민감도 분석 생략 (빠른 탐색만 실행)
python -m search.pipeline --no-sensitivity