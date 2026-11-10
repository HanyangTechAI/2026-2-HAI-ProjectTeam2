# Search Pipeline (탐색 엔진 베이스라인)

Overdrive 이펙터(Compressor 바이패스)의 파라미터(`drive`, `level`) 역추정을 위한 Gradient-free 탐색 파이프라인입니다.

## 주요 기능
- **탐색 대상**: Overdrive 노브 2종 (`drive`: 0~15dB, `level`: -12~12dB), Compressor는 무력화(Bypass) 고정
- **탐색 알고리즘**: Random Search 및 다중 재시작(Restart) CMA-ES 지원 (기본 총예산 200회를 50회 × 4회로 분할 탐색)
- **결과 비교**: 정답(Target) 노브 값과 탐색된(Found) 노브 값의 절대 오차 및 정규화 오차(%) 비교 출력
- **검증 도구**: 1D 파라미터 민감도 분석(`outputs/search/sensitivity.png`) 및 수렴 곡선(`outputs/search/convergence_*.png`) 자동 저장

## 사전 패키지 설치
```bash
pip install numpy scipy librosa soundfile pedalboard cma matplotlib