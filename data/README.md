# 데이터 관리

큰 용량 파일은 Drive에 보관하고 GitHub엔 작은 샘플만 올립니다. 원본 데이터셋, 합성 오디오, 음원 분리 결과 및 모델 체크포인트는 Drive에서 관리합니다.

- `dry/`: GuitarSet 등에서 준비한 dry 기타 샘플
- `synthetic/`: 합성한 (dry, params, wet) 페어
- `separated/`: 음원 분리 결과물

오디오와 체크포인트는 기본적으로 Git에서 제외합니다. `dry/sample_*.wav`만 테스트용 예외이며, 커밋 전 길이가 1~2초인 소량의 작은 샘플인지 확인하세요. `.gitignore`는 파일 길이나 용량을 검사하지 않습니다.
