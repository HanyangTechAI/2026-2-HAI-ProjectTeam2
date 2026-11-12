# 레퍼런스 기반 기타 이펙터 체인 추정

Reference-based Guitar Effector Chain Estimation

쌩 기타 톤과 목표 기타 톤을 입력받아, 보유한 이펙터 중 어떤 조합을 어떤 파라미터로 세팅해야 목표 톤을 가장 가깝게 재현할 수 있는지 추정합니다.

## 문제 정의

dry 신호 `x_dry`, 목표 신호 `x_ref`, 사용자 보유 이펙터 집합 `E`가 주어졌을 때,
유사도 `d(F_{C,P}(x_dry), x_ref)`를 최소화하는 이펙터 조합 `C`와 파라미터 `P`를 찾습니다.

## 시스템 구조

```
[dry 신호] + [후보 세팅] → 렌더링 → 유사도 계산(목표 신호와 비교) → 탐색·최적화 → 최적 세팅
```

## 폴더 구조

```
.
├── schemas/      # 공통 이펙터 스키마, 정의 데이터 및 검증 스크립트
├── effects/      # 이펙터 라이브러리, 체인 렌더링 엔진
├── metrics/      # 유사도 지표 및 평가 코드
├── search/       # 기존 Step 0 random search / CMA-ES 실험
├── estimation/   # CNN 데이터 준비·학습·추론·평가·CMA-ES 연결
├── configs/      # CNN 오디오 설정과 검증 환경의 패키지 버전
├── tests/        # CNN 파이프라인 테스트
├── data/         # 대용량 파일은 Drive, 작은 테스트 샘플만 GitHub에 보관
│   ├── dry/          # dry 기타 샘플
│   ├── synthetic/    # 합성한 (dry, params, wet) 페어
│   └── separated/    # 음원 분리 결과물
├── notebooks/    # 실험용 노트북
├── demo/         # 데모 UI
└── scripts/      # 실행 스크립트
```

## 실행 방법

처음이라면 [DRIVE_SETUP.md](DRIVE_SETUP.md)를 따라 환경을 세팅하세요.

```bash
pip install -r requirements.txt
python schemas/validate.py
```

검증 스크립트는 이펙터 정의가 JSON Schema를 만족하는지 확인합니다.
현재 스키마 기반 드라이브 렌더러, Step 0 탐색 실험과 CNN 초기 추정 파이프라인을 제공합니다.

초기 체인 순서는 컴프 → 드라이브류(overdrive/distortion/fuzz 중 하나) → 고정 앰프 → EQ입니다.
현재 CNN 버전은 드라이브 최대 한 개와 bypass만 출력하며 앰프, 컴프레서, EQ는 적용하지 않습니다.
종류 분류와 노브 예측을 분리한 CNN 실험은
[estimation/EXPERIMENTS.md](estimation/EXPERIMENTS.md)를 참고하세요.

## 개발 단계

| 단계 | 내용 |
|---|---|
| Step 0 | 소프트웨어 이펙트 3개, 순서 고정, 파라미터 복원 |
| Step 1 | 이펙터 10개로 확장, 조합 선택 문제 추가 |
| Step 2 | 실제 녹음 dry/wet 페어로 전환 |
| Step 3 | 실제 음원 분리 → 목표 톤 입력 |
| Step 4 | 실물 페달 캡처, 앰프/공간계 확장 |

## 문서

CNN 초기 추정 모델의 설치, 데이터 준비, 학습, 추론 및 평가 명령은
[estimation/README.md](estimation/README.md)를 참고하세요.

프로젝트 일정, 역할 분담, 개발 현황 및 논의 사항은 Notion에서 관리합니다.

## 참고 문헌

> 리서치 진행에 따라 추가 예정
