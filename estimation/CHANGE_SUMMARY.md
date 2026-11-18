# 공동 작업 변경점 — VST7 전환 및 CMA-ES 연결 (2026-11-18)

## estimation 내부

- 기존 합성 생성기와 3종 분리 실험 대신, 팀 생성기의 manifest/labels/audio/dry를 읽는 경로로 통일.
- catalog는 소문자 VST 7종과 compressor/eq_3band를 사용한다.
  이전 체크아웃의 표시명 key는 모듈 안에서 대응시키며 공통 스키마를 수정하지 않는다.
- 단일 joint CNN으로 7종 확률, 효과별 노브, 컴프·EQ 포함 여부를 출력.
- spectral + dynamics 복합 d, train-only scale, 거리 보조 학습과 실제 렌더링 평가를 추가.
- native-rate generated Dry를 캐시하여 학습·추론·탐색의 입력 기준을 유지.
- top-K 전체 체인과 CMA-ES 좌표를 동일한 candidate 형식으로 출력.
- 모델 checkpoint 포맷을 2로 변경. 이전 3종 checkpoint는 현재 추론에서 받지 않는다.

## 다른 파일

- `tests/test_estimation.py`: 현재 VST7 데이터·후보·loss 계약에 맞게 교체.
- 루트 README·CNN 오디오 설정: 현재 7종 joint 실험으로 갱신.
- `requirements-cnn.txt`: 파일 읽기·거리 계산의 BLAS 스레드 제어에 쓰는 threadpoolctl 명시.
- `outputs`의 이전 archive/datasets/experiments는 사용자 요청으로 제거.
  새 대용량 산출물은 `outputs/vst7`에 저장하며 Git에서 제외한다.
- 공유 Drive 원본, `effects/render.py`, `schemas`, `metrics`, 팀 데이터 생성기는 변경하지 않는다.

## 2026-11-18 팀 CMA-ES 연결

- `estimation/refine.py`: 팀 `search.pipeline.search_and_rank_chains`를 호출하도록 통일.
  `to_search_inputs`에서 후보 wrapper를 실제 체인과 tuple-keyed `initial_guesses`로 변환한다.
  정규화 x0를 실제값처럼 넘기지 않고, 전체 체인의 좌표·노브값 일치 여부를 먼저 검사한다.
- `search/pipeline.py`: 기존 기본 동작을 유지하는 선택 인자를 추가했다.
  렌더러 주입, 초기값 선평가·예산 포함·보존, 노브 반올림 해제, 오류 전파, base seed를 지원한다.
  공용 VST key/렌더러를 변경하지 않고 CNN catalog와 renderer를 주입한다.
- `estimation/cma_experiment.py`: validation/test 균형 표본 선택, 계획만 생성하는 모드,
  같은 구조·예산·seed의 CNN 초기값/기본 초기값 CMA 비교, oracle 재현 오차 기록을 추가했다.
- `estimation/renderer.py`: 캐시된 과거 상태 대신 실행 시 사용할 수 있는 7종 활성 검사를 추가했다.
- `estimation/infer.py`: 같은 팀 엔진 연결과 활성 검사를 사용하며 재시작 수·seed 옵션을 추가했다.
- `tests/test_cnn_cma.py`: 실제 pycma를 사용한 후보 변환·초기값·예산·실패 처리 검사를 추가했다.
- 자세한 비교 방법과 명령은 [CNN_CMA_EXPERIMENT.md](CNN_CMA_EXPERIMENT.md)를 참고한다.

## 탐색 담당자 연결 지점

`estimation.infer`의 candidates 배열을 `estimation.refine.to_search_inputs`로 변환한다.
팀 API의 `initial_guesses`는 실제 노브값 체인이며 `x0`는 정규화된 검증 좌표다.
거리 설정은 checkpoint의 `distance_settings`를 그대로 사용한다.
