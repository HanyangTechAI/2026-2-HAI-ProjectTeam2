# VST7 검증

- `tests/test_estimation.py`: canonical key, full-chain CMA 좌표, 보유 목록 필터,
  컴프·EQ threshold, 정답 체인 검증, 원본별 분할, train-only 거리 scale,
  비활성 노브 loss 마스킹, 거리 보조 head의 gradient를 검사한다.
- 렌더링 전 7종 정답 체인을 native-rate generated Dry로 재현한다.
  상대 RMS 오차 0.001 미만인지 `dataset_reproduced`로 기록한다.
  재현 오차가 있더라도 활성화된 VST의 실제 출력 d는 oracle d와 함께 비교한다.
  플러그인이 Dry 그대로 또는 무음을 출력하면 렌더링 평가는 실행하지 않는다.
- 공유 데이터에서 생성 Dry의 peak −12 dBFS, 파일 누락, 라벨 범위,
  원본·내용 해시 단위 train/val/test 누출을 확인한다.
- 실제 모델 출력 d와 Dry/target 거리 보조 추정 MAE는 별도로 기록한다.

실행 결과는 `outputs/vst7/dataset/renderer_preflight.json`과 모델·평가 summary에 남는다.

## 2026-11-17 실행 결과

```powershell
.\.conda-cnn\python.exe -m pytest tests/test_estimation.py metrics/test_combined.py metrics/test_dynamics.py -k 'not CompressorIntegration' -q
```

**61 passed, 6 deselected, 334 subtests passed**.
제외한 6개는 이전 `Create_Synthetic_pairs.py`를 import하는 기존 컴프 통합 검사이며,
현재 Conda 환경에 그 생성기의 `pyloudnorm` 의존성이 없어 제외했다. 전체 저장소 테스트 통과를 주장하지 않는다.
현재 CNN 테스트 8개에는 생성 폴더의 `render_input` 사용, 다른 PC 절대경로 무시,
공유 원본이 없어도 캐시로 학습 가능함, 70개 렌더링 표본의 종류별 균형 검사가 포함된다.

- 실제 VST7 데이터 2,730개를 읽었고 생성 Dry 78개의 peak·라벨·누락·분할을 검사했다.
- 30epoch 학습 완료. best epoch 20을 로드해 validation/test 각각 전체 420개를 평가했다.
- 각 split의 실제 렌더링 70개는 종류별 10개이며 Dry 원본 12개를 포함한다.
- 두 평가 파일의 총 2,520개 후보에서 체인·노브 범위·CMA 좌표 길이를 검증했다.
- 저장된 샘플로부터 accuracy와 평균 d를 다시 계산해 summary와 일치함을 확인했다.
- 학습된 checkpoint로 상위 3개 후보 JSON을 별도 출력하고 형식을 확인했다.

[산출물 검사 기록](../outputs/vst7/artifact_audit.json)에 모델·원본 manifest SHA-256,
실제 라이브러리 버전과 재계산 결과를 저장했다.
학습 이력에서 렌더링을 실행하지 않은 epoch의 상태 설명만 바로잡았으며 수치들은 변경하지 않았다.
최초 stdout은 `outputs/vst7/run.log`에 보존했다.

정답 체인의 재렌더링도 저장된 Wet과 완전히 같지는 않았다. 따라서 실제 d는
현재 설치 VST 아래의 비교 결과로 해석하고, 이 재현 오차를 CNN 오차와 구분한다.

## 컴프레서 추가 점검

컴프 ON 라벨 1,345개를 원래 표본율의 생성 Dry로 처리하고 대조 실험 312회를 수행했다.
컴프 구현은 작동하지만 ON 표본의 365개는 출력 변화가 없었고,
미세한 변화까지 포함하면 393개가 감사 기준상 bypass에 가까웠다.
측정 방법·기존 모델 결과와의 관계·후속 생성 권장 사항은
[COMPRESSOR_CHECK.md](COMPRESSOR_CHECK.md)에 기록한다.

## 2026-11-18 CNN → 팀 CMA-ES 연결 검증

```powershell
.\.conda-cnn\python.exe -m pytest tests/test_cnn_cma.py tests/test_estimation.py metrics/test_combined.py metrics/test_dynamics.py -k 'not CompressorIntegration' -q
```

**68 passed, 6 deselected, 334 subtests passed**.
제외한 6개는 위에 설명한 이전 생성기용 컴프 통합 검사다.
선택적 `cma.plot()`에 필요한 matplotlib이 없어 경고 1개가 발생했으나 탐색에는 필요하지 않다.

- 실제 pycma를 사용해 CNN 전체 체인 좌표와 팀 spec, 정규화/로그 범위, 잘못된 초기값 거부를 검사했다.
- 초기 후보 선평가·보존, 후보 전체 렌더 예산, 고정 seed 반복, 원래 표본율 전달을 검사했다.
- 오류/비유한 거리 전파, VST Dry pass-through 감지, 기존 공용 Overdrive 탐색 경로를 검사했다.
- 기존 best CNN으로 validation 7개·후보 21개를 준비하고 팀 입력 변환을 검증했다.
- 실제 활성 MultiDrive PRO로 validation 1개의 두 방법 CMA-ES를 실행했다.
  탐색 120회, 활성/정답 진단 8회이며 초기 Top-3 d 0.489065 → 탐색 후 0.346037이었다.
  같은 예산의 기본 노브 시작 탐색 결과는 0.373791이었다.
- 7개 전체 비교는 계획만 준비된 상태다. 단일 결과를 전체 성능 개선으로 해석하지 않는다.

실험 정의와 재현 명령은 [CNN_CMA_EXPERIMENT.md](CNN_CMA_EXPERIMENT.md),
실제 후보·예산·모델 hash 대조 결과는
[cma_smoke/artifact_audit.json](../outputs/vst7/cma_smoke/artifact_audit.json)에 기록했다.
