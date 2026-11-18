# CNN → 팀 CMA-ES 연결 실험

2026-11-18. 기존 epoch 20 CNN을 고정하고, CNN 노브 초기값이 CMA-ES 탐색에 도움이 되는지 비교한다.
새 학습이나 공유 데이터 변경은 하지 않는다.

## 연결 구조

`CNN → make_candidates → to_search_inputs → search.pipeline.search_and_rank_chains`
`→ run_cma_es_for_chain → estimation.Renderer → 기존 복합 거리 d`.

- `estimation/refine.py`가 별도 CMA 구현 대신 팀 `search/pipeline.py`를 호출한다.
- `estimation/cma_experiment.py`는 계획 생성과 반복 실험을 실행한다.
- `estimation/infer.py --refine-budget ...`도 같은 연결을 사용한다.

| CNN 출력 | 팀 탐색 입력 |
|---|---|
| `candidate['chain']` | `candidate_chains`의 한 항목 |
| 체인에 들어 있는 실제 노브값 | `initial_guesses[tuple(effect_types)]` |
| `x0` | 팀 spec에서 정규화한 값과 같은지 검사 |
| `vector_spec` | 효과 인덱스·효과명·노브 순서를 팀 spec과 대조 |
| `model_score` | CNN 종류 확률로 보존; 탐색의 목적함수로 사용하지 않음 |

```python
from estimation.refine import to_search_inputs
candidate_chains, initial_guesses = to_search_inputs(prediction['candidates'], definitions)
```

`initial_guesses`는 실제 노브값을 담은 전체 체인이다. 정규화된 `x0`를 실제 노브값처럼
넘기면 이중 정규화되므로 그렇게 전달하지 않는다. JSON 산출물에서는 tuple key 대신
`search_initial_guesses: [{effect_types: [...], chain: {...}}]`로 저장한다.

소문자 VST 7종 key는 CNN checkpoint의 catalog를 탐색에 주입해 처리한다.
VST 렌더링은 `estimation.Renderer`를 주입하므로 기존 공용 렌더러의 표시명 key와 충돌하지 않는다.
샘플링되지 않은 `input_gain_db`는 탐색 좌표에서 제외하고 렌더링에서 0으로 고정한다.
체인 순서는 선택적 Compressor → 드라이브 1종 → 선택적 EQ다.

## 실험 비교

같은 Dry/목표음, 같은 CNN Top-K 구조, 같은 렌더링 예산·seed·재시작 수로 다음을 비교한다.

| 방법 | 노브 초기값 | 보고하는 값 |
|---|---|---|
| CNN 초기 후보 | CNN 예측값 | Top-1과 Top-K 중 최소 d |
| CNN + CMA-ES | CNN 예측값 | 탐색 후 Top-K 중 최소 d |
| 기본 노브 + CMA-ES | 스키마 기본값 | 탐색 전·후 Top-K 중 최소 d |
| 정답 체인 재렌더링 | 라벨의 정답값 | 현재 렌더러의 저장 목표음 재현 오차 |

컴프·EQ의 ON/OFF와 후보 드라이브 종류는 각 CNN 후보에서 고정한다. CMA-ES는 포함된 효과들의
노브만 수정한다. 빠진 드라이브 후보나 틀린 컴프·EQ 구조를 새로 생성하지 않는다.
정답 체인은 두 탐색이 끝난 뒤 진단에만 쓰며, 탐색 초기값으로 전달하지 않는다.

입력은 생성 `dry/`에서 만든 로컬 native-rate 캐시만 사용한다.
VST는 원래 44.1 kHz에서 렌더링하고, 거리 계산 직전에만 22.05 kHz로 변환한다.
`d = 0.5 × spectral / spectral_scale + 0.5 × dynamics / dynamics_scale`이며,
scale은 기존 checkpoint의 train-only 값을 그대로 사용한다.

초기 후보도 예산에 포함하여 먼저 실제 평가하고 최적값 후보로 보존한다.
따라서 실행한 렌더러·거리 기준에서 최종 best d가 평가된 초기 Top-K 최솟값보다 커지지 않는다.
이는 구조/노브 정답 복원이나 청감 개선의 보장은 아니다.

## 실행

프로젝트 루트에서 기존 Conda 환경을 사용한다.

```powershell
# 플러그인 없이 CNN 후보와 팀 탐색용 초기값을 생성하고 검증
.\.conda-cnn\python.exe -m estimation.cma_experiment --prepare-only

# validation 7개, 종류별 1개로 작은 비교 실험 실행
.\.conda-cnn\python.exe -u -m estimation.cma_experiment `
  --split validation --samples 7 --k 3 --budget 180 --restarts 2 `
  --output outputs/vst7/cma_pilot

# 설정을 고정한 뒤 test 70개로 확대: 종류별 10개
.\.conda-cnn\python.exe -u -m estimation.cma_experiment `
  --split test --samples 70 --k 3 --budget 180 --restarts 2 `
  --output outputs/vst7/cma_test70
```

기본 cache는 `outputs/vst7/dataset`, checkpoint는 `outputs/vst7/model/best.pt`다.
다른 경로는 `--cache`, `--checkpoint`로 지정한다. provenance·오디오 설정·거리 설정이
일치하지 않으면 실행을 중단한다. `--samples -1`은 해당 split 전체다.
`--sample-id <ID>`는 특정 1개를 고르는 연결 점검용 옵션이며 균형 평가로 해석하지 않는다.
다른 seed는 `--seed`로 지정한다.

`--budget`은 **표본 1개·방법 1개에서 K개 후보가 나눠 쓰는 전체 렌더링 상한**이다.
기본 180, K=3이면 후보당 60회를 배정한다. 두 방법을 비교하므로 표본당 최대 360회다.
validation 7개는 탐색 최대 2,520회 + 활성 검사 7회 + 정답 재렌더링 7회 = 최대 2,534회다.
기존 팀 API의 `budget`은 후보·seed별 예산이므로 adapter가 후보별 몫을 계산해 호출한다.
조기 종료 시 사용 횟수는 상한보다 작을 수 있다. 실제 사용 횟수도 별도로 저장한다.

실제 실행은 활성화된 MultiDrive PRO와 라이선스에 접근 가능한 사용자 환경이 필요하다.
캐시된 과거 상태 대신 매 실행마다 드라이브 7종의 활성 상태를 검사한다.
Dry pass-through 또는 무음이면 탐색을 중단한다. UI를 열거나 체험판을 자동 시작하지 않는다.
기본 설치 위치가 다르면 `HAI_VST_DIR`를 지정한다.

기존 결과가 있는 `--output`으로 재실행하면 덮어쓰지 않는다. 새 output 경로를 사용한다.
플러그인이나 거리 계산 실패는 penalty로 감추지 않고 오류로 중단하며,
실행 중 완료된 표본은 `results/`에 남긴다. 현재 자동 재개 기능은 없다.

개별 파일 추론도 같은 엔진으로 연결된다.

```powershell
.\.conda-cnn\python.exe -m estimation.infer `
  --checkpoint outputs/vst7/model/best.pt `
  --dry '<생성 폴더>/dry/<파일>.wav' --reference '<목표>.wav' `
  --k 3 --refine-budget 180 --refine-restarts 2 --seed 42 `
  --output outputs/vst7/refined_prediction.json
```

## 산출물

| 파일 | 내용 |
|---|---|
| `plan.json` | 표본·방법·예산·모델/데이터 hash·거리 설정 |
| `tasks.jsonl` | CNN 예측, 후보, 팀 형식으로 변환한 체인·초기값 |
| `renderer_preflight.json` | 현재 VST 활성 검사 결과 |
| `results/0000.json` 등 | 방법별 초기값·탐색 후 후보·노브·d·수렴 이력·실제 렌더 수 |
| `summary.json` | 표본 평균 d, 초기 후보 대비 감소율, 기본값 탐색과의 승수 |

`results`의 탐색 후보는 원래 CNN 후보와 같은 `chain/x0/vector_spec` 형식이다.
`model_score`는 원래 CNN 확률이며, 탐색 이후 품질은 `distance`로 판단한다.
전체 결과 파일은 `outputs/` 아래라 Git에서 제외된다.

## 검증과 해석

- 후보 전체 체인의 좌표 순서, 로그 정규화, 초기값 불일치 거부를 확인한다.
- 실제 pycma를 사용해 초기값 보존, 예산 상한, 반복 seed, native-rate 전달을 검사한다.
- Dry pass-through와 렌더러 예외/비유한 거리의 실패 처리를 검사한다.
- 기존 CNN 학습·데이터 검사를 함께 실행한다.

```powershell
.\.conda-cnn\python.exe -m pytest tests/test_cnn_cma.py tests/test_estimation.py -q
```

작은 실험은 연결과 비용 점검용이다. 설정은 validation에서 고정하고 test를 평가한다.
컴프의 무효 압축 라벨 문제는 [COMPRESSOR_CHECK.md](COMPRESSOR_CHECK.md)에 별도로 기록했다.
현재 VST가 저장된 목표음을 완전히 재현하지 못하는 문제가 있으므로 oracle d를 함께 본다.
oracle은 재현 비교값이며 수학적인 최솟값 하한이 아니다.

## 2026-11-18 실제 VST 연결 확인

다음 1개 표본은 전체 성능 실험을 시작하기 전 연결 확인용으로 실행했다.
`wet_strat_sample_G53-40100-1111-00001_808_scream_001`이며,
정답은 808_scream이고 validation의 첫 준비 표본을 사용했다.

```powershell
.\.conda-cnn\python.exe -u -m estimation.cma_experiment `
  --sample-id wet_strat_sample_G53-40100-1111-00001_808_scream_001 `
  --output outputs/vst7/cma_smoke --budget 60 --restarts 1
```

| 비교 | 복합 d |
|---|---:|
| CNN Top-1 초기 후보 | 0.531998 |
| CNN Top-3 중 초기 최솟값 | 0.489065 |
| CNN 초기값 + CMA-ES | 0.346037 |
| 기본 노브 Top-3 초기 최솟값 | 1.174351 |
| 기본 노브 + CMA-ES | 0.373791 |
| 정답 전체 체인 재렌더링 | 0.241360 |

CNN Top-3 초기 최솟값 대비 **29.25% 감소**했다. 각 방법에서 60회씩 탐색했고,
후보당 초기 평가 포함 20회를 배정했다. 탐색 렌더 120회와 활성/정답 진단 8회를 실행했다.
탐색·정답 보고 구간은 약 168.8초였으며 시작 시 모델 로드와 활성 검사 시간은 별도다.

이번 CNN+CMA 승자 체인은 bat_distortion으로, 정답 종류 808_scream과 다르다.
따라서 이 결과는 오디오 d를 줄이는 연결이 작동함을 보여주며,
정답 이펙터 복원 또는 전체 데이터 성능 개선을 입증하지 않는다.
현재 저장 목표음의 재현 오차와 고정된 CNN 구조의 영향도 남아 있다.

- [단일 표본 요약](../outputs/vst7/cma_smoke/summary.json)
- [노브·후보별 수렴 기록](../outputs/vst7/cma_smoke/results/0000.json)
- [실제 플러그인 활성 검사](../outputs/vst7/cma_smoke/renderer_preflight.json)
- [준비된 validation 7개 계획](../outputs/vst7/cma_pilot/plan.json)
- [준비된 validation 7개 CNN·팀 탐색 입력](../outputs/vst7/cma_pilot/tasks.jsonl)

`cma_pilot`은 계획·입력만 준비된 상태이며 7개 전체 CMA-ES 비교는 아직 실행하지 않았다.
기존 30epoch 학습 및 420개 validation/test 평가 결과는 변경하지 않았다.

참고: [pycma 공식 API — 초기값, 정규화와 ask/tell](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html).
