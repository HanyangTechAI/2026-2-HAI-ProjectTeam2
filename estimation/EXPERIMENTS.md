# VST 드라이브 7종 전용 실험

2026-11-18. 컴프·EQ 없는 새 데이터로 기존 CNN encoder/fusion 구조를 재학습했다.
학습·평가·CMA-ES의 모든 실험 결과는 이 문서에 모은다. 다른 문서는 사용법과 검사 절차를 설명한다.
표본별 JSON·checkpoint·로그는 수치를 재검증하기 위한 원자료로 별도 유지한다.
과거 mixed 실험과는 데이터 생성 렌더러·학습 대상·표본율·d scale이 달라 직접 수치 비교하지 않는다.

## 설정

- 데이터: 생성 Dry 78개, 드라이브별 변형 5개, 총 2,730쌍.
- 효과: MultiDrive PRO 드라이브 한 개. 컴프·EQ·앰프 없음.
- 생성: main의 통합 Generator와 공용 renderer. seed=42, Dry peak −12 dBFS.
- 표본율: 생성/CNN/렌더링/거리 44,100 Hz.
- log-mel: 64 bands, n_fft=2048, hop=512, 2.5초.
- 원본 분할: train 54개/1,890쌍, validation 12개/420쌍, test 12개/420쌍.
- CNN: 공유 Conv(8,16,32,64) 4블록, 주파수 4구역 pooling, 256차원 fusion 유지.
- 출력: 7종 분류, 드라이브별 노브 초기값, Dry/target 거리 성분 보조값.
- 학습: Adam lr=0.001, cosine schedule, 30epochs, batch=16, CPU threads=4, seed=42.
- best: 전체 validation loss 최소 checkpoint. test로 모델을 선택하지 않는다.
- 분류/노브: 각 split 전체. 실제 렌더링 d: 종류별 10개, 70개/split.
- 학습 중 d: 고정 validation 14개, epoch 1/5/10/15/20/25/30.
- d: spectral+dynamics 동일 가중치, scale은 train에서만 계산한다.

## 결과

생성 직후 전수 검사에서 5개가 Dry 그대로 저장된 것을 발견했다.
해당 초기 학습은 중단하고 최종 결과에서 제외했다. 같은 정답 노브로 5개를 재렌더링한 뒤
전체 2,730개에서 무음/미처리 출력이 없음을 확인하고 새 캐시·새 초기 가중치로 다시 학습했다.
검사 및 수정 WAV hash는 원본 render_audit.json과 outputs/vst7_drive_only/source_quality.json에 남겼다.

검증된 데이터로 30epoch 재학습을 완료했고, 전체 validation loss가 가장 낮은
**epoch 29**를 best로 선택했다. 아래는 선택 시점의 학습 기록이다.

| 지표 | Train | Validation |
|---|---:|---:|
| 전체 loss | 0.010649 | 0.048634 |
| 드라이브 7종 accuracy | 100.00% | 98.81% |
| 정규화 노브 MAE | 0.091409 | 0.110851 |

노브 MAE는 정답 드라이브 head의 값이며 종류 오분류를 포함한 전체 음향 정확도가 아니다.
선택한 best 모델로 validation/test 각각 420개를 평가했다.

| 지표 | Validation | Test |
|---|---:|---:|
| 7종 Top-1 accuracy | 98.81% (415/420) | 98.33% (413/420) |
| Macro F1 | 0.988131 | 0.983412 |
| Top-3 정답 포함률 | 100% (420/420) | 100% (420/420) |

각 드라이브 60개씩의 균형 평가다. 원본 Dry는 split별 12개이며 모든 변형을 같은 split에 넣었다.

### 정답 드라이브 조건부 노브 오차

노브 범위의 0~1 좌표에서 MAE를 계산했다. 아래는 정답 드라이브 head의 오차이며,
종류 예측 실패까지 포함한 전체 음향 성공률로 해석하지 않는다.

| 드라이브 | Validation CNN | Test CNN | Test 기본 노브 |
|---|---:|---:|---:|
| 808_scream | 0.113672 | 0.126673 | 0.282835 |
| b_drive | 0.099002 | 0.093499 | 0.258051 |
| o_drive | 0.072830 | 0.096441 | 0.308205 |
| one_distortion | 0.101232 | 0.113374 | 0.245371 |
| bat_distortion | 0.110793 | 0.115065 | 0.266731 |
| f_fuzz | 0.140499 | 0.128181 | 0.258421 |
| m_fuzz | 0.135138 | 0.138823 | 0.265410 |

### 실제 VST 렌더링 거리

종류별 10개, split당 70개를 예측 결과를 보기 전에 선택했다. 각 split의 Dry 원본 12개를 포함한다.
CNN 후보 3개·같은 Top-1 드라이브의 기본 노브·정답 체인을 렌더링했으므로
split당 350회, 두 split 총 700회다. 최종 평가 중 무음/Dry 통과 재시도는 0회였다.

| 방법 | Validation 평균 d | Test 평균 d |
|---|---:|---:|
| CNN Top-1 노브 | 0.179136 | 0.128905 |
| CNN Top-3 중 d 최솟값 | 0.178009 | 0.128905 |
| 같은 예측 드라이브, 기본 노브 | 0.522359 | 0.350196 |
| 정답 체인 재렌더링 | 0.000365 | 0.000498 |

같은 드라이브의 기본 노브 대비 Top-1 평균 d는 validation **65.71%**, test **63.19%** 감소했다.
이 결과는 CNN 초기값의 비교이며 CMA-ES로 노브를 최적화한 결과는 아니다.

사전 검사 7개에서는 정답 체인 상대 RMS 오차 최댓값 4.63e-9로 재현 검사를 통과했다.
최종 70개씩의 oracle d에는 위와 같이 작은 재현 오차가 남아 있다.
oracle은 현재 렌더러의 재현 참고값이며 수학적인 최소 d 하한으로 간주하지 않는다.

복합 거리의 train-only scale은 spectral=0.14739062178676826,
dynamics=6.287869430713868, 각 가중치는 0.5다.
진행 기록은 outputs/vst7_drive_only/run.log와 model/history.json에 남긴다.

## 경로

- 원본: data/synthetic/vst7_drive_only/{dry,audio,labels,manifest.json}.
- 생성 이력: source의 generation_provenance.json.
- 캐시/분할/거리 scale: outputs/vst7_drive_only/dataset/.
- best/last 모델·학습 기록: outputs/vst7_drive_only/model/.
- 전체 평가: outputs/vst7_drive_only/{validation,test}/.
- Top-K 후보: 각 split의 samples.jsonl.
- 정답 재현 검사: dataset/renderer_preflight.json.
- 학습 이력 CSV: outputs/vst7_drive_only/model/training_metrics.csv.
- 실제 수량·후보·지표 재계산: outputs/vst7_drive_only/artifact_audit.json.
- 실제 best 모델의 단일 파일 추론 예시: outputs/vst7_drive_only/candidates_example.json.
  고정된 첫 validation 표본의 Top-3는 808_scream, one_distortion, bat_distortion이며
  각 후보는 VST 한 개와 노브 좌표 3개다. 컴프·EQ 사용 여부 출력은 없다.
- CNN → CMA-ES validation 실험: outputs/vst7_drive_only/cma_pilot/.
- CNN → CMA-ES test 실험: outputs/vst7_drive_only/cma_test/.

단일 기타·단음·seed 한 번의 결과다. 다른 악기나 실물 페달에 대한 일반화는 별도 검증이 필요하다.

## CNN → CMA-ES 실험

### 고정 설정과 비교 범위

- 현재 epoch 29 best 모델과 기존 검증된 캐시를 사용한다. CNN을 재학습하지 않는다.
- validation/test 각각 7개, 드라이브별 1개를 예측·탐색 결과를 보기 전에 균형 선택한다.
- CNN Top-3 드라이브 후보를 두 방법에 동일하게 제공한다. 종류 후보를 찾는 효과와
  CNN 노브 초기값의 효과를 구분하기 위한 비교다. 전체 7종을 무작위로 탐색하는 비교는 아니다.
- CNN 노브 시작 CMA-ES와 스키마 기본 노브 시작 CMA-ES를 비교한다.
- 표본·방법당 총 180회, 후보당 60회 렌더링을 배분한다. 최초 노브 평가도 예산에 포함한다.
- `--restarts 2`는 main의 `n_runs=2`, 즉 초기 실행 1회와 재시작 1회를 뜻한다.
  seed=42(후보별 42/43/44), sigma0=0.2, popsize=6을 고정한다.
  두 번째 실행은 두 방법에서 동일한 난수 규칙으로 새 초기점을 선택한다.
- `search.pipeline.run_cma_es_for_chain(..., x0_norm=...)`으로 연동한다.
  각 후보의 드라이브 종류는 고정하고 정규화 노브 3개만 탐색한다. 컴프·EQ는 사용하지 않는다.
- 생성 폴더의 −12 dBFS Dry를 그대로 사용하고 CNN·VST·거리 모두 44.1 kHz를 유지한다.
- 목적함수는 위의 train-only scale과 동일 가중치 d다. 정답 체인은 탐색 종료 후 재현 검사에만 사용한다.
- 초기 후보를 최종 선택지에 포함하여 탐색 후 d가 초기 최솟값보다 커지지 않도록 한다.
- 계획한 탐색 상한은 split당 2,520회이며, 활성 검사 7회와 정답 렌더 7회가 별도다.

이 절의 수치는 split당 7개에 대한 짝지은 비교다. 앞선 CNN 렌더 평가의 70개 평균과
표본 구성이 다르므로 두 절의 평균 차이를 CMA 개선량으로 사용하지 않는다.
모든 설정은 test 결과를 보기 전에 고정했다. 1개 seed·드라이브당 1개인 소규모 실험이므로
다른 표본·seed·탐색 예산에 대한 우위를 보장하지 않는다. d는 청감 점수나 정확도가 아니며,
낮은 d가 정답 드라이브·노브의 유일한 복원을 뜻하지 않는다.

### 실행 기록

격리 환경의 첫 시도는 VST가 Dry를 그대로 반환하여 활성 검사에서 중단됐다.
해당 출력은 거리나 탐색 결과에 포함하지 않았다. 일반 사용자 환경에서 7종 활성 검사를
통과한 뒤 동일 설정으로 다시 실행했다. 라이선스 활성화나 플러그인 GUI 조작은 하지 않았다.

validation과 test는 독립 프로세스에서 같은 설정으로 실행했다. 실행 시간은 시스템 부하를
통제한 속도 벤치마크로 해석하지 않는다. 실행 완료 여부는 summary.json의 status와
표본별 결과·예산·거리 재계산으로 확인한다.

<!-- CMA_RESULTS_START -->

### CMA-ES 실행 결과

| 방법 | Validation 평균 d | Test 평균 d |
|---|---:|---:|
| CNN Top-1 초기값 | 0.090417 | 0.118570 |
| CNN Top-3 초기값 중 최소 | 0.090417 | 0.118570 |
| 같은 Top-3 기본 노브 중 최소 | 0.315901 | 0.211778 |
| CNN 초기값 → CMA-ES | 0.044596 | 0.038138 |
| 기본 노브 → CMA-ES | 0.060268 | 0.064262 |
| 정답 체인 재렌더링 | 0.000880 | 0.000751 |

#### 정답 드라이브 후보의 노브 오차

각 표본의 정답 드라이브 후보에서 3개 정규화 노브 MAE를 계산했다. 최종 선택 드라이브가
오분류된 경우까지 포함하는 점수가 아니며, d 최소화와 정답 노브 복원은 별개다.

| 방법 | Validation 노브 MAE | Test 노브 MAE |
|---|---:|---:|
| CNN 초기값 | 0.092744 | 0.139008 |
| 기본 초기값 | 0.244625 | 0.220994 |
| CNN→CMA | 0.075323 | 0.066818 |
| 기본→CMA | 0.073358 | 0.111919 |

#### validation 표본별 결과

| 정답 드라이브 | CNN Top-3 초기 d | 기본 Top-3 초기 d | CNN→CMA d | 기본→CMA d | 정답 재렌더 d |
|---|---:|---:|---:|---:|---:|
| 808_scream | 0.024682 | 0.225459 | 0.024682 | 0.062937 | 0.000000 |
| b_drive | 0.119337 | 0.543267 | 0.058224 | 0.058224 | 0.000000 |
| o_drive | 0.088695 | 0.316617 | 0.016302 | 0.079220 | 0.000000 |
| one_distortion | 0.103548 | 0.342161 | 0.021014 | 0.018786 | 0.000000 |
| bat_distortion | 0.085508 | 0.346163 | 0.075417 | 0.089169 | 0.000000 |
| f_fuzz | 0.113864 | 0.251953 | 0.019254 | 0.039820 | 0.006161 |
| m_fuzz | 0.097282 | 0.185684 | 0.097282 | 0.073721 | 0.000000 |

- 원본 Dry 6개, 정답 Top-3 포함 7/7.
- CNN→CMA 평균 d는 CNN Top-3 초기값 대비 50.68% 감소.
- CNN→CMA 평균 d는 기본→CMA 대비 26.00% 감소.
- CNN 초기 최솟값보다 개선된 표본 5/7, 기본→CMA보다 낮은 d 4/7.
- 최종 선택 드라이브 정답: CNN→CMA 7/7, 기본→CMA 7/7.
- 탐색 렌더 2,520회 + 진단 14회. 실행 888.6초(사전 활성 검사 제외).

#### test 표본별 결과

| 정답 드라이브 | CNN Top-3 초기 d | 기본 Top-3 초기 d | CNN→CMA d | 기본→CMA d | 정답 재렌더 d |
|---|---:|---:|---:|---:|---:|
| 808_scream | 0.051664 | 0.111417 | 0.029564 | 0.019527 | 0.000000 |
| b_drive | 0.311874 | 0.360903 | 0.102988 | 0.226351 | 0.000004 |
| o_drive | 0.177630 | 0.085144 | 0.010502 | 0.010502 | 0.000000 |
| one_distortion | 0.021900 | 0.200216 | 0.017980 | 0.018197 | 0.000000 |
| bat_distortion | 0.075909 | 0.217074 | 0.044102 | 0.062287 | 0.000000 |
| f_fuzz | 0.040395 | 0.136522 | 0.006212 | 0.019024 | 0.005253 |
| m_fuzz | 0.150619 | 0.371170 | 0.055618 | 0.093944 | 0.000000 |

- 원본 Dry 6개, 정답 Top-3 포함 7/7.
- CNN→CMA 평균 d는 CNN Top-3 초기값 대비 67.84% 감소.
- CNN→CMA 평균 d는 기본→CMA 대비 40.65% 감소.
- CNN 초기 최솟값보다 개선된 표본 7/7, 기본→CMA보다 낮은 d 5/7.
- 최종 선택 드라이브 정답: CNN→CMA 7/7, 기본→CMA 7/7.
- 탐색 렌더 2,520회 + 진단 14회. 실행 864.8초(사전 활성 검사 제외).

#### 해석과 재검사

이번 표본에서는 CNN 초기값에 CMA-ES를 연결하면 두 split 모두 평균 d가 줄었다.
기본값 시작 CMA-ES보다 평균 d도 낮았지만, 모든 표본에서 우세하지는 않았다.
validation의 정답 드라이브 조건부 노브 MAE는 기본값 시작 CMA-ES가 조금 더 낮았다.
음향 거리 최소화와 정답 노브 복원을 같은 지표로 해석하지 않는다.

f_fuzz의 정답 재렌더 d는 validation 0.006161, test 0.005253으로 0이 아니다.
이 값은 저장된 Wet과 현재 렌더러의 재현 차이를 보여 준다. oracle은 하한으로 간주하지 않는다.

두 split의 탐색 5,040회와 성공한 진단 28회, 총 5,068회 렌더를 집계했다.
후보별 탐색 이력 84개의 유한값·단조 감소·초기값 보존·예산, 단일 VST/노브 범위,
checkpoint/cache provenance, 표본별 d와 모든 평균·감소율·노브 MAE 재계산을 통과했다.
CMA 원자료는 각 실행 폴더의 plan.json, tasks.jsonl, results/*.json, summary.json이다.
통합 검사 기록: `outputs/vst7_drive_only/cma_artifact_audit.json`.
재검사 및 이 문서의 CMA 결과 표 갱신: `python outputs/vst7_drive_only/audit_cma_results.py`.

<!-- CMA_RESULTS_END -->

## 코드·산출물 검증 결과

- CNN/CMA API·거리 회귀 검사: **70 passed, 6 deselected, 334 subtests passed**.
  제외한 6개는 학습 범위 밖인 공용 CompressorIntegration 검사다.
  matplotlib 부재 경고는 선택적 그래프 기능이며 탐색에 영향을 주지 않는다.
- 생성 데이터 전수 검사: 2,730쌍, Dry 78개, 원본별 split=54/12/12를 확인했다.
  무음/미처리 5개를 같은 정답 노브로 재렌더링하고 수정 전 실행은 폐기했다.
- 학습·평가 재검사: 전체 후보 2,520개의 단일 VST·노브 범위·좌표 순서를 확인했고,
  split당 70개 렌더 평가의 평균 거리와 분류 수치를 재계산했다.
- cache/checkpoint의 44.1 kHz 설정, 생성 manifest·전수 검사 hash를 확인했다.
  수정 이력과 hash는 반복 검사에서도 유지됐다.
- 이전 mixed 모델·결과·검사 파일과 폐기된 초기 실행, 혼합 데이터의 로컬 연결을 제거했다.
  현재 모델·데이터 hash는 유지했고 공유 Drive 원본은 변경하지 않았다.
- best checkpoint SHA-256: `72a401ad459f627b70cf0f2f3fd04ba0c5d83c3fdb8997a293312f3d58812c3a`.
- 검사 원자료: `outputs/vst7_drive_only/artifact_audit.json`.
