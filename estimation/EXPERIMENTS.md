# VST7 학습 결과

2026-11-17 완료. 이전 3종 CNN의 로컬 학습 결과·데이터 캐시·진행 문서는 제거했다.
현재 산출물은 `outputs/vst7/`만 사용한다. 공유 원본은 수정하지 않았다.

## 데이터와 학습 설정

`data/synthetic/step1_vst_v0_임시`의 VST7 2,730쌍, 생성 Dry 78개를 사용했다.
학습·렌더링 입력은 `manifest.dry_id_map[].render_input`의 peak −12 dBFS Dry다.
라벨의 다른 PC 절대경로는 사용하지 않고 현재 `audio/<target_id>.wav`를 연결했다.

| Split | 독립 Dry 원본 | 쌍 | 드라이브별 쌍 |
|---|---:|---:|---:|
| train | 54 | 1,890 | 270 |
| validation | 12 | 420 | 60 |
| test | 12 | 420 | 60 |x`

원본 ID와 생성 Dry의 내용 해시 모두 split 사이에 겹치지 않는다.
7종 외에 컴프·EQ의 사용 여부와 노브도 함께 학습했다. 앰프는 렌더링하지 않는다.

- 가중치를 공유하는 paired CNN, 채널 8/16/32/64, 주파수 영역 4개 보존, hidden 256.
- 학습 입력: 22,050 Hz, 2.5초, 64-band log-mel. 추가 peak 정규화 없음.
- 렌더링 입력: 생성 Dry의 원래 44,100 Hz를 유지하고, 렌더링 후 평가 표본율로 변환.
- Adam, 초기 LR 0.001, cosine decay, batch 16, seed 42, CPU 4 threads, 30epoch.
- loss: drive CE + stage BCE + active-knob Huber + 0.1 × distance auxiliary Huber.
- **best checkpoint는 validation 전체 loss 최소인 epoch 20**. test로 모델을 선택하지 않았다.

## 분류와 구조 복원 — 각 split 전체 420개

| 지표 | Validation | Test |
|---|---:|---:|
| 드라이브 Top-1 accuracy | 86.90% | **87.62%** |
| 드라이브 Top-3 정답 포함률 | 97.38% | **97.38%** |
| 드라이브 macro F1 | 0.8678 | 0.8764 |
| 컴프 사용 여부 accuracy | 65.95% | 65.48% |
| EQ 사용 여부 accuracy | 84.05% | 78.57% |
| 드라이브 + 컴프·EQ 사용 여부 모두 정답 | 47.14% | **47.38%** |

드라이브 분류 정확도와 전체 체인 구조 정확도는 다르다. 특히 컴프 포함 여부는 아직 개선이 필요하다.

## 노브 normalized MAE

0~1로 정규화한 노브의 평균 절대 오차이며 낮을수록 좋다. 아래 값은 정답 체인에 실제 있는
효과의 head를 평가한 조건부 지표다. 잘못 분류한 구조에 대한 패널티를 포함한 전체 체인 정확도가 아니다.

| 효과 | Validation CNN | Test CNN | Test 기본 노브 |
|---|---:|---:|---:|
| 808_scream | 0.2060 | 0.1931 | 0.2726 |
| b_drive | 0.1752 | 0.1686 | 0.2698 |
| o_drive | 0.1914 | 0.1413 | 0.2472 |
| one_distortion | 0.1633 | 0.1711 | 0.2539 |
| bat_distortion | 0.1851 | 0.1692 | 0.2562 |
| f_fuzz | 0.2245 | 0.2093 | 0.2618 |
| m_fuzz | 0.1972 | 0.1866 | 0.2589 |
| compressor | 0.2249 | 0.2342 | 0.2717 |
| eq_3band | 0.1981 | 0.2043 | 0.2459 |

## 실제 렌더링 유사도 d

공통 `metrics.combined`의 스펙트럼·다이내믹스 가중 평균을 사용했다.

`d = 0.5 × spectral / 0.1642928288 + 0.5 × dynamics / 6.984483499`.

두 scale은 **train Dry/target 거리의 중앙값**으로만 정했고 validation/test에서는 고정했다.
d는 무차원 거리이며 낮을수록 가깝다. 이전 spectral-only 점수와 직접 비교하지 않는다.

실제 렌더링 평가는 각 split에서 사전 고정한 **70개(종류별 10개, Dry 원본 12개 포함)**를 사용했다.
예측·점수를 본 뒤 쉬운 사례를 골라내지 않았다. 아래 모든 행은 같은 70개를 비교한다.

| 렌더링 방법 | Validation 평균 d | Test 평균 d |
|---|---:|---:|
| CNN 예측 구조 + 기본 노브 | 0.839685 | 0.731451 |
| CNN Top-1 구조·노브 | **0.651993** | **0.549682** |
| CNN Top-3를 렌더링해 d 최소 후보 선택 | 0.575242 | 0.492659 |
| 정답 전체 체인·정답 노브 재렌더링 | 0.215671 | 0.219903 |

같은 예측 구조의 기본 노브 대비 CNN Top-1의 d는 validation **22.35%**, test **24.85%** 감소했다.
이 비교에는 CMA-ES 노브 최적화를 사용하지 않았다.

**정답 체인 재렌더링도 0이 아니다.** 설치된 VST와 저장 데이터 사이의 재현 오차가 있으므로
전체 d를 CNN만의 오차로 해석하면 안 된다. 정답 재렌더링 점수도 이 렌더러 아래의 비교 기준이며,
모든 가능한 세팅의 이론적 최소값은 아니다. GitHub main에는 공지의 통합 생성기·소문자 key 수정본이
아직 확인되지 않아 현재 `estimation` adapter가 표시명 key를 대응시킨다.

전체 렌더링 평가는 `estimation.evaluate --render-samples -1`로 별도 실행할 수 있다.

## 학습 중 d 비교

학습에서는 기존 Dry/Wet에서 측정한 두 거리 성분을 보조 head의 정답으로 사용했다.
이는 `d(Dry, target)`의 추정 학습이며, 비미분 VST의 예측 노브 렌더링 d로 직접 역전파하지 않는다.
보조 d 추정 MAE는 validation 0.22434, test 0.17317이며 실제 출력 d와 다른 지표다.

이번 실행의 주기적 프로브는 고정 validation 원본 **1개에서 7종 × 첫 2변형 = 14개**를 사용했다.
전체 validation 대표값으로 보지 않고 학습 경과 확인에만 썼다. 최종 70개 평가는 위에 별도로 기록했다.

| Epoch | CNN 프로브 d | 같은 예측 구조의 기본 노브 d |
|---|---:|---:|
| 1 | 0.968790 | 1.008921 |
| 5 | 0.838547 | 1.014708 |
| 10 | 0.703671 | 1.077185 |
| 15 | 0.725558 | 0.990276 |
| 20 (best loss) | 0.613135 | 0.945666 |
| 25 | 0.701008 | 0.969010 |
| 30 | 0.648398 | 0.951230 |

거리·분류 지표가 매 epoch 단조 개선되는 것은 아니다. 현재 코드의 향후 학습 프로브는
원본을 round-robin으로 분산하며, 이번 실행의 실제 프로브 sample ID는 history에 보존했다.

## 산출물

- [best checkpoint](../outputs/vst7/model/best.pt): epoch 20.
- [학습 이력](../outputs/vst7/model/history.json), [CSV](../outputs/vst7/model/training_metrics.csv).
- [validation 요약](../outputs/vst7/validation/summary.json), [test 요약](../outputs/vst7/test/summary.json).
- [통합 요약](../outputs/vst7/summary.json), [상위 3개 후보 예시](../outputs/vst7/candidates_example.json).
- [정답 재현 검사](../outputs/vst7/dataset/renderer_preflight.json).

단일 기타·단음 데이터, seed 42 한 번의 결과다. 다른 기타·연주자·실물 페달 및
서로 다른 연주의 Dry/reference에 대한 일반화는 검증하지 않았다.

## 2026-11-18 후속 CNN → CMA-ES 연결 실험

위 2026-11-17 학습·평가 수치는 그대로 유지했다. 별도 validation 1개에 팀 CMA-ES를
연결한 결과, 초기 Top-3 최소 d 0.489065에서 0.346037로 감소했다.
같은 예산의 기본 노브 시작 CMA-ES는 0.373791이었다.
작은 연결 검증이며, 전체 성능 비교는 아직 실행하지 않았다.
실험 정의·정답 종류와의 차이·예산·결과 경로는
[CNN_CMA_EXPERIMENT.md](CNN_CMA_EXPERIMENT.md)에 기록한다.
