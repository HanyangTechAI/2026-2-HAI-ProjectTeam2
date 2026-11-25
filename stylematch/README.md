# stylematch – 같은 연주 dry 없이 목표 톤 맞추기 (Step 3)

곡에서 분리한 기타나 외부 기타 톤(목표)에 대해, **사용자의 다른 연주(dry)** 로
"내 페달로 목표 톤에 가장 가깝게 만드는 설정"을 찾습니다. 결과는 **상위 3개 설정**으로 제시합니다.

## 왜 따로 만들었나
Step 1·2의 CNN + 거리 d + CMA-ES는 **같은 연주의 dry/wet 쌍**을 전제로 합니다.
실제 사용(곡 음원 + 다른 연주)에서는 거리가 음·리듬과 녹음 환경(앰프·마이크) 차이에 반응해
실제 페달 데모에서 정답을 거의 못 맞혔습니다. 그래서 비교 기준을 **AFx-Rep**으로 바꿨습니다.
자세한 실험 기록: [EXPERIMENTS.md](EXPERIMENTS.md)

- **AFx-Rep**: 소리를 "어떤 이펙트가 걸렸는지" 512차원 지문으로 바꾸는 공개 모델.
  같은 이펙트면 연주가 달라도 지문이 비슷하도록 학습됨.
  출처: ST-ITO (Steinmetz et al., ISMIR 2024), https://github.com/csteinmetz1/st-ito

## 동작
```
목표 → AFx-Rep 지문
공통 dry → 7종 × 게인 × 톤 × 앰프 크런치 격자 렌더 → 지문 비교 → 페달 순위 → 상위 3개
각 페달 안에서만 CMA-ES로 게인·톤·크런치 보정 (거리: AFx-Rep, dropout 0.1)
level/volume은 음량 맞추기(이분 탐색)로 따로
체인: 드라이브(VST) → 앰프 크런치 → 캐비닛 IR
```
- 탐색이 페달까지 바꾸게 하면 거리 숫자만 줄이는 현상(metric hacking)이 생겨 페달은 격자 순위로 고정합니다.
- 보유 페달만 후보로: `--owned 808_scream b_drive ...`

## 준비
1. CNN 환경에 의존성: `pip install -r requirements-cnn.txt` (torchaudio, torchlibrosa, pyyaml 추가)
2. AFx-Rep 가중치(약 1.1GB, git 제외): 공유 드라이브 `model/afx-rep/`를 `data/local/afx-rep/`에 복사
   (또는 폴더를 환경변수 `HAI_AFX_DIR`로 지정). 가중치는 `weights_only=True`로만 로드합니다.
3. MultiDrive PRO 플러그인, 캐비닛 IR(`data/IR`), IDMT 원본(`data/raw`) 연결 – [DRIVE_SETUP.md](../DRIVE_SETUP.md)

## 실행 (저장소 루트)
```powershell
# 목표 소리 전처리 (2.5초 구간 · 모노 44.1 kHz · 음량 정규화)
python -m scripts.prepare_step3_targets --separated <드라이브>/data/step3_real_data `
  --isolated <드라이브>/data/step3_val_data --output data/local/step3_targets_v1

# 톤 맞추기 → 목표별 target / dry / top1~3.wav + result.json
python -m stylematch.run_targets --targets data/local/step3_targets_v1 `
  --raw-separated <드라이브>/data/step3_real_data --raw-isolated <드라이브>/data/step3_val_data `
  --out outputs/stylematch/step3_v1 [--owned 808_scream b_drive] [--resume]

# 평가: 독립 지표(탐색에 안 쓴 지표)로 기준선 대비 개선 / 거리 신뢰도
python -m stylematch.assess results --dir outputs/stylematch/step3_v1
python -m stylematch.assess reliability --data data/local/step2_real_v1 --per-drive 3

# 테스트
python -m unittest stylematch.test_stylematch
```
코드에서 쓰기: `from stylematch.match import match_tone` → `match_tone(dry, target, sr)["candidates"]`

## 파일
| 파일 | 내용 |
|---|---|
| `afx.py` | AFx-Rep 로드 · 지문 · 거리(dropout 옵션) |
| `match.py` | 격자 선별 → 페달 내 CMA-ES → 음량 맞추기, 앰프 크런치 |
| `run_targets.py` | Step 3 목표 일괄 실행, 데모·청취용 결과 저장 |
| `assess.py` | 독립 지표 평가, 거리 신뢰도(순위 상관) |
| `evaluate.py` | 정답 있는 데이터로 페달 일치 채점 (보조 지표) |
| `head.py` | AFx-Rep 보정층 실험 (실패 기록용, 사용 안 함) |
| `third_party/panns.py` | AFx-Rep 인코더 정의 (ST-ITO, Apache 2.0 / MIT) |
