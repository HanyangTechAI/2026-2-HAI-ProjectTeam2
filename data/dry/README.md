# Dry 기타 샘플

외부 데이터셋에서 가져온 dry(이펙터 없는) 기타 샘플을 보관하는 폴더입니다.

## 현재 데이터

- 출처: [IDMT-SMT-Guitar V2](https://zenodo.org/records/7544110) (Zenodo, 약 1.3GB)
- 사용 범위: `dataset1/Fender Strat Clean Neck SC`
- 구성: 단음 78개 (6줄 × 0~12프렛, 피킹)
- 형식: 44.1kHz, 모노, 2.5초 wav + 음 정보 xml
- 라이선스: CC BY-NC-ND 4.0 (비상업적 이용, 출처 표기, 변형물 재배포 불가)

## 폴더 구조

- `Fender_Strat_Clean/audio/`: `sample_*.wav` (Git에 포함되지 않음)
- `Fender_Strat_Clean/annotation/`: `sample_*.xml`

## 준비 방법

1. Zenodo에서 zip을 받아 압축을 풉니다.
2. `scripts/SaveAs_Sample_Files.py`의 `SOURCE_AUDIO_DIR`, `SOURCE_ANNO_DIR`를 내 PC의 원본 경로로 바꿉니다.
3. 저장소 루트에서 `python scripts/SaveAs_Sample_Files.py`를 실행합니다.

> wav 공유 방식(GitHub / Drive)은 논의 중입니다.
