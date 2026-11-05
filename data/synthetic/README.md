# 합성 데이터

`pedalboard`로 합성한 (dry, params, wet) 페어를 보관할 폴더입니다. 대용량 결과물은 Drive에서 관리합니다.

## 보류: 외부 VST 플러그인 (Step 1 이후)

Step 0을 먼저 끝내기 위해 pedalboard 내장 이펙트로 되돌렸습니다. VST 버전 코드는 커밋 `c211bfd`에 남아 있습니다.

| 종류 | 플러그인 | 다운로드 |
|---|---|---|
| 오버드라이브 | OverDiver (SD-1 스타일) | https://plugins4free.com/plugin/3096 |
| 디스토션 | ProF.E.T. | https://plugins4free.com/plugin/2922 |
| 퍼즈 | Analog Obsession PIG Pie | https://plugins4free.com/plugin/3095 |

아래를 모두 통과하면 VST로 전환하고, 하나라도 막히면 pedalboard 조합(`Distortion`/`Clipping` + 필터)으로 OD/DS/Fuzz를 구분합니다.

- [ ] Mac에서 로드 (`xattr -dr com.apple.quarantine <플러그인>.vst3`)
- [ ] Windows용 VST3 확보 및 로드
- [ ] 렌더링 속도 측정 (내장 이펙트 대비)
- [ ] 노브 0~1 값이 소리를 예상대로 바꾸는지 확인 → 스키마에 반영
- [ ] 팀 내부 공유 가능 여부 확인 (가능하면 저장소가 아니라 드라이브에 보관)
- [ ] `apply_effects` 입출력은 유지 (탐색·평가 코드가 그대로 쓰도록)
