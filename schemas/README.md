# 프로젝트 스키마

모든 스키마는 JSON Schema Draft 2020-12를 사용합니다. 각 `*.schema.json`에 대응하는 `*.data.json`이 있습니다. `effects.data.json`은 기존 이펙터 정의이며, 나머지 데이터 파일은 사용 형식을 보여주는 예제입니다.

| 스키마 | 역할 |
|---|---|
| `effects.schema.json` | 이펙터 종류와 파라미터 정의 |
| `chain.schema.json` | 앰프 ID와 순서 있는 이펙터별 실제 설정값 |
| `amp.schema.json` | 고정 앰프의 렌더링 방식과 파일 경로 |
| `dry_sample.schema.json` | dry 샘플 출처, 경로, 길이, 샘플레이트, 라이선스 |
| `target_sample.schema.json` | 타겟 경로와 합성 여부, 정답 체인 또는 실제 음원 출처 |
| `user_inventory.schema.json` | 사용자/세션 ID와 보유 이펙터 목록 |

## 적용 규칙

- 기존 `overdrive`(OD-1), `fuzz`(FZ-1W) 정의를 유지합니다. 첨부 문서의 `drive`(SD-1), FZ-5로 변경하지 않았습니다.
- 체인의 `chain` 배열은 이펙터의 상대적인 적용 순서입니다. 앰프는 `amp_id`로 별도 지정하며 MVP 렌더러는 드라이브 뒤, EQ 앞에 적용할 예정입니다. 현재 스키마는 특정 체인 순서나 길이를 강제하지 않습니다.
- 체인의 `values`에는 해당 이펙터의 모든 파라미터를 명시합니다. 기본값 자동 채우기는 수행하지 않습니다.
- 앰프의 `fixed`는 반드시 `true`입니다. `impulse_response`와 `nam_capture`는 비어 있지 않은 `file_path`가 필요하고, `pedalboard_synthetic`은 `null`을 허용합니다.
- 합성 타겟은 `dry_sample_id`와 `ground_truth_chain`이 필수입니다. 실제 음원 타겟은 `source_dataset`이 필수이며, dry 참조와 정답 체인은 생략하거나 `null`로 둡니다.
- 샘플 길이와 샘플레이트는 양수이며 샘플레이트는 정수입니다. 보유 이펙터 목록은 중복을 허용하지 않고, 빈 목록은 허용합니다. 선택 필드 `updated_at`은 시간대가 포함된 date-time 문자열입니다.
- 취소선이 표시된 `ExperimentRecord`는 이번 구현 범위에서 제외했습니다.

## 검증

저장소 루트에서 `python schemas/validate.py`로 검증합니다. 검증에 필요한 의존성은 `python -m pip install jsonschema`로 설치할 수 있습니다.

검증기는 모든 스키마와 예제 데이터를 검사하고, 외부 스키마 참조는 로컬 레지스트리에서 해결합니다. 스키마의 `https://hai-project.example/schemas/` ID는 식별용이며 네트워크에서 다운로드하지 않습니다.

JSON Schema 검사에 더해 Python 코드에서 이펙터 존재 여부, 파라미터 이름·타입·범위·옵션, 앰프 ID, 합성 타겟의 dry 샘플 ID, 보유 이펙터 참조를 확인합니다. 다른 프로그램에서 JSON Schema만 사용하면 이러한 데이터 간 일치 검사는 별도로 수행해야 합니다.

새 예제의 오디오 경로와 라이선스는 자리표시자입니다. 실제 파일을 생성하거나 파일 존재·오디오 길이·라이선스의 정확성을 검사하지 않습니다. 사용 전에 실제 데이터로 교체하세요.

검증 동작 테스트: `python -m unittest discover -s schemas -p "test_*.py"`
