# 평가와 실패 사례

```bash
make lab
```

Python 3.12에서 82개 `unittest` 테스트와 별도의 시나리오 평가를 실행합니다. ASGITransport는 프로세스 안에서 API를 호출하고, HTTP provider 검사는 MockTransport를 사용합니다. 실제 외부 모델 호출은 하지 않습니다.

`scripts/evaluate.py`는 실제 HTTP transport를 차단한 상태에서 다음 결과를 확인하고 `artifacts/evaluation.json`에 기록합니다. 비교용 실행 결과는 `examples/evaluation.json`에 있습니다.

| 시나리오 | 확인할 결과 |
| --- | --- |
| 기본 fixture | 입력 11, 제외 3, 검토 필요 1; 제외 항목도 보존 |
| JSON 저장·조회 | 같은 실행 ID 조회 결과가 최초 결과와 일치 |
| SSE | START·STAGE·CHUNK·DONE, 결과 데이터와 종료 상태 |
| 잘못된 입력 / 미지원 요청 | 각각 422 / 501 |
| 원문에 없는 사유 | reason 단계 실패, 이전 감정 결과 보존 |
| provider 타임아웃 | retry limit 1이면 총 시도 2회 후 TIMEOUT |

SSE 시나리오 평가는 버퍼링된 ASGI 응답에서 이벤트 순서와 내용을 확인합니다. 실시간 네트워크 지연이나 프록시 동작을 측정하지 않습니다. 연결 종료 시 취소와 미완료 결과 미저장은 별도의 ASGI 이벤트 테스트에서 확인합니다.

회귀 테스트에는 입력 별칭 충돌, 복합 키 누락, 페이지 간 중복, 근거 위반, 항목별 재시도, 동시 호출 제한, 인증·실행 주체 격리, 원자적 저장 충돌, CLI·SQLite 실행이 포함됩니다.

실행 ID·시각·소요 시간은 매번 달라집니다. 평가 요약은 이 값을 제외하고 고정 fixture의 건수·상태를 비교합니다. 테스트 개수는 기능 동작의 회귀 검사 수이며 분석 정확도, 부하 처리량, 운영 가용성을 나타내지 않습니다.
