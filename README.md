# Analysis API Lab

합성 피드백을 여러 단계로 분석하고, 실행 상태와 결과를 JSON 또는 SSE로 전달하는 FastAPI 실습입니다.
분석 단계가 실패했을 때 앞서 얻은 결과를 보존하고, 검토가 필요한 항목을 구분하는 흐름을 살펴봅니다.

기본 엔진은 작은 규칙·키워드 기반 `OfflineReferenceProvider`입니다. API 키와 모델 다운로드 없이 실행하며, fixture 11건과 분류표는 실습용으로 직접 작성했습니다.

## 다루는 흐름

```text
JSON / SQLite 합성 입력
  → 입력 계약·중복 확인 → 짧은 문장·관련성 필터
  → 측면 분류 → 감정 → 개체어 → 사유 → 발췌 요약
  → 단계별 상태·원문 근거 → 결과 보존 → JSON / SSE
```

- **입출력 계약:** Pydantic 검증, 요청별 상태, 원문에 존재하는 근거 확인
- **부분 결과:** 단계별 `success`, `partial_success`, `needs_review`, `failed`, `skipped` 구분
- **제한된 재시도:** 실패 항목만 다시 요청하고 동시 호출 수·시도 횟수·시간을 제한
- **결과 보존:** 실행 ID별 저장·조회, 동일 내용 재저장, 충돌 거부
- **스트리밍:** `START → STAGE → CHUNK → DONE`, 연결 종료 시 작업 취소

## 실행

Python 3.12에서 검증했습니다.

```bash
git clone https://github.com/jiyongKim615/analysis-api-lab.git
cd analysis-api-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make lab
python -m analysis_api demo --out output/demo.json --csv output/demo.csv
```

`make lab`은 82개 회귀 테스트, 문법 검사, 정상·실패 시나리오 평가를 실행합니다.
로컬 기본 예제에서 입력 11건, 필터 제외 3건, 검토 필요 1건을 관찰했습니다.
필터 제외 항목도 결과 목록에 남습니다. [평가 요약](examples/evaluation.json)은 같은 명령으로 재현할 수 있습니다.

## API 사용

```bash
CX_PROVIDER=offline CX_ALLOW_EXTERNAL=false \
  python -m uvicorn analysis_api.api:app --host 127.0.0.1 --port 8000
```

별도 터미널에서 실행합니다.

```bash
# 기본 합성 fixture 분석
curl -s http://127.0.0.1:8000/run_agent \
  -H 'Content-Type: application/json' \
  --data '{"analysis":{}}'

# 명시적인 합성 문장을 SSE로 분석
curl -N http://127.0.0.1:8000/run_agent \
  -H 'Content-Type: application/json' \
  --data @examples/request.json

# JSON 응답의 metadata.runId로 저장 결과 조회
curl -s http://127.0.0.1:8000/runs/REPLACE_WITH_RUN_ID
```

`requestType: "analysis"`는 전체 결과, `requestType: "voc"`는 간략 결과를 반환합니다.
지원 범위 밖의 보고서·설문 관련 라우트는 `501 UNSUPPORTED`로 응답합니다. API 문서는 서버 실행 후 `/docs`에서 확인할 수 있습니다.

## 결과 읽기

`metadata`에는 엔진·분류표 버전, 실행 ID, 처리 건수와 단계별 상태 집계를 기록합니다.
각 항목의 `stages`에는 시도 횟수, 오류 코드, 근거를 남깁니다. 원문에 없는 사유를 반환하는 모의 provider를 넣으면 해당 단계가 실패하며 앞선 성공 결과는 보존됩니다.

SSE의 `DONE.success`는 요청 처리가 끝났는지를 나타냅니다. 모든 항목의 분석 성공을 뜻하지 않으므로 `stages`와 `reviewCount`도 확인해야 합니다.

## 범위와 문서

- [설계와 API 계약](docs/design.md): 입력, 단계 실행, 저장, provider 연결 경계
- [평가와 실패 사례](docs/evaluation.md): SSE 종료, 근거 위반, 타임아웃, 선택 재시도
- [테스트](tests): 실제 네트워크 호출 없이 ASGI·MockTransport로 검증

기본 서버는 토큰 없는 원격 요청을 거부합니다. 외부 provider 어댑터는 명시적으로 활성화해야 하며, 실습 검증은 모의 응답으로만 수행했습니다. 정규식 마스킹은 일부 패턴만 처리하므로 이 저장소에는 합성 데이터만 사용합니다. 실제 모델 정확도나 운영 환경의 처리 성능은 측정하지 않았습니다.
