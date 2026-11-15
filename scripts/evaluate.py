"""Measure API and adverse-provider behavior without opening a network connection."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import httpx

with patch.dict(os.environ, {"CX_PROVIDER": "offline", "CX_ALLOW_EXTERNAL": "false",
                            "CX_REQUIRE_AUTH": "false", "CX_API_TOKEN": ""}):
    from analysis_api.api import create_app
from analysis_api.config import Settings
from analysis_api.models import AnalysisRequest, RequestContext, VocInput
from analysis_api.pipeline import AnalysisPipeline
from analysis_api.providers import OfflineReferenceProvider, ProviderGateway
from analysis_api.repository import JSONFixtureRepository
from analysis_api.taxonomy import Taxonomy


class UngroundedReason(OfflineReferenceProvider):
    async def complete(self, stage, payload, context):
        result = await super().complete(stage, payload, context)
        if stage == "reason":
            for value in result["results"].values():
                value["primary_reason"] = "This sentence is absent from the input."
        return result


class SlowProvider(OfflineReferenceProvider):
    async def complete(self, *args):
        await asyncio.sleep(1)


async def evaluate():
    context = RequestContext(request_id="synthetic-evaluation", actor_ref="evaluation")
    with tempfile.TemporaryDirectory(prefix="analysis-eval-") as temp:
        config = Settings(output_dir=Path(temp))
        app = create_app(config)
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 123))
        async with httpx.AsyncClient(transport=transport, base_url="http://lab") as client:
            response = await client.post("/run_agent", json={"analysis": {}})
            assert response.status_code == 200, response.text
            data = response.json()
            metrics = data["metadata"]["metrics"]
            counts = {key: metrics[key] for key in ("inputCount", "filteredCount", "reviewCount")}
            assert counts == {"inputCount": 11, "filteredCount": 3, "reviewCount": 1}
            loaded = await client.get("/runs/" + data["metadata"]["runId"])
            assert loaded.status_code == 200 and loaded.json() == data
            stream = await client.post("/run_agent", json={"isStream": True, "analysis": {}})
            assert stream.status_code == 200
            events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
            names = [event["event"] for event in events]
            assert names == ["START", *(["STAGE"] * 6), "CHUNK", "DONE"]
            assert events[-1]["content"]["success"] is True
            assert events[-2]["content"]["page"]["numberOfElements"] == 11
            invalid = await client.post("/run_agent", json={"analysis": {"pageSize": 0}})
            unsupported = await client.post("/run_agent", json={"requestType": "report"})
            assert (invalid.status_code, unsupported.status_code) == (422, 501)
        pipeline = AnalysisPipeline(JSONFixtureRepository([]), Taxonomy.load(config.taxonomy_path), UngroundedReason())
        failed = await pipeline.execute(AnalysisRequest(retry_limit=1, vocs=[VocInput(
            voc_id="SYN-FAILURE", text="앱 로그인 오류가 반복되어 정말 실망했습니다.", journey_code="18")]), context)
        item = failed.page.content[0]
        assert item.stages["reason"].status.value == "failed"
        assert item.primary_reason is None and item.emotion_success
        timeout = await ProviderGateway(SlowProvider(), timeout=.005, retry_limit=1).call(
            "emotion", [{"id": "synthetic-timeout", "text": "합성 입력"}], context, lambda value, item: value)
        timed_out = timeout["synthetic-timeout"]
        assert (timed_out.error_code, timed_out.attempts) == ("TIMEOUT", 2)
        return {"data_origin": "hand-authored synthetic fixtures", "network_calls": 0,
                "provider": data["metadata"]["engine"], "fixture": counts,
                "stored_result_matches": True, "sse_events": names,
                "invalid_request_status": invalid.status_code, "unsupported_request_status": unsupported.status_code,
                "ungrounded_reason": {"status": item.stages["reason"].status.value,
                                      "error_code": item.stages["reason"].error_code,
                                      "attempts": item.stages["reason"].attempts,
                                      "earlier_emotion_preserved": item.emotion_success},
                "timeout": {"error_code": timed_out.error_code, "attempts": timed_out.attempts}}


def main():
    with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", side_effect=AssertionError("NETWORK_DISABLED")):
        result = asyncio.run(evaluate())
    path = Path("artifacts/evaluation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Scenario checks passed: {path}")


if __name__ == "__main__":
    main()
