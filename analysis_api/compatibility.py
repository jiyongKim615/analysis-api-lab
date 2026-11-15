"""Explicit request envelopes and compact analysis output."""
from __future__ import annotations

from typing import Any
from .models import AnalysisRequest, AnalysisResponse, RunAgentRequest
from .providers import strict_json_loads

SUPPORTED = frozenset({"analysis", "voc"})
KNOWN_UNSUPPORTED = frozenset({"voc_suggestion", "voc_feedback", "governance_message", "report", "make_report", "sql", "doc_report", "governance"})
REQUEST_ALIASES = {"page": "page_number", "start_date": "date", "end_date": "date_end", "channels": "target_codes"}


def parse_run_request(raw: dict[str, Any]) -> RunAgentRequest:
    if set(raw) == {"input_value"}:
        if not isinstance(raw["input_value"], str) or len(raw["input_value"]) > 2_000_000:
            raise ValueError("INVALID_INPUT_VALUE")
        raw = strict_json_loads(raw["input_value"])
        if not isinstance(raw, dict):
            raise ValueError("INPUT_VALUE_MUST_CONTAIN_OBJECT")
    elif "input_value" in raw:
        raise ValueError("AMBIGUOUS_INPUT_ENVELOPE")
    raw = dict(raw)
    # Accept a flat request or an explicit analysis object, never both.
    route_keys = [key for key in ("requestType", "request_type", "agent_type") if key in raw]
    stream_keys = [key for key in ("isStream", "is_stream", "stream") if key in raw]
    if len(route_keys) > 1 or len(stream_keys) > 1:
        raise ValueError("DUPLICATE_ROUTING_ALIAS")
    route = raw.pop(route_keys[0]) if route_keys else "analysis"
    stream = raw.pop(stream_keys[0]) if stream_keys else False
    if type(stream) is not bool:
        raise ValueError("STREAM_MUST_BE_BOOLEAN")
    if "analysis" in raw:
        if set(raw) != {"analysis"}:
            raise ValueError("AMBIGUOUS_ANALYSIS_ENVELOPE")
        analysis = raw["analysis"]
    else:
        analysis = raw
    if not isinstance(analysis, dict):
        raise ValueError("ANALYSIS_MUST_BE_OBJECT")
    analysis = dict(analysis)
    for old, new in REQUEST_ALIASES.items():
        if old in analysis:
            if new in analysis:
                raise ValueError("DUPLICATE_REQUEST_ALIAS")
            analysis[new] = analysis.pop(old)
    for name in ("date", "date_end"):
        value = analysis.get(name)
        if isinstance(value, str) and len(value) == 8 and value.isdigit():
            analysis[name] = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    if "survey_mode_list" in analysis:
        analysis["survey_modes"] = analysis.pop("survey_mode_list") or []
    return RunAgentRequest(request_type=route, is_stream=stream, analysis=AnalysisRequest.model_validate(analysis))


def export_summary(response: AnalysisResponse) -> dict:
    """A compact view of validated results; missing analysis stays absent."""
    content = []
    for voc in response.page.content:
        emotion = voc.emotion
        content.append({
            "vocId": voc.voc_id,
            "recordedOn": voc.recorded_on.isoformat(),
            "recordKey": voc.get_pk_dict(),
            "channelCode": voc.channel_code,
            "journeyCode": voc.journey_code,
            "excluded": voc.excluded,
            "filteredReason": voc.filtered_reason,
            "aspectSuccess": voc.aspect_success,
            "aspects": [a.model_dump(mode="json", by_alias=True) for a in voc.aspects],
            "feedbackType": emotion.feedback_type if emotion else None,
            "emotion": emotion.model_dump(mode="json", by_alias=True) if emotion else None,
            "primaryReason": voc.primary_reason,
            "secondaryReasons": voc.secondary_reasons,
            "summary": voc.summary,
            "sourceHash": voc.source_hash,
            "visibility": voc.offer.model_dump(mode="json", by_alias=True),
            "stages": {key: value.model_dump(mode="json", by_alias=True) for key, value in voc.stages.items()},
        })
    page = response.page.model_dump(mode="json", by_alias=True, exclude={"content"})
    page["content"] = content
    return {"metadata": response.metadata.model_dump(mode="json", by_alias=True), "page": page}
