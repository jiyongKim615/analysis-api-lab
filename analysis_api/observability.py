"""Content-free metrics and safe event shape; no telemetry export or external SDK."""
from collections import Counter
from .models import VocInfo


def summarize(vocs: list[VocInfo], provider_calls: int = 0) -> dict:
    stages = {}
    for name in ("filter", "aspect", "emotion", "entity", "reason", "summary"):
        results = [v.stages[name] for v in vocs]
        stages[name] = {
            "statusCounts": dict(Counter(r.status.value for r in results)),
            "errorCounts": dict(Counter(r.error_code for r in results if r.error_code)),
            "maxDurationMs": max((r.duration_ms for r in results), default=0),
            "maxAttempts": max((r.attempts for r in results), default=0),
        }
    return {"inputCount": len(vocs), "filteredCount": sum(v.excluded for v in vocs),
            "reviewCount": sum(any(r.status.value in {"needs_review", "failed", "partial_success", "not_configured"} for r in v.stages.values()) for v in vocs),
            "providerCalls": provider_calls, "stages": stages}
