"""Limited deterministic redaction, not a comprehensive Korean PII solution."""
from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from .models import RequestContext, VocInput

_PATTERNS = (
    (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:01[016789]|02|0[3-6]\d)[- .]?\d{3,4}[- .]?\d{4}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{6}[- ]?[1-8]\d{6}(?!\d)"), "[PERSONAL_ID]"),
    (re.compile(r"(?<!\d)(?:\d{4}[- ]?){3}\d{4}(?!\d)"), "[CARD_LIKE]"),
)
_current: ContextVar[RequestContext | None] = ContextVar("cx_request", default=None)


def mask_pii(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_tree(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, list):
        return [redact_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_tree(v) for k, v in value.items()}
    return value


def redact_record(record: VocInput) -> tuple[VocInput, str]:
    digest = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
    # Preserve source identifiers for caller matching, but never send them to providers.
    payload = record.model_dump(mode="json", include=set(VocInput.model_fields))
    for key in ("text", "prompt_title", "prompt_description", "prompt_context", "channel_name", "journey_name"):
        payload[key] = mask_pii(payload[key])
    return VocInput.model_validate(payload), digest


@contextmanager
def use_context(context: RequestContext) -> Iterator[None]:
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)


def current_context() -> RequestContext | None:
    return _current.get()
