"""Provider ports plus an explicitly synthetic rule reference and opt-in JSON HTTP adapter.

No network call occurs in offline mode.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import httpx

from .models import RequestContext
from .prompts import STAGES, build_messages
from .security import redact_tree


class ProviderError(Exception):
    def __init__(self, code: str, retryable: bool = False):
        super().__init__(code)
        self.code, self.retryable = code, retryable


class OutputValidationError(ProviderError):
    def __init__(self, code: str = "INVALID_RESPONSE"):
        super().__init__(code, retryable=True)


def strict_json_loads(raw: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in items:
            if key in result:
                raise OutputValidationError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
    except (ValueError, TypeError) as exc:
        raise OutputValidationError() from exc


class AnalysisProvider(Protocol):
    name: str
    inference_kind: str
    async def complete(self, stage: str, payload: dict, context: RequestContext) -> dict: ...


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?。]\s*|(?<=지만)\s*", text) if s.strip()]


def choose_scope(item: dict) -> str:
    text = item["text"]
    keywords = item.get("keywords", [])
    return next((s for s in sentences(text) if any(k in s for k in keywords)), text)


def reference_emotion(text: str) -> str:
    # Deliberately tiny, documented lexicon. This is not general Korean sentiment NLP.
    if any(w in text for w in ("길지 않았", "느리지 않", "불편하지 않", "오류가 없")):
        return "02"
    rules = (
        ("06", ("짜증", "화가", "최악")),
        ("05", ("실망", "오류", "실패", "못했", "못해서", "불가")),
        ("04", ("아쉽", "아쉬", "느리", "느려", "오래", "복잡", "길었")),
        ("01", ("감동", "최고", "훌륭")),
        ("03", ("기대", "바랍니다", "생기길")),
        ("02", ("만족", "친절", "편리", "명확", "빠르", "좋았")),
    )
    return next((code for code, words in rules if any(w in text for w in words)), "07")


def polarity(text: str) -> str:
    code = reference_emotion(text)
    return "negative" if code in {"04", "05", "06"} else "neutral" if code == "07" else "positive"


class OfflineReferenceProvider:
    name = "offline-rule-reference-v1"
    inference_kind = "deterministic_synthetic_reference_not_llm"

    async def complete(self, stage: str, payload: dict, context: RequestContext) -> dict:
        if stage not in STAGES:
            raise ProviderError("UNSUPPORTED_STAGE")
        results = {}
        for item in payload["items"]:
            text = item["text"]
            scope = choose_scope(item)
            if stage == "filter":
                domain = ("앱", "서비스", "로그인", "인증", "결제", "주문", "검색", "메뉴", "대기", "직원", "상품", "상담", "수수료", "고객센터", "안내창구", "카드")
                if any(w in text for w in domain):
                    result = {"decision": "keep", "evidence_quote": text}
                elif any(w in text for w in ("날씨", "축구", "테스트테스트")):
                    result = {"decision": "exclude", "evidence_quote": text}
                else:
                    result = {"decision": "abstain", "evidence_quote": text}
            elif stage == "aspect":
                matches = []
                for entry in item["candidates"]:
                    candidate_scope = choose_scope({"text": text, "keywords": entry["keywords"]})
                    if any(k in text for k in entry["keywords"]):
                        matches.append({"aspect_code": entry["aspect_code"], "evidence_quote": candidate_scope,
                                        "polarity": polarity(candidate_scope)})
                matches.sort(key=lambda m: (m["polarity"] != "negative", text.find(m["evidence_quote"])))
                result = {"matches": matches}
            elif stage == "emotion":
                result = {"emotion_code": reference_emotion(scope), "evidence_quote": scope}
            elif stage == "entity":
                product = next((w for w in ("모바일 앱", "고객센터", "로그인", "인증", "결제", "주문", "안내창구", "직원", "검색", "수수료", "앱", "상품") if w in scope), "")
                quality = next((w for w in ("짜증", "오류", "실패", "아쉽", "아쉬", "느리", "느려", "친절", "명확", "편리", "대기", "설명", "만족") if w in scope), "")
                result = {"product_term": product, "quality_term": quality,
                          "product_evidence": product, "quality_evidence": quality}
            elif stage == "reason":
                allowed = {"05", "06"} if item.get("reason_policy", "complaints") == "complaints" else {"04", "05", "06"}
                negative = [s for s in sentences(text) if reference_emotion(s) in allowed]
                if scope in negative:
                    negative = [scope] + [s for s in negative if s != scope]
                result = {"primary_reason": negative[0] if negative else "", "secondary_reasons": negative[1:]}
            else:
                result = {"summary": scope, "evidence_quote": scope}
            results[item["id"]] = result
        await asyncio.sleep(0)  # cooperative scheduling, no hidden I/O
        return {"results": results}


class NotConfiguredProvider:
    name = "not-configured"
    inference_kind = "not_configured"
    async def complete(self, stage: str, payload: dict, context: RequestContext) -> dict:
        raise ProviderError("NOT_CONFIGURED")


class JsonHttpProvider:
    """Opt-in JSON adapter with a bounded, explicit transport contract.

    A separately authorized service receives {stage,messages} and returns {results:{id:...}}.
    Endpoint/allowlist come only from server configuration, never from API requests.
    """
    inference_kind = "external_json_service_unverified_quality"

    def __init__(self, endpoint: str, *, enabled: bool, allowed_hosts: set[str], token: str | None = None,
                 name: str = "configured-json-service", transport: httpx.AsyncBaseTransport | None = None):
        if not enabled:
            raise ProviderError("EXTERNAL_PROVIDER_DISABLED")
        parsed = urlparse(endpoint)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.hostname not in allowed_hosts or parsed.fragment or parsed.query):
            raise ProviderError("INVALID_PROVIDER_ENDPOINT")
        self.endpoint, self._token, self.name, self._transport = endpoint, token, name, transport

    async def complete(self, stage: str, payload: dict, context: RequestContext) -> dict:
        messages = build_messages(stage, payload)
        headers = {"Content-Type": "application/json", "X-Request-Id": context.request_id}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            # Request-local immutable headers. Never mutate a shared client/agent.
            async with httpx.AsyncClient(timeout=30, follow_redirects=False, verify=True,
                                         trust_env=False, transport=self._transport) as client:
                async with client.stream("POST", self.endpoint, json={"stage": stage, "messages": messages}, headers=headers) as response:
                    if response.status_code in {408, 429} or response.status_code >= 500:
                        raise ProviderError("PROVIDER_TRANSIENT", retryable=True)
                    if response.status_code != 200:
                        raise ProviderError("PROVIDER_HTTP_REJECTED")
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > 1_048_576:
                            raise ProviderError("PROVIDER_RESPONSE_TOO_LARGE")
            return strict_json_loads(data.decode("utf-8"))
        except httpx.TimeoutException as exc:
            raise ProviderError("TIMEOUT", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError("TRANSPORT_ERROR", retryable=True) from exc
        except UnicodeDecodeError as exc:
            raise OutputValidationError() from exc


@dataclass
class Invocation:
    value: Any = None
    error_code: str | None = None
    attempts: int = 0
    duration_ms: float = 0
    provider: str = ""


class ProviderGateway:
    """Bounded calls, timeout per attempt, selective retries, item-level failure isolation."""
    def __init__(self, provider: AnalysisProvider, *, concurrency: int = 4, timeout: float = 10,
                 retry_limit: int = 2, fallback: AnalysisProvider | None = None):
        if fallback is provider:
            raise ValueError("FALLBACK_MUST_BE_INDEPENDENT")
        self.provider, self.fallback = provider, fallback
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout, self.retry_limit = timeout, retry_limit
        self.calls = 0

    async def call(self, stage: str, items: list[dict], context: RequestContext,
                   validate: Callable[[dict, dict], Any], **metadata: Any) -> dict[str, Invocation]:
        if not items:
            return {}
        if len({item["id"] for item in items}) != len(items):
            raise ValueError("DUPLICATE_REQUEST_ITEM_ID")
        started = time.perf_counter()
        pending = {item["id"]: item for item in items}
        outcomes = {key: Invocation() for key in pending}
        for attempt in range(self.retry_limit + 1):
            if not pending:
                break
            if attempt:
                await asyncio.sleep(min(0.05 * 2 ** (attempt - 1), 1.0))
            provider = self.fallback if attempt == self.retry_limit and attempt > 0 and self.fallback else self.provider
            for key in pending:
                outcomes[key].attempts += 1
                outcomes[key].provider = provider.name
            try:
                async with self.semaphore:
                    self.calls += 1
                    raw = await asyncio.wait_for(provider.complete(stage, redact_tree({"items": list(pending.values()), **metadata}), context), self.timeout)
                if not isinstance(raw, dict) or set(raw) != {"results"} or not isinstance(raw["results"], dict):
                    raise OutputValidationError()
                results = raw["results"]
                # Reject the entire batch before applying anything if cross-item IDs appear.
                if set(results) - set(pending):
                    raise OutputValidationError("UNEXPECTED_RESULT_ID")
                for key in list(pending):
                    if key not in results:
                        outcomes[key].error_code = "MISSING_RESULT"
                        continue
                    try:
                        outcomes[key].value = validate(results[key], pending[key])
                        outcomes[key].error_code = None
                        pending.pop(key)
                    except (ValueError, TypeError, KeyError, ProviderError):
                        outcomes[key].error_code = "INVALID_ITEM_RESPONSE"
            except asyncio.CancelledError:
                raise
            except (TimeoutError, ProviderError) as exc:
                code = "TIMEOUT" if isinstance(exc, TimeoutError) else exc.code
                for key in pending:
                    outcomes[key].error_code = code
                if isinstance(exc, ProviderError) and not exc.retryable:
                    break
            except Exception:
                # Internal error details may contain provider bodies/credentials. Do not log them.
                for key in pending:
                    outcomes[key].error_code = "PROVIDER_ERROR"
                break
        elapsed = (time.perf_counter() - started) * 1000
        for result in outcomes.values():
            result.duration_ms = elapsed
        return outcomes
