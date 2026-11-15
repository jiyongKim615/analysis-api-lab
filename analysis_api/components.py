"""Analysis stages sharing a validated provider contract.

Methods mutate and return the same request-local list; no cross-request shared VOC state.
"""
from __future__ import annotations

import asyncio
import itertools
import re
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import AspectInfo, AnalysisRequest, EntityInfo, RequestContext, StageResult, StageStatus, VocInfo
from .providers import Invocation, ProviderGateway
from .taxonomy import Taxonomy, emotion_info, get_prev_word_set_key
from .security import mask_pii


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FilterOutput(StrictOutput):
    decision: Literal["keep", "exclude", "abstain"]
    evidence_quote: str


class MatchOutput(StrictOutput):
    aspect_code: str = Field(min_length=1, max_length=100)
    evidence_quote: str
    polarity: Literal["negative", "positive", "neutral"]


class AspectOutput(StrictOutput):
    matches: list[MatchOutput] = Field(max_length=30)


class EmotionOutput(StrictOutput):
    emotion_code: str
    evidence_quote: str


class EntityOutput(StrictOutput):
    product_term: str = Field(max_length=100)
    quality_term: str = Field(max_length=100)
    product_evidence: str = Field(max_length=500)
    quality_evidence: str = Field(max_length=500)


class ReasonOutput(StrictOutput):
    primary_reason: str = Field(max_length=12000)
    secondary_reasons: list[str] = Field(default_factory=list, max_length=20)


class SummaryOutput(StrictOutput):
    summary: str = Field(min_length=1, max_length=12000)
    evidence_quote: str = Field(min_length=1, max_length=12000)


def require_evidence(quote: str, text: str) -> None:
    if not quote.strip() or quote not in text:
        raise ValueError("UNGROUNDED_EVIDENCE")


def stage_from(outcome: Invocation, status: StageStatus = StageStatus.SUCCESS, *, evidence: list[str] | None = None,
               code: str | None = None, rejected: list[str] | None = None) -> StageResult:
    if outcome.error_code:
        status = StageStatus.NOT_CONFIGURED if outcome.error_code == "NOT_CONFIGURED" else StageStatus.FAILED
    return StageResult(status=status, error_code=outcome.error_code or code, attempts=outcome.attempts,
                       duration_ms=outcome.duration_ms, provider=outcome.provider,
                       evidence=evidence or [], rejected_codes=rejected or [])


def item_for(voc: VocInfo, item_id: str, taxonomy: Taxonomy, aspect: AspectInfo | None = None) -> dict:
    candidates = taxonomy.candidates(voc.channel_code, voc.journey_code)
    definition = next((d for d in candidates if aspect and d.aspect_code == aspect.aspect_code), None)
    return {"id": item_id, "text": voc.text,
            "target_code": voc.channel_code,
            "stage_code": voc.journey_code,
            "stage_name": voc.journey_name if voc.journey_code != "00" else "",
            "prompt_branch": "aspect" if aspect and definition else "no_aspect",
            "aspect_code": aspect.aspect_code if definition else None,
            "keywords": definition.keywords if definition else [],
            "candidates": [d.model_dump() for d in candidates],
            "survey": {"question": voc.prompt_title,
                       "description": voc.prompt_description,
                       "answer": voc.prompt_context}}


class Component:
    def __init__(self, gateway: ProviderGateway, taxonomy: Taxonomy, request: AnalysisRequest, context: RequestContext):
        self.gateway, self.taxonomy, self.request, self.context = gateway, taxonomy, request, context

    async def chunks(self, stage: str, groups: list[list[dict]], validate: Any, **metadata: Any) -> dict[str, Invocation]:
        outcomes: dict[str, Invocation] = {}
        for offset in range(0, len(groups), self.request.batch_size):
            batch = groups[offset:offset + self.request.batch_size]
            results = await asyncio.gather(*(self.gateway.call(stage, chunk, self.context, validate, **metadata) for chunk in batch))
            for result in results:
                outcomes.update(result)
            if offset + self.request.batch_size < len(groups) and self.request.batch_sleep:
                await asyncio.sleep(self.request.batch_sleep)
        return outcomes

    def split(self, values: list[dict]) -> list[list[dict]]:
        n = self.request.voc_chunk_size
        return [values[i:i + n] for i in range(0, len(values), n)]


class VocFilter(Component):
    @staticmethod
    def _collapse_repeats(text: str) -> str:
        return "".join(k for k, _ in itertools.groupby(text))

    @staticmethod
    def _length_check(text: str, limit: int) -> bool:
        return len(text) > limit  # strict length boundary

    def filter_feedback_text(self, voc: VocInfo) -> bool:
        meaningful = re.sub(r"[^가-힣A-Za-z]", "", self._collapse_repeats(voc.text))
        return self._length_check(meaningful, self.request.voc_length_limit)

    @staticmethod
    def distinct_voc_list(vocs: list[VocInfo]) -> list[VocInfo]:
        seen = set()
        for voc in vocs:
            if voc.duplicate or voc.pk() in seen:
                voc.excluded, voc.filtered_reason = True, "DUPLICATE_PK"
            seen.add(voc.pk())
        return vocs

    @staticmethod
    def convert(vocs: list[VocInfo]) -> dict[str, dict[str, list[VocInfo]]]:
        grouped: dict = {}
        for voc in vocs:
            if voc.excluded:
                continue
            target = voc.channel_code
            grouped.setdefault(target, {"STAGE_EXIST": [], "STAGE_NOT_EXIST": []})
            stage = "STAGE_NOT_EXIST" if voc.journey_code == "00" else "STAGE_EXIST"
            grouped[target][stage].append(voc)
        return grouped

    async def execute(self, vocs: list[VocInfo]) -> list[VocInfo]:
        # Reset only this stage so re-execution cannot retain stale exclusions.
        for voc in vocs:
            voc.excluded, voc.filtered_reason = False, None
            voc.stages["filter"] = StageResult()
        self.distinct_voc_list(vocs)
        for voc in vocs:
            if not voc.excluded and not self.filter_feedback_text(voc):
                voc.excluded, voc.filtered_reason = True, "TEXT_FILTER"
            if voc.excluded:
                voc.stages["filter"] = StageResult(status=StageStatus.SUCCESS, error_code=voc.filtered_reason, provider="deterministic-pre-filter")
        refs, chunks = {}, []
        next_id = 0
        for target_group in self.convert(vocs).values():
            for records in target_group.values():
                items = []
                for voc in records:
                    key = f"f{next_id}"; next_id += 1
                    refs[key] = voc
                    item = item_for(voc, key, self.taxonomy)
                    item["prompt_branch"] = "stage" if voc.journey_code != "00" else "no_stage"
                    items.append(item)
                chunks.extend(self.split(items))
        def validate(raw: dict, item: dict) -> FilterOutput:
            out = FilterOutput.model_validate(raw)
            require_evidence(out.evidence_quote, item["text"])
            return out
        outcomes = await self.chunks("filter", chunks, validate)
        for key, outcome in outcomes.items():
            voc = refs[key]
            if outcome.error_code:
                voc.stages["filter"] = stage_from(outcome)
                continue
            out = outcome.value
            if out.decision == "exclude":
                voc.excluded, voc.filtered_reason = True, "CHANNEL_JOURNEY_UNRELATED"
            status = StageStatus.NEEDS_REVIEW if out.decision == "abstain" else StageStatus.SUCCESS
            voc.stages["filter"] = stage_from(outcome, status, evidence=[out.evidence_quote],
                                             code="RELEVANCE_UNCERTAIN" if out.decision == "abstain" else voc.filtered_reason)
        return vocs


class AspectMapper(Component):
    async def execute(self, vocs: list[VocInfo]) -> list[VocInfo]:
        refs, items = {}, []
        for i, voc in enumerate(vocs):
            voc.aspects, voc.aspect_success, voc.aspect_error = [], False, None
            item = item_for(voc, f"c{i}", self.taxonomy)
            if not item["candidates"]:
                voc.aspect_error = "NO_ASPECT_DICTIONARY"
                voc.stages["aspect"] = StageResult(status=StageStatus.NEEDS_REVIEW, error_code=voc.aspect_error)
            else:
                refs[item["id"]] = voc; items.append(item)
        def validate(raw: dict, item: dict) -> tuple[list[MatchOutput], list[str]]:
            output = AspectOutput.model_validate(raw)
            allowed = {d["aspect_code"] for d in item["candidates"]}
            accepted, rejected, seen = [], [], set()
            for match in output.matches:
                if match.aspect_code in seen:
                    raise ValueError("DUPLICATE_ASPECT_RESULT")
                seen.add(match.aspect_code)
                if match.aspect_code not in allowed:
                    rejected.append(mask_pii(match.aspect_code))
                    continue
                require_evidence(match.evidence_quote, item["text"])
                accepted.append(match)
            accepted.sort(key=lambda m: (m.polarity != "negative", item["text"].find(m.evidence_quote)))
            return accepted, rejected
        outcomes = await self.chunks("aspect", self.split(items), validate)
        for key, outcome in outcomes.items():
            voc = refs[key]
            if outcome.error_code:
                voc.aspect_error = outcome.error_code
                voc.stages["aspect"] = stage_from(outcome)
                continue
            accepted, rejected = outcome.value
            for seq, match in enumerate(accepted, 1):
                entry = self.taxonomy.lookup(voc.channel_code, voc.journey_code, match.aspect_code)
                voc.aspects.append(AspectInfo(seq=seq, quality_code=entry.quality_code, quality_name=mask_pii(entry.quality_name), aspect_code=entry.aspect_code,
                    aspect_name=mask_pii(entry.aspect_name), evidence_quote=match.evidence_quote, polarity=match.polarity))
            voc.aspect_success = bool(accepted)
            voc.aspect_error = "UNKNOWN_ASPECT" if rejected else None if accepted else "NOT_CLASSIFIED"
            status = StageStatus.PARTIAL_SUCCESS if accepted and rejected else StageStatus.SUCCESS if accepted else StageStatus.NEEDS_REVIEW
            voc.stages["aspect"] = stage_from(outcome, status, evidence=[m.evidence_quote for m in accepted], code=voc.aspect_error, rejected=rejected)
        return vocs


def combine_status(outcomes: list[Invocation], evidence: list[str]) -> StageResult:
    good = [o for o in outcomes if not o.error_code]
    first = outcomes[0]
    if not good:
        return stage_from(first)
    return StageResult(status=StageStatus.SUCCESS if len(good) == len(outcomes) else StageStatus.PARTIAL_SUCCESS,
                       error_code=None if len(good) == len(outcomes) else "PARTIAL_ITEM_FAILURE",
                       attempts=max(o.attempts for o in outcomes), duration_ms=max(o.duration_ms for o in outcomes),
                       provider=first.provider, evidence=evidence)


class EmotionAnalyzer(Component):
    async def execute(self, vocs: list[VocInfo]) -> list[VocInfo]:
        refs, items = {}, []
        for i, voc in enumerate(vocs):
            voc.emotion, voc.emotion_success, voc.emotion_error = None, False, None
            for j, aspect in enumerate(voc.aspects or [None]):
                if aspect:
                    aspect.emotion = None
                key = f"e{i}_{j}"
                refs[key] = (voc, aspect)
                items.append(item_for(voc, key, self.taxonomy, aspect))
        def validate(raw: dict, item: dict):
            out = EmotionOutput.model_validate(raw)
            require_evidence(out.evidence_quote, item["text"])
            return emotion_info(out.emotion_code, out.evidence_quote, item["aspect_code"])
        outcomes = await self.chunks("emotion", self.split(items), validate)
        by_voc = defaultdict(list)
        for key, outcome in outcomes.items():
            voc, aspect = refs[key]
            by_voc[id(voc)].append(outcome)
            if outcome.error_code:
                continue
            if aspect:
                aspect.emotion = outcome.value
            if voc.emotion is None:
                voc.emotion = outcome.value  # backward-facing representative; per-ASPECT result also retained
        for voc in vocs:
            group = by_voc[id(voc)]
            voc.emotion_success = voc.emotion is not None
            voc.stages["emotion"] = combine_status(group, [o.value.evidence_quote for o in group if not o.error_code])
            voc.emotion_error = voc.stages["emotion"].error_code
        return vocs


class EntityWordDetector(Component):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: dict[tuple, EntityInfo] = {}
        self._terms: dict[str, list[str]] = {}

    async def execute(self, vocs: list[VocInfo]) -> list[VocInfo]:
        refs, grouped = {}, defaultdict(list)
        for i, voc in enumerate(vocs):
            voc.entity_result, voc.entity_success = None, False
            for j, aspect in enumerate(voc.aspects or [None]):
                if aspect:
                    aspect.entity, aspect.entity_success = None, False
                key = f"w{i}_{j}"
                item = item_for(voc, key, self.taxonomy, aspect)
                scope = get_prev_word_set_key(voc.channel_code, aspect.aspect_code if aspect else None)
                refs[key] = (voc, aspect, scope)
                item["previous_terms"] = self._terms.get(scope, [])[-self.request.prev_word_input_size:] if self.request.prev_word_input_size else []
                grouped[scope].append(item)
        def validate(raw: dict, item: dict) -> EntityInfo:
            out = EntityOutput.model_validate(raw)
            # Normalized synonym claims are not accepted without a versioned approved synonym lexicon.
            for term, quote in ((out.product_term, out.product_evidence), (out.quality_term, out.quality_evidence)):
                if term:
                    require_evidence(quote, item["text"])
                    if term not in quote:
                        raise ValueError("UNGROUNDED_ENTITY")
                elif quote:
                    raise ValueError("EVIDENCE_WITHOUT_ENTITY")
            return EntityInfo(**out.model_dump())
        chunks = [chunk for values in grouped.values() for chunk in self.split(values)]
        outcomes: dict[str, Invocation] = {}
        for chunk in chunks:
            pending = []
            for item in chunk:
                voc, aspect, scope = refs[item["id"]]
                cache_key = (voc.pk(), voc.journey_code, self.taxonomy.version, scope, voc.source_hash, voc.text)
                cached = self._cache.get(cache_key)
                if cached:
                    copy = cached.model_copy(deep=True); copy.reused = True
                    outcomes[item["id"]] = Invocation(value=copy, provider="validated-request-local-cache")
                else:
                    item["previous_terms"] = self._terms.get(scope, [])[-self.request.prev_word_input_size:] if self.request.prev_word_input_size else []
                    pending.append(item)
            result = await self.gateway.call("entity", pending, self.context, validate)
            outcomes.update(result)
            for item in pending:
                outcome = result[item["id"]]
                if not outcome.error_code:
                    voc, aspect, scope = refs[item["id"]]
                    self._cache[(voc.pk(), voc.journey_code, self.taxonomy.version, scope, voc.source_hash, voc.text)] = outcome.value.model_copy(deep=True)
                    for term in (outcome.value.product_term, outcome.value.quality_term):
                        if term and term not in self._terms.setdefault(scope, []):
                            self._terms[scope].append(term)
        by_voc = defaultdict(list)
        for key, outcome in outcomes.items():
            voc, aspect, _ = refs[key]
            by_voc[id(voc)].append(outcome)
            if outcome.error_code:
                continue
            entity = outcome.value
            has_value = bool(entity.product_term or entity.quality_term)
            if aspect:
                aspect.entity, aspect.entity_success = entity, has_value
            if voc.entity_result is None or (has_value and not voc.entity_success):
                voc.entity_result = entity
            voc.entity_success = voc.entity_success or has_value
        for voc in vocs:
            group = by_voc[id(voc)]
            voc.stages["entity"] = combine_status(group, [e for o in group if not o.error_code for e in (o.value.product_evidence, o.value.quality_evidence) if e])
            if all(o.error_code or not (o.value.product_term or o.value.quality_term) for o in group) and any(not o.error_code for o in group):
                voc.stages["entity"].status = StageStatus.NEEDS_REVIEW
                voc.stages["entity"].error_code = "NO_ENTITY_FOUND"
        return vocs


class VocProblemReasonDetector(Component):
    async def execute(self, vocs: list[VocInfo]) -> list[VocInfo]:
        refs, items = {}, []
        for i, voc in enumerate(vocs):
            voc.reason_success, voc.primary_reason, voc.secondary_reasons = False, None, []
            def policy_matches(emotion):
                return emotion is not None and (emotion.feedback_type == "02" if self.request.reason_policy == "complaints" else emotion.polarity_code == "02")
            matched = [aspect for aspect in voc.aspects if policy_matches(aspect.emotion)]
            allowed = bool(matched) or policy_matches(voc.emotion)
            if not allowed:
                voc.stages["reason"] = StageResult(status=StageStatus.SKIPPED, error_code="EMOTION_UNAVAILABLE" if not voc.emotion else "REASON_POLICY_NOT_MATCHED")
                continue
            key = f"r{i}"; refs[key] = voc
            selected = matched[0] if matched else (voc.aspects[0] if voc.aspects else None)
            item = item_for(voc, key, self.taxonomy, selected)
            item["reason_policy"] = self.request.reason_policy
            items.append(item)
        def validate(raw: dict, item: dict) -> ReasonOutput:
            out = ReasonOutput.model_validate(raw)
            require_evidence(out.primary_reason, item["text"])
            for reason in out.secondary_reasons:
                require_evidence(reason, item["text"])
            if len(set([out.primary_reason, *out.secondary_reasons])) != 1 + len(out.secondary_reasons):
                raise ValueError("DUPLICATE_REASON")
            return out
        outcomes = await self.chunks("reason", self.split(items), validate)
        for key, outcome in outcomes.items():
            voc = refs[key]
            if not outcome.error_code:
                voc.reason_success = True
                voc.primary_reason = outcome.value.primary_reason
                voc.secondary_reasons = outcome.value.secondary_reasons
            voc.stages["reason"] = stage_from(outcome, evidence=[voc.primary_reason] if voc.primary_reason else [])
        return vocs


class VocSummaryAgent(Component):
    async def execute(self, vocs: list[VocInfo]) -> list[VocInfo]:
        refs = {f"s{i}": voc for i, voc in enumerate(vocs)}
        items = [item_for(voc, key, self.taxonomy, voc.aspects[0] if voc.aspects else None) for key, voc in refs.items()]
        def validate(raw: dict, item: dict) -> SummaryOutput:
            out = SummaryOutput.model_validate(raw)
            require_evidence(out.evidence_quote, item["text"])
            if out.summary != out.evidence_quote:
                raise ValueError("REFERENCE_SUMMARY_MUST_BE_EXTRACTIVE")
            return out
        outcomes = await self.chunks("summary", self.split(items), validate)
        for key, outcome in outcomes.items():
            voc = refs[key]
            voc.summary = outcome.value.summary if not outcome.error_code else None
            voc.stages["summary"] = stage_from(outcome, evidence=[voc.summary] if voc.summary else [])
        return vocs


# Alternate spelling delegates to the same implementation.
VocSummerizer = VocSummaryAgent
VocSummarizer = VocSummaryAgent
