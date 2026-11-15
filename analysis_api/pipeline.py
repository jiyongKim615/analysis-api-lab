"""One request-scoped orchestration for API, CLI and tests."""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Awaitable, Callable

from .components import AspectMapper, EmotionAnalyzer, EntityWordDetector, VocFilter, VocProblemReasonDetector, VocSummaryAgent
from .models import AnalysisRequest, AnalysisResponse, RunMetadata, PageResult, RequestContext, StageResult, StageStatus, VocInfo, utc_now
from .observability import summarize
from .providers import AnalysisProvider, OfflineReferenceProvider, ProviderGateway
from .repository import VocGatherer, VocRepository
from .security import redact_record, redact_tree, use_context
from .taxonomy import Taxonomy, review_visibility

EventSink = Callable[[str, dict], Awaitable[None]]


class AnalysisPipeline:
    def __init__(self, repository: VocRepository, taxonomy: Taxonomy, provider: AnalysisProvider | None = None,
                 reference_data: dict | None = None):
        self.repository, self.taxonomy = repository, taxonomy
        self.provider = provider or OfflineReferenceProvider()
        self.reference_data = reference_data

    async def execute(self, request: AnalysisRequest, context: RequestContext, on_event: EventSink | None = None) -> AnalysisResponse:
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        with use_context(context):
            gathered = await asyncio.wait_for(VocGatherer(self.repository, self.taxonomy, self.reference_data).execute(request), timeout=15)
            records: list[VocInfo] = []
            for source in gathered.voc_list:
                clean, source_hash = redact_record(source)
                voc = VocInfo.model_validate(clean.model_dump())
                voc.source_hash = source_hash
                voc.duplicate = source.duplicate
                # Apply source-text length policy before masking changes length, keep no raw text.
                voc.offer = review_visibility(source, gathered.exception_words)
                records.append(voc)
            gateway = ProviderGateway(self.provider, concurrency=request.concurrency,
                                      timeout=request.timeout_seconds, retry_limit=request.retry_limit)
            args = (gateway, self.taxonomy, request, context)
            await VocFilter(*args).execute(records)
            if on_event:
                await on_event("STAGE", {"stage": "filter", "metrics": summarize(records, gateway.calls)["stages"]["filter"]})
            eligible = []
            for voc in records:
                if not voc.excluded and voc.stages["filter"].status == StageStatus.SUCCESS:
                    eligible.append(voc)
                else:
                    for name in ("aspect", "emotion", "entity", "reason", "summary"):
                        voc.stages[name] = StageResult(status=StageStatus.SKIPPED,
                            error_code="FILTER_EXCLUDED" if voc.excluded else "FILTER_UNVALIDATED")
            # Run stages sequentially; bound calls within each stage.
            components = [("aspect", AspectMapper(*args)), ("emotion", EmotionAnalyzer(*args)),
                          ("entity", EntityWordDetector(*args)), ("reason", VocProblemReasonDetector(*args))]
            # Historical previous-term dictionaries are context only, not trusted predictions.
            entity = components[2][1]
            for key in sorted(set(gathered.prior_product_terms) | set(gathered.prior_quality_terms)):
                entity._terms[key] = list(dict.fromkeys(gathered.prior_product_terms.get(key, []) + gathered.prior_quality_terms.get(key, [])))
            if request.include_summary:
                components.append(("summary", VocSummaryAgent(*args)))
            else:
                for voc in eligible:
                    voc.stages["summary"] = StageResult(status=StageStatus.SKIPPED, error_code="SUMMARY_DISABLED")
            for name, component in components:
                if eligible:
                    await component.execute(eligible)
                if on_event:
                    await on_event("STAGE", {"stage": name, "metrics": summarize(records, gateway.calls)["stages"][name]})
            count = len(records)
            metadata = RunMetadata(run_id=run_id, request_id=context.request_id, created_at=utc_now(),
                elapsed_ms=(time.perf_counter() - started) * 1000, engine=self.provider.name,
                model_version=self.provider.name, taxonomy_version=self.taxonomy.version,
                inference_kind=self.provider.inference_kind, reason_gate=request.reason_policy,
                metrics={**summarize(records, gateway.calls), "repositoryId": gathered.repository_id,
                         "referenceVersion": gathered.reference_version})
            return AnalysisResponse(metadata=metadata, page=PageResult(page_number=request.page_number,
                page_size=request.page_size, total_elements=gathered.total_count,
                total_pages=math.ceil(gathered.total_count / request.page_size), number_of_elements=count,
                first_yn=request.page_number == 0,
                last_yn=(request.page_number + 1) * request.page_size >= gathered.total_count,
                content=records))

    async def execute_unit(self, function_name: str, vocs: list[VocInfo], request: AnalysisRequest,
                           context: RequestContext) -> list[VocInfo]:
        """Run one selected stage only; caller supplies any prerequisite analysis state.

        Unlike the full pipeline this never invents missing prerequisite outputs.
        Returns request-local copies; individual component.execute keeps in-place semantics.
        """
        factories = {"voc_filtering": VocFilter, "aspect_mapping": AspectMapper,
                     "emotion_analysis": EmotionAnalyzer, "entity_word_detect": EntityWordDetector,
                     "voc_problem_reason_detect": VocProblemReasonDetector}
        if function_name not in factories:
            raise ValueError("UNSUPPORTED_UNIT_FUNCTION")
        records = []
        for source in vocs:
            clean, digest = redact_record(source)
            payload = source.model_dump(mode="json", by_alias=True)
            identity_fields = {"vocId", "sourceSystem", "sourceHash", "datasetId", "surveyId", "responseId", "questionId"}
            payload = {key: value if key in identity_fields else redact_tree(value) for key, value in payload.items()}
            copy = VocInfo.model_validate(payload)
            for key in type(clean).model_fields:
                setattr(copy, key, getattr(clean, key))
            copy.source_hash = digest
            records.append(copy)
        gateway = ProviderGateway(self.provider, concurrency=request.concurrency,
                                  timeout=request.timeout_seconds, retry_limit=request.retry_limit)
        with use_context(context):
            return await factories[function_name](gateway, self.taxonomy, request, context).execute(records)

    async def execute_data_anlysis(self, request: AnalysisRequest, context: RequestContext) -> AnalysisResponse:
        """Compatibility spelling dispatches to the same implementation."""
        return await self.execute(request, context)
