"""Typed contracts for synthetic feedback, analysis stages and run records."""
from __future__ import annotations

from datetime import date as Date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid", validate_assignment=True)


class StageStatus(str, Enum):
    NOT_RUN = "not_run"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    NOT_CONFIGURED = "not_configured"


class StageResult(CamelModel):
    status: StageStatus = StageStatus.NOT_RUN
    error_code: str | None = None
    attempts: int = 0
    duration_ms: float = 0
    provider: str | None = None
    prompt_version: str = "integrated-json-v1"
    evidence: list[str] = Field(default_factory=list)
    rejected_codes: list[str] = Field(default_factory=list)


class RequestContext(CamelModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    actor_ref: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")


class VocInput(CamelModel):
    # Synthetic records may optionally use a complete composite key.
    # If any source-key field is supplied, all four plus recorded_on are required.
    voc_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")
    source_system: str = Field(default="SYNTHETIC", min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    recorded_on: Date = Date(2026, 10, 29)
    dataset_id: str | None = Field(default=None, max_length=40)
    survey_id: str | None = Field(default=None, max_length=100)
    response_id: str | None = Field(default=None, max_length=100)
    question_id: str | None = Field(default=None, max_length=100)
    survey_mode: str | None = Field(default=None, max_length=32)
    survey_kind: str | None = Field(default=None, max_length=32)
    service_code: str | None = Field(default=None, max_length=32)
    intent_code: str | None = Field(default=None, max_length=32)
    quality_group: str | None = Field(default=None, max_length=32)
    text: str = Field(max_length=12000, validation_alias=AliasChoices("text", "feedback_text", "feedbackText"))
    channel_code: str = Field(default="02", min_length=1, max_length=32)
    channel_name: str = Field(default="합성 디지털 채널", max_length=200)
    journey_code: str = Field(default="00", max_length=32)
    journey_name: str = Field(default="", max_length=200)
    prompt_title: str = Field(default="", max_length=1000)
    prompt_description: str = Field(default="", max_length=1000)
    prompt_context: str = Field(default="", max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def require_source_date(cls, value: Any) -> Any:
        if isinstance(value, dict):
            source_fields = ("dataset_id", "datasetId", "survey_id", "surveyId", "response_id", "responseId", "question_id", "questionId")
            if any(value.get(k) is not None for k in source_fields) and "recorded_on" not in value and "recordedOn" not in value:
                raise ValueError("source composite key requires explicit recorded_on")
        return value

    @field_validator("journey_code", mode="before")
    @classmethod
    def normalize_stage(cls, value: Any) -> str:
        return str(value).strip() if value is not None and str(value).strip() else "00"

    @model_validator(mode="after")
    def clear_stale_stage_name(self) -> "VocInput":
        if self.journey_code == "00":
            object.__setattr__(self, "journey_name", "")
        return self

    @field_validator("recorded_on", mode="before")
    @classmethod
    def parse_ymd(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) == 8 and value.isdigit():
            return Date(int(value[:4]), int(value[4:6]), int(value[6:]))
        return value

    @model_validator(mode="after")
    def validate_source_key(self) -> "VocInput":
        fields = (self.dataset_id, self.survey_id, self.response_id, self.question_id)
        if any(v is not None for v in fields) and not all(isinstance(v, str) and v.strip() for v in fields):
            raise ValueError("source composite key must be complete")
        return self

    def get_pk_dict(self) -> dict[str, str]:
        if self.dataset_id is not None:
            return {"dataset_id": self.dataset_id, "recorded_on": self.recorded_on.isoformat(),
                    "survey_id": self.survey_id, "response_id": self.response_id,
                    "question_id": self.question_id}
        return {"source_system": self.source_system, "voc_id": self.voc_id}

    def pk(self) -> tuple[str, ...]:
        return tuple(self.get_pk_dict().values())


class EmotionInfo(CamelModel):
    emotion_code: str
    emotion_name: str
    polarity_code: str | None = None
    polarity_name: Literal["긍정", "부정", "중립"]
    feedback_type: str | None = None
    feedback_label: Literal["칭찬", "개선", "불만", "기타"]
    evidence_quote: str
    aspect_code: str | None = None


class EntityInfo(CamelModel):
    product_term: str = ""
    quality_term: str = Field(default="", validation_alias=AliasChoices("quality_term", "qualityTerm", "quality_word", "qualityWord"))
    product_evidence: str = ""
    quality_evidence: str = ""
    reused: bool = False


class AspectInfo(CamelModel):
    seq: int
    quality_code: str
    quality_name: str
    aspect_code: str
    aspect_name: str
    evidence_quote: str
    polarity: Literal["negative", "positive", "neutral"] = "neutral"
    emotion: EmotionInfo | None = None
    entity_success: bool = False
    entity: EntityInfo | None = None


class ReviewVisibility(CamelModel):
    reviewer_visible: bool | None = None
    operator_visible: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    policy_version: str = "partial-evidence-v1"
    review_required: bool = True


class VocInfo(VocInput):
    source_hash: str = ""
    duplicate: bool = False
    excluded: bool = False
    filtered_reason: str | None = None
    aspect_success: bool = False
    aspect_error: str | None = None
    aspects: list[AspectInfo] = Field(default_factory=list)
    emotion_success: bool = False
    emotion_error: str | None = None
    emotion: EmotionInfo | None = None
    entity_success: bool = False
    entity_result: EntityInfo | None = None
    reason_success: bool = False
    primary_reason: str | None = None
    secondary_reasons: list[str] = Field(default_factory=list)
    summary: str | None = None
    offer: ReviewVisibility = Field(default_factory=ReviewVisibility)
    stages: dict[str, StageResult] = Field(default_factory=lambda: {s: StageResult() for s in ("filter", "aspect", "emotion", "entity", "reason", "summary")})


class AnalysisRequest(CamelModel):
    agent_id: str = Field(default="cx-analysis-reference-v1", max_length=80)
    page_number: int = Field(default=0, ge=0, le=100000)
    page_size: int = Field(default=500, ge=1, le=500)
    date: Date | None = None
    date_end: Date | None = None
    target_codes: list[str] = Field(default_factory=list, max_length=20)
    survey_modes: list[str] = Field(default_factory=list, max_length=20)
    vocs: list[VocInput] | None = Field(default=None, max_length=500)
    batch_size: int = Field(default=150, ge=1, le=500)
    voc_chunk_size: int = Field(default=50, ge=1, le=100)
    batch_sleep: float = Field(default=0.0, ge=0, le=5)
    retry_limit: int = Field(default=2, ge=0, le=3)
    concurrency: int = Field(default=4, ge=1, le=16)
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    voc_length_limit: int = Field(default=10, ge=0, le=100)
    prev_word_input_size: int = Field(default=15, ge=0, le=100)
    include_summary: bool = True
    reason_policy: Literal["complaints", "negative_emotion"] = "complaints"

    @model_validator(mode="after")
    def validate_dates_and_inline(self) -> "AnalysisRequest":
        if self.date_end is not None and self.date is None:
            raise ValueError("date_end requires date")
        if self.date and self.date_end and self.date_end < self.date:
            raise ValueError("date_end precedes date")
        if self.vocs is not None and (self.page_number != 0 or self.date is not None or self.target_codes or self.survey_modes):
            raise ValueError("inline vocs cannot be combined with repository filters or a nonzero page")
        if self.vocs is not None and len(self.vocs) > self.page_size:
            raise ValueError("inline voc count exceeds page_size; no silent truncation")
        return self


class PageResult(CamelModel):
    page_number: int
    page_size: int
    total_elements: int
    total_pages: int
    number_of_elements: int
    first_yn: bool
    last_yn: bool
    content: list[VocInfo]


class RunMetadata(CamelModel):
    run_id: str
    request_id: str
    created_at: str
    elapsed_ms: float
    engine: str
    model_version: str
    taxonomy_version: str
    prompt_version: str = "integrated-json-v1"
    code_version: str = "1.0.0"
    evidence_basis: str = "hand-authored-synthetic-fixtures-v1"
    inference_kind: str
    privacy_mode: str = "redacted_input_and_output; limited_regex_not_full_anonymization"
    reason_gate: str = "complaints"
    metrics: dict[str, Any]


class AnalysisResponse(CamelModel):
    metadata: RunMetadata
    page: PageResult


class RunAgentRequest(CamelModel):
    request_type: str = "analysis"
    analysis: AnalysisRequest = Field(default_factory=AnalysisRequest)
    is_stream: bool = False


class SerializedRequest(CamelModel):
    input_value: str = Field(max_length=2_000_000)
