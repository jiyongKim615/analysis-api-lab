"""Small illustrative codebook and a synthetic aspect catalog."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from pydantic import Field, field_validator

from .models import CamelModel, EmotionInfo, ReviewVisibility, VocInput

NO_CX_STAGE_CD = "00"
NO_ASPECT_ID = "9999"
NEGATIVE_EMOTION_CD = "02"
# Codes and labels are examples for local exercises.
EMOTION_CODEBOOK = {
    "01": ("감동", "01", "긍정", "01", "칭찬"),
    "02": ("만족", "01", "긍정", "01", "칭찬"),
    "03": ("기대", "01", "긍정", "03", "개선"),
    "04": ("아쉬움", "02", "부정", "03", "개선"),
    "05": ("실망", "02", "부정", "02", "불만"),
    "06": ("짜증", "02", "부정", "02", "불만"),
    "07": ("감정 없음", "03", "중립", "04", "기타"),
}


def emotion_info(code: str, evidence: str, aspect_code: str | None = None) -> EmotionInfo:
    if code not in EMOTION_CODEBOOK:
        raise ValueError("UNKNOWN_EMOTION")
    mid, lag, lag_name, typ, typ_name = EMOTION_CODEBOOK[code]
    return EmotionInfo(emotion_code=code, emotion_name=mid, polarity_code=lag, polarity_name=lag_name,
                       feedback_type=typ, feedback_label=typ_name, evidence_quote=evidence, aspect_code=aspect_code)


class AspectDefinition(CamelModel):
    target_code: str
    stage_code: str
    stage_name: str = ""
    aspect_code: str
    aspect_name: str
    description: str
    quality_code: str
    quality_name: str
    keywords: list[str] = Field(min_length=1)

    @field_validator("target_code", "stage_code", "aspect_code", "aspect_name", "description", "quality_code", "quality_name")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("EMPTY_ASPECT_DEFINITION_FIELD")
        if value != value.strip():
            raise ValueError("NONCANONICAL_ASPECT_FIELD")
        return value

    @field_validator("aspect_code")
    @classmethod
    def reject_sentinel(cls, value: str) -> str:
        if value == NO_ASPECT_ID:
            raise ValueError("NO_MATCH_SENTINEL_IS_NOT_A_CANDIDATE")
        return value

    @field_validator("keywords")
    @classmethod
    def valid_keywords(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or value != value.strip() for value in values):
            raise ValueError("EMPTY_OR_NONCANONICAL_KEYWORD")
        if len(set(values)) != len(values):
            raise ValueError("DUPLICATE_KEYWORD")
        return values


class Taxonomy:
    def __init__(self, entries: Iterable[AspectDefinition], version: str = "synthetic-aspect-v1"):
        self.version = version
        self.entries = tuple(entries)
        seen: dict[tuple[str, str], AspectDefinition] = {}
        exact: set[tuple[str, str, str]] = set()
        for entry in self.entries:
            key = (entry.target_code, entry.aspect_code)
            triple = (*key, entry.stage_code)
            if triple in exact:
                raise ValueError("DUPLICATE_ASPECT_DEFINITION")
            exact.add(triple)
            if key in seen:
                a = seen[key].model_dump(exclude={"stage_code", "stage_name"})
                b = entry.model_dump(exclude={"stage_code", "stage_name"})
                if a != b:
                    raise ValueError("CONFLICTING_ASPECT_DEFINITION")
            seen[key] = entry

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([AspectDefinition.model_validate(x) for x in payload["entries"]], payload["version"])

    def candidates(self, target: str, stage: str) -> list[AspectDefinition]:
        values = [e for e in self.entries if e.target_code == target and (stage == NO_CX_STAGE_CD or e.stage_code == stage)]
        return list({e.aspect_code: e for e in values}.values())

    def lookup(self, target: str, stage: str, code: str) -> AspectDefinition | None:
        return next((e for e in self.candidates(target, stage) if e.aspect_code == code), None)


def get_prev_word_set_key(target: str, aspect: str | None) -> str:
    return f"{target}_{aspect or NO_ASPECT_ID}"


def review_visibility(voc: VocInput, exception_words: Iterable[str] = ()) -> ReviewVisibility:
    """Illustrative visibility rule; always requires review and grants no access."""
    length = len("".join(voc.text.split()))
    manager = employee = True
    reasons: list[str] = []
    if length <= 20:
        manager = employee = False
        reasons.append("WORD_LENGTH_20")
    elif length <= 40:
        manager = employee = False
        reasons.append("WORD_LENGTH_40")
    if voc.channel_code in {"07", "08"}:
        if any(word.strip() and word.strip() in voc.text for word in exception_words):
            manager = employee = False
            reasons.append("EXCEPT_WORD_INCLUDE")
        if voc.journey_code in {"04", "15", "06"}:
            employee = False
            reasons.append("JOURNEY_OPERATOR_HIDDEN")  # Keep the reason separate from the decision.
    return ReviewVisibility(reviewer_visible=manager, operator_visible=employee, reasons=reasons,
                         policy_version="synthetic-visibility-v1", review_required=True)
