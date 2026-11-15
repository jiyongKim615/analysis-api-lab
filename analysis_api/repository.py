"""Read-only, bounded access to synthetic JSON and SQLite fixtures."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import quote

from pydantic import ValidationError

from .models import AnalysisRequest, EmotionInfo, VocInfo, VocInput
from .taxonomy import EMOTION_CODEBOOK, Taxonomy, emotion_info


class RepositoryError(RuntimeError):
    """Safe, stable error code; exception messages never contain VOC text or SQL."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(slots=True)
class GatheredContext:
    voc_list: list[VocInfo]
    total_count: int
    repository_id: str
    channel_aspects: dict[str, dict[str, dict[str, dict[str, Any]]]] = field(default_factory=dict)
    journey_names: dict[str, dict[str, str]] = field(default_factory=dict)
    emotion_dict: dict[str, EmotionInfo] = field(default_factory=dict)
    prior_product_terms: dict[str, list[str]] = field(default_factory=dict)
    prior_quality_terms: dict[str, list[str]] = field(default_factory=dict)
    exception_words: list[str] = field(default_factory=list)
    reference_version: str = "not-loaded"


class VocRepository(Protocol):
    async def page(self, request: AnalysisRequest) -> GatheredContext: ...


def _input_copy(record: VocInput | Mapping[str, Any]) -> VocInput:
    try:
        if isinstance(record, VocInput):
            # Drop derived analysis fields if a VocInfo was provided; never reuse
            # another request's processing state or mutable nested lists.
            payload = record.model_dump(include=set(VocInput.model_fields))
        elif isinstance(record, Mapping):
            payload = dict(record)
        else:
            raise TypeError("not a record")
        return VocInput.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise RepositoryError("INVALID_SOURCE_RECORD") from exc


def _fresh_voc(record: VocInput) -> VocInfo:
    return VocInfo.model_validate(record.model_dump(mode="json"))


def _materialize_snapshot(records: Sequence[VocInput], offset: int = 0, limit: int | None = None) -> list[VocInfo]:
    """Mark later logical duplicates before slicing, including across page boundaries."""
    seen: set[tuple[str, ...]] = set()
    result = []
    end = len(records) if limit is None else offset + limit
    for index, record in enumerate(records):
        if offset <= index < end:
            voc = _fresh_voc(record)
            voc.duplicate = record.pk() in seen
            result.append(voc)
        seen.add(record.pk())
    return result


class JSONFixtureRepository:
    """Immutable validated snapshot; file order is the deterministic row order."""

    def __init__(self, source: str | Path | Sequence[VocInput | Mapping[str, Any]]):
        if isinstance(source, (str, Path)):
            path = Path(source)
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise RepositoryError("FIXTURE_TOO_LARGE")
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RepositoryError("INVALID_FIXTURE") from exc
        else:
            data = source
        if not isinstance(data, (list, tuple)) or len(data) > 50000:
            raise RepositoryError("INVALID_FIXTURE")
        self._records = tuple(_input_copy(row) for row in data)

    async def page(self, request: AnalysisRequest) -> GatheredContext:
        if request.vocs is not None:
            raise RepositoryError("INLINE_REQUIRES_GATHERER")
        end = request.date_end or request.date
        targets = frozenset(request.target_codes)
        survey_modes = frozenset(request.survey_modes)
        matched = [
            row for row in self._records
            if (request.date is None or row.recorded_on >= request.date)
            and (end is None or row.recorded_on <= end)
            and (not targets or row.channel_code in targets)
            and (not survey_modes or row.survey_mode in survey_modes)
        ]
        offset = request.page_number * request.page_size
        return GatheredContext(
            _materialize_snapshot(matched, offset, request.page_size),
            len(matched),
            "synthetic-json-snapshot-v1",
        )


# Conventional spelling accepted without maintaining a second implementation.
JsonFixtureRepository = JSONFixtureRepository


_TARGET_PLACEHOLDERS = ",".join(f":target_{i}" for i in range(20))
_SURVEY_MODE_PLACEHOLDERS = ",".join(f":survey_mode_{i}" for i in range(20))
_FILTER_SQL = f"""
 WHERE (:date_from IS NULL OR recorded_on >= :date_from)
   AND (:date_to IS NULL OR recorded_on <= :date_to)
   AND (:all_targets = 1 OR target_code IN ({_TARGET_PLACEHOLDERS}))
   AND (:all_survey_modes = 1 OR COALESCE(json_extract(payload, '$.survey_mode'), json_extract(payload, '$.surveyMode')) IN ({_SURVEY_MODE_PLACEHOLDERS}))
"""
# Only constant application-owned SQL is executable. Values are always bound.
_QUERIES = {
    "voc_count": "SELECT COUNT(*) AS total_count FROM demo_voc" + _FILTER_SQL,
    "voc_page": "SELECT row_id, recorded_on, target_code, payload FROM demo_voc" + _FILTER_SQL
                + " ORDER BY row_id ASC LIMIT :limit OFFSET :offset",
    "voc_snapshot": "SELECT row_id, recorded_on, target_code, payload FROM demo_voc" + _FILTER_SQL
                    + " ORDER BY row_id ASC LIMIT 50001",
}
_REQUIRED_PARAMS = {"date_from", "date_to", "all_targets", "all_survey_modes", "limit", "offset"} | {
    f"target_{i}" for i in range(20)
} | {f"survey_mode_{i}" for i in range(20)}


def _bound_parameters(request: AnalysisRequest) -> dict[str, Any]:
    params: dict[str, Any] = {
        "date_from": request.date.isoformat() if request.date else None,
        "date_to": (request.date_end or request.date).isoformat() if request.date else None,
        "all_targets": int(not request.target_codes),
        "all_survey_modes": int(not request.survey_modes),
        "limit": request.page_size,
        "offset": request.page_number * request.page_size,
    }
    params.update({f"target_{i}": request.target_codes[i] if i < len(request.target_codes) else None for i in range(20)})
    params.update({f"survey_mode_{i}": request.survey_modes[i] if i < len(request.survey_modes) else None for i in range(20)})
    return params


def _validate_parameters(params: Mapping[str, Any]) -> None:
    if set(params) != _REQUIRED_PARAMS:
        raise RepositoryError("INVALID_QUERY_PARAMETERS")
    if type(params["limit"]) is not int or not 1 <= params["limit"] <= 500:
        raise RepositoryError("INVALID_QUERY_PARAMETERS")
    if type(params["offset"]) is not int or not 0 <= params["offset"] <= 50_000_000:
        raise RepositoryError("INVALID_QUERY_PARAMETERS")
    for flag in ("all_targets", "all_survey_modes"):
        if type(params[flag]) is not int or params[flag] not in (0, 1):
            raise RepositoryError("INVALID_QUERY_PARAMETERS")
    # AnalysisRequest validates dates for normal callers. The tool boundary
    # validates them again because it can be called independently.
    from datetime import date
    for field in ("date_from", "date_to"):
        value = params[field]
        if value is not None:
            if not isinstance(value, str):
                raise RepositoryError("INVALID_QUERY_PARAMETERS")
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError("noncanonical")
            except ValueError as exc:
                raise RepositoryError("INVALID_QUERY_PARAMETERS") from exc
    if params["date_from"] is None and params["date_to"] is not None:
        raise RepositoryError("INVALID_QUERY_PARAMETERS")
    if params["date_from"] and params["date_to"] and params["date_from"] > params["date_to"]:
        raise RepositoryError("INVALID_QUERY_PARAMETERS")
    for prefix in ("target", "survey_mode"):
        for i in range(20):
            value = params[f"{prefix}_{i}"]
            if value is not None and (not isinstance(value, str) or len(value) > 128):
                raise RepositoryError("INVALID_QUERY_PARAMETERS")


class ReadOnlyNamedQueryExecutor:
    """Local query port restricted to named, parameterized statements."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 5.0):
        self.path = Path(path).resolve()
        if not 0 < timeout_seconds <= 60:
            raise ValueError("timeout_seconds out of range")
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> sqlite3.Connection:
        # mode=ro must remain in place: missing files must not silently become DBs.
        uri = "file:" + quote(str(self.path), safe="/") + "?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            # Bound CPU as well as lock wait time for a malformed/oversized fixture.
            import time
            deadline = time.monotonic() + self.timeout_seconds
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            return conn
        except sqlite3.Error as exc:
            raise RepositoryError("DATABASE_UNAVAILABLE") from exc

    @staticmethod
    def _execute(conn: sqlite3.Connection, query_id: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        if query_id not in _QUERIES:
            raise RepositoryError("QUERY_NOT_ALLOWED")
        _validate_parameters(parameters)
        try:
            return [dict(row) for row in conn.execute(_QUERIES[query_id], dict(parameters)).fetchall()]
        except sqlite3.Error as exc:
            raise RepositoryError("QUERY_FAILED") from exc

    def _query(self, query_id: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        if query_id not in _QUERIES:
            raise RepositoryError("QUERY_NOT_ALLOWED")
        _validate_parameters(parameters)
        conn = self._connect()
        try:
            return self._execute(conn, query_id, parameters)
        finally:
            conn.close()

    async def execute_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> list[dict[str, Any]]:
        if tool_name != "query_voc" or set(arguments) != {"query_id", "parameters"}:
            raise RepositoryError("TOOL_NOT_ALLOWED")
        query_id, parameters = arguments["query_id"], arguments["parameters"]
        if not isinstance(query_id, str) or not isinstance(parameters, Mapping):
            raise RepositoryError("INVALID_QUERY_PARAMETERS")
        return await asyncio.to_thread(self._query, query_id, dict(parameters))


class SQLiteVocRepository:
    """Count and page are read in one transaction on a read-only connection."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 5.0):
        self.executor = ReadOnlyNamedQueryExecutor(path, timeout_seconds=timeout_seconds)

    def _page(self, request: AnalysisRequest) -> GatheredContext:
        if request.vocs is not None:
            raise RepositoryError("INLINE_REQUIRES_GATHERER")
        params = _bound_parameters(request)
        conn = self.executor._connect()
        try:
            conn.execute("BEGIN")
            count_rows = self.executor._execute(conn, "voc_count", params)
            total_count = count_rows[0]["total_count"]
            if total_count > 50000:
                raise RepositoryError("SNAPSHOT_LIMIT_EXCEEDED")
            # This deliberately bounded fixture adapter reads the stable matching
            # snapshot so a duplicate in page N remembers its first appearance in
            # page N-1. Production adapters should provide server-side business-key
            # ranking rather than scanning a whole production table.
            rows = self.executor._execute(conn, "voc_snapshot", params)
            if len(rows) != total_count:
                raise RepositoryError("SNAPSHOT_COUNT_MISMATCH")
            records: list[VocInput] = []
            for row in rows:
                try:
                    record = _input_copy(json.loads(row["payload"]))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise RepositoryError("INVALID_SOURCE_RECORD") from exc
                if record.recorded_on.isoformat() != row["recorded_on"] or record.channel_code != row["target_code"]:
                    raise RepositoryError("SOURCE_INDEX_MISMATCH")
                records.append(record)
            offset = request.page_number * request.page_size
            vocs = _materialize_snapshot(records, offset, request.page_size)
            return GatheredContext(vocs, total_count, "synthetic-sqlite-readonly-v1")
        except sqlite3.Error as exc:
            raise RepositoryError("QUERY_FAILED") from exc
        finally:
            conn.close()

    async def page(self, request: AnalysisRequest) -> GatheredContext:
        # sqlite connections are created and closed inside the same worker thread.
        return await asyncio.to_thread(self._page, request.model_copy(deep=True))


def _reference_words(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 1000:
        raise RepositoryError("INVALID_REFERENCE_CONTEXT")
    if any(not isinstance(word, str) or len(word) > 200 for word in value):
        raise RepositoryError("INVALID_REFERENCE_CONTEXT")
    return sorted({word.strip() for word in value if word.strip()})


def _reference_terms(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict) or len(value) > 1000:
        raise RepositoryError("INVALID_REFERENCE_CONTEXT")
    if any(not isinstance(key, str) or not key or len(key) > 128 for key in value):
        raise RepositoryError("INVALID_REFERENCE_CONTEXT")
    return {key: _reference_words(value[key]) for key in sorted(value)}


class VocGatherer:
    """Explicit context containing records, taxonomy, and prior terms.

    Taxonomy/codebook/reference terms are explicit supplied or local fixture data.
    All reference terms come from the supplied synthetic fixtures.
    """

    def __init__(self, repository: VocRepository, taxonomy: Taxonomy | None = None,
                 reference_data: dict[str, Any] | None = None):
        self.repository = repository
        fixture_root = Path(__file__).resolve().parent.parent / "fixtures"
        try:
            self.taxonomy = taxonomy if taxonomy is not None else Taxonomy.load(fixture_root / "aspect_catalog.json")
            data = reference_data if reference_data is not None else json.loads((fixture_root / "reference_context.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RepositoryError("REFERENCE_CONTEXT_UNAVAILABLE") from exc
        if not isinstance(data, dict) or set(data) - {"version", "exception_words", "previous_product_terms", "previous_quality_terms"}:
            raise RepositoryError("INVALID_REFERENCE_CONTEXT")
        version = data.get("version")
        if not isinstance(version, str) or not version or len(version) > 100:
            raise RepositoryError("INVALID_REFERENCE_CONTEXT")
        self._version = version
        self._exceptions = _reference_words(data.get("exception_words", []))
        self._products = _reference_terms(data.get("previous_product_terms", {}))
        self._qualities = _reference_terms(data.get("previous_quality_terms", {}))
        self._aspect: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        self._stages: dict[str, dict[str, str]] = {}
        for entry in self.taxonomy.entries:
            self._aspect.setdefault(entry.target_code, {}).setdefault(entry.stage_code, {})[entry.aspect_code] = entry.model_dump(mode="json")
            names = self._stages.setdefault(entry.target_code, {})
            supplied_name = getattr(entry, "stage_name", "").strip()
            previous_name = names.get(entry.stage_code, entry.stage_code)
            if supplied_name and previous_name not in (entry.stage_code, supplied_name):
                raise RepositoryError("CONFLICTING_STAGE_NAME")
            names[entry.stage_code] = supplied_name or previous_name

    async def execute(self, request: AnalysisRequest) -> GatheredContext:
        if request.vocs is not None:
            result = GatheredContext(_materialize_snapshot(request.vocs), len(request.vocs), "inline-request-v1")
        else:
            result = await self.repository.page(request)
        result.channel_aspects = deepcopy(self._aspect)
        result.journey_names = deepcopy(self._stages)
        result.emotion_dict = {code: emotion_info(code, "") for code in EMOTION_CODEBOOK}
        result.prior_product_terms = deepcopy(self._products)
        result.prior_quality_terms = deepcopy(self._qualities)
        result.exception_words = list(self._exceptions)
        result.reference_version = self._version
        return result


def create_demo_database(path: str | Path, records: Sequence[VocInput | Mapping[str, Any]]) -> Path:
    """Explicit CLI setup only; refuses overwrite and non-SYNTHETIC source data."""
    if len(records) > 50000:
        raise RepositoryError("SNAPSHOT_LIMIT_EXCEEDED")
    validated = [_input_copy(row) for row in records]
    if any(row.source_system != "SYNTHETIC" for row in validated):
        raise RepositoryError("DEMO_REQUIRES_SYNTHETIC_DATA")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RepositoryError("DATABASE_ALREADY_EXISTS") from exc
    os.close(fd)
    try:
        conn = sqlite3.connect(target)
        try:
            conn.execute("CREATE TABLE demo_voc (row_id INTEGER PRIMARY KEY, recorded_on TEXT NOT NULL, target_code TEXT NOT NULL, payload TEXT NOT NULL)")
            conn.executemany(
                "INSERT INTO demo_voc (recorded_on, target_code, payload) VALUES (?, ?, ?)",
                [(row.recorded_on.isoformat(), row.channel_code, row.model_dump_json()) for row in validated],
            )
            conn.execute("CREATE INDEX demo_voc_date_target ON demo_voc (recorded_on, target_code, row_id)")
            conn.commit()
        finally:
            conn.close()
    except Exception:
        target.unlink(missing_ok=True)  # Only the file exclusively created above.
        raise
    return target
