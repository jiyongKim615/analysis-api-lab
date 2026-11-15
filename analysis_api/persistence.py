"""Atomic, actor-scoped local result persistence, disabled unless explicitly used.

The directory is a private application workspace, not a multi-user filesystem
security boundary. Authentication/authorization belongs to the API. Opaque source
identifiers remain unchanged for traceability; text masking is not anonymization. Result files
are never placed under a user-supplied filename, and retention never deletes files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any
from uuid import UUID

from .models import AnalysisResponse, RequestContext
from .security import mask_pii


class PersistenceError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


_IDENTITY_FIELDS = frozenset({"runId", "requestId", "sourceHash", "vocId", "sourceSystem", "datasetId", "surveyId", "responseId", "questionId"})
_FORBIDDEN_KEYS = frozenset({"apikey", "authorization", "token", "accesstoken", "secret", "password", "credential", "credentials"})


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        if any(str(key).lower().replace("_", "").replace("-", "") in _FORBIDDEN_KEYS for key in value):
            raise PersistenceError("SECRET_FIELD_NOT_ALLOWED")
        return {key: item if key in _IDENTITY_FIELDS else _sanitize(item) for key, item in value.items()}
    return value


def _run_key(run_id: str) -> str:
    if not isinstance(run_id, str) or len(run_id) not in (32, 36):
        raise PersistenceError("INVALID_RUN_ID")
    try:
        parsed = UUID(run_id)
    except (ValueError, AttributeError) as exc:
        raise PersistenceError("INVALID_RUN_ID") from exc
    if run_id not in (parsed.hex, str(parsed)):
        raise PersistenceError("INVALID_RUN_ID")
    return parsed.hex


class ResultStore:
    MAX_BYTES = 32 * 1024 * 1024

    def __init__(self, root: str | Path):
        supplied = Path(root)
        if supplied.is_symlink():
            raise PersistenceError("UNSAFE_STORAGE_PATH")
        supplied.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not supplied.is_dir():
            raise PersistenceError("UNSAFE_STORAGE_PATH")
        self.root = supplied.resolve()

    def _actor_dir(self, context: RequestContext, *, create: bool) -> Path:
        name = hashlib.sha256(context.actor_ref.encode("utf-8")).hexdigest()
        path = self.root / name
        if path.is_symlink():
            raise PersistenceError("UNSAFE_STORAGE_PATH")
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
        if path.exists() and (not path.is_dir() or path.resolve().parent != self.root):
            raise PersistenceError("UNSAFE_STORAGE_PATH")
        return path

    def _read_bytes(self, path: Path) -> bytes:
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > self.MAX_BYTES:
                    raise PersistenceError("INVALID_RESULT_FILE")
                content = stream.read(self.MAX_BYTES + 1)
                if len(content) > self.MAX_BYTES:
                    raise PersistenceError("INVALID_RESULT_FILE")
                return content
        except FileNotFoundError as exc:
            raise PersistenceError("RESULT_NOT_FOUND") from exc
        except OSError as exc:
            raise PersistenceError("RESULT_READ_FAILED") from exc

    def save(self, response: AnalysisResponse, context: RequestContext) -> Path:
        if response.metadata.request_id != context.request_id:
            raise PersistenceError("REQUEST_CONTEXT_MISMATCH")
        key = _run_key(response.metadata.run_id)
        data = _sanitize(response.model_dump(mode="json", by_alias=True))
        encoded = (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > self.MAX_BYTES:
            raise PersistenceError("RESULT_TOO_LARGE")
        parent = self._actor_dir(context, create=True)
        target = parent / f"{key}.json"
        fd, name = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=parent)
        pending = Path(name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Atomic, exclusive publication. Unlike replace(), conflicting
                # concurrent results cannot silently overwrite an existing run.
                os.link(pending, target)
            except FileExistsError:
                if self._read_bytes(target) != encoded:
                    raise PersistenceError("RUN_ID_CONFLICT")
            # Flush directory metadata for crash durability where supported.
            dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            return target
        except OSError as exc:
            raise PersistenceError("RESULT_WRITE_FAILED") from exc
        finally:
            pending.unlink(missing_ok=True)

    def load(self, run_id: str, context: RequestContext) -> dict[str, Any]:
        key = _run_key(run_id)
        target = self._actor_dir(context, create=False) / f"{key}.json"
        content = self._read_bytes(target)
        try:
            parsed = json.loads(content)
            response = AnalysisResponse.model_validate(parsed)
            if _run_key(response.metadata.run_id) != key:
                raise ValueError("run mismatch")
        except (ValueError, TypeError) as exc:
            raise PersistenceError("INVALID_RESULT_FILE") from exc
        return response.model_dump(mode="json", by_alias=True)
