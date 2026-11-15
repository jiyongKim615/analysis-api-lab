"""FastAPI JSON/SSE facade. Default demo binds loopback; deployment is not configured."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .compatibility import KNOWN_UNSUPPORTED, SUPPORTED, export_summary, parse_run_request
from .config import Settings
from .models import RequestContext
from .persistence import PersistenceError, ResultStore
from .pipeline import AnalysisPipeline
from .providers import AnalysisProvider, ProviderError, strict_json_loads
from .repository import JSONFixtureRepository, RepositoryError, VocRepository
from .taxonomy import Taxonomy


def create_app(settings: Settings | None = None, *, repository: VocRepository | None = None,
               provider: AnalysisProvider | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    if config.require_auth and not config.api_token:
        raise ValueError("CX_API_TOKEN_REQUIRED")
    app = FastAPI(title="Analysis API Lab", version="1.0.0")
    repo = repository or JSONFixtureRepository(config.fixture_path)
    taxonomy = Taxonomy.load(config.taxonomy_path)
    reference_data = json.loads(config.reference_path.read_text(encoding="utf-8"))
    selected_provider = provider or config.provider()
    store = ResultStore(config.output_dir)
    admission = asyncio.Semaphore(8)
    app.state.settings = config

    def identity(authorization: str | None, request: Request) -> RequestContext:
        if config.api_token:
            supplied = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
            if not hmac.compare_digest(supplied.encode("utf-8"), config.api_token.encode("utf-8")):
                raise HTTPException(401, detail="UNAUTHORIZED", headers={"WWW-Authenticate": "Bearer"})
            # Identity is derived from server-controlled validated credential, not a user_id body/header.
            actor = "token_" + hashlib.sha256(config.api_token.encode()).hexdigest()[:24]
        else:
            # No unauthenticated remote access even when someone starts uvicorn with 0.0.0.0.
            host = request.client.host if request.client else ""
            if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
                raise HTTPException(403, detail="LOCAL_DEMO_ONLY")
            actor = "local_demo"
        return RequestContext(request_id=uuid.uuid4().hex, actor_ref=actor)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # FastAPI's default detail can echo unredacted offending values.
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_REQUEST"}})

    @app.get("/health")
    async def health():
        return {"status": "ok", "engine": selected_provider.name, "inferenceKind": selected_provider.inference_kind,
                "supportedRequests": sorted(SUPPORTED), "unsupportedRequests": sorted(KNOWN_UNSUPPORTED)}

    @app.get("/runs/{run_id}")
    async def read_run(run_id: str, request: Request, authorization: str | None = Header(default=None)):
        context = identity(authorization, request)
        try:
            return await asyncio.to_thread(store.load, run_id, context)
        except (PersistenceError, ValueError) as exc:
            raise HTTPException(404, detail="RUN_NOT_FOUND") from exc

    @app.post("/run_agent")
    async def run_agent(request: Request, authorization: str | None = Header(default=None)):
        context = identity(authorization, request)
        raw_bytes = bytearray()
        async for chunk in request.stream():
            raw_bytes.extend(chunk)
            if len(raw_bytes) > 2_000_000:
                raise HTTPException(413, detail="REQUEST_TOO_LARGE")
        try:
            raw = strict_json_loads(raw_bytes.decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("OBJECT_REQUIRED")
            # Reject unsupported routes before validating their payload.
            routing = raw
            if set(raw) == {"input_value"} and isinstance(raw["input_value"], str):
                routing = strict_json_loads(raw["input_value"])
            route = routing.get("requestType", routing.get("request_type", routing.get("agent_type", "analysis"))) if isinstance(routing, dict) else "analysis"
            if route in KNOWN_UNSUPPORTED:
                return JSONResponse(status_code=501, content={"error": {"code": "UNSUPPORTED", "requestType": route,
                    "message": "This route is outside this lab scope."}})
            payload = parse_run_request(raw)
            if payload.request_type not in SUPPORTED:
                raise ValueError("UNKNOWN_REQUEST_TYPE")
        except (ValueError, TypeError, UnicodeError, ValidationError, ProviderError) as exc:
            raise HTTPException(422, detail="INVALID_REQUEST") from exc
        # Create request-local pipeline state; do not mutate a shared provider.
        agent = AnalysisPipeline(repo, taxonomy, selected_provider, reference_data)

        async def execute(sink=None):
            async with admission:
                result = await agent.execute(payload.analysis, context, sink)
                await asyncio.to_thread(store.save, result, context)
                return result

        def outward(result):
            return export_summary(result) if payload.request_type == "voc" else result.model_dump(mode="json", by_alias=True)

        if not payload.is_stream:
            try:
                return outward(await execute())
            except RepositoryError as exc:
                return JSONResponse(status_code=503, content={"error": {"code": exc.code, "requestId": context.request_id}})
            except (PersistenceError, OSError):
                return JSONResponse(status_code=500, content={"error": {"code": "RESULT_PERSISTENCE_FAILED", "requestId": context.request_id}})
            except (TimeoutError, ProviderError):
                return JSONResponse(status_code=503, content={"error": {"code": "ANALYSIS_UNAVAILABLE", "requestId": context.request_id}})

        async def events():
            queue: asyncio.Queue = asyncio.Queue(maxsize=16)
            async def sink(event: str, content: dict):
                await queue.put({"event": event, "content": content, "requestId": context.request_id})
            async def producer():
                try:
                    await sink("START", {"engine": selected_provider.name})
                    result = await execute(sink)
                    await sink("CHUNK", outward(result))
                    await sink("DONE", {"success": True, "runId": result.metadata.run_id})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    code = exc.code if isinstance(exc, (RepositoryError, ProviderError, PersistenceError)) else "ANALYSIS_FAILED"
                    await sink("ERROR", {"code": code})
                    await sink("DONE", {"success": False})
                finally:
                    if not asyncio.current_task().cancelling():
                        await queue.put(None)
            task = asyncio.create_task(producer())
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield "event: " + event["event"] + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            finally:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return app


app = create_app()
