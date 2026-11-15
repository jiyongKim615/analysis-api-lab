"""Offline CLI: same AnalysisPipeline used by the HTTP API."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import sys
import uuid

from .compatibility import export_summary
from .config import ROOT, Settings
from .models import AnalysisRequest, RequestContext, VocInput
from .persistence import ResultStore
from .pipeline import AnalysisPipeline
from .repository import JSONFixtureRepository, SQLiteVocRepository, create_demo_database
from .taxonomy import Taxonomy


def safe_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def export_csv(path: Path, response) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["voc_id", "masked_voc", "filtered_reason", "aspect_codes", "emotion", "voc_type", "reason", "summary", "stage_statuses"])
        for voc in response.page.content:
            emotion = voc.emotion
            writer.writerow([safe_cell(v) for v in (voc.voc_id, voc.text, voc.filtered_reason,
                ",".join(c.aspect_code for c in voc.aspects), emotion.emotion_name if emotion else None,
                emotion.feedback_label if emotion else None, voc.primary_reason, voc.summary,
                json.dumps({k: s.status.value for k, s in voc.stages.items()}, ensure_ascii=False))])


async def run(args) -> int:
    if args.command == "init-db":
        records = [VocInput.model_validate(v) for v in json.loads(args.input.read_text(encoding="utf-8"))]
        create_demo_database(args.path, records)
        print(json.dumps({"status": "created", "records": len(records)}, ensure_ascii=False))
        return 0
    settings = Settings.from_env()
    # CLI demo is always offline regardless of environment provider selection.
    from .providers import OfflineReferenceProvider
    repository = SQLiteVocRepository(args.database) if args.database else JSONFixtureRepository(args.input)
    agent = AnalysisPipeline(repository, Taxonomy.load(settings.taxonomy_path), OfflineReferenceProvider(),
                              json.loads(settings.reference_path.read_text(encoding="utf-8")))
    context = RequestContext(request_id=uuid.uuid4().hex, actor_ref="cli_demo")
    request = AnalysisRequest(page_number=args.page, page_size=args.page_size,
                                    reason_policy=args.reason_policy, include_summary=not args.no_summary,
                                    voc_length_limit=args.min_length)
    response = await agent.execute(request, context)
    store = ResultStore(args.store)
    path = store.save(response, context)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        data = export_summary(response) if args.summary else response.model_dump(mode="json", by_alias=True)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
    if args.csv:
        export_csv(args.csv, response)
    print(json.dumps({"runId": response.metadata.run_id, "input": response.page.number_of_elements,
        "filtered": response.metadata.metrics["filteredCount"], "needsReview": response.metadata.metrics["reviewCount"],
        "engine": response.metadata.engine, "resultFile": str(path)}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic feedback analysis lab")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-db", help="Create a NEW synthetic SQLite fixture; never overwrites")
    init.add_argument("path", type=Path)
    init.add_argument("--input", type=Path, default=ROOT / "fixtures/vocs.json")
    demo = sub.add_parser("demo", help="Run the complete offline pipeline")
    demo.add_argument("--input", type=Path, default=ROOT / "fixtures/vocs.json")
    demo.add_argument("--database", type=Path)
    demo.add_argument("--out", type=Path)
    demo.add_argument("--csv", type=Path)
    demo.add_argument("--store", type=Path, default=Path.cwd() / "output/runs")
    demo.add_argument("--page", type=int, default=0)
    demo.add_argument("--page-size", type=int, default=500)
    demo.add_argument("--min-length", type=int, default=10)
    demo.add_argument("--reason-policy", choices=["complaints", "negative_emotion"], default="complaints")
    demo.add_argument("--no-summary", action="store_true")
    demo.add_argument("--summary", action="store_true", help="Compact analysis summary")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        # Do not echo untrusted inputs, provider response bodies, or credentials.
        code = getattr(exc, "code", "CLI_FAILED")
        print(json.dumps({"error": code}), file=sys.stderr)
        return 1
