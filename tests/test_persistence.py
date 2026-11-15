from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from analysis_api.models import AnalysisResponse, RunMetadata, PageResult, RequestContext, VocInfo
from analysis_api.persistence import PersistenceError, ResultStore


def make_response(context: RequestContext) -> AnalysisResponse:
    return AnalysisResponse(
        metadata=RunMetadata(
            run_id=str(uuid4()), request_id=context.request_id, created_at="2026-10-29T12:00:00+00:00", elapsed_ms=1.0,
            engine="rule-reference", model_version="rules-v1", taxonomy_version="synthetic-v1", inference_kind="deterministic-reference", metrics={"processed": 1},
        ),
        page=PageResult(
            page_number=0, page_size=500, total_elements=1, total_pages=1, number_of_elements=1, first_yn=True, last_yn=True,
            content=[VocInfo(voc_id="V1", text="로그인이 느려요. user@example.com 010-0000-0000")],
        ),
    )


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "results"
        self.store = ResultStore(self.root)
        self.context = RequestContext(request_id="request-1", actor_ref="actor-1")
        self.response = make_response(self.context)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_redacts_text_preserves_identity(self):
        path = self.store.save(self.response, self.context)
        data = self.store.load(self.response.metadata.run_id, self.context)
        self.assertNotIn("user@example.com", path.read_text())
        self.assertNotIn("010-0000-0000", path.read_text())
        self.assertIn("[EMAIL]", data["page"]["content"][0]["text"])
        self.assertEqual(data["page"]["content"][0]["vocId"], "V1")
        self.assertEqual(data["metadata"]["runId"], self.response.metadata.run_id)
        self.assertEqual(path.parent.name, hashlib.sha256(b"actor-1").hexdigest())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_actor_isolation(self):
        self.store.save(self.response, self.context)
        other = RequestContext(request_id="request-2", actor_ref="actor-2")
        with self.assertRaisesRegex(PersistenceError, "RESULT_NOT_FOUND"):
            self.store.load(self.response.metadata.run_id, other)

    def test_context_mismatch_is_rejected(self):
        other = RequestContext(request_id="request-2", actor_ref="actor-1")
        with self.assertRaisesRegex(PersistenceError, "REQUEST_CONTEXT_MISMATCH"):
            self.store.save(self.response, other)

    def test_path_traversal_and_malformed_ids(self):
        for value in ("../result", "../../secret.json", "", "f" * 31, "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz", "/tmp/result"):
            with self.subTest(run_id=value):
                with self.assertRaisesRegex(PersistenceError, "INVALID_RUN_ID"):
                    self.store.load(value, self.context)

    def test_repeat_save_is_idempotent_conflict_not_overwritten(self):
        path = self.store.save(self.response, self.context)
        original = path.read_bytes()
        self.assertEqual(self.store.save(self.response, self.context), path)
        changed = self.response.model_copy(deep=True)
        changed.metadata.metrics["processed"] = 99
        with self.assertRaisesRegex(PersistenceError, "RUN_ID_CONFLICT"):
            self.store.save(changed, self.context)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(path.parent.glob(".pending-*")))

    def test_concurrent_same_run_is_atomic_and_idempotent(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda _: self.store.save(self.response, self.context), range(20)))
        self.assertEqual(len(set(paths)), 1)
        parsed = json.loads(paths[0].read_text())
        self.assertEqual(parsed["metadata"]["runId"], self.response.metadata.run_id)
        self.assertFalse(list(paths[0].parent.glob(".pending-*")))

    def test_conflicting_concurrent_runs_publish_one_complete_result(self):
        variants = []
        for count in range(8):
            item = self.response.model_copy(deep=True)
            item.metadata.metrics["processed"] = count
            variants.append(item)
        def save(item):
            try:
                self.store.save(item, self.context)
                return "saved"
            except PersistenceError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=8) as pool:
            states = list(pool.map(save, variants))
        self.assertEqual(states.count("saved"), 1)
        self.assertEqual(states.count("RUN_ID_CONFLICT"), 7)
        data = self.store.load(self.response.metadata.run_id, self.context)
        self.assertIn(data["metadata"]["metrics"]["processed"], range(8))

    def test_secret_fields_are_not_saved(self):
        self.response.metadata.metrics["Authorization"] = "Bearer secret"
        with self.assertRaisesRegex(PersistenceError, "SECRET_FIELD_NOT_ALLOWED"):
            self.store.save(self.response, self.context)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_corrupt_result_does_not_look_like_missing(self):
        path = self.store.save(self.response, self.context)
        path.write_text("broken", encoding="utf-8")
        with self.assertRaisesRegex(PersistenceError, "INVALID_RESULT_FILE"):
            self.store.load(self.response.metadata.run_id, self.context)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unsupported")
    def test_symlink_storage_path_is_rejected(self):
        alias = Path(self.tmp.name) / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(PersistenceError, "UNSAFE_STORAGE_PATH"):
            ResultStore(alias)
        actor_dir = self.root / hashlib.sha256(self.context.actor_ref.encode()).hexdigest()
        actor_dir.symlink_to(Path(self.tmp.name), target_is_directory=True)
        with self.assertRaisesRegex(PersistenceError, "UNSAFE_STORAGE_PATH"):
            self.store.save(self.response, self.context)

    def test_technical_identifiers_are_not_regex_redacted(self):
        self.response.metadata.run_id = "12345678-1234-5678-1234-567812345678"
        self.response.page.content[0].source_hash = "0" * 64
        self.response.page.content[0].voc_id = "01012345678"
        self.store.save(self.response, self.context)
        data = self.store.load(self.response.metadata.run_id, self.context)
        self.assertEqual(data["metadata"]["runId"], self.response.metadata.run_id)
        self.assertEqual(data["page"]["content"][0]["sourceHash"], "0" * 64)
        self.assertEqual(data["page"]["content"][0]["vocId"], "01012345678")


if __name__ == "__main__":
    unittest.main()
