from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from analysis_api.models import AnalysisRequest, VocInput
from analysis_api.repository import (
    JSONFixtureRepository, SQLiteVocRepository, ReadOnlyNamedQueryExecutor,
    RepositoryError, VocGatherer, _bound_parameters, create_demo_database,
)


def row(voc_id: str, day: str = "2026-10-29", target: str = "02") -> VocInput:
    return VocInput(voc_id=voc_id, text="앱 로그인 과정이 반복해서 느려 불편했습니다.", recorded_on=day, channel_code=target)


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = [row("V1", "2026-10-28"), row("V2"), row("V2"), row("V3", target="07")]
        self.json_repository = JSONFixtureRepository(self.rows)
        self.path = create_demo_database(self.root / "fixture.db", self.rows)
        self.sqlite_repository = SQLiteVocRepository(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_empty_page_retains_total(self):
        for repository in (self.json_repository, self.sqlite_repository):
            result = await repository.page(AnalysisRequest(page_number=10, page_size=2))
            self.assertEqual(result.total_count, 4)
            self.assertEqual(result.voc_list, [])

    async def test_date_and_target_filters_match_for_both_adapters(self):
        request = AnalysisRequest(date="2026-10-29", target_codes=["02"])
        for repository in (self.json_repository, self.sqlite_repository):
            result = await repository.page(request)
            self.assertEqual(result.total_count, 2)
            self.assertEqual([v.voc_id for v in result.voc_list], ["V2", "V2"])

    async def test_range_filters_and_stable_pagination(self):
        request = AnalysisRequest(date="2026-10-28", date_end="2026-10-29", page_number=1, page_size=2)
        for repository in (self.json_repository, self.sqlite_repository):
            result = await repository.page(request)
            self.assertEqual([v.voc_id for v in result.voc_list], ["V2", "V3"])
            self.assertEqual(result.total_count, 4)

    async def test_request_state_is_not_reused(self):
        for repository in (self.json_repository, self.sqlite_repository):
            first = await repository.page(AnalysisRequest())
            first.voc_list[0].excluded = True
            first.voc_list[0].stages["aspect"].rejected_codes.append("bogus")
            second = await repository.page(AnalysisRequest())
            self.assertFalse(second.voc_list[0].excluded)
            self.assertEqual(second.voc_list[0].stages["aspect"].rejected_codes, [])
            self.assertEqual(len(second.voc_list), 4)

    async def test_inline_empty_is_explicit_not_repository_fallback(self):
        result = await VocGatherer(self.json_repository).execute(AnalysisRequest(vocs=[]))
        self.assertEqual(result.voc_list, [])
        self.assertEqual(result.total_count, 0)
        self.assertEqual(result.repository_id, "inline-request-v1")

    async def test_inline_records_are_fresh(self):
        request = AnalysisRequest(vocs=[self.rows[0]])
        result = await VocGatherer(self.json_repository).execute(request)
        result.voc_list[0].text = "changed"
        self.assertNotEqual(request.vocs[0].text, "changed")

    async def test_json_path_and_invalid_source_errors(self):
        path = self.root / "input.json"
        path.write_text(json.dumps([v.model_dump(mode="json") for v in self.rows]), encoding="utf-8")
        repo = JSONFixtureRepository(path)
        self.assertEqual((await repo.page(AnalysisRequest())).total_count, 4)
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with self.assertRaisesRegex(RepositoryError, "INVALID_FIXTURE"):
            JSONFixtureRepository(path)
        path.write_text("broken json", encoding="utf-8")
        with self.assertRaisesRegex(RepositoryError, "INVALID_FIXTURE"):
            JSONFixtureRepository(path)
        with self.assertRaisesRegex(RepositoryError, "INVALID_SOURCE_RECORD"):
            JSONFixtureRepository([{"voc_id": "only-id"}])

    async def test_missing_database_is_error_not_empty(self):
        path = self.root / "missing.db"
        with self.assertRaisesRegex(RepositoryError, "DATABASE_UNAVAILABLE"):
            await SQLiteVocRepository(path).page(AnalysisRequest())
        self.assertFalse(path.exists())

    async def test_sql_injection_is_bound_value(self):
        request = AnalysisRequest(target_codes=["02') OR 1=1; DROP TABLE demo_voc;--"])
        result = await self.sqlite_repository.page(request)
        self.assertEqual(result.total_count, 0)
        self.assertEqual((await self.sqlite_repository.page(AnalysisRequest())).total_count, 4)

    async def test_query_and_tool_allowlists(self):
        executor = ReadOnlyNamedQueryExecutor(self.path)
        parameters = _bound_parameters(AnalysisRequest())
        for name in ("DROP TABLE demo_voc", "mysql_query", "unknown"):
            with self.assertRaisesRegex(RepositoryError, "QUERY_NOT_ALLOWED"):
                await executor.execute_tool("query_voc", {"query_id": name, "parameters": parameters})
        with self.assertRaisesRegex(RepositoryError, "TOOL_NOT_ALLOWED"):
            await executor.execute_tool("mysql_query", {"query": "SELECT * FROM demo_voc"})
        actual = await executor.execute_tool("query_voc", {"query_id": "voc_count", "parameters": parameters})
        self.assertEqual(actual, [{"total_count": 4}])

    async def test_invalid_query_parameters_rejected(self):
        executor = ReadOnlyNamedQueryExecutor(self.path)
        base = _bound_parameters(AnalysisRequest())
        invalid = [{**base, "limit": True}, {**base, "offset": -1}, {**base, "limit": 501}, {**base, "date_from": "2026-02-30"}, {**base, "extra": "value"}]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(RepositoryError, "INVALID_QUERY_PARAMETERS"):
                    await executor.execute_tool("query_voc", {"query_id": "voc_page", "parameters": parameters})

    async def test_database_connection_is_readonly(self):
        connection = self.sqlite_repository.executor._connect()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM demo_voc")
        finally:
            connection.close()

    async def test_source_index_mismatch_is_error(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE demo_voc SET target_code='99' WHERE row_id=1")
        with self.assertRaisesRegex(RepositoryError, "SOURCE_INDEX_MISMATCH"):
            await self.sqlite_repository.page(AnalysisRequest())

    async def test_invalid_database_payload_is_error(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE demo_voc SET payload='not-json' WHERE row_id=1")
        with self.assertRaisesRegex(RepositoryError, "INVALID_SOURCE_RECORD"):
            await self.sqlite_repository.page(AnalysisRequest())

    async def test_parallel_reads_independent(self):
        results = await asyncio.gather(*(self.sqlite_repository.page(AnalysisRequest(page_size=1, page_number=i)) for i in range(4)))
        self.assertEqual([r.voc_list[0].voc_id for r in results], ["V1", "V2", "V2", "V3"])
        self.assertTrue(all(r.total_count == 4 for r in results))

    async def test_duplicate_marked_across_page_boundaries(self):
        for repository in (self.json_repository, self.sqlite_repository):
            first = await repository.page(AnalysisRequest(page_size=2, page_number=0))
            second = await repository.page(AnalysisRequest(page_size=2, page_number=1))
            self.assertFalse(first.voc_list[1].duplicate)
            self.assertTrue(second.voc_list[0].duplicate)
            self.assertFalse(second.voc_list[1].duplicate)
            self.assertEqual(second.total_count, 4)

    async def test_composite_business_key_duplicate_across_pages(self):
        common = dict(recorded_on="2026-10-29", dataset_id="G", survey_id="Q", response_id="P", question_id="I")
        records = [VocInput(voc_id="one", text="합성 고객 의견 첫 번째입니다.", **common),
                   VocInput(voc_id="different-id", text="동일 원천 키의 두 번째 합성 의견입니다.", **common)]
        other_db = create_demo_database(self.root / "composite.db", records)
        for repository in (JSONFixtureRepository(records), SQLiteVocRepository(other_db)):
            result = await repository.page(AnalysisRequest(page_size=1, page_number=1))
            self.assertTrue(result.voc_list[0].duplicate)

    async def test_duplicate_scope_is_filtered_snapshot(self):
        records = [row("same", "2026-10-28"), row("same", "2026-10-29")]
        other_db = create_demo_database(self.root / "filtered.db", records)
        for repository in (JSONFixtureRepository(records), SQLiteVocRepository(other_db)):
            result = await repository.page(AnalysisRequest(date="2026-10-29"))
            self.assertFalse(result.voc_list[0].duplicate)
            self.assertEqual(result.total_count, 1)

    async def test_gatherer_full_context_available_even_on_empty_page(self):
        result = await VocGatherer(self.json_repository).execute(AnalysisRequest(page_number=100))
        self.assertEqual(result.voc_list, [])
        self.assertEqual(result.total_count, 4)
        self.assertIn("02", result.channel_aspects)
        self.assertIn("18", result.journey_names["02"])
        self.assertEqual(set(result.emotion_dict), {f"0{i}" for i in range(1, 8)})
        self.assertEqual(result.emotion_dict["04"].feedback_type, "03")
        self.assertEqual(result.emotion_dict["04"].polarity_code, "02")
        self.assertIn("02_SYN_LOGIN", result.prior_product_terms)
        self.assertEqual(result.reference_version, "synthetic-reference-context-v1")
        self.assertEqual(result.exception_words, ["합성제외용어"])

    async def test_gatherer_context_normalization_and_request_isolation(self):
        reference = {"version": "test-reference-v1", "exception_words": ["z", "", " z ", "a"],
                     "previous_product_terms": {"02_SYN_LOGIN": ["앱", "로그인", "앱"]},
                     "previous_quality_terms": {"02_SYN_LOGIN": ["오류", "느림"]}}
        gatherer = VocGatherer(self.json_repository, reference_data=reference)
        first = await gatherer.execute(AnalysisRequest())
        self.assertEqual(first.exception_words, ["a", "z"])
        self.assertEqual(first.prior_product_terms["02_SYN_LOGIN"], ["로그인", "앱"])
        first.prior_product_terms["02_SYN_LOGIN"].append("mutated")
        first.channel_aspects["02"]["18"].clear()
        first.emotion_dict["01"].emotion_name = "mutated"
        reference["previous_product_terms"]["02_SYN_LOGIN"].append("outside mutation")
        second = await gatherer.execute(AnalysisRequest())
        self.assertEqual(second.prior_product_terms["02_SYN_LOGIN"], ["로그인", "앱"])
        self.assertTrue(second.channel_aspects["02"]["18"])
        self.assertEqual(second.emotion_dict["01"].emotion_name, "감동")

    async def test_inline_duplicates_marked(self):
        result = await VocGatherer(self.json_repository).execute(AnalysisRequest(vocs=[row("same"), row("same")]))
        self.assertEqual([v.duplicate for v in result.voc_list], [False, True])

    def test_invalid_reference_context_fails_closed(self):
        for reference in ({}, {"version": "v", "exception_words": [123]}, {"version": "v", "previous_product_terms": []}, {"version": "v", "unknown": []}):
            with self.assertRaisesRegex(RepositoryError, "INVALID_REFERENCE_CONTEXT"):
                VocGatherer(self.json_repository, reference_data=reference)

    async def test_sqlite_snapshot_limit_is_explicit(self):
        with sqlite3.connect(self.path) as connection:
            payload = row("repeated").model_dump_json()
            connection.executemany("INSERT INTO demo_voc (recorded_on, target_code, payload) VALUES (?, ?, ?)",
                                   (("2026-10-29", "02", payload) for _ in range(49997)))
        with self.assertRaisesRegex(RepositoryError, "SNAPSHOT_LIMIT_EXCEEDED"):
            await self.sqlite_repository.page(AnalysisRequest(page_size=1))

    async def test_survey_mode_filter_count_and_page_are_consistent(self):
        records = [row("BU-1").model_copy(update={"survey_mode": "BU"}),
                   row("TD-1").model_copy(update={"survey_mode": "TD"}),
                   row("BU-2", target="07").model_copy(update={"survey_mode": "BU"}),
                   row("NO-MODE")]
        other_db = create_demo_database(self.root / "pol-modes.db", records)
        for repository in (JSONFixtureRepository(records), SQLiteVocRepository(other_db)):
            request = AnalysisRequest(date="2026-10-29", survey_modes=["BU"], target_codes=["02"], page_size=1)
            first = await repository.page(request)
            self.assertEqual(first.total_count, 1)
            self.assertEqual(first.voc_list[0].voc_id, "BU-1")
            second = await repository.page(request.model_copy(update={"page_number": 1}))
            self.assertEqual(second.total_count, 1)
            self.assertEqual(second.voc_list, [])
            union = await repository.page(AnalysisRequest(survey_modes=["BU", "TD"]))
            self.assertEqual(union.total_count, 3)
            injection = await repository.page(AnalysisRequest(survey_modes=["BU') OR 1=1--"]))
            self.assertEqual(injection.total_count, 0)

    def test_demo_setup_refuses_overwrite_and_real_source(self):
        before = self.path.read_bytes()
        with self.assertRaisesRegex(RepositoryError, "DATABASE_ALREADY_EXISTS"):
            create_demo_database(self.path, [])
        self.assertEqual(self.path.read_bytes(), before)
        real = row("real").model_copy(update={"source_system": "PRODUCTION"})
        target = self.root / "not-created.db"
        with self.assertRaisesRegex(RepositoryError, "DEMO_REQUIRES_SYNTHETIC_DATA"):
            create_demo_database(target, [real])
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
