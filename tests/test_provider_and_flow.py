"""New integration regressions. All network-like responses use MockTransport."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from analysis_api.cli import main, safe_cell
from analysis_api.compatibility import export_summary, parse_run_request
from analysis_api.components import AspectMapper, VocFilter
from analysis_api.config import ROOT, Settings
from analysis_api.models import AspectInfo, AnalysisRequest, RequestContext, VocInfo, VocInput
from analysis_api.pipeline import AnalysisPipeline
from analysis_api.providers import JsonHttpProvider, OfflineReferenceProvider, ProviderError, ProviderGateway, strict_json_loads
from analysis_api.repository import JSONFixtureRepository
from analysis_api.taxonomy import Taxonomy

CONTEXT = RequestContext(request_id="test-request", actor_ref="test-user")

class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_disabled_or_nonapproved_endpoints_blocked(self):
        for endpoint, enabled, hosts in [("https://example.invalid/a", False, {"example.invalid"}),
                                          ("http://example.invalid/a", True, {"example.invalid"}),
                                          ("https://unapproved.invalid/a", True, {"example.invalid"}),
                                          ("https://user:pw@example.invalid/a", True, {"example.invalid"})]:
            with self.assertRaises(ProviderError):
                JsonHttpProvider(endpoint, enabled=enabled, allowed_hosts=hosts)

    async def test_headers_are_request_scoped_and_no_credentials_in_body(self):
        seen = []
        async def handler(request):
            seen.append((dict(request.headers), json.loads(request.content)))
            await asyncio.sleep(0.001)
            return httpx.Response(200, json={"results": {}})
        p = JsonHttpProvider("https://approved.invalid/analyze", enabled=True, allowed_hosts={"approved.invalid"},
                             token="fake-test-token", transport=httpx.MockTransport(handler))
        await asyncio.gather(*(p.complete("emotion", {"items": []}, RequestContext(request_id=f"req-{i}", actor_ref=f"u-{i}")) for i in range(5)))
        self.assertEqual({h["x-request-id"] for h, _ in seen}, {f"req-{i}" for i in range(5)})
        self.assertTrue(all(h["authorization"] == "Bearer fake-test-token" for h, _ in seen))
        self.assertNotIn("fake-test-token", str([b for _, b in seen]))

    async def test_redirect_and_auth_rejections_are_not_retried(self):
        for status in [302, 401, 403]:
            count = 0
            def handler(request):
                nonlocal count; count += 1
                return httpx.Response(status, headers={"Location": "https://unapproved.invalid"})
            p = JsonHttpProvider("https://approved.invalid/a", enabled=True, allowed_hosts={"approved.invalid"}, transport=httpx.MockTransport(handler))
            result = await ProviderGateway(p, retry_limit=3).call("emotion", [{"id":"x","text":"합성"}], CONTEXT, lambda value,item:value)
            self.assertEqual(result['x'].error_code, 'PROVIDER_HTTP_REJECTED')
            self.assertEqual(count,1)

    async def test_transient_http_retry_recovers(self):
        count = 0
        def handler(request):
            nonlocal count; count += 1
            return httpx.Response(429) if count == 1 else httpx.Response(200,json={"results":{"x":{"ok":True}}})
        p=JsonHttpProvider("https://approved.invalid/a",enabled=True,allowed_hosts={"approved.invalid"},transport=httpx.MockTransport(handler))
        out=await ProviderGateway(p,retry_limit=1).call("emotion",[{"id":"x","text":"합성"}],CONTEXT,lambda value,item:value)
        self.assertEqual(out['x'].attempts,2)
        self.assertIsNone(out['x'].error_code)

    async def test_timeout_is_explicit_and_bounded(self):
        class Slow(OfflineReferenceProvider):
            async def complete(self,*args):await asyncio.sleep(1)
        out=await ProviderGateway(Slow(),timeout=.005,retry_limit=1).call("emotion",[{"id":"x","text":"합성"}],CONTEXT,lambda value,item:value)
        self.assertEqual((out['x'].error_code,out['x'].attempts),('TIMEOUT',2))

    async def test_concurrency_limit(self):
        class Tracking(OfflineReferenceProvider):
            active=0; maximum=0
            async def complete(self,stage,payload,context):
                self.active+=1;self.maximum=max(self.maximum,self.active)
                try:
                    await asyncio.sleep(.005)
                    return {"results":{i['id']:{"ok":True} for i in payload['items']}}
                finally:self.active-=1
        p=Tracking();g=ProviderGateway(p,concurrency=2,retry_limit=0)
        await asyncio.gather(*(g.call('emotion',[{'id':str(i),'text':'합성'}],CONTEXT,lambda v,i:v) for i in range(8)))
        self.assertEqual(p.maximum,2)

    def test_duplicate_json_and_nonfinite_values_rejected(self):
        for text in ['{"a":1,"a":2}', '{"a":NaN}', 'not-json']:
            with self.assertRaises(ProviderError):strict_json_loads(text)

class IntegratedFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.taxonomy=Taxonomy.load(ROOT/'fixtures/aspect_catalog.json')
        self.agent=AnalysisPipeline(JSONFixtureRepository(ROOT/'fixtures/vocs.json'),self.taxonomy)

    async def test_default_demo_has_no_http_calls(self):
        with patch.object(httpx.AsyncClient,'send',side_effect=AssertionError('UNEXPECTED_NETWORK')):
            result=await self.agent.execute(AnalysisRequest(),CONTEXT)
        self.assertEqual(result.page.number_of_elements,11)
        self.assertNotIn('010-0000-0000',result.model_dump_json())
        self.assertNotIn('demo@example.com',result.model_dump_json())
        self.assertTrue(all(s.duration_ms>=0 for v in result.page.content for s in v.stages.values()))

    async def test_complaint_vs_negative_reason_policy(self):
        record=VocInput(voc_id='regret',text='앱 로그인 화면이 느려서 정말 아쉽습니다.',journey_code='18')
        a=await self.agent.execute(AnalysisRequest(vocs=[record],reason_policy='complaints'),CONTEXT)
        b=await self.agent.execute(AnalysisRequest(vocs=[record],reason_policy='negative_emotion'),CONTEXT)
        self.assertFalse(a.page.content[0].reason_success)
        self.assertTrue(b.page.content[0].reason_success)

    async def test_mixed_aspect_reason_uses_complaint_not_first_regret(self):
        record=VocInput(voc_id='mixed',text='로그인 화면이 느려 아쉬웠습니다. 결제가 실패해서 실망했습니다.')
        result=await self.agent.execute(AnalysisRequest(vocs=[record]),CONTEXT)
        self.assertIn('실망',result.page.content[0].primary_reason)
        self.assertNotIn('아쉬',result.page.content[0].primary_reason)
        exported=export_summary(result)
        self.assertEqual(len(exported['page']['content'][0]['aspects']),len(result.page.content[0].aspects))

    async def test_empty_out_of_range_page_keeps_total(self):
        result=await self.agent.execute(AnalysisRequest(page_number=99,page_size=2),CONTEXT)
        self.assertEqual(result.page.total_elements,11)
        self.assertEqual(result.page.number_of_elements,0)
        self.assertTrue(result.page.last_yn)

    async def test_partial_aspect_and_ungrounded_evidence(self):
        class Partial(OfflineReferenceProvider):
            async def complete(self,stage,payload,context):
                response=await super().complete(stage,payload,context)
                if stage=='aspect':
                    for value in response['results'].values():
                        value['matches'].append({'aspect_code':'UNKNOWN','evidence_quote':'bad','polarity':'neutral'})
                return response
        agent=AnalysisPipeline(JSONFixtureRepository([]),self.taxonomy,Partial())
        result=await agent.execute(AnalysisRequest(vocs=[VocInput(voc_id='partial',text='모바일 앱 로그인 오류가 나서 실망했습니다.',journey_code='18')]),CONTEXT)
        voc=result.page.content[0]
        self.assertEqual(voc.stages['aspect'].status.value,'partial_success')
        self.assertTrue(voc.aspect_success)
        self.assertEqual(voc.stages['aspect'].rejected_codes,['UNKNOWN'])

    async def test_summary_export_and_batch_request_aliases(self):
        request=parse_run_request({'input_value':json.dumps({'requestType':'voc','page':0,'start_date':'20261029','channels':['02'],'survey_mode_list':[]})})
        result=await self.agent.execute(request.analysis,CONTEXT)
        output=export_summary(result)
        self.assertIn('modelVersion',output['metadata'])
        first=output['page']['content'][0]
        self.assertIn('aspectSuccess',first)
        self.assertEqual(first['feedbackType'],'02')
        self.assertEqual(first['recordKey']['source_system'], 'SYNTHETIC')

    async def test_execute_unit_only_runs_selected_tool(self):
        records=[VocInfo(voc_id='unit',text='모바일 앱 로그인 오류가 나서 실망했습니다.')]
        result=await self.agent.execute_unit('emotion_analysis',records,AnalysisRequest(),CONTEXT)
        self.assertIsNot(result,records)
        self.assertEqual(result[0].stages['emotion'].status.value,'success')
        self.assertEqual(result[0].stages['filter'].status.value,'not_run')
        self.assertIsNone(records[0].emotion)
        with self.assertRaises(ValueError):
            await self.agent.execute_unit('write_database',records,AnalysisRequest(),CONTEXT)

    async def test_unit_redacts_existing_derived_evidence(self):
        record=VocInfo(voc_id='unit-pii',text='모바일 앱 로그인 화면은 만족했습니다.')
        record.aspects=[AspectInfo(seq=1,quality_code='SYN',quality_name='합성',aspect_code='SYN_LOGIN',aspect_name='합성',evidence_quote='person@example.com')]
        record.stages['aspect'].evidence=['person@example.com']
        result=await self.agent.execute_unit('voc_filtering',[record],AnalysisRequest(),CONTEXT)
        self.assertNotIn('person@example.com',result[0].model_dump_json())
        self.assertIn('person@example.com',record.model_dump_json())

    def test_filter_exact_length_boundary_and_group_chunks(self):
        self.assertFalse(VocFilter._length_check('12345',5))
        self.assertTrue(VocFilter._length_check('123456',5))
        rows=[VocInfo(voc_id=str(i),text='합성 로그인 경험 문장입니다.',journey_code='18' if i<2 else '00')for i in range(4)]
        grouped=VocFilter.convert(rows)['02']
        self.assertEqual([len(v)for v in grouped.values()],[2,2])

class CliTests(unittest.TestCase):
    def test_demo_and_sqlite_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('builtins.print'):
                self.assertEqual(main(['init-db',str(root/'demo.db')]),0)
                self.assertEqual(main(['demo','--database',str(root/'demo.db'),'--out',str(root/'demo.json'),'--csv',str(root/'demo.csv'),'--store',str(root/'runs')]),0)
            self.assertEqual(json.loads((root/'demo.json').read_text())['page']['numberOfElements'],11)
            self.assertTrue((root/'demo.csv').read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_csv_formula_escape(self):
        self.assertEqual(safe_cell('=1+1'),"'=1+1")
        self.assertEqual(safe_cell('정상'), '정상')
