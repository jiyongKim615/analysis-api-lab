"""Independent end-to-end pipeline tests, synthetic in-memory providers only."""
import unittest
from pathlib import Path
from analysis_api.models import AnalysisRequest, RequestContext, VocInput
from analysis_api.pipeline import AnalysisPipeline
from analysis_api.providers import OfflineReferenceProvider,NotConfiguredProvider
from analysis_api.repository import JSONFixtureRepository
from analysis_api.taxonomy import Taxonomy
from analysis_api.compatibility import parse_run_request

ROOT=Path(__file__).resolve().parent.parent
PII='audit.person@example.com'

class PipelineAudit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.taxonomy=Taxonomy.load(ROOT/'fixtures/aspect_catalog.json')
        self.context=RequestContext(request_id='audit-pipeline',actor_ref='tester')
        self.repo=JSONFixtureRepository(ROOT/'fixtures/vocs.json')

    async def run_agent(self,provider=None,request=None,reference=None):
        return await AnalysisPipeline(self.repo,self.taxonomy,provider,reference_data=reference).execute(request or AnalysisRequest(retry_limit=0),self.context)

    async def test_fixture_pipeline_terminal(self):
        result=await self.run_agent()
        self.assertEqual(result.page.number_of_elements,11)
        self.assertTrue(all(s.status.value!='not_run' for v in result.page.content for s in v.stages.values()))
        self.assertEqual(result.page.content[6].filtered_reason,'DUPLICATE_PK')

    async def test_failure_is_not_neutral(self):
        result=await self.run_agent(NotConfiguredProvider())
        for v in result.page.content:
            self.assertIsNone(v.emotion)
            self.assertEqual(v.stages['emotion'].status.value,'skipped')

    async def test_all_provider_payload_fields_are_redacted(self):
        class Capture(OfflineReferenceProvider):
            def __init__(self): self.payloads=[]
            async def complete(self,stage,payload,context):
                self.payloads.append(payload)
                return await super().complete(stage,payload,context)
        p=Capture()
        reference={'version':'audit-v1','previous_product_terms':{'02_SYN_LOGIN':[PII]},'previous_quality_terms':{},'exception_words':[]}
        request=AnalysisRequest(retry_limit=0,vocs=[VocInput(voc_id='private',text='앱 로그인 오류가 반복되어 정말 실망했습니다.')])
        await self.run_agent(p,request,reference)
        self.assertNotIn(PII,str(p.payloads))

    async def test_unknown_model_code_is_redacted_in_output(self):
        class BadCode(OfflineReferenceProvider):
            async def complete(self,stage,payload,context):
                if stage=='aspect':
                    return {'results':{i['id']:{'matches':[{'aspect_code':PII,'evidence_quote':i['text'],'polarity':'negative'}]} for i in payload['items']}}
                return await super().complete(stage,payload,context)
        request=AnalysisRequest(retry_limit=0,vocs=[VocInput(voc_id='private',text='앱 로그인 오류가 반복되어 정말 실망했습니다.')])
        result=await self.run_agent(BadCode(),request)
        self.assertNotIn(PII,result.model_dump_json())

    def test_conflicting_route_alias_rejected(self):
        with self.assertRaises(ValueError):parse_run_request({'requestType':'voc','request_type':'report'})

    def test_conflicting_stream_alias_rejected(self):
        with self.assertRaises(ValueError):parse_run_request({'stream':True,'is_stream':False})

if __name__=='__main__':unittest.main()
