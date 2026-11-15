"""Independent component probes; no live model/network calls."""
import unittest
from pathlib import Path
from analysis_api.components import AspectMapper, EmotionAnalyzer, EntityWordDetector, VocProblemReasonDetector
from analysis_api.models import AnalysisRequest, VocInfo, RequestContext
from analysis_api.providers import ProviderGateway, OfflineReferenceProvider, NotConfiguredProvider
from analysis_api.taxonomy import Taxonomy

ROOT=Path(__file__).resolve().parent.parent

class ComponentAudit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.taxonomy=Taxonomy.load(ROOT/'fixtures/aspect_catalog.json')
        self.req=AnalysisRequest(retry_limit=0,voc_chunk_size=1)
        self.context=RequestContext(request_id='audit-components',actor_ref='tester')
        self.gateway=ProviderGateway(OfflineReferenceProvider(),retry_limit=0)

    def component(self,cls,gateway=None):
        return cls(gateway or self.gateway,self.taxonomy,self.req,self.context)

    async def test_later_complaint_not_hidden_by_first_regret_aspect(self):
        rows=[VocInfo(voc_id='mixed',text='로그인 화면이 느려 아쉬웠습니다. 결제가 실패해서 실망했습니다.')]
        await self.component(AspectMapper).execute(rows)
        await self.component(EmotionAnalyzer).execute(rows)
        self.assertTrue(any(c.emotion.feedback_type=='02' for c in rows[0].aspects))
        await self.component(VocProblemReasonDetector).execute(rows)
        self.assertTrue(rows[0].reason_success, rows[0].stages['reason'])

    async def test_entity_no_cross_voc_reuse(self):
        rows=[VocInfo(voc_id=x,text='앱 로그인 화면이 느려서 아쉬웠습니다.') for x in ['one','two']]
        await self.component(EntityWordDetector).execute(rows)
        self.assertFalse(rows[1].entity_result.reused)

    async def test_entity_failed_rerun_clears_per_aspect_old_value(self):
        rows=[VocInfo(voc_id='rerun',text='앱 로그인 화면이 느려서 아쉬웠습니다.')]
        await self.component(AspectMapper).execute(rows)
        await self.component(EntityWordDetector).execute(rows)
        self.assertIsNotNone(rows[0].aspects[0].entity)
        failing=ProviderGateway(NotConfiguredProvider(),retry_limit=0)
        await self.component(EntityWordDetector,failing).execute(rows)
        self.assertIsNone(rows[0].aspects[0].entity)
        self.assertFalse(rows[0].aspects[0].entity_success)

    async def test_previous_terms_reach_later_chunk(self):
        class Provider:
            name='mock'; inference_kind='mock'
            def __init__(self):self.items=[]
            async def complete(self,stage,payload,context):
                self.items.extend(payload['items'])
                return {'results': {i['id']:{'product_term':'앱','quality_term':'','product_evidence':'앱','quality_evidence':''} for i in payload['items']}}
        p=Provider(); gate=ProviderGateway(p,retry_limit=0)
        rows=[VocInfo(voc_id='one',text='앱 첫번째 합성 의견'),VocInfo(voc_id='two',text='앱 두번째 합성 의견')]
        await self.component(EntityWordDetector,gate).execute(rows)
        self.assertIn('앱',p.items[1]['previous_terms'])

if __name__=='__main__':unittest.main()
