"""Independent audit regressions; all inputs synthetic and providers in memory."""
import asyncio
import unittest
from analysis_api.models import VocInput, RequestContext
from analysis_api.taxonomy import AspectDefinition, Taxonomy, review_visibility, emotion_info
from analysis_api.providers import ProviderGateway
from analysis_api.security import current_context, use_context


def voc(text='가'*41, **kwargs):
    return VocInput(voc_id='audit', text=text, **kwargs)


def definition(**kwargs):
    payload = dict(target_code='07', stage_code='01', aspect_code='SYN_A', aspect_name='이름', description='합성 설명', quality_code='SQ_A', quality_name='품질', keywords=['합성'])
    payload.update(kwargs)
    return AspectDefinition(**payload)


class BoundaryAudit(unittest.TestCase):
    def test_full_source_key_requires_explicit_date(self):
        with self.assertRaises(ValueError):
            voc(dataset_id='G', survey_id='Q', response_id='U', question_id='I')

    def test_no_stage_clears_stale_stage_label(self):
        self.assertEqual(voc(journey_code='00',journey_name='stale').journey_name,'')

    def test_emotion_negative_is_not_complaint(self):
        regret=emotion_info('04','아쉽다')
        complaint=emotion_info('05','실망했다')
        self.assertEqual((regret.polarity_code,regret.feedback_type),('02','03'))
        self.assertEqual((complaint.polarity_code,complaint.feedback_type),('02','02'))

    def test_offer_length_boundaries_and_no_manager_reenable(self):
        for length,reason in [(20,'WORD_LENGTH_20'),(21,'WORD_LENGTH_40'),(40,'WORD_LENGTH_40')]:
            decision=review_visibility(voc('가 '*length,channel_code='07',journey_code='04'))
            self.assertFalse(decision.reviewer_visible)
            self.assertFalse(decision.operator_visible)
            self.assertIn(reason,decision.reasons)
        decision=review_visibility(voc(channel_code='07',journey_code='04'))
        self.assertTrue(decision.reviewer_visible)
        self.assertFalse(decision.operator_visible)

    def test_offer_exception_only_07_08_and_blank_ignored(self):
        for target in ['02','06','09']:
            decision=review_visibility(voc(channel_code=target,journey_code='04'),['','가'])
            self.assertTrue(decision.operator_visible)
            self.assertTrue(decision.reviewer_visible)
        for target in ['07','08']:
            self.assertFalse(review_visibility(voc(channel_code=target),['가']).operator_visible)
            self.assertTrue(review_visibility(voc(channel_code=target),['','  ']).operator_visible)

    def test_stage_zero_union_and_explicit_stage_restriction(self):
        t=Taxonomy([definition(),definition(stage_code='02'),definition(stage_code='02',aspect_code='SYN_B')])
        self.assertEqual({e.aspect_code for e in t.candidates('07','00')},{'SYN_A','SYN_B'})
        self.assertEqual([e.aspect_code for e in t.candidates('07','01')],['SYN_A'])
        self.assertEqual(t.candidates('08','00'),[])

    def test_blank_taxonomy_keyword_rejected(self):
        with self.assertRaises(ValueError):
            definition(keywords=[''])

    def test_duplicate_conflicting_taxonomy_rejected(self):
        with self.assertRaises(ValueError):
            Taxonomy([definition(),definition()])
        with self.assertRaises(ValueError):
            Taxonomy([definition(),definition(stage_code='02',aspect_name='conflict')])


class GatewayAudit(unittest.IsolatedAsyncioTestCase):
    async def test_cross_item_id_invalidates_whole_batch(self):
        class Provider:
            name='mock'; inference_kind='mock'
            async def complete(self,stage,payload,context):
                return {'results':{'a':{'value':'good'},'unknown':{'value':'bad'}}}
        called=[]
        out=await ProviderGateway(Provider(),retry_limit=0).call('filter',[{'id':'a'},{'id':'b'}],RequestContext(request_id='audit',actor_ref='tester'),lambda r,i: called.append(i['id']))
        self.assertEqual(called,[])
        self.assertTrue(all(x.error_code=='UNEXPECTED_RESULT_ID' for x in out.values()))

    async def test_retry_only_pending(self):
        class Provider:
            name='mock'; inference_kind='mock'
            def __init__(self): self.calls=[]
            async def complete(self,stage,payload,context):
                self.calls.append([x['id'] for x in payload['items']])
                return {'results': {'a':{'value':1}} if len(self.calls)==1 else {'b':{'value':2}}}
        p=Provider()
        out=await ProviderGateway(p,retry_limit=1).call('entity',[{'id':'a'},{'id':'b'}],RequestContext(request_id='audit',actor_ref='tester'),lambda r,i:r['value'])
        self.assertEqual(p.calls,[['a','b'],['b']])
        self.assertEqual([out[k].value for k in ['a','b']],[1,2])

    async def test_context_reset_on_cancellation(self):
        entered=asyncio.Event()
        async def work():
            with use_context(RequestContext(request_id='audit',actor_ref='tester')):
                entered.set()
                await asyncio.Event().wait()
        task=asyncio.create_task(work())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertIsNone(current_context())

if __name__=='__main__': unittest.main()
