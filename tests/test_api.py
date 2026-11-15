"""Independent in-process ASGI probes; no sockets or live providers."""
import os
from pathlib import Path
import json
import asyncio
import tempfile
import unittest
import httpx
from unittest.mock import patch
with patch.dict(os.environ, {"CX_PROVIDER":"offline", "CX_ALLOW_EXTERNAL":"false", "CX_REQUIRE_AUTH":"false", "CX_API_TOKEN":"", "CX_OUTPUT_DIR":str(Path(__file__).resolve().parent.parent/"output/test-import")}):
    from analysis_api.api import create_app
from analysis_api.config import Settings

class ApiAudit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cx-api-test-')
        self.config=Settings(output_dir=Path(self.tmp.name),api_token='audit-user-one',require_auth=True)
        self.app=create_app(self.config)
        self.headers={'Authorization':'Bearer audit-user-one'}
    def tearDown(self):self.tmp.cleanup()
    def client(self,app=None,host='127.0.0.1'):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app or self.app,client=(host,123),raise_app_exceptions=False),base_url='http://test')

    async def test_auth_required_and_user_body_cannot_impersonate(self):
        async with self.client() as client:
            self.assertEqual((await client.post('/run_agent',json={})).status_code,401)
            result=await client.post('/run_agent',headers=self.headers,json={'user_id':'other'})
            self.assertEqual(result.status_code,422)
            self.assertNotIn('other',result.text)

    async def test_json_persist_get_and_actor_scope(self):
        async with self.client() as client:
            result=await client.post('/run_agent',headers=self.headers,json={'analysis':{'vocs':[]}})
            self.assertEqual(result.status_code,200,result.text)
            data=result.json(); run=data['metadata']['runId']
            loaded=await client.get('/runs/'+run,headers=self.headers)
            self.assertEqual(loaded.json(),data)
        other=create_app(Settings(output_dir=Path(self.tmp.name),api_token='audit-user-two',require_auth=True))
        async with self.client(other) as client:
            response=await client.get('/runs/'+run,headers={'Authorization':'Bearer audit-user-two'})
            self.assertEqual(response.status_code,404)

    async def test_sse_completion_and_no_placeholder_routes(self):
        async with self.client() as client:
            result=await client.post('/run_agent',headers=self.headers,json={'isStream':True,'analysis':{'vocs':[]}})
            self.assertEqual(result.status_code,200)
            events=[json.loads(x[6:]) for x in result.text.splitlines() if x.startswith('data: {')]
            self.assertIn('CHUNK',[e['event'] for e in events])
            self.assertEqual(events[-1]['event'],'DONE')
            self.assertTrue(events[-1]['content']['success'])
            for route in ['report','make_report','governance_message','voc_suggestion','voc_feedback']:
                response=await client.post('/run_agent',headers=self.headers,json={'requestType':route,'unsupportedPayload':'whatever'})
                self.assertEqual(response.status_code,501,(route,response.text))
                self.assertEqual(response.json()['error']['code'],'UNSUPPORTED')

    async def test_non_ascii_bearer_is_clean_401(self):
        async with self.client() as client:
            response=await client.post('/run_agent',headers=[(b'Authorization',b'Bearer \xff')],json={})
            self.assertEqual(response.status_code,401,response.text)

    async def test_unauthenticated_remote_denied(self):
        app=create_app(Settings(output_dir=Path(self.tmp.name),api_token=None))
        async with self.client(app,host='203.0.113.8') as client:
            response=await client.post('/run_agent',json={})
            self.assertEqual(response.status_code,403)

    async def test_malformed_payload_no_raw_value_echo(self):
        async with self.client() as client:
            response=await client.post('/run_agent',headers=self.headers,json={'analysis':{'vocs':[{'voc_id':'bad','text':'person@example.com','unknown':'other'}]}})
            self.assertEqual(response.status_code,422)
            self.assertNotIn('person@example.com',response.text)

    async def test_sse_disconnect_cancels_provider_and_no_partial_save(self):
        entered=asyncio.Event(); cancelled=asyncio.Event()
        class BlockingProvider:
            name='audit-blocking'; inference_kind='mock'
            async def complete(self,stage,payload,context):
                entered.set()
                try: await asyncio.Event().wait()
                finally: cancelled.set()
        app=create_app(self.config,provider=BlockingProvider())
        body=json.dumps({'isStream':True,'analysis':{'vocs':[{'voc_id':'disconnect','text':'앱 로그인 오류가 반복되어 정말 실망했습니다.'}]}}).encode()
        scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.0'},'http_version':'1.1','method':'POST','scheme':'http','path':'/run_agent','raw_path':b'/run_agent','query_string':b'','root_path':'','headers':[(b'authorization',b'Bearer audit-user-one'),(b'content-type',b'application/json')],'client':('127.0.0.1',123),'server':('test',80)}
        first=True; sent=[]
        async def receive():
            nonlocal first
            if first:
                first=False
                return {'type':'http.request','body':body,'more_body':False}
            await entered.wait()
            return {'type':'http.disconnect'}
        async def send(message):sent.append(message)
        await asyncio.wait_for(app(scope,receive,send),timeout=2)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(list(Path(self.tmp.name).rglob('*.json')),[])

if __name__=='__main__':unittest.main()
