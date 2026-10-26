"""Real lightweight subprocesses prove reuse, isolation, expiry and restart."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from modules.persistent_observation_client import PersistentObservationClient

STUB = '''import hashlib,json,pathlib,sys,time
for line in sys.stdin:
    transport=json.loads(line);path=pathlib.Path(transport['request_path']);request=json.loads(path.read_text())
    if request.get('delay'):time.sleep(request['delay'])
    if request.get('crash'):sys.exit(7)
    data={'call_input':request['input_sha256']}
    artifact=path.parent/'observed.json';artifact.write_text(json.dumps(data))
    reply={k:request[k] for k in ('protocol','request_id','backend_id','input_sha256')}
    reply.update(status='ok',artifacts=[{'path':artifact.name,'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}])
    (path.parent/'reply.json').write_text(json.dumps(reply))
    rid='wrong' if request.get('wrong_id') else request['request_id']
    print(json.dumps({'request_id':rid,'status':'completed'}),flush=True)
'''


class PersistentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.script=self.root/'stub.py';self.script.write_text(STUB)
        self.source=self.root/'input.wav';self.source.write_bytes(b'first actual input')
        self.client=PersistentObservationClient([sys.executable,'-B',self.script],self.root/'worker',
            backend_id='sensevoice',allowed_root=self.root)

    def tearDown(self):
        self.client.close();self.temp.cleanup()

    def submit(self,**kwargs):
        return self.client.submit(self.source,[0,1],timeout_s=kwargs.pop('timeout_s',2),
            mode='online',available_until_s=1,**kwargs).result(timeout=4)

    def test_one_process_distinct_inputs_and_request_ids(self):
        first=self.submit();self.source.write_bytes(b'changed actual input');second=self.submit()
        self.assertEqual(first['status'],'ok');self.assertEqual(second['status'],'ok')
        self.assertNotEqual(first['request_id'],second['request_id'])
        self.assertNotEqual(first['input_sha256'],second['input_sha256'])
        self.assertEqual(first['worker_transport']['pid'],second['worker_transport']['pid'])
        self.assertEqual(self.client.start_count,1)
        self.assertNotEqual(json.loads(Path(first['artifacts'][0]['resolved_path']).read_text()),
                            json.loads(Path(second['artifacts'][0]['resolved_path']).read_text()))

    def test_timeout_reaps_old_process_and_next_request_uses_new_one(self):
        self.assertEqual(self.submit(delay=.3,timeout_s=.05)['reason'],'timeout')
        self.assertIsNone(self.client.process)
        result=self.submit()
        self.assertEqual(result['status'],'ok');self.assertEqual(self.client.start_count,2)

    def test_mismatched_reply_and_worker_crash_are_not_reused(self):
        for kwargs in ({'wrong_id':True},{'crash':True}):
            self.assertEqual(self.submit(**kwargs)['status'],'inference_failure')
            self.assertIsNone(self.client.process)
            self.assertEqual(self.submit()['status'],'ok')

    def test_future_interval_rejected_before_worker_starts(self):
        with self.assertRaisesRegex(ValueError,'Future input'):
            self.client.submit(self.source,[0,2],mode='online',available_until_s=1)
        self.assertEqual(self.client.start_count,0)

    def test_close_releases_owned_process(self):
        self.submit();process=self.client.process
        self.client.close()
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.client.process)

    def test_visual_persistence_requires_separate_tracking_contract(self):
        with self.assertRaisesRegex(ValueError,'audio_backends_only'):
            PersistentObservationClient([],self.root/'visual',backend_id='visual',allowed_root=self.root)


if __name__=='__main__':unittest.main()
