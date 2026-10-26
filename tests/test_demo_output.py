"""Speech ownership, revision/age gates, duplicate suppression and short ARA wire."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from modules.demo_output import SpeechOutput,avatar_image
from modules.demo_response_transport import ResponseTextClient


class Process:
    def __init__(self):self.returncode=None;self.stopped=False
    def poll(self):return self.returncode
    def terminate(self):self.stopped=True;self.returncode=0
    def wait(self,timeout=None):self.returncode=0;return 0
    def kill(self):self.stopped=True;self.returncode=-9


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.wave=self.root/'voice.wav';self.wave.write_bytes(b'fixture-not-played')
        self.catalog={'records':[{'generated':True,'path':str(self.wave),'text':'案内します。',
                                  'sha256':hashlib.sha256(self.wave.read_bytes()).hexdigest()}]}
        self.processes=[]
    def launch(self,args,**kwargs):
        self.assertEqual(args[:4],['aplay','-q','-D','null']);p=Process();self.processes.append(p);return p
    def controller(self,device='null'):
        obj=SpeechOutput(self.root,self.catalog,device=device,cooldown_seconds=0,popen=self.launch)
        self.addCleanup(obj.stop);return obj
    def snapshot(self,revision='s:0',text='案内します。',now=10_000_000_000):
        return dict(preference_revision=revision,preference_history=[dict(source='explicit_user',preference='minimal_guidance')],
                    response=dict(preference_revision=revision,text=text,status='accepted',source_end_ns=now,
                                  available_ns=now,window_id='w1'))
    def test_play_once_and_stop_on_preference_change(self):
        obj=self.controller();snap=self.snapshot();obj.update(snap,10_000_000_000);obj.update(snap,10_000_000_001)
        self.assertEqual(len(self.processes),1)
        snap=self.snapshot('s:1',text='');obj.update(snap,10_000_000_002)
        self.assertTrue(self.processes[0].stopped)
    def test_stale_result_never_plays(self):
        obj=self.controller();snap=self.snapshot();snap['response']['preference_revision']='old'
        self.assertEqual(obj.update(snap,10_000_000_000)['status'],'suppressed');self.assertFalse(self.processes)
    def test_future_expired_or_faulted_never_plays(self):
        for now,extra in [(9_000_000_000,{}),(400_000_000_000,{}),(10_000_000_000,{'input_faults':['camera']})]:
            obj=self.controller();self.assertEqual(obj.update({**self.snapshot(),**extra},now)['status'],'suppressed')
        self.assertFalse(self.processes)
    def test_no_conversation_and_empty_text_stop_audio(self):
        obj=self.controller();snap=self.snapshot();obj.update(snap,10_000_000_000)
        snap['preference_history']=[dict(source='explicit_user',preference='no_conversation')]
        obj.update(snap,10_000_000_001);self.assertTrue(self.processes[0].stopped)
    def test_same_text_new_window_does_not_repeat_but_explicit_revision_can(self):
        obj=self.controller();snap=self.snapshot();obj.update(snap,10_000_000_000)
        self.processes[0].returncode=0;snap['response']['window_id']='w2';snap['response']['available_ns']+=1
        obj.update(snap,10_000_000_002);self.assertEqual(len(self.processes),1)
        obj.update(self.snapshot('s:1'),10_000_000_003);self.assertEqual(len(self.processes),2)
    def test_muted_and_unlisted_text_do_not_launch(self):
        self.assertEqual(self.controller(None).update(self.snapshot(),10_000_000_000)['status'],'muted')
        self.assertEqual(self.controller().update(self.snapshot(text='別の文'),10_000_000_000)['status'],'text_only')
        self.assertFalse(self.processes)
    def test_catalog_hash_rejected(self):
        self.catalog['records'][0]['sha256']='bad'
        with self.assertRaisesRegex(ValueError,'hash_mismatch'):self.controller()
    def test_short_wire_normalizes_and_preserves_raw(self):
        class Client:
            def generate(self,messages,schema,**kwargs):
                self.schema=schema
                return dict(status='generated',text=json.dumps('案内します。'),raw_http_body='original_http')
        client=Client();expected=dict(schema='ara-response-v1',policy='brief_guidance',text='案内します。')
        result=ResponseTextClient(client).generate([dict(content=json.dumps(expected))],{},deadline=20)
        self.assertEqual(json.loads(result['text']),expected);self.assertEqual(result['raw_http_body'],'original_http')
        self.assertEqual(client.schema,{'type':'string','const':'案内します。'})
    def test_wire_different_text_rejected(self):
        class Client:
            def generate(self,*args,**kwargs):return dict(status='generated',text='"different"')
        result=ResponseTextClient(Client()).generate([dict(content=json.dumps(dict(text='案内します。')))],{})
        self.assertEqual(result['status'],'invalid_output')


if __name__=='__main__':unittest.main()
