"""Visual-only environment and post-result reclamation, with lightweight backends."""
import ast
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from modules.observation_worker_client import digest,validate_request,write_json
from modules.persistent_observation_client import PersistentVisualClient,PersistentObservationClient


class VisualMemoryPolicyTests(unittest.TestCase):
    def test_visual_arena_default_does_not_change_audio_environment(self):
        with patch('sys.platform','linux'),patch.dict(os.environ,{},clear=True):
            self.assertEqual(PersistentVisualClient.environment()['MALLOC_ARENA_MAX'],'2')
            self.assertNotIn('MALLOC_ARENA_MAX',PersistentObservationClient.environment())

    def test_visual_preserves_explicit_arena_setting(self):
        with patch('sys.platform','linux'),patch.dict(os.environ,{'MALLOC_ARENA_MAX':'7'},clear=True):
            self.assertEqual(PersistentVisualClient.environment()['MALLOC_ARENA_MAX'],'7')

    def test_other_platform_does_not_set_glibc_options(self):
        with patch('sys.platform','win32'),patch.dict(os.environ,{},clear=True):
            self.assertNotIn('MALLOC_ARENA_MAX',PersistentVisualClient.environment())

    def execute(self,backend):
        source=Path(__file__).resolve().parents[1]/'scripts/observation_worker.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'request';folder.mkdir();video=root/'saved.mp4';video.write_bytes(b'saved file')
            request=dict(protocol='observation-worker-v1',request_id='request1',backend_id=backend,input_path=str(video),input_sha256=digest(video),
                interval_seconds=[0,1],deadline_monotonic_ns=time.monotonic_ns()+10**10,mode='offline',available_until_s=1)
            path=folder/'request.json';write_json(path,request)
            events=[]
            def visual(request,folder):
                events.append('visual');(folder/'observed.json').write_bytes(b'{"meaning":"preserved"}');return ['observed.json']
            def audio(request,folder,adapters):
                events.append('audio');(folder/'observed.json').write_bytes(b'{"meaning":"preserved"}');return ['observed.json']
            def reclaim():events.append('reclaim');return dict(status='released',elapsed_ns=1)
            namespace=dict(Path=Path,WORKSPACE=root,json=json,time=time,validate_request=validate_request,digest=digest,
                visual=visual,audio=audio,process_memory=lambda:dict(status='not_measured'),write_json=write_json)
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
            with patch('modules.native_memory_reclaim.release_idle_pages',side_effect=reclaim):
                self.assertEqual(namespace['execute'](path),'request1')
            reply=json.loads((folder/'reply.json').read_text())
            self.assertEqual(reply['status'],'ok');self.assertEqual(reply['artifacts'][0]['sha256'],digest(folder/'observed.json'))
            return events,reply

    def test_reclaim_only_after_visual_result_is_written(self):
        events,reply=self.execute('visual')
        self.assertEqual(events,['visual','reclaim']);self.assertEqual(reply['native_memory_reclamation']['status'],'released')

    def test_audio_keeps_original_reply_and_never_reclaims(self):
        events,reply=self.execute('sensevoice')
        self.assertEqual(events,['audio']);self.assertNotIn('native_memory_reclamation',reply)


if __name__=='__main__':unittest.main()
