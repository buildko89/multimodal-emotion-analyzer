"""Prove overlap, once-only audio reuse, errors and ownership cleanup."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from modules.observation_integration import ObservationIntegrationSession


class ParallelObservationTests(unittest.TestCase):
    def make(self,root,invoke):
        session=ObservationIntegrationSession.__new__(ObservationIntegrationSession)
        session.config={'asr_backend':'sensevoice','ser_backends':['sensevoice','emotion2vec']}
        session.audio_replies={};session.backend_futures={};session.backend_executor=None
        session.directory=Path(root);session.invoke=invoke
        return session

    def test_all_independent_backends_overlap_and_audio_is_never_submitted_twice(self):
        barrier=threading.Barrier(3);calls=[]
        def invoke(backend):
            calls.append(backend);barrier.wait(timeout=2)
            return {'status':'ok','backend':backend}
        with tempfile.TemporaryDirectory() as root:
            session=self.make(root,invoke)
            try:
                session.start_backend_tasks()
                sense=session.audio_reply('sensevoice')
                self.assertIs(sense,session.audio_reply('sensevoice'))
                self.assertEqual(session.audio_reply('emotion2vec')['status'],'ok')
                self.assertEqual(session.backend_reply('visual')['status'],'ok')
            finally:session.close()
            self.assertCountEqual(calls,['visual','sensevoice','emotion2vec'])
            status=json.loads((Path(root)/'parallel_backends.json').read_text(encoding='utf-8'))
            self.assertTrue(status['all_backend_tasks_done'])
            self.assertTrue(status['owned_threads_stopped'])
            session.close()

    def test_exception_still_drains_other_jobs_and_closes_owned_threads(self):
        barrier=threading.Barrier(3)
        def invoke(backend):
            barrier.wait(timeout=2)
            if backend=='sensevoice':raise ValueError('measured_failure')
            return {'status':'ok'}
        with tempfile.TemporaryDirectory() as root:
            session=self.make(root,invoke);session.start_backend_tasks()
            try:
                with self.assertRaisesRegex(ValueError,'measured_failure'):session.audio_reply('sensevoice')
            finally:session.close()
            status=json.loads((Path(root)/'parallel_backends.json').read_text(encoding='utf-8'))
            self.assertTrue(status['all_backend_tasks_done'] and status['owned_threads_stopped'])

    def test_sequential_default_and_failed_reply_keep_existing_cache_semantics(self):
        calls=[]
        with tempfile.TemporaryDirectory() as root:
            session=self.make(root,lambda backend:calls.append(backend) or {'status':'inference_failure'})
            self.assertEqual(session.audio_reply('sensevoice')['status'],'inference_failure')
            self.assertEqual(session.audio_reply('sensevoice')['status'],'inference_failure')
            self.assertEqual(calls,['sensevoice']);session.close()
