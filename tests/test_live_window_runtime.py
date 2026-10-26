"""Window isolation and evidence ownership when live processes are reused."""
from collections import deque
from concurrent.futures import Future
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from models.detector import EmotionDetector
from models.tracking import IoUTracker
from modules.live_window_runtime import LiveWindowRuntime
from modules.observation_integration import ObservationIntegrationSession, WORKSPACE, REPO
from modules.observation_worker_client import digest


def detector():
    result = EmotionDetector.__new__(EmotionDetector)
    result.config = {'face_detection': {'tracking': {'iou_threshold': .4, 'max_missed_frames': 2}}}
    result.tracker = IoUTracker(.4, 2)
    result.emotion_history = {}
    result.next_untracked_face_id = 0
    result.timings = {'total': []}
    result.face_model = types.SimpleNamespace(close=lambda: None)
    result.emotion_model = object()
    return result


class LiveRuntimeTests(unittest.TestCase):
    def test_worker_pool_purge_keeps_only_newest_uuid_requests(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            output = Path(temporary)
            pool = output/'worker_pool/sensevoice'
            pool.mkdir(parents=True)
            names = [f'{value:032x}' for value in range(4)]
            for value, name in enumerate(names):
                folder = pool/name
                folder.mkdir()
                (folder/'reply.json').write_text('{}')
                os.utime(folder, ns=(value + 1, value + 1))
            (pool/'persistent_001.log').write_text('keep log')
            runtime = LiveWindowRuntime.__new__(LiveWindowRuntime)
            runtime.output = output
            runtime.worker_request_retention = 2
            runtime.clients = {'sensevoice': types.SimpleNamespace(directory=pool)}
            state = runtime.purge_worker_requests()
            self.assertEqual(sorted(path.name for path in pool.iterdir() if path.is_dir()), names[-2:])
            self.assertTrue((pool/'persistent_001.log').is_file())
            self.assertEqual(state['backends']['sensevoice']['requests_after'], 2)

    def test_new_window_resets_people_and_smoothing_but_keeps_models(self):
        instance = detector()
        model, face = instance.emotion_model, instance.face_model
        instance.tracker.update([(0, 0, 10, 10)])
        instance.emotion_history[0] = deque([('happy', .9)])
        instance.next_untracked_face_id = 7
        instance.timings['total'].append(.3)
        instance.reset_window_state()
        self.assertIs(instance.emotion_model, model)
        self.assertIs(instance.face_model, face)
        self.assertFalse(instance.emotion_history)
        self.assertFalse(instance.tracker._tracks)
        self.assertEqual(instance.next_untracked_face_id, 0)
        self.assertEqual(instance.tracker.update([(100, 100, 110, 110)]), [0])
        self.assertEqual(instance.timings, {'total': []})

    def test_two_apps_share_model_with_independent_window_state_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            output = Path(temporary)
            config = {'live': {'output': str(output)}, 'observation_integration': {
                'audio_python': sys.executable, 'visual_python': sys.executable}}
            def app(config_path, *, emotion_detector, observation_clients):
                return types.SimpleNamespace(emotion_detector=emotion_detector or detector())
            with patch.dict(sys.modules, {'multimodal_app': types.SimpleNamespace(MultiModalEmotionApp=app)}):
                runtime = LiveWindowRuntime(config, WORKSPACE, REPO)
                try:
                    paths = [output/'first/config.yaml', output/'second/config.yaml']
                    for path in paths: path.parent.mkdir()
                    first = runtime.new_app(paths[0], preparation=True)
                    self.assertEqual(runtime.windows_processed,0)
                    first.emotion_detector.tracker.update([(0, 0, 20, 20)])
                    first.emotion_detector.emotion_history[0] = deque([('sad', .7)])
                    second = runtime.new_app(paths[1])
                    self.assertIs(first.emotion_detector, second.emotion_detector)
                    a, b = [json.loads((p.parent/'runtime_state.json').read_text()) for p in paths]
                    self.assertEqual(a['detector_instance_id'], b['detector_instance_id'])
                    self.assertNotEqual(a['window_state_id'], b['window_state_id'])
                    self.assertEqual(b['tracks_at_start'], 0)
                    self.assertEqual(b['smoothing_entries_at_start'], 0)
                    self.assertEqual(a['purpose'],'startup_preparation')
                    self.assertEqual(b['window_number'],1)
                    (output/'preparation_state.json').write_text(json.dumps({'ready':True,'state':'ready'}))
                finally:
                    runtime.close()
                self.assertTrue(all(not c.thread.is_alive() for c in runtime.clients.values()))
                self.assertTrue(json.loads((output/'window_runtime_final.json').read_text())['all_owned_workers_exited'])
                self.assertFalse(json.loads((output/'preparation_state.json').read_text())['ready'])

    def test_shared_worker_evidence_remains_inside_its_window(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as temporary:
            root = Path(temporary)
            source = root/'pool/audio_observation.json'
            source.parent.mkdir()
            source.write_text('{"observation": "unique-window-value"}')
            future = Future()
            future.set_result({'request_id': 'abc123', 'status': 'ok', 'artifacts': [{
                'path': source.name, 'resolved_path': str(source), 'sha256': digest(source)}]})
            calls = []
            client = types.SimpleNamespace(submit=lambda *a, **k: (calls.append((a, k)) or future))
            session = ObservationIntegrationSession.__new__(ObservationIntegrationSession)
            session.config = {'audio_python': sys.executable, 'worker_mode': 'online'}
            session.directory = root/'window/observation_integration'
            session.directory.mkdir(parents=True)
            session.clients = {'sensevoice': client}
            session.duration = 1
            session.media_begin, session.media_end, session.samples = 0, 1, 16000
            session.pcm_sha, session.clip_id = 'pcm', 'window1'
            session.common_audio = root/'audio.wav'
            result = session.invoke('sensevoice')
            artifact = Path(result['artifacts'][0]['resolved_path'])
            self.assertTrue(artifact.is_relative_to(session.directory))
            self.assertNotEqual(source, artifact)
            source.unlink()
            self.assertEqual(digest(artifact), result['artifacts'][0]['sha256'])
            self.assertEqual(calls[0][1]['mode'], 'online')


if __name__ == '__main__':
    unittest.main()
