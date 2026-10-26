"""Contract, conservative association and real subprocess failure regression."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

from modules.observation_integration import SCHEMA, QUESTIONS, REPO, WORKSPACE, ObservationIntegrationSession, person_windows, validate_envelope
from modules.observation_worker_client import SubprocessObservationClient, digest, validate_request


FAKE_WORKER = '''import argparse,hashlib,json,pathlib,time,sys
p=argparse.ArgumentParser();p.add_argument('--request');a=p.parse_args()
f=pathlib.Path(a.request);r=json.loads(f.read_text(encoding='utf-8'))
time.sleep(r.get('sleep_seconds',0))
if r.get('exit_failure'): sys.exit(3)
t=f.parent/'observation.json';t.write_text('{}')
reply={k:r[k] for k in ('protocol','request_id','backend_id','input_sha256')}
reply.update(status='ok',artifacts=[{'path':'observation.json','sha256':hashlib.sha256(t.read_bytes()).hexdigest()}])
if r.get('wrong_id'): reply['request_id']='wrong'
if r.get('outside_path'): reply['artifacts'][0]['path']='../observation.json'
if r.get('bad_hash'): reply['artifacts'][0]['sha256']='0'*64
(f.parent/'reply.json').write_text(json.dumps(reply))
'''


class WorkerClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = self.root / 'fake.py'
        self.worker.write_text(FAKE_WORKER, encoding='utf-8')
        self.input = self.root / 'input.bin'
        self.input.write_bytes(b'actual input')
        self.client = SubprocessObservationClient([sys.executable, '-B', self.worker], self.root / 'ipc',
                                                   backend_id='fake', allowed_root=self.root)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def call(self, **payload):
        return self.client.submit(self.input, [0, 1], timeout_s=5, **payload).result()

    def test_real_subprocess_reply_and_artifact_integrity(self):
        result = self.call()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['input_sha256'], digest(self.input))
        self.assertTrue(Path(result['artifacts'][0]['resolved_path']).is_file())

    def test_mismatched_reply_cannot_be_adopted(self):
        self.assertEqual(self.call(wrong_id=True)['reason'], 'reply_request_id_mismatch')

    def test_bad_hash_and_path_escape_cannot_be_adopted(self):
        self.assertEqual(self.call(bad_hash=True)['reason'], 'artifact_integrity_failure')
        self.assertEqual(self.call(outside_path=True)['reason'], 'artifact_path_invalid')

    def test_backend_exit_and_timeout_are_failure_not_absence(self):
        self.assertEqual(self.call(exit_failure=True)['status'], 'inference_failure')
        result = self.client.submit(self.input, [0, 1], timeout_s=0.15, sleep_seconds=0.6).result()
        self.assertEqual(result['reason'], 'timeout')
        self.assertEqual(result['gap']['interval_seconds'], [0, 1])

    def test_running_plus_one_pending_and_expired_pending(self):
        first = self.client.submit(self.input, [0, 1], timeout_s=5, sleep_seconds=0.5)
        self.assertTrue(self.client.running.wait(2))
        second = self.client.submit(self.input, [1, 2], timeout_s=0.1)
        third = self.client.submit(self.input, [2, 3], timeout_s=5)
        self.assertEqual(third.result()['reason'], 'queue_full')
        self.assertEqual(first.result()['status'], 'ok')
        self.assertEqual(second.result()['reason'], 'deadline_expired_in_queue')

    def test_future_online_input_is_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'Future input'):
            self.client.submit(self.input, [0, 2], mode='online', available_until_s=1)
        self.assertEqual(self.client.audit, [])

    def test_input_outside_allowed_root_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Input escapes'):
            self.client.submit(Path(__file__), [0, 1])


class AssociationTests(unittest.TestCase):
    def test_worker_thread_budget_is_bounded_and_boolean_is_not_an_integer(self):
        request={'protocol':'observation-worker-v1','request_id':'bounded','backend_id':'sensevoice',
            'input_path':'closed.wav','input_sha256':'0'*64,'interval_seconds':[0,1],
            'deadline_monotonic_ns':1,'mode':'online','available_until_s':1}
        for value in (True,False,0,-1,9,'2',2.5):
            with self.assertRaisesRegex(ValueError,'thread budget'):validate_request({**request,'torch_num_threads':value})
        for value in (1,2,4,8):validate_request({**request,'torch_num_threads':value})

    def rows(self, t=0.5, count=1, risk='no_positive_overlap'):
        visual = {'frame_time_s': t, 'face_count': count, 'status': 'ok',
                  'hand_mouth': {'status': risk}, 'eye_hand_risk': {'hold_eye_scores': False}}
        face = {'source_pts_seconds': t, 'face_count': count, 'faces': [{'face_id': 1}]}
        return visual, face

    def test_time_overlap_never_verifies_speaker_or_driver_state(self):
        v, f = self.rows()
        result = person_windows([{'start': 0, 'end': 1}], [v], [f], offset_s=0, duration_s=2)[0]
        self.assertEqual(result['time_alignment'], 'overlapping')
        self.assertEqual(result['speaker_link']['status'], 'unknown')
        self.assertFalse(result['driver_state_usable'])

    def test_half_open_boundary_and_negative_offset(self):
        v, f = self.rows(t=1)
        results = person_windows([{'start': 0.5, 'end': 1.5}, {'start': 1.5, 'end': 2.5}], [v], [f],
                                 offset_s=-0.5, duration_s=2)
        self.assertEqual([r['visual_samples'] for r in results], [0, 1])

    def test_multiple_faces_overlap_risk_and_track_change_hold(self):
        v, f = self.rows(count=2, risk='hand_overlap_risk')
        f['faces'].append({'face_id': 2})
        result = person_windows([{'start': 0, 'end': 1}], [v], [f], offset_s=0, duration_s=1)[0]
        for reason in ('multiple_faces', 'projected_hand_mouth_overlap', 'track_changed_or_unknown'):
            self.assertIn(reason, result['uncertainty'])

    def test_missing_mouth_eye_and_incomplete_window_hold(self):
        v, f = self.rows(risk='unknown')
        v['eye_hand_risk']['hold_eye_scores'] = True
        result = person_windows([{'start': 0, 'end': 2}], [v], [f], offset_s=0, duration_s=1)[0]
        for reason in ('mouth_geometry_unknown', 'eye_visibility_unavailable', 'incomplete_media_window'):
            self.assertIn(reason, result['uncertainty'])


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'frame.jsonl'
        self.path.write_text(json.dumps({'source_pts_seconds': 0.0}) + '\n')
        self.env = {'schema': SCHEMA, 'mode': 'offline', 'run_id': 'test', 'clip_id': 'test',
                    'backend_manifest': {}, 'backend_health': {}, 'human_reference_used': False,
                    'state_status': 'not_implemented', 'preferences': None, 'input': {'interval_seconds': [0, 1]},
                    'evidence_registry': [{'evidence_id': 'f0', 'relative_path': 'frame.jsonl',
                                          'file_sha256': digest(self.path), 'line_index': 0,
                                          'media_interval_seconds': [0, 1]}],
                    'observations': [{'kind': 'test', 'backend_id': 'test', 'value': {}, 'status': 'observed',
                                      'quality': {'status': 'not_measured'}, 'evidence_ids': ['f0']}],
                    'event_predictions': {k: {'status': 'not_implemented', 'value': None, 'evidence_ids': []} for k in QUESTIONS},
                    'person_windows': [], 'state_candidates': [], 'self_report': None, 'response': None}

    def tearDown(self):
        self.temp.cleanup()

    def test_evidence_hash_locator_and_schema_valid(self):
        self.assertTrue(validate_envelope(self.env, self.root)['schema_valid'])

    def test_required_provenance_and_reference_contract(self):
        bad = deepcopy(self.env)
        del bad['backend_manifest']
        with self.assertRaisesRegex(ValueError, 'Missing envelope'):
            validate_envelope(bad, self.root)
        self.env['human_reference_used'] = True
        with self.assertRaisesRegex(ValueError, 'reference contract'):
            validate_envelope(self.env, self.root)

    def test_changed_raw_evidence_fails(self):
        self.path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_envelope(self.env, self.root)

    def test_unknown_evidence_id_and_wrong_time_fail(self):
        bad = deepcopy(self.env)
        bad['observations'][0]['evidence_ids'] = ['invented']
        with self.assertRaisesRegex(ValueError, 'Missing observation evidence'):
            validate_envelope(bad, self.root)
        self.env['evidence_registry'][0]['media_interval_seconds'] = [0.5, 1]
        with self.assertRaisesRegex(ValueError, 'frame time mismatch'):
            validate_envelope(self.env, self.root)

    def test_missing_backend_is_not_no_and_states_are_not_manufactured(self):
        self.env['event_predictions']['nod']['value'] = 'no'
        with self.assertRaisesRegex(ValueError, 'cannot become absence'):
            validate_envelope(self.env, self.root)
        self.env['event_predictions']['nod']['value'] = None
        self.env['state_candidates'] = ['sleepy']
        with self.assertRaisesRegex(ValueError, 'must not manufacture'):
            validate_envelope(self.env, self.root)


class SharedInputAndFailureTests(unittest.TestCase):
    def setUp(self):
        import yaml
        temporary = WORKSPACE / '.tmp_devai_ara'
        temporary.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.root = Path(self.temp.name)
        self.video = self.root / 'video.mp4'
        self.video.write_bytes(b'probe supplied by test')
        self.audio = self.root / 'input.wav'
        self.pcm = b'\x01\x00' * 32085
        with wave.open(str(self.audio), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(self.pcm)
        self.config = yaml.safe_load((REPO / 'config.yaml').read_text(encoding='utf-8'))
        self.config['observation_integration'] = {'clip_id': 'test', 'visual_python': sys.executable,
                                                'audio_python': sys.executable}
        self.probe = {'frames': [{'best_effort_timestamp_time': '0'}, {'best_effort_timestamp_time': '1'}],
                      'streams': [{'duration': '2'}]}

    def tearDown(self):
        self.temp.cleanup()

    def session(self, offset):
        self.config['synchronization']['audio_offset_seconds'] = offset
        with patch('modules.observation_integration.subprocess.check_output', return_value=json.dumps(self.probe).encode()):
            return ObservationIntegrationSession(self.config, self.root, self.video, self.audio)

    def test_common_pcm_clips_tail_and_preserves_positive_offset(self):
        session = self.session(0.25)
        self.assertEqual(session.inputs['audio_samples'], [0, 28000])
        self.assertEqual([session.media_begin, session.media_end], [0.25, 2.0])
        with wave.open(str(session.common_audio), 'rb') as stream:
            self.assertEqual(stream.readframes(stream.getnframes()), self.pcm[:56000])
        self.assertEqual(self.config['synchronization']['audio_offset_seconds'], 0.25)

    def test_negative_offset_crops_prefix_and_restores_media_origin(self):
        session = self.session(-0.25)
        self.assertEqual(session.inputs['audio_samples'], [4000, 32085])
        self.assertEqual(session.media_begin, 0)
        self.assertEqual(self.config['synchronization']['audio_offset_seconds'], 0)
        with wave.open(str(session.common_audio), 'rb') as stream:
            self.assertEqual(stream.readframes(stream.getnframes()), self.pcm[8000:])

    def test_failed_workers_still_emit_valid_envelope_without_absence_or_states(self):
        session = self.session(0)
        self.assertEqual(session.inputs['audio_samples'], [0, 32000])
        frames = [{'source_pts_seconds': t, 'face_count': 1, 'faces': [{'face_id': 1}]} for t in (0, 1)]
        (self.root / 'frame_observations.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
        speech = {'segments': [{'start': 0, 'end': 1, 'text': 'synthetic regression'}]}
        (self.root / 'speech_result.json').write_text(json.dumps({'result': speech}))
        session.invoke = lambda backend: {'status': 'inference_failure', 'reason': 'test_timeout',
                                          'gap': {'interval_seconds': [0, 2]}}
        summary = session.finish(speech, 'whisper')
        self.assertEqual(summary['worker_failures'], 3)
        envelope = json.loads((self.root / 'observation_envelope_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(envelope['state_candidates'], [])
        self.assertTrue(all(p['value'] is None for p in envelope['event_predictions'].values()))
        self.assertEqual(envelope['person_windows'][0]['speaker_link']['status'], 'unknown')
        self.assertEqual(sum(o['status'] == 'inference_failure' for o in envelope['observations']), 6)
        self.assertTrue(validate_envelope(envelope, self.root)['schema_valid'])

    def test_optional_av_visual_failure_stays_a_failure_and_preserves_unknown_link(self):
        self.config['observation_integration']['av_sync_enabled'] = True
        session = self.session(0)
        frames = [{'source_pts_seconds': t, 'face_count': 1, 'faces': [{'face_id': 1}]} for t in (0, 1)]
        (self.root / 'frame_observations.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
        speech = {'segments': [{'start': 0, 'end': 1, 'text': 'machine regression'}]}
        (self.root / 'speech_result.json').write_text(json.dumps({'result': speech}))
        session.invoke = lambda backend: {'status': 'inference_failure', 'reason': 'test_timeout',
                                          'gap': {'interval_seconds': [0, 2]}}
        summary = session.finish(speech, 'whisper')
        envelope = json.loads((self.root / 'observation_envelope_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['backend_health']['syncnet'], 'inference_failure')
        av = next(o for o in envelope['observations'] if o['kind'] == 'av_correspondence')
        self.assertEqual(av['failure_reason'], 'visual_geometry_unavailable')
        self.assertEqual(av['status'], 'inference_failure')
        self.assertEqual(envelope['person_windows'][0]['speaker_link']['status'], 'unknown')
        self.assertTrue(validate_envelope(envelope, self.root)['schema_valid'])


if __name__ == '__main__':
    unittest.main()
