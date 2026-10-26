"""Meaningful failures, preference precedence and local HTTP regression."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from modules.driver_observation_features import compact_observations, rule_recognition
from modules.driver_recognition import parse_json, validate_recognition, infer_recognition, normalize_model_recognition, recognition_schema
from modules.driver_response_policy import select_policy, produce_response, validate_response, resolve_preferences, apply_recognition_correction, DriverCooperativePipeline
from modules.local_llm_client import LocalLLMClient

COMPACT = {'features': [{'evidence_id': 'E_head'}], 'experimental_candidates': {
    'visible_person_speaks': [], 'offscreen_voice': [], 'hand_over_mouth': [], 'nod': ['E_head']}}


class FakeClient:
    def __init__(self, result):
        self.result, self.calls = result, 0
    def generate(self, *args, **kwargs):
        self.calls += 1
        return self.result


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.value = rule_recognition(COMPACT)

    def test_machine_candidate_and_unknown_state_contract(self):
        self.assertTrue(validate_recognition(self.value, COMPACT)['support_scope_valid'])
        self.assertEqual(self.value['events']['nod']['value'], 'yes')
        self.assertEqual(self.value['events']['offscreen_voice']['value'], 'unknown')

    def test_duplicate_and_nonfinite_json_are_invalid(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":'):
            with self.assertRaises(ValueError):
                parse_json(text)

    def test_unknown_missing_and_extra_fields_rejected(self):
        for change in ('missing', 'extra', 'enum'):
            bad = deepcopy(self.value)
            if change == 'missing': del bad['events']['nod']
            if change == 'extra': bad['events']['nod']['verified'] = True
            if change == 'enum': bad['events']['nod']['value'] = 'maybe'
            with self.assertRaises(ValueError): validate_recognition(bad, COMPACT)

    def test_existing_but_irrelevant_evidence_is_not_support(self):
        self.value['events']['hand_over_mouth'] = deepcopy(self.value['events']['nod'])
        with self.assertRaisesRegex(ValueError, 'unsupported_positive'):
            validate_recognition(self.value, COMPACT)

    def test_invented_evidence_rejected(self):
        self.value['events']['nod']['evidence_ids'] = ['invented']
        with self.assertRaisesRegex(ValueError, 'invented_evidence'):
            validate_recognition(self.value, COMPACT)

    def test_no_is_not_inferred_from_missing_positive(self):
        self.value['events']['offscreen_voice']['value'] = 'no'
        with self.assertRaisesRegex(ValueError, 'negative_event_unverified'):
            validate_recognition(self.value, COMPACT)

    def test_inner_state_is_not_inferred_from_ser_or_nod(self):
        self.value['states']['drowsiness_signs']['value'] = 'candidate'
        with self.assertRaisesRegex(ValueError, 'driver_state_unverified'):
            validate_recognition(self.value, COMPACT)

    def test_failure_invalid_and_accepted_are_separate(self):
        model_value = {'schema': 'ara-recognition-model-v2', 'events': {
            q: {k: v for k, v in e.items() if k != 'uncertainty'} for q, e in self.value['events'].items()}}
        for raw, expected in (({'status': 'inference_failure', 'reason': 'timeout'}, 'inference_failure'),
                              ({'status': 'generated', 'text': 'broken'}, 'invalid_output'),
                              ({'status': 'generated', 'text': json.dumps(model_value)}, 'accepted')):
            result = infer_recognition(FakeClient(raw), COMPACT, time.monotonic() + 5)
            self.assertEqual(result['status'], expected)
            self.assertEqual(result['accepted'] is not None, expected == 'accepted')

    def test_missing_candidate_uncertainty_rejected(self):
        self.value['events']['nod']['uncertainty'] = ['insufficient_observation']
        with self.assertRaisesRegex(ValueError, 'uncertainty_missing'):
            validate_recognition(self.value, COMPACT)

    def test_v2_adds_application_limitations_but_never_repairs_v1_raw(self):
        raw = {'schema': 'ara-recognition-model-v2', 'events': {
            q: {k: v for k, v in e.items() if k != 'uncertainty'} for q, e in self.value['events'].items()}}
        normalized = normalize_model_recognition(raw)
        self.assertNotIn('uncertainty', raw['events']['nod'])
        self.assertEqual(normalized['events']['nod']['uncertainty'], ['uncalibrated_geometry'])
        self.assertTrue(all(e['value'] == 'unknown' for e in normalized['states'].values()))
        with self.assertRaisesRegex(ValueError, 'model_recognition_schema'):
            normalize_model_recognition(self.value)

    def test_speaking_requires_matched_audio_mouth_pair(self):
        compact = {'features': [{'evidence_id': k} for k in ['E_asr0','E_mouth0','E_asr1','E_mouth1']],
                   'experimental_candidates': {**COMPACT['experimental_candidates'], 'nod': [],
                                               'visible_person_speaks': ['E_asr0','E_mouth0','E_asr1','E_mouth1']}}
        value = rule_recognition(compact)
        for ids in (['E_asr0'], ['E_mouth0'], ['E_asr0','E_mouth1']):
            value['events']['visible_person_speaks']['evidence_ids'] = ids
            with self.assertRaisesRegex(ValueError, 'matched_audio_mouth'):
                validate_recognition(value, compact)
        value['events']['visible_person_speaks']['evidence_ids'] = ['E_asr1','E_mouth1']
        self.assertTrue(validate_recognition(value, compact)['support_scope_valid'])

    def test_v3_schema_gates_unsupported_answers_before_generation(self):
        schema = recognition_schema(['E_head'], COMPACT)
        entries = schema['properties']['events']['properties']
        self.assertEqual(entries['offscreen_voice']['properties']['value']['const'], 'unknown')
        self.assertEqual([branch['properties']['value']['const'] for branch in entries['nod']['anyOf']], ['unknown','yes'])
        raw = {'schema':'ara-recognition-model-v3','events':{
            q: {'value':'unknown','evidence_ids':[]} for q in COMPACT['experimental_candidates']}}
        raw['events']['nod']['evidence_ids'] = ['E_head']
        with self.assertRaisesRegex(ValueError, 'unknown_evidence_schema'):
            normalize_model_recognition(raw)


class ResponseTests(unittest.TestCase):
    def preference(self, name):
        return {'source': 'explicit_user', 'preference': name}

    def test_unverified_state_default_is_wait(self):
        policy = select_policy({'status': 'accepted'})
        client = FakeClient({})
        response = produce_response(client, policy, deadline=time.monotonic()+2)
        self.assertEqual(response['accepted']['text'], '')
        self.assertEqual(client.calls, 0)

    def test_no_conversation_overrides_guidance_and_skips_generation(self):
        policy = select_policy({}, [self.preference('no_conversation')], '案内。')
        client = FakeClient({})
        self.assertEqual(produce_response(client, policy, deadline=time.monotonic()+2)['accepted']['text'], '')
        self.assertEqual(client.calls, 0)

    def test_explicit_preference_correction_preserves_history(self):
        history = [self.preference('rest_information'), self.preference('no_conversation')]
        result = resolve_preferences(history)
        self.assertEqual(result['active'], 'no_conversation')
        self.assertEqual(len(result['history']), 2)
        self.assertEqual(select_policy({}, history)['policy'], 'reduce_dialogue')

    def test_repeat_without_history_does_not_invent_guidance(self):
        policy = select_policy({}, [self.preference('repeat_guidance')])
        self.assertEqual(policy['policy'], 'wait')
        self.assertEqual(policy['text'], '')

    def test_repeat_uses_saved_guidance(self):
        policy = select_policy({}, [self.preference('repeat_guidance')], '案内を確認してください。')
        self.assertEqual(policy['policy'], 'repeat_guidance')
        self.assertEqual(policy['text'], '案内を確認してください。')

    def test_timeout_bad_json_unsupported_diagnosis_fall_back(self):
        policy = select_policy({}, [self.preference('rest_information')])
        for raw in ({'status': 'inference_failure', 'reason': 'timeout'}, {'status': 'generated', 'text': 'bad'},
                    {'status': 'generated', 'text': json.dumps({'schema': 'ara-response-v1', 'policy': 'rest_suggestion', 'text': 'あなたは眠いです。'})}):
            result = produce_response(FakeClient(raw), policy, deadline=time.monotonic()+3)
            self.assertEqual(result['status'], 'fallback')
            self.assertEqual(result['accepted']['text'], policy['text'])

    def test_accepted_guidance_and_unicode_sentence_limits(self):
        policy = select_policy({}, [self.preference('minimal_guidance')], 'シートベルトを確認してください。')
        value = {'schema': 'ara-response-v1', 'policy': policy['policy'], 'text': policy['text']}
        self.assertEqual(produce_response(FakeClient({'status': 'generated', 'text': json.dumps(value)}), policy,
                                         deadline=time.monotonic()+3)['status'], 'accepted')
        for text in ('あ'*81, '一。二。三。'):
            with self.assertRaisesRegex(ValueError, 'response_length'):
                validate_response({**value, 'text': text}, policy)

    def test_unreceived_preferences_cannot_be_manufactured(self):
        with self.assertRaises(ValueError): resolve_preferences([{'source': 'model', 'preference': 'no_conversation'}])

    def test_recognition_correction_is_separate_and_machine_value_unchanged(self):
        value = rule_recognition(COMPACT)
        result = apply_recognition_correction(value, 'nod', 'no', text='regression correction')
        self.assertEqual(value['events']['nod']['value'], 'yes')
        self.assertFalse(result['changes_scoring_reference'])
        self.assertEqual(result['answer'], 'no')

    def test_correction_effective_view_preserves_invalid_machine_output(self):
        result = DriverCooperativePipeline(FakeClient({'status':'generated','text':'bad'})).process(
            COMPACT, corrections=[{'question':'nod','answer':'no','text':'explicit regression correction'}])
        self.assertEqual(result['recognition']['status'], 'invalid_output')
        self.assertIsNone(result['recognition']['accepted'])
        self.assertEqual(result['effective_events']['nod']['source'], 'explicit_user')
        self.assertFalse(result['recognition_corrections'][0]['machine_prediction_available'])


class LocalHTTPTests(unittest.TestCase):
    def setUp(self):
        test = self
        self.delay, self.finish = 0, 'stop'
        self.requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                test.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                time.sleep(test.delay)
                body = json.dumps({'choices': [{'finish_reason': test.finish, 'message': {'content': '{}'}}]}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers()
                try: self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError): pass
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = LocalLLMClient(f'http://127.0.0.1:{self.server.server_port}', 'test')

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_local_real_http_records_raw_body_and_disables_cache(self):
        result = self.client.generate([], {}, deadline=time.monotonic()+2)
        self.assertEqual(result['status'], 'generated')
        self.assertFalse(self.requests[0]['cache_prompt'])
        self.assertIn('choices', result['raw_http_body'])

    def test_timeout_and_expired_request_are_never_adopted(self):
        self.delay = 0.1
        self.assertEqual(self.client.generate([], {}, deadline=time.monotonic()+0.03)['status'], 'inference_failure')
        count = len(self.requests)
        result = self.client.generate([], {}, deadline=time.monotonic()-1)
        self.assertEqual(result['reason'], 'deadline_before_request')
        self.assertEqual(len(self.requests), count)

    def test_truncated_generation_is_invalid(self):
        self.finish = 'length'
        self.assertEqual(self.client.generate([], {}, deadline=time.monotonic()+2)['status'], 'invalid_output')

    def test_remote_inference_endpoint_rejected(self):
        with self.assertRaises(ValueError): LocalLLMClient('https://example.com', 'test')


class FeatureTests(unittest.TestCase):
    def test_raw_motion_candidates_and_missing_modality_hold(self):
        protocol = json.loads((Path(__file__).resolve().parents[2] / 'devai/devai_ara/runs/ara03_local_llm_20261011_v1/protocol.json').read_text())
        rows = []
        for n in range(25):
            angle = math.radians(8 * math.sin(math.pi * n / 24))
            matrix = [[1,0,0,0], [0,math.cos(angle),-math.sin(angle),0], [0,math.sin(angle),math.cos(angle),0], [0,0,0,1]]
            rows.append({'frame_index': n, 'frame_time_s': n/24, 'status': 'ok', 'face_count': 1, 'hands': [],
                         'head_transform': matrix, 'hand_mouth': {'status': 'no_positive_overlap', 'aperture': n/200}})
        registry = [{'evidence_id': f'geometry.{n}', 'relative_path': 'rows.jsonl'} for n in range(25)]
        registry.append({'evidence_id': 'asr.0'})
        env = {'input': {'interval_seconds': [0, 2], 'common_audio_interval_seconds': [0, 2]}, 'evidence_registry': registry,
               'observations': [{'kind': 'asr', 'status': 'observed', 'value': {'start': 0, 'end': 1, 'text': 'test'}, 'evidence_ids': ['asr.0']}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'envelope.json'; path.write_text('{}')
            with patch('modules.driver_observation_features.read_json', return_value=env), \
                 patch('modules.driver_observation_features.validate_envelope'), \
                 patch('modules.driver_observation_features.read_lines', return_value=rows):
                compact, mapping = compact_observations(path, protocol['candidate_thresholds'])
                self.assertTrue(compact['experimental_candidates']['nod'])
                self.assertTrue(compact['experimental_candidates']['visible_person_speaks'])
                self.assertFalse(compact['experimental_candidates']['hand_over_mouth'])
                audio, _ = compact_observations(path, protocol['candidate_thresholds'], modalities=('audio',))
                self.assertFalse(audio['experimental_candidates']['nod'])
                self.assertFalse(audio['experimental_candidates']['visible_person_speaks'])
                text_only, _ = compact_observations(path, protocol['candidate_thresholds'], modalities=('asr',))
                self.assertTrue(any(f['kind']=='asr' for f in text_only['features']))
                self.assertFalse(text_only['experimental_candidates']['visible_person_speaks'])


if __name__ == '__main__':
    unittest.main()
