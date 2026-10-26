"""Transport guards for the ARA bridge; native inference is verified separately."""
from http.server import ThreadingHTTPServer
from copy import deepcopy
import http.client
import json
import threading
import time
import unittest

from modules.ara_llm_service import MODEL_ID, ServiceRuntime, StructuredProcessorMixin, build_messages, handler_class, validate_request
from modules.local_llm_client import LocalLLMClient

SCHEMA = {'type': 'object', 'properties': {'status': {'const': 'ok'}},
          'required': ['status'], 'additionalProperties': False}


class Backend:
    def __init__(self):
        self.budgets = []
        self.delay = 0
        self.finish = 'stop'

    def generate(self, value, schema, maximum, deadline, cancelled):
        self.budgets.append(deadline - time.monotonic())
        self.assert_connected = not cancelled()
        time.sleep(self.delay)
        return {'text': '{"status":"ok"}', 'finish_reason': self.finish,
                'prompt_tokens': 8, 'completion_tokens': 6}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.runtime = ServiceRuntime(self.backend)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_class(self.runtime))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = LocalLLMClient(f'http://127.0.0.1:{self.server.server_port}', MODEL_ID)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def generate(self, seconds=1):
        return self.client.generate([{'role': 'user', 'content': 'ok'}], SCHEMA,
                                    deadline=time.monotonic() + seconds)

    def raw(self, body, budget='1'):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        connection.request('POST', '/v1/chat/completions', body,
                           {'Content-Type': 'application/json', 'X-ARA-Timeout-Seconds': budget})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_common_client_preserves_json_and_propagates_remaining_budget(self):
        result = self.generate()
        self.assertEqual(result['status'], 'generated')
        self.assertEqual(result['text'], '{"status":"ok"}')
        self.assertTrue(self.backend.assert_connected)
        self.assertTrue(0 < self.backend.budgets[0] <= 1)

    def test_duplicate_or_nonfinite_input_rejected_without_backend_call(self):
        for body in ('{"model":"a","model":"b"}', '{"model":NaN}'):
            self.assertEqual(self.raw(body)[0], 400)
        self.assertEqual(self.backend.budgets, [])

    def test_invalid_deadlines_rejected(self):
        for budget in ('nan', 'inf', '0', '-1', '61', 'bad'):
            self.assertEqual(self.raw('{}', budget)[0], 400)
        self.assertEqual(self.backend.budgets, [])

    def test_backend_busy_and_loading_fail_without_queueing(self):
        self.runtime.lock.acquire()
        try:
            self.assertEqual(self.generate()['http_status'], 503)
        finally:
            self.runtime.lock.release()
        self.runtime.state = 'loading'
        self.assertEqual(self.generate()['http_status'], 503)
        self.assertEqual(self.backend.budgets, [])

    def test_truncated_json_is_not_accepted(self):
        self.backend.finish = 'length'
        self.assertEqual(self.generate()['status'], 'invalid_output')

    def test_late_native_output_is_never_adopted(self):
        self.backend.delay = .08
        result = self.generate(.03)
        self.assertEqual(result['status'], 'inference_failure')
        time.sleep(.1)
        self.assertFalse(self.runtime.lock.locked())
        self.backend.delay = 0
        self.assertEqual(self.generate()['status'], 'generated')

    def test_health_reports_busy_until_timed_out_native_call_releases_lock(self):
        self.backend.delay = .25
        self.assertEqual(self.generate(.03)['status'], 'inference_failure')
        self.assertEqual(self.runtime.health()['state'], 'busy')
        self.assertTrue(self.runtime.health()['inference_active'])
        with self.assertRaisesRegex(TimeoutError, 'readiness_deadline'):
            self.client.wait_until_ready(deadline=time.monotonic() + .04, poll_seconds=.01)
        self.assertEqual(len(self.backend.budgets), 1)
        ready = self.client.wait_until_ready(deadline=time.monotonic() + 1, poll_seconds=.01)
        self.assertTrue(ready['ready'])
        self.assertEqual(self.runtime.health()['state'], 'ready')
        self.backend.delay = 0
        self.assertEqual(self.generate()['status'], 'generated')

    def test_opt_in_readiness_wait_shares_generation_deadline(self):
        self.client.wait_for_backend = True
        self.runtime.lock.acquire()
        try:
            started = time.monotonic()
            result = self.generate(.06)
            self.assertEqual(result['status'], 'inference_failure')
            self.assertIn('backend_readiness_deadline', result['reason'])
            self.assertLess(time.monotonic() - started, .3)
            self.assertEqual(self.backend.budgets, [])
        finally:
            self.runtime.lock.release()
        self.assertEqual(self.generate()['status'], 'generated')

    def test_failed_health_never_starts_generation(self):
        self.client.wait_for_backend = True
        self.runtime.state = 'failed'
        result = self.generate()
        self.assertIn('backend_not_recoverable', result['reason'])
        self.assertEqual(self.backend.budgets, [])

    def test_bad_nested_schema_and_format_rejected(self):
        value = {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': 'ok'}],
                 'response_format': {'type': 'json_schema', 'json_schema': {'strict': True, 'schema': SCHEMA}}}
        for schema in ({'$ref': 'file:///tmp/no'}, {'properties': []}, {'anyOf': {}}, None):
            value['response_format']['json_schema']['schema'] = schema
            with self.assertRaises(ValueError):
                validate_request(value, MODEL_ID)
        value['response_format'] = []
        with self.assertRaises(ValueError):
            validate_request(value, MODEL_ID)


class BSPProcessorTests(unittest.TestCase):
    def test_prompt_forward_cannot_discard_custom_grammar_processors(self):
        class BSP:
            def _generate_first_token(self, *args, **kwargs):
                self.forward_calls = getattr(self, 'forward_calls', 0) + 1
                self.logits_processor = ['bsp_default']
                return 123
        class Model(StructuredProcessorMixin, BSP):
            pass
        model = Model()
        model.structured_processors = [object()]
        self.assertEqual(model._generate_first_token('native_input'), 123)
        self.assertEqual(model.forward_calls, 1)
        self.assertIs(model.logits_processor, model.structured_processors)


class PromptTests(unittest.TestCase):
    def test_json_compaction_preserves_sensor_text_and_schema_values(self):
        data = {'text': '空白  と改行\nを保持', 'limits': [False, None, 6.0], 'evidence': 'E_asr0'}
        messages = [{'role': 'system', 'content': 'Keep all limitations.'},
                    {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]
        before = deepcopy(messages)
        legacy = build_messages(messages, SCHEMA)
        compact = build_messages(messages, SCHEMA, 'compact-json-v1')
        self.assertEqual(messages, before)
        self.assertEqual(json.loads(compact[1]['content']), data)
        self.assertEqual(json.loads(compact[0]['content'].split('Schema: ')[1]), SCHEMA)
        self.assertLess(len(compact[0]['content'] + compact[1]['content']),
                        len(legacy[0]['content'] + legacy[1]['content']))

    def test_plain_or_invalid_json_user_text_is_unchanged(self):
        for content in ('status ok', '{"x":1,"x":2}', '{"x":NaN}', '"  text  "'):
            result = build_messages([{'role': 'user', 'content': content}], SCHEMA, 'compact-json-v1')
            self.assertEqual(result[-1]['content'], content)


if __name__ == '__main__':
    unittest.main()
