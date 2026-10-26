"""Local-only structured generation with absolute deadlines and raw audit."""
import http.client
import json
import threading
import time
from urllib.parse import urlsplit


class LocalLLMClient:
    def __init__(self, base_url, model, *, max_tokens=512, wait_for_backend=False):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path not in {'', '/'}:
            raise ValueError('Only local 127.0.0.1 HTTP inference is allowed')
        self.port, self.model = parsed.port or 80, model
        self.max_tokens = max_tokens
        self.wait_for_backend = wait_for_backend
        self.lock = threading.Lock()

    def wait_until_ready(self, *, deadline, poll_seconds=.2):
        """Wait for the ARA lock to clear without issuing speculative inference."""
        observations = []
        started = time.monotonic()
        while time.monotonic() < deadline:
            connection = http.client.HTTPConnection('127.0.0.1', self.port,
                timeout=max(.001, min(2, deadline - time.monotonic())))
            try:
                connection.request('GET', '/health')
                transport = connection.sock
                transport.settimeout(max(.001, deadline - time.monotonic()))
                response = connection.getresponse()
                transport.settimeout(max(.001, deadline - time.monotonic()))
                raw = response.read(65537)
                if response.status != 200 or len(raw) > 65536:
                    raise ValueError('health_http_failure')
                health = json.loads(raw)
                observations.append(health)
                if health.get('backend') != 'ARA' or health.get('model') != self.model:
                    raise ValueError('health_backend_or_model_mismatch')
                if health.get('state') == 'ready' and health.get('inference_active') is False:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('backend_readiness_deadline')
                    return {'ready': True, 'observations': observations,
                            'wall_seconds': time.monotonic() - started}
                if health.get('state') not in {'busy', 'loading'}:
                    raise ValueError('backend_not_recoverable_or_health_protocol_old')
            finally:
                connection.close()
            time.sleep(min(max(.01, poll_seconds), max(0, deadline - time.monotonic())))
        raise TimeoutError('backend_readiness_deadline')

    def generate(self, messages, schema, *, deadline, max_tokens=None, seed=42):
        started = time.monotonic()
        record = {'model': self.model, 'started_monotonic': started, 'deadline_monotonic': deadline,
                  'status': 'inference_failure', 'raw_http_body': None, 'text': None}
        remaining = deadline - started
        if remaining <= 0:
            return {**record, 'reason': 'deadline_before_request', 'wall_seconds': 0}
        if not self.lock.acquire(timeout=remaining):
            return {**record, 'reason': 'queue_deadline', 'wall_seconds': time.monotonic() - started}
        connection = None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('deadline_after_queue')
            if self.wait_for_backend:
                record['readiness'] = self.wait_until_ready(deadline=deadline)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('deadline_after_readiness')
            body = {'model': self.model, 'messages': messages, 'temperature': 0, 'seed': seed,
                    'max_tokens': max_tokens or self.max_tokens, 'stream': False, 'cache_prompt': False,
                    'response_format': {'type': 'json_schema', 'json_schema': {'name': 'ara_output', 'strict': True, 'schema': schema}}}
            record['request'] = body
            record['queue_wait_seconds'] = time.monotonic() - started
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=remaining)
            connection.request('POST', '/v1/chat/completions', json.dumps(body, ensure_ascii=False).encode(),
                               {'Content-Type': 'application/json', 'X-ARA-Timeout-Seconds': f'{min(remaining, 60):.6f}'})
            transport = connection.sock
            transport.settimeout(max(0.001, deadline - time.monotonic()))
            response = connection.getresponse()
            record['http_status'] = response.status
            transport.settimeout(max(0.001, deadline - time.monotonic()))
            raw = response.read(1024 * 1024 + 1)
            record['raw_http_body'] = raw.decode('utf-8')
            if time.monotonic() > deadline:
                raise TimeoutError('late_result')
            if response.status != 200 or len(raw) > 1024 * 1024:
                raise ValueError('http_failure_or_oversized')
            data = json.loads(raw)
            choice = data['choices'][0]
            record.update(text=choice['message']['content'], finish_reason=choice.get('finish_reason'),
                          usage=data.get('usage'), timings=data.get('timings'))
            if choice.get('finish_reason') != 'stop':
                record.update(status='invalid_output', reason='truncated_generation')
            else:
                record['status'] = 'generated'
        except Exception as error:
            record['reason'] = f'{type(error).__name__}: {error}'
        finally:
            if connection:
                connection.close()
            self.lock.release()
            record['wall_seconds'] = time.monotonic() - started
            record['available_monotonic'] = time.monotonic()
        return record
