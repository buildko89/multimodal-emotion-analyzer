"""Bounded single-consumer subprocess workers with verifiable file IPC."""
from concurrent.futures import Future
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_request(request):
    required = {'protocol', 'request_id', 'backend_id', 'input_path', 'input_sha256',
                'interval_seconds', 'deadline_monotonic_ns', 'mode', 'available_until_s'}
    if not required <= request.keys() or request['protocol'] != 'observation-worker-v1':
        raise ValueError('Invalid worker protocol')
    start, end = request['interval_seconds']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (start, end, request['available_until_s'])) or not 0 <= start < end:
        raise ValueError('Invalid media interval')
    if request['mode'] not in {'offline', 'online'}:
        raise ValueError('Invalid mode')
    if request['mode'] == 'online' and end > request['available_until_s']:
        raise ValueError('Future input is forbidden')
    if not isinstance(request['deadline_monotonic_ns'], int):
        raise ValueError('Invalid deadline')
    if not isinstance(request['request_id'], str) or not request['request_id'].isalnum():
        raise ValueError('Invalid request ID')
    threads=request.get('torch_num_threads',4)
    if isinstance(threads,bool) or not isinstance(threads,int) or not 1<=threads<=8:
        raise ValueError('Invalid torch thread budget')


class SubprocessObservationClient:
    """At most one running and one pending request; all drops are recorded.

    A fresh backend process is used for each request, so environments remain
    isolated. This is an offline prototype, not a persistent low-latency service.
    """
    def __init__(self, command, directory, *, backend_id, allowed_root):
        self.command = [str(v) for v in command]
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.backend_id = backend_id
        self.allowed_root = Path(allowed_root).resolve()
        if not self.directory.is_relative_to(self.allowed_root):
            raise ValueError('Output escapes allowed workspace')
        self.pending = queue.Queue(maxsize=1)
        self.audit = []
        self.closed = False
        self.running = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def submit(self, input_path, interval_seconds, *, timeout_s=120, mode='offline', available_until_s=None, **payload):
        if self.closed:
            raise RuntimeError('Client is closed')
        path = Path(input_path).resolve()
        if not path.is_relative_to(self.allowed_root):
            raise ValueError('Input escapes allowed workspace')
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Positive finite timeout required')
        request = dict(payload)
        request.update(protocol='observation-worker-v1', request_id=uuid.uuid4().hex, backend_id=self.backend_id,
                       input_path=str(path), input_sha256=digest(path), interval_seconds=list(interval_seconds),
                       deadline_monotonic_ns=time.monotonic_ns() + round(timeout_s * 1e9), mode=mode,
                       available_until_s=interval_seconds[1] if available_until_s is None else available_until_s)
        validate_request(request)
        future = Future()
        try:
            self.pending.put_nowait((request, future, time.monotonic_ns()))
        except queue.Full:
            future.set_result(self._failure(request, 'queue_full', 'input_gap'))
        return future

    def _failure(self, request, reason, gap='no_adopted_observation'):
        result = {'protocol': request['protocol'], 'request_id': request['request_id'], 'backend_id': self.backend_id,
                  'input_sha256': request['input_sha256'], 'status': 'inference_failure', 'reason': reason,
                  'gap': {'interval_seconds': request['interval_seconds'], 'reason': gap}, 'artifacts': []}
        with self.lock:
            self.audit.append(result)
        return result

    def _consume(self):
        while True:
            item = self.pending.get()
            if item is None:
                self.pending.task_done()
                return
            request, future, submitted_ns = item
            self.running.set()
            try:
                if future.set_running_or_notify_cancel():
                    future.set_result(self._execute(request, submitted_ns))
            except Exception as error:
                future.set_result(self._failure(request, f'{type(error).__name__}: {error}'))
            finally:
                self.running.clear()
                self.pending.task_done()

    def _execute(self, request, submitted_ns):
        started_ns = time.monotonic_ns()
        remaining = (request['deadline_monotonic_ns'] - started_ns) / 1e9
        if remaining <= 0:
            return self._failure(request, 'deadline_expired_in_queue')
        folder = self.directory / request['request_id']
        folder.mkdir(exist_ok=False)
        request_path = folder / 'request.json'
        write_json(request_path, request)
        try:
            returncode = self._run_process(request_path, folder, remaining)
        except subprocess.TimeoutExpired:
            return self._failure(request, 'timeout')
        if returncode:
            return self._failure(request, f'worker_exit_{returncode}')
        if time.monotonic_ns() > request['deadline_monotonic_ns']:
            return self._failure(request, 'late_reply')
        path = folder / 'reply.json'
        if not path.exists() or path.stat().st_size > 4 * 1024 * 1024:
            return self._failure(request, 'missing_or_oversized_reply')
        reply = json.loads(path.read_text(encoding='utf-8'))
        for field in ('protocol', 'request_id', 'backend_id', 'input_sha256'):
            if reply.get(field) != request[field]:
                return self._failure(request, f'reply_{field}_mismatch')
        if digest(request['input_path']) != request['input_sha256']:
            return self._failure(request, 'input_changed')
        if reply.get('status') != 'ok':
            return self._failure(request, reply.get('reason', 'backend_failure'))
        if not isinstance(reply.get('artifacts'), list) or not reply['artifacts']:
            return self._failure(request, 'missing_artifacts')
        for artifact in reply['artifacts']:
            artifact_path = (folder / artifact['path']).resolve()
            if not artifact_path.is_relative_to(folder) or not artifact_path.is_file():
                return self._failure(request, 'artifact_path_invalid')
            if artifact_path.stat().st_size > 64 * 1024 * 1024 or digest(artifact_path) != artifact['sha256']:
                return self._failure(request, 'artifact_integrity_failure')
            artifact['resolved_path'] = str(artifact_path)
        reply['client_timing'] = {'submitted_ns': submitted_ns, 'started_ns': started_ns,
                                  'available_ns': time.monotonic_ns(), 'queue_wait_seconds': (started_ns - submitted_ns) / 1e9,
                                  'deadline_monotonic_ns': request['deadline_monotonic_ns']}
        # Validation and file hashing count toward the deadline too.
        if reply['client_timing']['available_ns'] > request['deadline_monotonic_ns']:
            return self._failure(request, 'late_validated_reply')
        with self.lock:
            self.audit.append(reply)
        return reply

    @staticmethod
    def environment():
        env = os.environ.copy()
        env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        return env

    def _run_process(self, request_path, folder, remaining):
        with (folder / 'worker.log').open('xb') as log:
            return subprocess.run(self.command + ['--request', str(request_path)], env=self.environment(),
                                  stdout=log, stderr=subprocess.STDOUT, timeout=remaining).returncode

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.join()
            self.pending.put(None)
            self.thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
