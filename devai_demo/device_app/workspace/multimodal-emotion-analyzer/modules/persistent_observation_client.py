"""Opt-in cached audio worker; reuse the existing bounded/hash-checked IPC."""
import json
import sys
import queue
import subprocess
import threading
import time

from modules.observation_worker_client import SubprocessObservationClient


class PersistentObservationClient(SubprocessObservationClient):
    backends = {'sensevoice', 'emotion2vec'}
    server_argument = '--serve-audio'

    def __init__(self, *args, **kwargs):
        backend = kwargs.get('backend_id')
        if backend not in self.backends:
            raise ValueError('persistent_audio_backends_only')
        self.process = None
        self.worker_log = None
        self.reader = None
        self.responses = None
        self.start_count = 0
        super().__init__(*args, **kwargs)

    def _start(self):
        self.start_count += 1
        self.responses = queue.Queue(maxsize=4)
        self.worker_log = (self.directory / f'persistent_{self.start_count:03d}.log').open('xb')
        try:
            self.process = subprocess.Popen(self.command + [self.server_argument],
                env=self.environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.worker_log)
        except Exception:
            self.worker_log.close()
            self.worker_log = None
            raise
        process, responses = self.process, self.responses
        def read():
            while True:
                line = process.stdout.readline(65537)
                try:
                    responses.put_nowait(line)
                except queue.Full:
                    break
                if not line or len(line) > 65536:
                    break
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def _stop(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if self.reader:
                self.reader.join(timeout=2)
            for pipe in (process.stdin, process.stdout):
                pipe.close()
            self.process = None
        if self.worker_log:
            self.worker_log.close()
            self.worker_log = None

    def _run_process(self, request_path, folder, remaining):
        deadline = time.monotonic() + remaining
        if self.process is None or self.process.poll() is not None:
            self._stop()
            self._start()
        request = json.loads(request_path.read_text(encoding='utf-8'))
        transport = {'request_path': str(request_path), 'request_id': request['request_id']}
        try:
            self.process.stdin.write((json.dumps(transport)+'\n').encode('utf-8'))
            self.process.stdin.flush()
            response = self.responses.get(timeout=max(.001, deadline-time.monotonic()))
            if not response or len(response) > 65536:
                raise ValueError('persistent_transport_eof_or_oversized')
            response = json.loads(response)
            if response != {'request_id': request['request_id'], 'status': 'completed'}:
                raise ValueError('persistent_response_identity_or_status')
            return 0
        except queue.Empty:
            self._stop()
            raise subprocess.TimeoutExpired(self.command, remaining)
        except Exception:
            self._stop()
            raise

    def _execute(self, request, submitted_ns):
        try:
            result = super()._execute(request, submitted_ns)
        except Exception:
            self._stop()
            raise
        if result['status'] != 'ok':
            self._stop()
        result['worker_transport'] = {'mode': 'persistent_visual_reused_graph_v2' if self.backend_id == 'visual' else 'persistent_audio_v1',
            'start_count': self.start_count, 'pid': self.process.pid if self.process else None}
        return result

    def close(self):
        try:
            super().close()
        finally:
            self._stop()


class PersistentVisualClient(PersistentObservationClient):
    """Keep imported libraries and verified VIDEO graphs warm across clips."""
    backends = {'visual'}
    server_argument = '--serve-visual'

    @staticmethod
    def environment():
        env = PersistentObservationClient.environment()
        if sys.platform == 'linux':
            env.setdefault('MALLOC_ARENA_MAX', '2')
        return env

