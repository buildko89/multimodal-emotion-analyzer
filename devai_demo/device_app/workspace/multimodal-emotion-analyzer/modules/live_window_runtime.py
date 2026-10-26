"""Own models across closed windows; scope all person state to one window."""
import json
import os
from pathlib import Path
import signal
import sys
import uuid

from modules.observation_worker_client import write_json
from modules.persistent_observation_client import PersistentObservationClient, PersistentVisualClient


class LiveWindowRuntime:
    def __init__(self, config, workspace, repo):
        self.config = config
        self.output = Path(config['live']['output']).resolve()
        self.detector = None
        self.detector_instance_id = uuid.uuid4().hex
        self.windows_processed = 0
        self.clients = {}
        try:
            for backend in ('sensevoice', 'emotion2vec', 'visual'):
                executable = config['observation_integration'][
                    'visual_python' if backend == 'visual' else 'audio_python']
                client_type = PersistentVisualClient if backend == 'visual' else PersistentObservationClient
                self.clients[backend] = client_type(
                    [executable, '-B', repo/'scripts/observation_worker.py'],
                    self.output/'worker_pool'/backend, backend_id=backend, allowed_root=workspace)
        except BaseException:
            self.close()
            raise

    def new_app(self, config_path, *, preparation=False):
        from multimodal_app import MultiModalEmotionApp
        if self.detector is not None:
            self.detector.reset_window_state()
        app = MultiModalEmotionApp(str(config_path), emotion_detector=self.detector,
                                   observation_clients=self.clients)
        self.detector = app.emotion_detector
        if not preparation:
            self.windows_processed += 1
        state = {'runtime_pid': os.getpid(), 'detector_instance_id': self.detector_instance_id,
                 'detector_model_loads': 1, 'window_number': self.windows_processed,
                 'purpose': 'startup_preparation' if preparation else 'closed_window',
                 'window_state_id': uuid.uuid4().hex, 'person_track_scope': 'this_window_only',
                 'smoothing_entries_at_start': len(self.detector.emotion_history),
                 'tracks_at_start': len(self.detector.tracker._tracks)}
        write_json(Path(config_path).parent/'runtime_state.json', state)
        return app

    def close(self):
        preparation = self.output/'preparation_state.json'
        if preparation.exists():
            from modules.live_preparation import publish, preparation_status
            state = json.loads(preparation.read_text(encoding='utf-8'))
            if state.get('ready'):
                state.update(ready=False,state='stopped',status=preparation_status('stopped'),
                             reason='owning_runtime_closed')
                publish(preparation,state)
        workers = {}
        for backend, client in self.clients.items():
            process = client.process
            client.close()
            workers[backend] = {'start_count': client.start_count,
                'pid': process.pid if process else None,
                'exited': process is None or process.poll() is not None}
        if self.detector is not None:
            self.detector.face_model.close()
            self.detector = None
        final = self.output/'window_runtime_final.json'
        if not final.exists():
            write_json(final, {'runtime_pid': os.getpid(), 'windows_processed': self.windows_processed,
                'ready_after_close':False,
                'workers': workers, 'all_owned_workers_exited': all(w['exited'] for w in workers.values())})


def serve_windows(config_path):
    import yaml
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    from modules.live_pipeline import process_window, REPO, WORKSPACE
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    output = Path(config['live']['output']).resolve()
    if not output.is_relative_to(WORKSPACE):
        raise ValueError('runtime_output_outside_workspace')
    def terminate(*_):
        raise KeyboardInterrupt('owned_window_runtime_stopped')
    signal.signal(signal.SIGTERM, terminate)
    runtime = LiveWindowRuntime(config, WORKSPACE, REPO)
    try:
        if config['live'].get('startup_preparation'):
            from modules.live_preparation import prepare_runtime
            prepare_runtime(runtime, config, WORKSPACE)
        for line in sys.stdin:
            request = json.loads(line)
            manifest = Path(request['manifest']).resolve()
            if not manifest.is_relative_to(output/'windows') or manifest.name != 'capture_manifest.json':
                raise ValueError('window_manifest_outside_owned_run')
            request_config = Path(request['config']).resolve() if 'config' in request else Path(config_path)
            if 'config' in request and (request_config.parent != manifest.parent or request_config.name != 'processing_config.json'):
                raise ValueError('window_config_outside_owned_window')
            process_window(manifest, request_config, runtime=runtime)
    finally:
        runtime.close()
