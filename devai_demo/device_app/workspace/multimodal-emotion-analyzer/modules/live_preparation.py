"""Prepare reusable models using declared saved input, never a capture window."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time


def preparation_status(state):
    return {
        'perception': '知覚モデルを準備中（保存素材・発話不要）',
        'ara': 'ARAを準備中（準備用入力・発話不要）',
        'checking': '認識・応答の準備を確認中（発話不要）',
        'ready': '準備完了。録音は準備回答後に開始します',
        'failed': '準備に失敗しました。録音は開始しません',
        'stopped': '準備を停止しました。録音は開始しません',
    }.get(state, '準備中（発話不要）')


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def prepare_runtime(runtime, configuration, workspace):
    """Only ready after all backends and the bounded ARA contract succeed.

    Artifacts live outside windows; preparation never contributes to live metrics.
    The owning persistent runtime must remain alive for readiness to be valid.
    """
    import yaml
    from modules.compact_driver_cycle import warmup_compact_cycle
    from modules.local_llm_client import LocalLLMClient
    config = deepcopy(configuration)
    output = runtime.output
    folder = output/'preparation'
    folder.mkdir(exist_ok=False)
    started = time.monotonic_ns()
    record = {'state': 'perception', 'started_ns': started, 'ready': False,
              'used_live_input': False, 'new_recordings': 0, 'human_reference_used': False,
              'owner_pid': __import__('os').getpid(), 'phases': []}
    def phase(state):
        record.update(state=state, status=preparation_status(state), updated_ns=time.monotonic_ns())
        publish(output/'preparation_state.json', record)
    phase('perception')
    try:
        inputs = config['live']['startup_preparation']['saved_inputs']
        paths = {}
        for name in ('video', 'audio'):
            path = Path(inputs[name]['path']).resolve()
            if not path.is_relative_to(Path(workspace).resolve()):
                raise ValueError('preparation_input_outside_workspace')
            if hashlib.sha256(path.read_bytes()).hexdigest() != inputs[name]['sha256']:
                raise ValueError('preparation_input_hash_mismatch')
            paths[name] = path
        record['saved_inputs'] = deepcopy(inputs)
        config['cooperative_ai'] = {'enabled': False}
        config['output'].update(save_video=False, save_results=True, save_frame_observations=True, output_dir=str(folder))
        config['display']['show_window'] = False
        config['logging']['output_dir'] = str(folder/'logs')
        config.setdefault('debug', {})['save_face_crops'] = False
        config['observation_integration'].update(clip_id='startup_preparation', worker_mode='offline')
        cfg = folder/'config.yaml'
        cfg.write_text(yaml.safe_dump(config), encoding='utf-8')
        app = runtime.new_app(cfg, preparation=True)
        try:
            app.run(video_file=str(paths['video']), audio_file=str(paths['audio']))
        finally:
            session = getattr(app, 'observation_session', None)
            if session is not None: session.close()
            runtime.detector.reset_window_state()
        envelope = json.loads((folder/'observation_envelope_v1.json').read_text(encoding='utf-8'))
        if envelope['backend_health'] != {'visual':'ok','sensevoice':'ok','emotion2vec':'ok'}:
            raise RuntimeError('preparation_backend_unhealthy')
        record['phases'].append({'phase':'perception', 'completed_ns':time.monotonic_ns(),
                                 'backend_health':envelope['backend_health']})
        phase('ara')
        cooperative = configuration['cooperative_ai']
        client = LocalLLMClient(cooperative['base_url'], cooperative['model'],
                               wait_for_backend=cooperative.get('wait_for_backend_ready',True))
        for name, budget in (('ara_initialization', 60), ('ara_bounded_check', 10)):
            if budget == 10: phase('checking')
            result = warmup_compact_cycle(client, preferences=cooperative.get('preferences'),
                timeout_s=budget, transport=cooperative.get('transport','compact-cycle-v4'))
            publish(folder/(name+'.json'), result)
            cycle = result['cycle']
            record['phases'].append({'phase':name, 'completed_ns':time.monotonic_ns(),
                'seconds':cycle['wall_seconds'], 'budget_seconds':budget,
                'recognition_status':cycle['recognition']['status'],
                'response_status':cycle['response']['status']})
            if cycle['recognition']['status'] != 'accepted' or cycle['response']['status'] not in ('accepted','policy_output'):
                raise RuntimeError('preparation_ara_contract_failed')
        record.update(ready=True, elapsed_seconds=(time.monotonic_ns()-started)/1e9)
        phase('ready')
        publish(folder/'completed.json', record)
        return record
    except BaseException as error:
        record.update(ready=False, error=f'{type(error).__name__}: {error}')
        phase('stopped' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed')
        raise
