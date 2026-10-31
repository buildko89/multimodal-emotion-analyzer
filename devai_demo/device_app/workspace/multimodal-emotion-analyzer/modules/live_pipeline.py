import os as _demo_thread_env
_demo_thread_env.environ.update(OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
"""Live acquisition/display with bounded, causally closed perception windows.

Existing file backends process only a closed window. Their offline envelope
remains versioned as offline; the outer live manifest records native clocks,
availability, drops, and result age instead of claiming streaming model speed.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave

from modules.live_capture import SpeechPriorityWindowQueue, TimedPCM, NativeMicrophone, camera_frames, validate_closed_window

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def purge_processed_windows(windows_dir, retention=3):
    """Keep the strongest-audio window plus the newest completed windows."""
    if isinstance(retention, bool) or not isinstance(retention, int) or not 3 <= retention <= 16:
        raise ValueError('processed_window_retention_out_of_range')
    root = Path(windows_dir).resolve()
    if not root.is_dir():
        return {'retention': retention, 'retained': [], 'removed': []}
    entries = []
    for path in root.iterdir():
        if not (path.is_dir() and not path.is_symlink()
                and path.name.startswith('live_') and path.name[5:].isdigit()):
            continue
        manifest_path = path/'capture_manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        energy = float(manifest.get('audio_rms', -1))
        interval = manifest.get('interval_ns')
        if (not math.isfinite(energy) or energy < 0 or not isinstance(interval, list)
                or len(interval) != 2 or not all(isinstance(value, int) for value in interval)):
            raise ValueError('invalid_processed_window_retention_metadata')
        entries.append({'path': path, 'window_id': path.name, 'audio_rms': energy,
                        'source_end_ns': interval[1]})
    retained = []
    if entries:
        retained.append(max(entries, key=lambda item: (item['audio_rms'], item['source_end_ns'])))
        for item in sorted(entries, key=lambda item: item['source_end_ns'], reverse=True):
            if item not in retained:
                retained.append(item)
            if len(retained) >= retention:
                break
    retained_paths = {item['path'] for item in retained}
    removed = []
    for item in entries:
        if item['path'] in retained_paths:
            continue
        # A window still being processed is never removed.  The speech-priority
        # queue can hand out an older window after newer ones, so a just-encoded
        # window may otherwise look stale and be deleted under its own processing.
        if not (item['path']/'result.json').is_file():
            continue
        resolved = item['path'].resolve()
        if resolved.parent != root:
            raise ValueError('processed_window_path_escape')
        shutil.rmtree(resolved)
        removed.append({key: value for key, value in item.items() if key != 'path'})
    return {'retention': retention,
            'policy': 'strongest_audio_plus_newest_completed_windows',
            'retained': [{key: value for key, value in item.items() if key != 'path'} for item in retained],
            'removed': removed}


def asr_energy_limits(gate, history):
    """Silence limits for one window, from windows closed before it only.

    The absolute limits rise toward the recent noise floor (a low quantile, so
    occasional speech does not lift it) and stay capped below speech levels.
    """
    limits = {'rms_below': gate['rms_below'], 'peak_below': gate['peak_below'], 'basis': 'absolute'}
    if gate.get('activity'):
        limits['activity'] = dict(gate['activity'])
    adaptive = gate.get('adaptive')
    if not adaptive or len(history) < adaptive.get('minimum_windows', 10):
        return limits
    quantile = adaptive.get('floor_quantile', 0.2)
    floors = {}
    for index, name in enumerate(('rms', 'peak')):
        values = sorted(item[index] for item in history)
        floors[name] = values[int(quantile * (len(values) - 1))]
    limits.update(
        rms_below=min(adaptive['rms_below_cap'], max(gate['rms_below'], floors['rms']*adaptive['rms_floor_ratio'])),
        peak_below=min(adaptive['peak_below_cap'], max(gate['peak_below'], floors['peak']*adaptive['peak_floor_ratio'])),
        basis='recent_noise_floor', history_windows=len(history), floor_quantile=quantile,
        rms_floor=floors['rms'], peak_floor=floors['peak'])
    return limits


def face_detection_summary(path):
    """Primary vs NPU-shadow face detection over one window's frames."""
    from models.blazeface_decode import iou
    rows = [json.loads(line).get('face_detection') for line in path.read_text(encoding='utf-8').splitlines()]
    rows = [row for row in rows if row]
    result = {'frames': len(rows), 'primary_face_frames': sum(len(r['primary_faces']) == 1 for r in rows),
              'primary_mean_seconds': sum(r['primary_seconds'] for r in rows) / len(rows) if rows else None}
    shadow = [r for r in rows if 'npu_shadow' in r]
    if shadow:
        ok = [r for r in shadow if 'faces' in r['npu_shadow']]
        agree = sum(1 for r in ok if len(r['primary_faces']) == len(r['npu_shadow']['faces']) and
                    all(iou(p, s[:4]) >= 0.5 for p, s in zip(r['primary_faces'], r['npu_shadow']['faces'])))
        result['npu_shadow'] = {'frames': len(ok), 'errors': len(shadow) - len(ok),
            'face_frames': sum(len(r['npu_shadow']['faces']) == 1 for r in ok),
            'agree_frames': agree,
            'best_score_max': max((r['npu_shadow']['best_score'] for r in ok), default=None),
            'mean_seconds': sum(r['npu_shadow']['seconds'] for r in ok) / len(ok) if ok else None}
    return result


def perception_summary(folder):
    """Gate decision and backend timings, logged before the window can be purged."""
    integration = Path(folder)/'app/observation_integration'
    summary = {'window_id': Path(folder).name, 'asr_energy_gate': None, 'backends': {}}
    gate = integration/'asr_energy_gate.json'
    if gate.exists():
        value = json.loads(gate.read_text(encoding='utf-8'))
        summary['asr_energy_gate'] = {k: value.get(k) for k in ('skipped', 'policy', 'active_seconds', 'rms', 'peak', 'rms_below', 'peak_below', 'asr_sample_interval')}
        summary['asr_energy_gate']['basis'] = value.get('limits_basis', {}).get('basis')
    for audit in sorted(integration.glob('*_client_audit.json')):
        reply = json.loads(audit.read_text(encoding='utf-8'))[-1]
        summary['backends'][reply['backend_id']] = {'status': reply['status'],
            'worker_seconds': (reply['completed_ns']-reply['started_ns'])/1e9 if 'completed_ns' in reply else None}
    frames = Path(folder)/'app/frame_observations.jsonl'
    if frames.exists():
        summary['face_detection'] = face_detection_summary(frames)
    timing = Path(folder)/'app/pipeline_timing.json'
    if timing.exists():
        value = json.loads(timing.read_text(encoding='utf-8'))
        summary['pipeline_seconds'] = {k[:-3]: (v-value['started_ns'])/1e9 for k, v in value.items()
                                       if k.endswith('_ns') and k != 'started_ns'}
    return summary


def window_configuration(configuration, interaction, folder, energy_gate=None):
    """Freeze the exact preference revision used by a dispatched perception window."""
    config = deepcopy(configuration)
    live = config['live']
    realtime_backends = live.get('realtime_ser_backends')
    if realtime_backends is not None:
        if not isinstance(realtime_backends, list) or not realtime_backends:
            raise ValueError('realtime_ser_backends_required')
        config['observation_integration']['ser_backends'] = list(realtime_backends)
    realtime_threads = live.get('realtime_audio_torch_threads')
    if realtime_threads is not None:
        config['observation_integration']['audio_torch_threads_by_backend'] = dict(realtime_threads)
    # Live windows only; startup preparation keeps checking every backend.
    if energy_gate is None and live.get('asr_energy_gate') is not None:
        energy_gate = asr_energy_limits(live['asr_energy_gate'], [])
    if energy_gate is not None:
        config['observation_integration']['asr_energy_gate'] = dict(energy_gate)
    if interaction is not None:
        config['cooperative_ai']['preferences'] = deepcopy(interaction.history)
        config['live']['preference_revision'] = interaction.preference_revision
    path = Path(folder)/'processing_config.json'
    write(path, config)
    return path


def adopt_response(shared, interaction, result):
    if interaction is not None and not interaction.accept_result(result):
        return False
    shared['response'] = deepcopy(result)
    return True


def encode_window(window, output):
    import av
    from fractions import Fraction
    validate_closed_window(window, time.monotonic_ns())
    output.mkdir(parents=True, exist_ok=False)
    start, end = window['interval_ns']
    native_frames = window['frames']
    with av.open(str(output / 'video.mp4'), 'w') as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height = 640, 480
        stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, 1000000)
        stream.codec_context.time_base = stream.time_base
        stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'crf': '18'}
        for index, source in enumerate(native_frames):
            frame = av.VideoFrame.from_ndarray(source['bgr'], format='bgr24')
            frame.pts = round((source['native_ns'] - start) / 1000)
            frame.time_base = stream.time_base
            next_ns = native_frames[index+1]['native_ns'] if index+1 < len(native_frames) else end
            for packet in stream.encode(frame):
                packet.duration = round((next_ns-source['native_ns']) / 1e9 / packet.time_base)
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    audio = window['audio']
    with wave.open(str(output / 'audio.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(audio['pcm'])
    manifest = {k: v for k, v in window.items() if k not in ('frames', 'audio')}
    manifest['frames'] = [{k: v for k, v in f.items() if k != 'bgr'} for f in native_frames]
    manifest['audio'] = {k: v for k, v in audio.items() if k != 'pcm'}
    manifest.update(mode='live_closed_window', video=str(output/'video.mp4'), audio_file=str(output/'audio.wav'),
        pcm_sha256=hashlib.sha256(audio['pcm']).hexdigest(),
        video_origin_monotonic_ns=start, video_timestamp_quantization_seconds=1e-6,
        audio_offset_seconds=(audio['first_sample_monotonic_ns']-start)/1e9,
        absolute_av_sync_measured=False, human_reference_used=False)
    from modules.media_timeline import probe_video_native
    probe = probe_video_native(output/'video.mp4', 'pyav')
    pts = [float(f['best_effort_timestamp_time']) for f in probe['frames']]
    if len(pts) != len(native_frames) or max(abs(t-(f['native_ns']-start)/1e9) for t, f in zip(pts, native_frames)) > 2e-6:
        raise ValueError('mux_changed_native_timestamps')
    if abs(float(probe['streams'][0]['duration'])-(end-start)/1e9) > 2e-6:
        raise ValueError('mux_changed_closed_interval')
    manifest['native_mux_verified'] = True
    write(output / 'capture_manifest.json', manifest)
    return manifest


def process_window(manifest_path, config_path, *, runtime=None, defer=False):
    """Process one closed window.

    With ``defer`` the frame loop runs now and the remainder (waiting on this
    window's workers, the envelope and ARA) is returned as a callable, so the
    owner can start the next window's frame loop meanwhile.  Otherwise it runs
    to completion and returns None.
    """
    import yaml
    if runtime is None:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    from multimodal_app import MultiModalEmotionApp
    from modules.driver_response_policy import run_saved_observations
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['closed_available_ns'] > time.monotonic_ns():
        raise ValueError('future_window_request')
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    live = config['live']
    output = Path(manifest_path).parent / 'app'
    output.mkdir(exist_ok=False)
    config['output'].update(save_results=True, output_dir=str(output), save_frame_observations=True)
    config['display']['show_window'] = False
    config['logging']['output_dir'] = str(output/'logs')
    config['synchronization']['audio_offset_seconds'] = manifest['audio_offset_seconds']
    config['observation_integration'].update(clip_id=manifest['window_id'], worker_mode='online')
    cooperative = config['cooperative_ai']
    config['cooperative_ai'] = {'enabled': False}
    app_config = output / 'config.yaml'
    app_config.write_text(yaml.safe_dump(config), encoding='utf-8')
    started = time.monotonic_ns()
    app = runtime.new_app(app_config) if runtime is not None else MultiModalEmotionApp(str(app_config))
    app.defer_window_finish = defer and runtime is not None
    def release():
        session = getattr(app,'observation_session',None)
        if session is not None and hasattr(session,'close'): session.close()
        if runtime is None:
            app.emotion_detector.face_model.close()
    try:
        app.run(video_file=manifest['video'], audio_file=manifest['audio_file'])
    except BaseException:
        release()
        raise
    deferred = getattr(app, 'deferred_finish', None)
    def finish():
        try:
            if deferred is not None:
                deferred()
        finally:
            release()
        complete_window(manifest_path, manifest, output, live, cooperative, started)
    if deferred is not None:
        write(Path(manifest_path).parent/'frames_done.json',
              {'window_id': manifest['window_id'], 'frames_done_ns': time.monotonic_ns()})
        return finish
    finish()
    return None


def complete_window(manifest_path, manifest, output, live, cooperative, started):
    from modules.driver_response_policy import run_saved_observations
    envelope = output/'observation_envelope_v1.json'
    # Additional metadata cannot turn a batch worker's offline mode into live.
    data = json.loads(envelope.read_text())
    data['input']['live_capture_manifest_sha256'] = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    write(envelope, data)
    production_timeout = float(live.get('production_timeout_seconds', 10))
    if not 1 <= production_timeout <= 10:
        raise ValueError('production_timeout_seconds_out_of_range')
    production = run_saved_observations(envelope, output/'production_10s',
                                        {**cooperative, 'timeout_seconds': production_timeout})
    write(output.parent/'production_display.json', {
        'source_end_ns': manifest['interval_ns'][1], 'text': production['response']['accepted']['text'],
        'status': production['response']['status'], 'label':live.get('llm_label','ARA短文'),
        'available_ns': time.monotonic_ns(), 'window_id': manifest['window_id'],
        'preference_revision':live.get('preference_revision')})
    # Diagnostic replay is opt-in per run; production keeps its ten seconds.
    replay = live.get('diagnostic_replay', True)
    diagnostic = run_saved_observations(envelope, output/'diagnostic_120s',
        {**cooperative, 'timeout_seconds': live.get('diagnostic_seconds', 120)}) if replay else production
    result = {'source_end_ns': manifest['interval_ns'][1], 'text': diagnostic['response']['accepted']['text'],
        'label':'ARA診断短文' if replay else live.get('llm_label','ARA短文'),
        'status': diagnostic['response']['status'], 'recognition_status': diagnostic['recognition']['status'],
        'available_ns': time.monotonic_ns(), 'window_id': manifest['window_id'],
        'processing_seconds': (time.monotonic_ns()-started)/1e9,
        'production_response_status': production['response']['status'],
        'diagnostic_executed':replay,'result_source':'diagnostic_120s' if replay else 'production_10s',
        'diagnostic_budget_seconds': live.get('diagnostic_seconds', 120) if replay else None,
        'preference_revision':live.get('preference_revision')}
    write(output.parent/'result.json', result)


def run_live(configuration):
    import numpy as np
    from modules.demo_output import DemoFramebufferDisplay as FramebufferDisplay
    config = deepcopy(configuration)
    live = config['live']
    if live.get('speech_output'):
        import os
        speech = live['speech_output']
        os.environ['ARA_DEMO_SPEECH_CATALOG'] = speech['catalog']
        os.environ['ARA_DEMO_AUDIO_DEVICE'] = speech.get('device') or ''

    preparation = bool(live.get('startup_preparation'))
    if preparation and (not live.get('persistent_windows')
                        or live.get('capture_only') or live.get('ara_startup_warmup')):
        raise ValueError('preparation_requires_persistent_manual_capture_without_legacy_warmup')
    output = Path(live['output']).resolve()
    if not output.is_relative_to(WORKSPACE) or output.exists():
        raise ValueError('fresh_live_workspace_output_required')
    output.mkdir(parents=True)
    config_path = output / 'live_config.json'
    write(config_path, config)
    stop = threading.Event()
    inference_stop = threading.Event()
    preparation_ready = threading.Event()
    post_ready_capture = threading.Event()
    owned_runtime = {}
    if not preparation: preparation_ready.set()
    if not live.get('post_ready_start_gate', False): post_ready_capture.set()
    queue = SpeechPriorityWindowQueue(capacity=2)
    ring = TimedPCM(retain_seconds=max(30, live.get('window_seconds', 3)+10))
    shared = {'status': 'カメラ・マイクを起動中', 'camera_frames': 0, 'audio_samples': 0,
              'audio_rms': 0, 'max_result_age_seconds': live.get('max_result_age_seconds', 300),
              'sensor_capture_enabled':live.get('sensor_capture', True),
              'post_ready_capture_armed':post_ready_capture.is_set()}
    capture_camera,microphone_type=camera_frames,NativeMicrophone
    if live.get('capture_backend')=='windows':
        from modules.pc_live import windows_camera_frames,WindowsMicrophone
        capture_camera,microphone_type=windows_camera_frames,WindowsMicrophone
        shared['capture_clock_basis']='host_receive_camera_and_callback_sample_timeline_not_native'
    if 'preference_label' in live: shared['preference_label'] = live['preference_label']
    interaction = None
    if live.get('explicit_interactions', False):
        from modules.live_interaction import LiveInteraction
        interaction = LiveInteraction(config['cooperative_ai'].get('preferences', []),
            config['cooperative_ai'].get('guidance'), scope=live.get('interaction_scope','actual'))
        initial = live.get('fixture_initial_asr')
        if initial is not None:
            if interaction.scope != 'fixture' or live.get('sensor_capture', True):
                raise ValueError('fixture_observation_requires_fixture_display_session')
            interaction.observe_asr(initial['text'],initial['window_id'],time.monotonic_ns())
        shared.update(interaction.snapshot())
    if not live.get('sensor_capture', True) and interaction is None:
        raise ValueError('display_session_requires_explicit_interactions')
    lock, audit_lock = threading.Lock(), threading.Lock()
    audit = (output/'live_events.jsonl').open('x', encoding='utf-8')
    def emit(event):
        if event.get('event') in ('audio_chunk', 'video_frame'):
            return
        event = {**event, 'logged_ns': time.monotonic_ns()}
        with audit_lock:
            audit.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
            audit.flush()
    def register_fault(role, error):
        from modules.live_health import capture_fault
        with lock: fault=capture_fault(shared,role,error,time.monotonic_ns())
        emit({'event':'capture_failure',**fault})
        stop.set()
        inference_stop.set()
        queue.close()
    def guarded(function, role='capture'):
        try:
            function()
        except Exception as error:
            register_fault(role,f'{type(error).__name__}: {error}')
    def audio_event(event):
        emit(event)
        if event['event'] in ('audio_idle_recovery', 'audio_xrun_recovery'):
            with lock:
                shared.update(audio_recoveries=event['recovery_count'], audio_samples=0,
                              audio_available_ns=event['available_ns'])
            return
        with ring.lock:
            pcm = ring.chunks[-1]['pcm']
        signal_value = np.frombuffer(pcm, dtype='<i2').astype(np.float32)/32768
        with lock:
            shared['audio_samples'] = event['sample_interval'][1]
            shared['audio_available_ns'] = event['available_ns']
            shared['audio_rms'] = float(np.sqrt(np.mean(signal_value**2)))
    def camera():
        frames, counter = [], 0
        energy_gate = live.get('asr_energy_gate')
        audio_history = deque(maxlen=(energy_gate or {}).get('adaptive', {}).get('history_windows', 30))
        learn_before_arm = bool((energy_gate or {}).get('adaptive', {}).get('learn_before_arm'))
        preview = {'last_end_ns': 0, 'samples': 0}
        extension = live.get('window_extension')
        activity = (energy_gate or {}).get('activity')
        def speech_continues(start, end, base_seconds):
            """Keep a window open, within a bound, while speech runs past its boundary.

            On the device, fixed 4 s windows split utterances (「すね」, 「二識がおかしいぞ」).
            The same sustained-sound threshold as the ASR gate decides whether the
            last tail_seconds still sound; the window then closes in a pause instead.
            """
            if not extension or not activity:
                return False
            if (end - start) / 1e9 >= base_seconds + extension['max_extra_seconds']:
                return False
            if not ring.chunks:
                return False
            available = min(end, ring.chunks[-1]['end_ns'])
            if available - start < 1e9:
                return False
            try:
                audio = ring.slice(start, available, time.monotonic_ns())
            except ValueError:
                return False
            signal_value = np.frombuffer(audio['pcm'], dtype='<i2').astype(np.float32) / 32768
            frame_size = round(activity['frame_seconds'] * 16000)
            frame_rms = np.sqrt(np.mean(signal_value[:len(signal_value) // frame_size * frame_size]
                                        .reshape(-1, frame_size) ** 2, axis=1))
            threshold = max(activity['frame_rms_min'],
                            float(np.quantile(frame_rms, activity['floor_quantile'])) * activity['floor_ratio'])
            tail = frame_rms[-max(1, round(extension['tail_seconds'] / activity['frame_seconds'])):]
            return int(np.count_nonzero(tail > threshold)) >= extension['min_tail_active_frames']
        def learn_noise_floor():
            """Loudness only, during the pre-arm wait: no audio is saved or recognized."""
            span = int(live.get('window_seconds', 3) * 1e9)
            now = time.monotonic_ns()
            end = now - 500_000_000
            if end - preview['last_end_ns'] < span:
                return
            try:
                audio = ring.slice(end - span, end, now)
            except ValueError:
                return
            signal_value = np.frombuffer(audio['pcm'], dtype='<i2').astype(np.float32) / 32768
            if signal_value.size:
                audio_history.append((float(np.sqrt(np.mean(signal_value ** 2))),
                                      float(np.max(np.abs(signal_value)))))
                preview['samples'] += 1
            preview['last_end_ns'] = end
        from modules.live_capture import ManualCaptureGate
        gate = ManualCaptureGate() if live.get('record_on_request',False) else None
        for frame in capture_camera(stop, live.get('camera', '/dev/video4')):
            age = frame['available_ns']-frame['native_ns']
            emit({'event': 'video_frame', **{k:v for k,v in frame.items() if k != 'bgr'}})
            if age > 500_000_000:
                emit({'event': 'input_gap', 'reason': 'stale_camera_startup', 'native_ns': frame['native_ns'], 'age_ns': age})
                continue
            with lock:
                shared.update(frame=frame['bgr'], camera_native_ns=frame['native_ns'],
                              camera_frames=shared['camera_frames']+1)
            if inference_stop.is_set() or time.monotonic()-started < live.get('warmup_seconds', 10):
                frames = []
                continue
            if not preparation_ready.is_set():
                frames = []
                continue
            if not post_ready_capture.is_set():
                command_path = output/'continuous_start_command.json'
                if not command_path.exists():
                    with lock: shared['status'] = '準備完了。現在の発話準備確認を待っています'
                    if learn_before_arm:
                        learn_noise_floor()
                    frames = []
                    continue
                command = json.loads(command_path.read_text(encoding='utf-8'))
                authorized_ns = command.get('authorized_ns')
                if (command.get('explicit_user_ready') is not True or not isinstance(authorized_ns, int)
                        or authorized_ns > time.monotonic_ns()):
                    raise ValueError('invalid_post_ready_start_command')
                post_ready_capture.set()
                with lock:
                    shared['post_ready_capture_armed'] = True
                    shared['status'] = '実入力の連続取得を開始しました'
                emit({'event':'post_ready_capture_armed','authorized_ns':authorized_ns,
                      'request_id':command.get('request_id')})
                if energy_gate is not None:
                    emit({'event':'asr_noise_floor_preview','samples':preview['samples'],
                          'history_windows':len(audio_history),'audio_saved':False,
                          'limits':asr_energy_limits(energy_gate, audio_history)})
                frames = []
            if gate is not None:
                command = output/'capture_command.json'
                if command.exists() and gate.pending is None:
                    gate.arm(json.loads(command.read_text(encoding='utf-8')),time.monotonic_ns())
                was_active = gate.active
                if not gate.recording(frame['native_ns']):
                    with lock:
                        if gate.pending:
                            remaining=max(0,(gate.pending['not_before_ns']-time.monotonic_ns())/1e9)
                            shared['status']=f'発話録音まで {remaining:.0f} 秒。開始表示をお待ちください'
                        elif not gate.seen: shared['status']=live.get('idle_capture_status','発話試験の準備完了。録音開始の案内をお待ちください')
                    frames=[]
                    continue
                if not was_active:
                    emit({'event':'manual_recording_started','request_id':gate.pending['request_id'],
                          'native_ns':frame['native_ns']})
                    with lock:
                        shared.pop('asr_text',None)
                        shared.pop('asr_correction',None)
                        if interaction is not None: interaction.asr = {}
                        shared['status']='録音中：'+gate.pending.get('prompt','今日はいい天気ですね、と話してください')
            if frames and frame['native_ns']-frames[-1]['native_ns'] > 150_000_000:
                emit({'event': 'input_gap', 'reason': 'camera_frame_gap',
                      'interval_ns': [frames[-1]['native_ns'], frame['native_ns']]})
                frames = []
            target_win_sec = live.get('window_seconds', 3)
            if gate is not None and gate.pending and 'window_seconds' in gate.pending:
                try:
                    c_sec = float(gate.pending['window_seconds'])
                    if 1 <= c_sec <= 30: target_win_sec = c_sec
                except (ValueError, TypeError): pass
            if (frames and frame['native_ns']-frames[0]['native_ns'] >= target_win_sec*1e9
                    and not speech_continues(frames[0]['native_ns'], frame['native_ns'], target_win_sec)):
                start, end = frames[0]['native_ns'], frame['native_ns']
                # No future samples. Wait briefly for the microphone's same-time data.
                timeout = time.monotonic()+.4
                while True:
                    now = time.monotonic_ns()
                    try:
                        audio = ring.slice(start, end, now)
                        break
                    except ValueError:
                        if time.monotonic() >= timeout or stop.is_set():
                            audio = None
                            break
                        time.sleep(.02)
                if audio is not None:
                    signal_value = np.frombuffer(audio['pcm'], dtype='<i2').astype(np.float32) / 32768
                    audio_rms = float(np.sqrt(np.mean(signal_value ** 2)))
                    audio_peak = float(np.max(np.abs(signal_value)))
                    counter += 1
                    window = {'window_id': f'live_{counter:06d}', 'interval_ns': [start, end],
                        'closed_available_ns': time.monotonic_ns(), 'frames': frames, 'audio': audio,
                        'audio_rms': audio_rms, 'audio_peak': audio_peak}
                    if extension and activity:
                        window['boundary_extension_seconds'] = max(0.0, round((end-start)/1e9 - target_win_sec, 3))
                    if energy_gate is not None:
                        window['asr_energy_gate'] = asr_energy_limits(energy_gate, audio_history)
                    audio_history.append((audio_rms, audio_peak))
                    window['capture_clock_basis']=shared.get('capture_clock_basis','linux_native_monotonic')
                    if gate is not None: window['capture_request_id']=gate.pending['request_id']
                    validate_closed_window(window, time.monotonic_ns())
                    queue.put(window)
                    emit({'event': 'window_closed', 'window_id': window['window_id'],
                          'interval_ns': window['interval_ns'], 'frames': len(frames),
                          'audio_samples': audio['sample_count'], 'audio_rms': audio_rms,
                          'boundary_extension_seconds': window.get('boundary_extension_seconds'),
                          'audio_peak': audio_peak, 'closed_available_ns': window['closed_available_ns']})
                else:
                    emit({'event': 'input_gap', 'interval_ns': [start, end], 'reason': 'audio_window_not_available'})
                frames = []
                if gate is not None:
                    ident=gate.finish()
                    emit({'event':'manual_recording_finished','request_id':ident,'saved':audio is not None})
                    continue
            frames.append(frame)
    def consumer():
        owned = owned_runtime
        try:
            if preparation:
                launch_runtime(owned, config_path, persistent=True)
                deadline = time.monotonic()+live.get('preparation_deadline_seconds', 600)
                previous = None
                while not stop.is_set():
                    path = output/'preparation_state.json'
                    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'state':'perception','ready':False}
                    from modules.live_preparation import preparation_status
                    with lock:
                        shared['preparation_state'] = state['state']
                        shared['status'] = preparation_status(state['state'])
                    if state['state'] != previous:
                        emit({'event':'preparation_state', **state})
                        previous = state['state']
                    if owned['process'].poll() is not None:
                        raise RuntimeError('preparation_runtime_exited')
                    if state['ready']:
                        preparation_ready.set()
                        break
                    if state['state'] in ('failed','stopped'):
                        raise RuntimeError('preparation_failed')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('preparation_deadline')
                    stop.wait(.2)
                if stop.is_set(): return
            if live.get('ara_startup_warmup', False) and not live.get('capture_only', False):
                from modules.compact_driver_cycle import warmup_compact_cycle
                from modules.local_llm_client import LocalLLMClient
                cooperative = config['cooperative_ai']
                client = LocalLLMClient(cooperative['base_url'],cooperative['model'],wait_for_backend=True)
                for index in range(2):
                    if stop.is_set(): break
                    with lock: shared['status'] = 'ARA起動準備中（実入力の結果ではありません）'
                    result = warmup_compact_cycle(client,preferences=cooperative.get('preferences'),
                        transport=cooperative.get('transport','compact-cycle-v3'))
                    write(output/f'ara_startup_warmup_{index+1}.json',result)
                    emit({'event':'ara_startup_warmup','index':index+1,
                          'status':result['cycle']['response']['status'],
                          'seconds':result['cycle']['wall_seconds'],'used_sensor_input':False})
            consume_windows(owned)
        except Exception as error:
            preparation_ready.clear()
            if preparation:
                with lock:
                    shared.update(preparation_state='failed',status='準備に失敗しました。録音は開始しません',
                                  preparation_error=f'{type(error).__name__}: {error}')
            emit({'event':'window_failure','error':f'{type(error).__name__}: {error}','phase':'startup_or_consumer'})
            stop.set()
            inference_stop.set()
            queue.close()
        finally:
            preparation_ready.clear()
            process = owned.get('process')
            if process is not None:
                if process.poll() is None:
                    process.stdin.close()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        terminate_group(process)
                emit({'event': 'window_runtime_exited', 'pid': process.pid, 'returncode': process.returncode})
            if 'log' in owned:
                owned['log'].close()

    def terminate_group(process):
        if process.poll() is None:
            if os.name=='nt':
                import psutil
                parent=psutil.Process(process.pid)
                children=parent.children(recursive=True)
                for child in reversed(children):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                process.terminate();process.wait(timeout=10)
                _,alive=psutil.wait_procs(children,timeout=5)
                for child in alive:child.kill()
                psutil.wait_procs(alive,timeout=5)
                return
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def launch_runtime(owned, request_config, *, persistent, folder=None):
        log = (output/'window_runtime.log' if persistent else folder/'process.log').open('x',encoding='utf-8')
        owned['log'] = log
        command = [sys.executable, '-B', '-u', '-m', 'modules.live_pipeline', '--config', str(request_config)]
        command += ['--serve-windows'] if persistent else ['--process-window', str(folder/'capture_manifest.json')]
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.PIPE,
            stdout=log, stderr=log, start_new_session=True)
        owned['process'] = process
        emit({'event':'window_runtime_started','pid':process.pid,'persistent':persistent})
        return process

    def consume_windows(owned):
        completed = 0
        persistent = live.get('persistent_windows', False)
        # Overlap needs the persistent runtime; it sends one window ahead at most.
        overlap = persistent and live.get('cross_window_overlap', False)
        state = {'process': owned.get('process')}

        def dispatch(window, *, ahead=False):
            """Encode and send one window; None if expired, 'capture_only' to stop."""
            now = time.monotonic_ns()
            if now-window['interval_ns'][1] > live.get('window_max_queue_age_seconds', 8)*1e9:
                emit({'event':'input_gap','window_id':window['window_id'],'reason':'queued_window_expired'})
                return None
            folder = output/'windows'/window['window_id']
            with lock: shared['status'] = '実入力窓の知覚・ARA診断を処理中（遅延あり）'
            encode_started = time.monotonic_ns()
            manifest = encode_window(window, folder)
            encode_seconds = (time.monotonic_ns() - encode_started) / 1e9
            retention = purge_processed_windows(
                output/'windows', int(live.get('processed_window_retention', 3)))
            write(output/'processed_window_retention.json', retention)
            for removed in retention['removed']:
                emit({'event': 'processed_window_purged', **removed,
                      'policy': retention['policy']})
            if not ahead:
                # A window sent ahead must not hide the result still due before it.
                with lock:
                    shared['visual_result'] = dict(state='processing', window_id=manifest['window_id'],
                                                   source_end_ns=manifest['interval_ns'][1])
            emit({'event': 'processing_started', 'window_id': manifest['window_id'], 'sent_ahead': ahead,
                  'encode_seconds': encode_seconds})
            if live.get('capture_only', False):
                emit({'event': 'capture_window_verified', 'window_id': manifest['window_id'],
                      'native_mux_verified': manifest['native_mux_verified']})
                inference_stop.set()
                queue.close()
                with lock: shared['status'] = 'カメラ・マイクのnative窓保存を検証。ライブ表示継続中'
                return 'capture_only'
            with lock:
                request_config = window_configuration(config, interaction, folder,
                                                      manifest.get('asr_energy_gate'))
            process = state['process']
            if not persistent or process is None:
                process = launch_runtime(owned, config_path if persistent else request_config,
                                         persistent=persistent, folder=folder)
                state['process'] = process
            if persistent:
                process.stdin.write((json.dumps({'manifest':str(folder/'capture_manifest.json'),
                    'config':str(request_config)})+'\n').encode())
                process.stdin.flush()
            return {'folder': folder, 'manifest': manifest,
                    'deadline': time.monotonic()+live.get('worker_deadline_seconds', 600)}

        def overlap_allowed(folder):
            """On the device, a next window run beside SenseVoice slowed ASR by up to 2 s.

            Send ahead only while the current window skipped ASR, unless configured otherwise.
            """
            if not live.get('overlap_only_when_asr_skipped', True):
                return True
            path = folder/'app/observation_integration/asr_energy_gate.json'
            try:
                return json.loads(path.read_text(encoding='utf-8')).get('skipped') is True
            except (OSError, ValueError):
                return False

        ahead = None
        while not inference_stop.is_set():
            folder = None
            try:
                if ahead is None:
                    window = queue.get()
                    if window is None:
                        break
                    current = dispatch(window)
                    del window
                    if current is None:
                        continue
                    if current == 'capture_only':
                        break
                else:
                    current, ahead = ahead, None
                folder, manifest, deadline = current['folder'], current['manifest'], current['deadline']
                process = state['process']
                adopted_production = False
                adopted_speech = False
                adopted_visual = False

                def publish_visual():
                    from modules.live_visual_display import visual_observation_result
                    path = folder/'app/production_10s/compact_observations.json'
                    compact = json.loads(path.read_text(encoding='utf-8'))
                    visual = visual_observation_result(compact, manifest['window_id'],
                                                       manifest['interval_ns'][1], time.monotonic_ns())
                    with lock: shared['visual_result'] = visual
                    emit({'event': 'visual_observations_available', **visual})

                while True:
                    if (overlap and ahead is None and not inference_stop.is_set()
                            and (folder/'frames_done.json').exists() and overlap_allowed(folder)):
                        waiting = queue.get_nowait()
                        if waiting is not None:
                            sent = dispatch(waiting, ahead=True)
                            del waiting
                            if isinstance(sent, dict):
                                ahead = sent
                    if not adopted_visual and (folder/'app/production_10s/compact_observations.json').exists():
                        try: publish_visual()
                        except (ValueError, KeyError, OSError, TypeError): pass
                        else: adopted_visual = True
                    speech = folder/'app/speech_result.json'
                    if speech.exists() and not adopted_speech:
                        try: transcript=json.loads(speech.read_text(encoding='utf-8'))['result']['text']
                        except (ValueError,KeyError): pass
                        else:
                            with lock:
                                if interaction is not None:
                                    interaction.observe_asr(transcript,manifest['window_id'],manifest['interval_ns'][1])
                                    shared.pop('asr_correction',None)
                                    shared.update(interaction.snapshot())
                                else:
                                    shared.update(asr_text=transcript, asr_source_end_ns=manifest['interval_ns'][1],
                                                  asr_window_id=manifest['window_id'])
                            emit({'event':'asr_text_available','window_id':manifest['window_id'],
                                  'text':transcript,'speaker_verified':False,'changes_preferences':False})
                            adopted_speech=True
                    production = folder/'production_display.json'
                    if production.exists() and not adopted_production:
                        try:
                            result = json.loads(production.read_text(encoding='utf-8'))
                        except ValueError:
                            time.sleep(.02)
                            continue
                        with lock: adopted = adopt_response(shared, interaction, result)
                        emit({'event': 'production_response_available', **result, 'display_adopted':adopted})
                        adopted_production = True
                    if (folder/'result.json').exists():
                        try:
                            result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
                            break
                        except ValueError:
                            pass
                    if process.poll() is not None:
                        raise RuntimeError(f'window_process_exit_{process.returncode}')
                    if time.monotonic() >= deadline or stop.is_set():
                        terminate_group(process)
                        raise TimeoutError('perception_worker_stopped_or_deadline')
                    time.sleep(.2)
                if not persistent:
                    process.wait(timeout=10)
                    owned['log'].close()
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'window_process_exit_{process.returncode}')
                if not adopted_visual:
                    try: publish_visual()
                    except (ValueError, KeyError, OSError, TypeError) as error:
                        with lock:
                            shared['visual_result'] = dict(state='unavailable', window_id=manifest['window_id'],
                                                           source_end_ns=manifest['interval_ns'][1])
                        emit({'event': 'visual_display_unavailable', 'window_id': manifest['window_id'],
                              'error': f'{type(error).__name__}: {error}'})
                with lock:
                    adopted = adopt_response(shared, interaction, result)
                    shared['status'] = '実入力窓のARA短文の処理完了。カメラ・マイクは取得継続中'
                emit({'event': 'diagnostic_response_available', **result, 'display_adopted':adopted})
                try:
                    emit({'event': 'window_perception_summary', **perception_summary(folder)})
                except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
                    emit({'event': 'window_perception_summary_unavailable', 'window_id': manifest['window_id'],
                          'error': f'{type(error).__name__}: {error}'})
                completed += 1
                if completed >= live.get('maximum_processed_windows', 1):
                    inference_stop.set()
                    queue.close()
                    if live.get('stop_after_success', False):
                        stop.wait(1)
                        stop.set()
                    break
            except Exception as error:
                emit({'event': 'window_failure', 'error': f'{type(error).__name__}: {error}', 'window': str(folder),
                      'window_sent_ahead': ahead['manifest']['window_id'] if isinstance(ahead, dict) else None})
                with lock:
                    shared['status'] = '処理失敗。入力は継続中：'+str(error)
                    if shared.get('visual_result'):
                        shared['visual_result'] = {**shared['visual_result'], 'state': 'unavailable'}
                inference_stop.set()
                queue.close()
                if live.get('stop_on_failure', False): stop.set()
                break
    def request_stop(*_):
        emit({'event': 'stop_requested'})
        stop.set()
        inference_stop.set()
        queue.close()
    def resources():
        from modules.live_resources import memory_snapshot
        sample_seconds = float(live.get('resource_sample_seconds', 60))
        if not 5 <= sample_seconds <= 300:
            raise ValueError('resource_sample_seconds_out_of_range')
        next_sample = 0.0
        with (output/'memory_samples.jsonl').open('x', encoding='utf-8') as stream:
            while not stop.is_set():
                snapshot = memory_snapshot(WORKSPACE)
                now = time.monotonic()
                if now >= next_sample:
                    stream.write(json.dumps(snapshot, allow_nan=False)+'\n')
                    stream.flush()
                    next_sample = now + sample_seconds
                if snapshot['mem_available_bytes'] < live.get('minimum_mem_available_bytes', 400*1024**2):
                    emit({'event':'resource_headroom_stop', 'mem_available_bytes':snapshot['mem_available_bytes']})
                    request_stop()
                import shutil
                if shutil.disk_usage(output).free < live.get('minimum_disk_free_bytes',150*1024**2):
                    register_fault('storage','minimum_disk_headroom')
                stop.wait(5)
    old_handler = signal.signal(signal.SIGINT, request_stop)
    old_term_handler = signal.signal(signal.SIGTERM, request_stop)
    def microphone():
        device = live.get('microphone', 'hw:3,0')
        allow_continuous = live.get('allow_capture_xrun_recovery', True)
        max_recoveries = int(live.get('max_xrun_recoveries', 100))
        if live.get('capture_backend') == 'windows':
            microphone_type(device).run(stop, ring, audio_event)
        else:
            microphone_type(
                device,
                idle_recovery_allowed=lambda: (
                    live.get('record_on_request', False) and not stop.is_set()
                    and not (output/'capture_command.json').exists()
                    and not (output.parent/'recording_operator_ready.json').exists()
                ),
                allow_continuous_recovery=allow_continuous,
                max_recoveries=max_recoveries
            ).run(stop, ring, audio_event)
    threads = [threading.Thread(target=lambda:guarded(microphone,'microphone'),name='microphone'),
               threading.Thread(target=lambda:guarded(camera,'camera'),name='camera'),
               threading.Thread(target=consumer,name='perception')]
    if not live.get('sensor_capture', True):
        threads = [threading.Thread(target=consumer,name='perception')] if preparation else []
        shared['status'] = '希望変更の表示試験（録音・推論なし／試験値）' if interaction.scope == 'fixture' else '希望の設定・表示（録音・推論なし）'
        if preparation:
            shared.update(preparation_state='perception',status='知覚モデルを準備中（保存素材・発話不要）')
    if live.get('sample_resources', False):
        threads.append(threading.Thread(target=lambda:guarded(resources),name='resources'))
    if live.get('display_backend')=='windows':
        from modules.pc_live import WindowsDisplay
        display=WindowsDisplay(output,live['font_path'],headless=live.get('headless_display',False))
    else:
        display = FramebufferDisplay(output, live['font_path'])
    started, last_save, display_records = time.monotonic(), 0, []
    started_ns=time.monotonic_ns()
    last_response_key = None
    try:
        for thread in threads: thread.start()
        while not stop.is_set() and time.monotonic()-started < live.get('duration_seconds', 600):
            if (output/'stop_requested.json').exists():
                request_stop()
                break
            if preparation_ready.is_set() and preparation and owned_runtime['process'].poll() is not None:
                register_fault('preparation','prepared_runtime_exited')
                break
            from modules.live_health import stale_capture
            with lock: missing=stale_capture(shared,time.monotonic_ns(),started_ns,
                startup_seconds=live.get('capture_startup_deadline_seconds',15),
                timeout_seconds=live.get('capture_stale_seconds',3))
            if missing:
                register_fault(missing,'capture_heartbeat_deadline')
                break
            with lock:
                if interaction is not None:
                    from modules.live_interaction import poll_commands
                    poll_commands(interaction, output, time.monotonic_ns(), emit)
                    shared.update(interaction.snapshot())
                snapshot = dict(shared)
            if 'camera_native_ns' in snapshot:
                snapshot['camera_age_seconds']=(time.monotonic_ns()-snapshot['camera_native_ns'])/1e9
            snapshot['dropped_windows']=len(queue.gaps)
            response = snapshot.get('response', {})
            visual_result = snapshot.get('visual_result', {})
            response_key = (response.get('window_id'), response.get('available_ns'), snapshot.get('interaction_revision'), snapshot.get('preparation_state'),
                            visual_result.get('window_id'), visual_result.get('state'), visual_result.get('available_ns'))
            save = time.monotonic()-last_save > 10 or response_key != last_response_key
            record = display.draw(snapshot, save=save)
            if save:
                record.update(interaction_revision=snapshot.get('interaction_revision'),
                              preference_revision=snapshot.get('preference_revision'))
                if interaction is not None:
                    saved = output/f'display_revision_{interaction.revision:04d}.png'
                    if not saved.exists(): saved.write_bytes((output/'display_latest.png').read_bytes())
                if preparation and snapshot.get('preparation_state'):
                    saved = output/('display_preparation_'+snapshot['preparation_state']+'.png')
                    if not saved.exists(): saved.write_bytes((output/'display_latest.png').read_bytes())
                display_records.append(record)
                emit({'event':'display_readback',**record})
                last_save = time.monotonic()
                last_response_key = response_key
                write(output/'status.json', {k:v for k,v in snapshot.items() if k != 'frame'})
            time.sleep(.2)
    finally:
        request_stop()
        for thread in threads:
            thread.join(timeout=live.get('worker_deadline_seconds',600)+10 if thread.name=='perception' else 8)
        with lock: snapshot=dict(shared)
        if preparation:
            snapshot['preparation_state'] = 'failed' if snapshot.get('preparation_state') == 'failed' else 'stopped'
        snapshot['dropped_windows'] = len(queue.gaps)
        if not snapshot.get('input_faults'):
            snapshot['status']='試験終了：カメラ・マイク停止。最後の結果を表示中'
        display_record=None
        report={'mode':'live_closed_windows','elapsed_seconds':time.monotonic()-started,
            'camera_frames':snapshot['camera_frames'],'audio_samples':snapshot['audio_samples'],
            'response':snapshot.get('response'),'gaps':queue.gaps,'display_frames':display.frames,
            'threads_stopped':{thread.name:not thread.is_alive() for thread in threads},
            'input_faults':snapshot.get('input_faults',[]),'absolute_av_sync_measured':False,
            'preparation_state':snapshot.get('preparation_state'),
            'preparation_error':snapshot.get('preparation_error'),
            'independent_quality_evaluated':False,'ten_second_generation_achieved':False}
        try:
            display_record=display.draw(snapshot,save=True)
            emit({'event':'display_final',**display_record})
        finally:
            report['display_final']=display_record
            write(output/'status.json', {k:v for k,v in snapshot.items() if k != 'frame'})
            write(output/'final_report.json',report)
            try: display.close()
            finally:
                audit.close()
                signal.signal(signal.SIGINT,old_handler)
                signal.signal(signal.SIGTERM,old_term_handler)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--process-window')
    parser.add_argument('--serve-windows', action='store_true')
    args=parser.parse_args()
    if args.serve_windows:
        from modules.live_window_runtime import serve_windows
        serve_windows(args.config)
    elif args.process_window:
        process_window(args.process_window,args.config)
    else:
        import yaml
        result=run_live(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))
        if result['input_faults'] or result['preparation_error'] or not all(result['threads_stopped'].values()):
            raise SystemExit(2)
