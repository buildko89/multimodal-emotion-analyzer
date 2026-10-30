"""File IPC worker using existing isolated driver_eval research adapters."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import uuid
import wave

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
DRIVER = WORKSPACE / 'devai/driver_eval'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DRIVER / 'scripts'))

from modules.observation_worker_client import digest, validate_request, write_json


def process_memory():
    """Individual worker process; never a summed system peak."""
    if sys.platform != 'win32':
        return {'status': 'not_measured'}
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set', 'quota_peak_paged',
            'quota_paged', 'quota_peak_non_paged', 'quota_non_paged', 'pagefile', 'peak_pagefile', 'private_usage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'status': 'measured', 'peak_working_set_bytes': counters.peak_working_set,
            'working_set_bytes': counters.working_set, 'private_usage_bytes': counters.private_usage,
            'method': 'Windows GetProcessMemoryInfo; process lifetime including imports/load'}


def visual_models(models):
    """Verify and build the face/hand graphs once per worker process.

    On the device, per-window creation and model hashing ran in every 4 s window.
    """
    import mediapipe as mp
    from face_landmarker_adapter import FaceLandmarkerAdapter
    if models is not None and 'visual' in models:
        return models['visual'], True, 0.0
    started = time.perf_counter()
    hand_manifest = json.loads((DRIVER / 'hand_landmarker_experiment_manifest.json').read_text(encoding='utf-8'))
    hand_path = DRIVER / hand_manifest['path']
    if digest(hand_path) != hand_manifest['sha256']:
        raise ValueError('Hand model changed')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        **{k: v for k, v in hand_manifest['config'].items() if k != 'running_mode'})
    adapter = FaceLandmarkerAdapter()
    try:
        hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    except BaseException:
        adapter.close()
        raise
    built = {'face': adapter, 'hand': hand_detector, 'hand_manifest': hand_manifest,
             'instance_id': uuid.uuid4().hex, 'calls': 0}
    if models is not None:
        models['visual'] = built
    return built, False, time.perf_counter() - started


def close_visual_models(built, models):
    if models is not None:
        models.pop('visual', None)
    built['hand'].close()
    built['face'].close()


def visual(request, folder, models=None):
    import cv2
    import mediapipe as mp
    import numpy as np
    from eye_event_detector import EyeEventConfig, detect_eye_events
    from eye_hand_overlap import classify, gate
    from mm03_mouth_gate import mouth_risk, CONFIG
    if request['interval_seconds'][0] != 0:
        raise ValueError('Visual prototype processes a complete zero-origin recording')
    stride = request.get('visual_stride', 2)
    if isinstance(stride, bool) or not isinstance(stride, int) or not 1 <= stride <= 8:
        raise ValueError('visual_stride_out_of_range')
    built, reused, setup_seconds = visual_models(models)
    adapter, hand_detector, hand_manifest = built['face'], built['hand'], built['hand_manifest']
    # A reused VIDEO graph needs increasing timestamps; each recording restarts at 0.
    graph_offset_ms = 0 if adapter.last_timestamp_ms is None else adapter.last_timestamp_ms + 1000
    cap = cv2.VideoCapture(request['input_path'])
    if not cap.isOpened():
        if models is None:
            close_visual_models(built, models)
        raise ValueError('Video could not be opened')
    rows, risks, last_pts = [], [], -1.0
    failed = True
    try:
        with (folder / 'visual_observations.jsonl').open('x', encoding='utf-8') as stream:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(pts) or pts <= last_pts or pts >= request['interval_seconds'][1]:
                    raise ValueError('Invalid native PTS; no unlabeled timestamp fallback')
                last_pts = pts
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if len(rows) % stride != 0 and face is not None:
                    face = dict(face, frame_index=len(rows), timestamp_s=pts, frame_time_s=pts, timestamp_ms=round(pts * 1000))
                    eye_risk = dict(eye_risk, frame_time_s=pts)
                else:
                    face = adapter.analyze(rgb, asset_id=request['clip_id'], timestamp_s=pts, graph_offset_ms=graph_offset_ms)
                    if face['status'] == 'backend_error':
                        raise RuntimeError(face.get('error'))
                    hand_result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
                    hands = [[[p.x, p.y, p.z] for p in points] for points in hand_result.hand_landmarks]
                    height, width = bgr.shape[:2]
                    eye_risk = classify(face, hands, width, height)
                    mouth = mouth_risk(face, hands, width, height)
                    face.update(frame_index=len(rows), hands=hands, hand_mouth=mouth, eye_hand_risk=eye_risk,
                                hand_model_sha256=hand_manifest['sha256'])
                stream.write(json.dumps(face, allow_nan=False) + '\n')
                rows.append(face)
                risks.append(eye_risk)
        failed = False
    finally:
        cap.release()
        # A failed window may leave graph state unknown; rebuild for the next one.
        if failed or models is None:
            close_visual_models(built, models)
    built['calls'] += 1
    if not rows:
        raise ValueError('No video frames')
    eye_events = detect_eye_events(gate(rows, risks), duration_s=request['interval_seconds'][1])
    # No new event threshold is fitted to the three human reference labels.
    geometry = {'frame_count': len(rows), 'inference_stride': stride,
                'inference_frames': (len(rows) + stride - 1) // stride, 'eye_events': eye_events,
                'temporal_state': {'scope': 'worker_graph_across_windows' if reused else 'this_window_only', 'state_id': uuid.uuid4().hex,
                                   'fresh_video_graph': not reused, 'timestamp_origin_seconds': 0,
                                   'graph_timestamp_offset_ms': graph_offset_ms},
                'worker_models': {'reused': reused, 'setup_seconds': setup_seconds,
                                  'instance_id': built['instance_id'], 'calls': built['calls']},
                'head_motion': {'status': 'raw_transform_only', 'nod_classifier': 'not_implemented'},
                'mouth': {'status': 'raw_aperture_and_projected_hand_risk', 'visibility_verified': False},
                'models': {'face': adapter.provenance, 'hand': hand_manifest},
                'experimental_config': {'eye': asdict(EyeEventConfig()), 'mouth': CONFIG},
                'accuracy_measured': False}
    write_json(folder / 'visual_summary.json', geometry)
    return ['visual_observations.jsonl', 'visual_summary.json']


def audio(request, folder, adapters=None):
    import numpy as np
    import torch
    torch.set_num_threads(request.get('torch_num_threads',4))
    with wave.open(request['input_path'], 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Common input must be 16kHz mono PCM16')
        total = stream.getnframes()
        begin, end = request['sample_interval']
        if not (isinstance(begin, int) and isinstance(end, int) and 0 <= begin < end <= total):
            raise ValueError('Invalid sample interval')
        stream.setpos(begin)
        pcm = stream.readframes(end - begin)
    import hashlib
    if hashlib.sha256(pcm).hexdigest() != request['pcm_sha256']:
        raise ValueError('PCM hash mismatch')
    offset = request['audio_offset_seconds']
    if request['interval_seconds'] != [begin / 16000 + offset, end / 16000 + offset]:
        raise ValueError('Sample/media interval mismatch')
    backend = request['backend_id']
    from modules.funasr_scoped_import import prepare_funasr
    import_policy = prepare_funasr(backend)
    cached = adapters is not None and backend in adapters
    if cached:
        adapter = adapters[backend]
        if getattr(adapter, 'quantization', None) != request.get('quantization'):
            raise ValueError('cached_adapter_quantization_mismatch')
    elif backend == 'sensevoice':
        from sensevoice_adapter import SenseVoiceAdapter
        adapter = SenseVoiceAdapter(quantization=request.get('quantization'))
    else:
        from emotion2vec_adapter import Emotion2vecAdapter
        adapter = Emotion2vecAdapter()
    # FunASR resets torch's thread count from backend.kwargs during generate.
    # Loading first also prevents its constructor's ncpu=4 overriding the request.
    adapter.load()
    adapter.backend.kwargs['ncpu'] = request.get('torch_num_threads',4)
    # Pinned FunASR also restores a frozen kwargs snapshot before every call.
    for config in getattr(adapter.backend,'_base_kwargs_map',{}).values():
        if isinstance(config,dict):config['ncpu'] = request.get('torch_num_threads',4)
    torch.set_num_threads(request.get('torch_num_threads',4))
    if not hasattr(adapter, '_worker_instance_id'):
        adapter._worker_instance_id = uuid.uuid4().hex
        adapter._worker_calls = 0
    observation = adapter.analyze(np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768,
                                  asset_id=request['clip_id'], audio_start_s=begin / 16000,
                                  audio_offset_s=offset, signal_scope='unverified', speaker='speaker_unverified')
    if observation['status'] != 'ok':
        raise RuntimeError(observation)
    if torch.get_num_threads() != request.get('torch_num_threads',4):
        raise RuntimeError('audio_thread_budget_not_applied')
    adapter._worker_calls += 1
    if adapters is not None:
        adapters[backend] = adapter
    write_json(folder / 'audio_observation.json', {'pcm_sha256': request['pcm_sha256'], 'sample_interval': [begin, end],
        'torch_num_threads':torch.get_num_threads(),
        'audio_import_policy': import_policy,
        'observation': observation, 'load_report': adapter.load_report,
        'model_reused': cached, 'model_initializations': 1,
        'model_instance_id': adapter._worker_instance_id, 'model_calls': adapter._worker_calls})
    return ['audio_observation.json']


def execute(request_path, adapters=None):
    request_path = request_path.resolve()
    if not request_path.is_relative_to(WORKSPACE) or request_path.stat().st_size > 1024 * 1024:
        raise ValueError('Request location/size invalid')
    request = json.loads(request_path.read_text(encoding='utf-8'))
    validate_request(request)
    folder = request_path.parent
    started = time.monotonic_ns()
    path = Path(request['input_path']).resolve()
    if not path.is_relative_to(WORKSPACE) or digest(path) != request['input_sha256']:
        raise ValueError('Input location/hash invalid')
    if started > request['deadline_monotonic_ns']:
        raise TimeoutError('Expired before backend execution')
    reclamation = None
    if request['backend_id'] == 'visual':
        files = visual(request, folder, adapters)
        from modules.native_memory_reclaim import release_idle_pages
        reclamation = release_idle_pages()
    elif request['backend_id'] in {'sensevoice', 'emotion2vec'}:
        files = audio(request, folder, adapters)
    else:
        raise ValueError('Unsupported backend')
    reply = {k: request[k] for k in ('protocol', 'request_id', 'backend_id', 'input_sha256')}
    reply.update(status='ok', started_ns=started, completed_ns=time.monotonic_ns(),
                 process_memory=process_memory(),
                 artifacts=[{'path': name, 'sha256': digest(folder / name)} for name in files])
    if reclamation is not None:
        reply['native_memory_reclamation'] = reclamation
    write_json(folder / 'reply.json', reply)
    return request['request_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--request', type=Path)
    mode.add_argument('--serve-audio', action='store_true')
    mode.add_argument('--serve-visual', action='store_true')
    args = parser.parse_args()
    if args.request:
        execute(args.request)
        return
    adapters = {}
    output = sys.stdout
    while True:
        line = sys.stdin.buffer.readline(65537)
        if not line:
            return
        if len(line) > 65536:
            raise ValueError('persistent_request_line_oversized')
        transport = json.loads(line)
        path = Path(transport['request_path']).resolve()
        if not path.is_relative_to(WORKSPACE) or path.stat().st_size > 1024 * 1024:
            raise ValueError('persistent_request_location_or_size')
        request = json.loads(path.read_text(encoding='utf-8'))
        allowed = {'visual'} if args.serve_visual else {'sensevoice', 'emotion2vec'}
        if request.get('request_id') != transport['request_id'] or request.get('backend_id') not in allowed:
            raise ValueError('persistent_audio_request_identity')
        with redirect_stdout(sys.stderr):
            request_id = execute(path, adapters)
        output.write(json.dumps({'request_id': request_id, 'status': 'completed'})+'\n')
        output.flush()


if __name__ == '__main__':
    main()
