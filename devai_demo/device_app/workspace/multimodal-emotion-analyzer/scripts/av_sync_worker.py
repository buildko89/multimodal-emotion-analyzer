"""CPU SyncNet file-IPC worker using local assets and machine ASR intervals only."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import wave

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
RUNTIME = WORKSPACE / 'devai/driver_eval/runtime/syncnet'
sys.path.insert(0, str(REPO))
from modules.av_sync_backend import CONFIG, SCHEMA, geometry_tracks, segment_geometry, validate_windows, scan_embeddings, correspondence
from modules.observation_worker_client import digest, validate_request, write_json

REQUEST_KEYS = {'protocol', 'request_id', 'backend_id', 'input_path', 'input_sha256',
                'interval_seconds', 'deadline_monotonic_ns', 'mode', 'available_until_s',
                'clip_id', 'audio_input_path', 'audio_sha256', 'audio_offset_seconds',
                'geometry_path', 'geometry_sha256', 'classification_path', 'classification_sha256', 'asr_windows'}


def inputs(request):
    validate_request(request)
    if set(request) - REQUEST_KEYS or request['mode'] != 'offline' or request['backend_id'] != 'syncnet':
        raise ValueError('Only the offline SyncNet machine-input contract is supported')
    if request['interval_seconds'][0] != 0:
        raise ValueError('Video timeline must start at zero')
    validate_windows(request['asr_windows'], request['interval_seconds'][1])
    offset = request['audio_offset_seconds']
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0:
        raise ValueError('Common audio needs a finite nonnegative media origin')
    paths = {}
    for key, hash_key in [('input_path', 'input_sha256'), ('audio_input_path', 'audio_sha256'),
                          ('geometry_path', 'geometry_sha256'), ('classification_path', 'classification_sha256')]:
        path = Path(request[key]).resolve()
        if not path.is_relative_to(WORKSPACE) or not path.is_file() or digest(path) != request[hash_key]:
            raise ValueError('AV input path/hash mismatch: ' + key)
        paths[key] = path
    return paths


def load_runtime():
    acquisition = json.loads((RUNTIME / 'acquisition.json').read_text(encoding='utf-8'))
    for name in ('upstream/SyncNetModel.py', 'syncnet_v2.model'):
        if digest(RUNTIME / name) != acquisition['assets'][name]['sha256']:
            raise ValueError('Existing SyncNet asset changed: ' + name)
    sys.path.insert(0, str(RUNTIME / 'vendor'))
    sys.path.insert(0, str(RUNTIME / 'upstream'))
    import torch
    from SyncNetModel import S
    torch.set_num_threads(CONFIG['cpu_threads'])
    model = S().cpu().eval()
    state = torch.load(RUNTIME / 'syncnet_v2.model', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'scipy', 'opencv-python-headless', 'python_speech_features')}
    return model, {'device': 'cpu', 'strict_state_dict': True, 'versions': versions,
                   'weight_sha256': digest(RUNTIME / 'syncnet_v2.model'),
                   'code_sha256': digest(RUNTIME / 'upstream/SyncNetModel.py'),
                   'repository_commit': acquisition['repository_commit']}


def crops(video, rows, tracks, boxes, duration):
    import cv2
    import numpy as np
    from scipy.signal import medfilt
    cap = cv2.VideoCapture(str(video))
    native_crops, rects = [], []
    pixel_boxes = [None] * len(boxes)
    # Frame dimensions come from decoding, never a fixed 720p assumption.
    decoded = []
    try:
        for row, box in zip(rows, boxes):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video decode ended before native geometry')
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if abs(pts - row['frame_time_s']) > 1e-3:
                raise ValueError('Decoded PTS differs from native geometry')
            decoded.append(frame)
            if box is not None:
                height, width = frame.shape[:2]
                lo, hi = np.asarray(box[:2]) * [width, height], np.asarray(box[2:]) * [width, height]
                center = (lo + hi) / 2
                size = float(max(hi - lo) / 2)
                if size < 1 or size > max(width, height) or not np.isfinite(center).all():
                    raise ValueError('Invalid face crop bounds')
                pixel_boxes[row['frame_index']] = [*center, size]
        if cap.read()[0]:
            raise ValueError('Unexpected additional decoded frames')
    finally:
        cap.release()
    start = 0
    kernels = []
    while start < len(rows):
        if pixel_boxes[start] is None:
            start += 1
            continue
        end = start + 1
        while end < len(rows) and pixel_boxes[end] is not None and tracks[end] == tracks[start] and rows[end]['frame_time_s'] - rows[end - 1]['frame_time_s'] <= CONFIG['maximum_frame_gap_seconds']:
            end += 1
        kernel = min(CONFIG['median_filter_kernel'], (end - start - 1) // 2 * 2 + 1)
        smoothed = np.stack([medfilt(np.asarray(pixel_boxes[start:end])[:, column], kernel) for column in range(3)], axis=1)
        for index, value in enumerate(smoothed, start):
            pixel_boxes[index] = value.tolist()
        kernels.append({'native_start': start, 'native_end': end, 'track_id': tracks[start], 'kernel': kernel})
        start = end
    for frame, box in zip(decoded, pixel_boxes):
        if box is None:
            native_crops.append(None)
            rects.append(None)
            continue
        mx, my, bs = box
        pad = math.ceil(bs * 1.8)
        padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), constant_values=110)
        x0, x1 = int(mx + pad - bs * 1.4), int(mx + pad + bs * 1.4)
        y0, y1 = int(my + pad - bs), int(my + pad + bs * 1.8)
        face = padded[y0:y1, x0:x1]
        if not face.size:
            raise ValueError('Empty decoded face crop')
        native_crops.append(cv2.resize(face, (CONFIG['crop_size'], CONFIG['crop_size']), interpolation=cv2.INTER_LINEAR))
        rects.append([x0 - pad, y0 - pad, x1 - pad, y1 - pad])
    pts = np.asarray([r['frame_time_s'] for r in rows], dtype=np.float64)
    if len(pts) < 2 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError('Strict native PTS required')
    targets = np.arange(math.floor(duration * CONFIG['fps'])) / CONFIG['fps']
    right = np.clip(np.searchsorted(pts, targets), 0, len(pts) - 1)
    left = np.maximum(right - 1, 0)
    mapping = np.where(np.abs(pts[left] - targets) <= np.abs(pts[right] - targets), left, right)
    frames = np.zeros((len(targets), CONFIG['crop_size'], CONFIG['crop_size'], 3), dtype=np.uint8)
    valid, model_tracks = [], []
    for i, native in enumerate(mapping):
        is_valid = native_crops[native] is not None and abs(targets[i] - pts[native]) <= CONFIG['maximum_frame_gap_seconds']
        if is_valid:
            frames[i] = native_crops[native]
        valid.append(bool(is_valid))
        model_tracks.append(tracks[native] if is_valid else None)
    return frames, valid, model_tracks, mapping.tolist(), {
        'native_pts_seconds': pts.tolist(), 'native_crop_xyxy': rects, 'smoothing_runs': kernels,
        'model_fps': CONFIG['fps'], 'model_to_native': mapping.tolist(),
        'max_native_sampling_delta_ms': float(np.max(np.abs(targets - pts[mapping])) * 1000),
        'model_valid_frames': valid, 'model_track_ids': model_tracks,
        'model_frames_bgr_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
        'invalid_frames_not_forwarded_to_model': True}


def embeddings(model, frames, valid, tracks, audio):
    import numpy as np
    import torch
    import python_speech_features
    lip_count = max(0, len(frames) - CONFIG['video_window_frames'])
    audio_count = max(0, len(audio) // 640 - CONFIG['video_window_frames'])
    visual = np.zeros((lip_count, 1024), dtype=np.float32)
    acoustic = np.zeros((audio_count, 1024), dtype=np.float32)
    eligible = [i for i in range(lip_count) if all(valid[i:i + 5]) and len(set(tracks[i:i + 5])) == 1]
    lip = torch.from_numpy(frames.transpose(3, 0, 1, 2).astype(np.float32))
    mfcc = torch.from_numpy(python_speech_features.mfcc(audio, 16000).T.astype(np.float32))
    calls = {'visual_batches': 0, 'audio_batches': 0}
    with torch.inference_mode():
        for start in range(0, len(eligible), CONFIG['batch_size']):
            indices = eligible[start:start + CONFIG['batch_size']]
            batch = torch.stack([lip[:, i:i + 5] for i in indices])
            visual[indices] = model.forward_lip(batch).numpy()
            calls['visual_batches'] += 1
        for start in range(0, audio_count, CONFIG['batch_size']):
            indices = list(range(start, min(start + CONFIG['batch_size'], audio_count)))
            batch = torch.stack([mfcc[:, i * 4:i * 4 + 20] for i in indices])[:, None]
            acoustic[indices] = model.forward_aud(batch).numpy()
            calls['audio_batches'] += 1
    if not np.isfinite(visual).all() or not np.isfinite(acoustic).all():
        raise ValueError('Nonfinite SyncNet embeddings')
    return visual, acoustic, eligible, calls


def run_backend(request, folder):
    paths = inputs(request)
    duration = request['interval_seconds'][1]
    result = {'schema': SCHEMA, 'method': CONFIG, 'source_hashes': {
        key: digest(path) for key, path in paths.items()}, 'human_reference_used': False,
        'transcript_contents_used': False, 'ready': False, 'method_accuracy_validated': False,
        'speaker_verified': False, 'records': [], 'model_calls': {'visual_batches': 0, 'audio_batches': 0}}
    if not request['asr_windows']:
        result['status'] = 'no_machine_asr_intervals'
        write_json(folder / 'av_correspondence.json', result)
        return ['av_correspondence.json']
    geometry = [json.loads(line) for line in paths['geometry_path'].read_text(encoding='utf-8').splitlines()]
    classification = [json.loads(line) for line in paths['classification_path'].read_text(encoding='utf-8').splitlines()]
    tracks, boxes = geometry_tracks(geometry, classification)
    # Vendored dependencies are selected before importing OpenCV/SciPy/NumPy.
    model, provenance = load_runtime()
    result['model'] = provenance
    with wave.open(str(paths['audio_input_path']), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError('SyncNet requires common mono PCM16 at 16kHz')
        import numpy as np
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').copy()
    if request['audio_offset_seconds'] + len(audio) / 16000 > duration + 1 / 16000:
        raise ValueError('Common audio extends outside video timeline')
    frames, valid, model_tracks, mapping, sampling = crops(paths['input_path'], geometry, tracks, boxes, duration)
    visual, acoustic, eligible, calls = embeddings(model, frames, valid, model_tracks, audio)
    result['model_calls'] = calls
    for window in request['asr_windows']:
        gate = segment_geometry(geometry, tracks, window, duration)
        left, right = window['interval_seconds']
        chosen = [i for i in eligible if i / 25 >= left - 1e-9 and (i + 5) / 25 <= right + 1e-9
                  and model_tracks[i] == gate['track_id']
                  and all(left <= geometry[mapping[k]]['frame_time_s'] < right for k in range(i, i + 5))]
        raw = scan_embeddings(visual, acoustic, chosen, request['audio_offset_seconds']) if not gate['hold_reasons'] else {
            'status': 'geometry_held', 'candidate_offset_ms': None, 'accepted_offset_ms': None, 'curve': []}
        record = correspondence(window, gate, raw)
        record['geometry_frame_indices'] = gate['geometry_frame_indices']
        record['raw_track_identity_verified'] = False
        result['records'].append(record)
    result['status'] = 'diagnosed'
    # Validate all secondary inputs again, not only the primary video IPC hash.
    inputs(request)
    write_json(folder / 'sampling_map.json', sampling)
    for name, array in [('visual_embeddings.npy', visual), ('audio_embeddings.npy', acoustic)]:
        with (folder / name).open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
    write_json(folder / 'av_correspondence.json', result)
    return ['av_correspondence.json', 'sampling_map.json', 'visual_embeddings.npy', 'audio_embeddings.npy']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    path = args.request.resolve()
    if not path.is_relative_to(WORKSPACE) or path.stat().st_size > 1024 * 1024:
        raise ValueError('Worker request path/size invalid')
    request = json.loads(path.read_text(encoding='utf-8'))
    started = time.monotonic_ns()
    if started > request['deadline_monotonic_ns']:
        raise TimeoutError('Expired before SyncNet execution')
    files = run_backend(request, path.parent)
    if time.monotonic_ns() > request['deadline_monotonic_ns']:
        raise TimeoutError('SyncNet result exceeded deadline')
    reply = {key: request[key] for key in ('protocol', 'request_id', 'backend_id', 'input_sha256')}
    reply.update(status='ok', started_ns=started, completed_ns=time.monotonic_ns(),
                 artifacts=[{'path': name, 'sha256': digest(path.parent / name)} for name in files])
    write_json(path.parent / 'reply.json', reply)


if __name__ == '__main__':
    main()
