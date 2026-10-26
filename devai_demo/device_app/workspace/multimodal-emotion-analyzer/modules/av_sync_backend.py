"""Offline machine-ASR/track AV correspondence contracts; no human references.

SyncNet scores are research diagnostics, never speaker identity or probabilities.
The inherited MM03 gates are development defaults, not independent calibration.
"""
import math

from modules.speech_mouth_diagnostic import speech_mouth_diagnostic

CONFIG = {'fps': 25, 'video_window_frames': 5, 'sample_rate': 16000, 'mfcc_frames': 20,
          'scan_max_shift_frames': 25, 'minimum_pairs': 8, 'batch_size': 8, 'cpu_threads': 4,
          'crop_size': 224, 'crop_scale': 0.4, 'median_filter_kernel': 13,
          'maximum_frame_gap_seconds': 0.15, 'mouth_minimum_samples': 5,
          'mouth_aperture_range': 0.02, 'sync_confidence_min': 3.0, 'sync_distance_max': 10.0}
SCHEMA = 'ara-av-correspondence-v1'


def machine_windows(segments, offset, duration):
    """Return intervals only; transcript contents never enter the AV estimator."""
    windows = []
    for index, segment in enumerate(segments):
        if not segment.get('text', '').strip():
            continue
        left, right = segment['start'] + offset, segment['end'] + offset
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                   for t in (left, right)) or not 0 <= left < right <= duration:
            raise ValueError('Machine ASR interval outside common video timeline')
        windows.append({'segment_index': index, 'interval_seconds': [left, right]})
    return windows


def validate_windows(windows, duration):
    seen = set()
    for window in windows:
        if set(window) != {'segment_index', 'interval_seconds'}:
            raise ValueError('Only machine segment index and interval are accepted')
        index = window['segment_index']
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
            raise ValueError('Unique nonnegative ASR segment indices required')
        seen.add(index)
        left, right = window['interval_seconds']
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                   for t in (left, right)) or not 0 <= left < right <= duration:
            raise ValueError('ASR interval outside media')


def geometry_tracks(geometry, classification):
    """Match single-face geometry to original detector track at the same PTS.

    Multiple faces, missing rows and changed tracks are held; this initial
    backend does not assign ambiguous geometry to a face or verify identity.
    """
    if len(geometry) != len(classification):
        raise ValueError('Geometry/classification frame count mismatch')
    tracks, boxes = [], []
    for index, (row, classified) in enumerate(zip(geometry, classification)):
        if row['frame_index'] != index or abs(row['frame_time_s'] - classified['source_pts_seconds']) > 1e-6:
            raise ValueError('Geometry/classification native timeline mismatch')
        native = [o for o in row.get('observations', []) if o.get('kind') == 'face_landmarks']
        faces = classified.get('faces', [])
        valid = (row.get('status') == 'ok' and row.get('face_count') == 1 and len(native) == 1
                 and classified.get('face_count') == 1 and len(faces) == 1)
        landmarks = native[0].get('landmarks', []) if valid else []
        valid = valid and len(landmarks) == 478 and all(
            len(p) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                               for v in p) for p in landmarks)
        track = faces[0].get('face_id') if valid else None
        if isinstance(track, bool) or not isinstance(track, int) or track < 0:
            valid = False
        if valid:
            x, y = [p[0] for p in landmarks[:468]], [p[1] for p in landmarks[:468]]
            box = [min(x), min(y), max(x), max(y)]
            valid = box[2] > box[0] and box[3] > box[1]
        tracks.append(track if valid else None)
        boxes.append(box if valid else None)
    return tracks, boxes


def segment_geometry(rows, tracks, window, duration, config=CONFIG):
    diagnostic = speech_mouth_diagnostic(rows, window['interval_seconds'], duration, config)
    indices = diagnostic.pop('source_geometry_frame_indices')
    track_ids = {tracks[i] for i in indices}
    reasons = list(diagnostic['geometry_blockers'])
    if None in track_ids or len(track_ids) != 1:
        reasons.append('track_changed_or_ambiguous')
    return {'diagnostic': diagnostic, 'hold_reasons': reasons,
            'track_id': next(iter(track_ids)) if len(track_ids) == 1 and None not in track_ids else None,
            'geometry_frame_indices': indices}


def scan_embeddings(video, audio, indices, audio_offset_seconds=0.0, config=CONFIG):
    """Same real pairs at every lag, with an explicit common-audio media origin."""
    import numpy as np
    video, audio = np.asarray(video), np.asarray(audio)
    if video.ndim != 2 or audio.ndim != 2 or video.shape[1] != audio.shape[1] or not np.isfinite(video).all() or not np.isfinite(audio).all():
        raise ValueError('Invalid embeddings')
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(video)):
        raise ValueError('Invalid visual chunk indices')
    if not math.isfinite(audio_offset_seconds):
        raise ValueError('Finite audio media origin required')
    fps, shift = config['fps'], config['scan_max_shift_frames']
    # Round half upward, recording the sub-frame timing quantization separately.
    base = np.floor(indices - audio_offset_seconds * fps + 0.5).astype(np.int64)
    keep = (base >= shift) & (base + shift < len(audio))
    indices, base = indices[keep], base[keep]
    if len(indices) < config['minimum_pairs']:
        return {'status': 'insufficient_context', 'candidate_offset_ms': None,
                'accepted_offset_ms': None, 'pairs': len(indices), 'curve': []}
    curve = []
    for lag in range(-shift, shift + 1):
        distance = np.sqrt(np.sum((video[indices] - audio[base - lag] + np.float32(1e-6)) ** 2, axis=1))
        curve.append({'offset_ms': lag * 1000 / fps, 'mean_distance': float(distance.mean()), 'pairs': len(indices)})
    values = np.asarray([point['mean_distance'] for point in curve])
    best = int(values.argmin())
    flat = float(values.max() - values.min()) < 1e-6
    return {'status': 'flat_curve' if flat else 'diagnostic_candidate',
            'candidate_offset_ms': None if flat else curve[best]['offset_ms'],
            'accepted_offset_ms': None, 'min_distance': float(values[best]),
            'confidence_margin': float(np.median(values) - values[best]),
            'peak_at_search_boundary': best in (0, len(curve) - 1), 'pairs': len(indices),
            'visual_window_indices': indices.tolist(), 'curve': curve,
            'max_audio_origin_quantization_ms': float(np.max(np.abs(indices / fps - (audio_offset_seconds + base / fps))) * 1000),
            'embedding_zero_padding': False, 'method_accuracy_validated': False}


def correspondence(window, geometry, scan, config=CONFIG):
    reasons = list(geometry['hold_reasons'])
    if geometry['diagnostic']['aperture_range'] is None or geometry['diagnostic']['aperture_range'] < config['mouth_aperture_range']:
        reasons.append('insufficient_visible_mouth_motion')
    if scan.get('status') != 'diagnostic_candidate':
        reasons.append('no_reliable_sync_candidate')
    if scan.get('peak_at_search_boundary'):
        reasons.append('sync_peak_at_search_boundary')
    for key, test, reason in (
            ('confidence_margin', lambda v: v >= config['sync_confidence_min'], 'weak_sync_distance_margin'),
            ('min_distance', lambda v: v <= config['sync_distance_max'], 'high_sync_distance')):
        value = scan.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not test(value):
            reasons.append(reason)
    return {'schema': SCHEMA, **window,
            'correspondence_status': 'in_frame_candidate' if not reasons else 'unknown',
            'candidate_track_id': geometry['track_id'] if not reasons else None,
            'geometry_track_id': geometry['track_id'], 'hold_reasons': sorted(set(reasons)),
            'mouth_diagnostic': geometry['diagnostic'], 'scan': scan,
            'speaker_link': {'status': 'unknown', 'track_id': None}, 'screen_status': 'unknown',
            'speaker_verified': False, 'mouth_visibility_verified': False,
            'absolute_sync_measured': False, 'thresholds_independently_calibrated': False,
            'driver_state_usable': False, 'offscreen_source_identified': False}


def invoke_file_backend(video, audio, geometry, classification, segments, offset, duration,
                        output, executable, timeout=120):
    from pathlib import Path
    from modules.observation_worker_client import SubprocessObservationClient, digest, write_json
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    windows = machine_windows(segments, offset, duration)
    if geometry is None:
        result = {'status': 'inference_failure', 'reason': 'visual_geometry_unavailable', 'artifacts': []}
        write_json(Path(output) / 'syncnet_client_audit.json', [result])
        return result
    with SubprocessObservationClient([executable, '-B', repo / 'scripts/av_sync_worker.py'],
                                    Path(output) / 'syncnet', backend_id='syncnet', allowed_root=workspace) as client:
        result = client.submit(video, [0, duration], timeout_s=timeout,
            clip_id='machine_av_observation', asr_windows=windows,
            audio_input_path=str(Path(audio).resolve()), audio_sha256=digest(audio), audio_offset_seconds=offset,
            geometry_path=str(Path(geometry).resolve()), geometry_sha256=digest(geometry),
            classification_path=str(Path(classification).resolve()), classification_sha256=digest(classification)).result()
        write_json(Path(output) / 'syncnet_client_audit.json', client.audit)
    return result


def validate_result(result, expected_windows, source_hashes):
    if result.get('schema') != SCHEMA or result.get('human_reference_used') is not False or result.get('source_hashes') != source_hashes:
        raise ValueError('AV result schema/source/reference mismatch')
    if result.get('speaker_verified') is not False or result.get('method_accuracy_validated') is not False:
        raise ValueError('Uncalibrated AV backend cannot verify a speaker')
    if len(result.get('records', [])) != len(expected_windows):
        raise ValueError('AV result must preserve every machine ASR window')
    for actual, expected in zip(result['records'], expected_windows):
        if actual.get('segment_index') != expected['segment_index'] or actual.get('interval_seconds') != expected['interval_seconds']:
            raise ValueError('AV segment index/time mismatch')
        if actual.get('screen_status') != 'unknown' or actual.get('speaker_link') != {'status': 'unknown', 'track_id': None} or actual.get('driver_state_usable') is not False:
            raise ValueError('AV diagnostic cannot manufacture attribution/state')
        if actual.get('correspondence_status') not in ('unknown', 'in_frame_candidate'):
            raise ValueError('Invalid correspondence candidate')
    return True
