"""Offline local AV support within machine ASR; never whole-window speech.

Keep every native missing/ambiguous sample. Only maximal runs with a single
observed track and usable mouth geometry can be scanned. The original full
window contract remains unchanged; local support is diagnostic evidence only.
"""
from collections import Counter
import math

from modules.av_sync_backend import CONFIG, geometry_tracks, validate_windows
from modules.av_visible_speech import assess, same_interval

SCHEMA = 'ara-local-av-diagnostic-v1'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_timeline(rows, classification, duration):
    if not finite(duration) or duration <= 0 or not rows:
        raise ValueError('Positive duration and native geometry required')
    last = -1.0
    for index, row in enumerate(rows):
        stamp = row.get('frame_time_s')
        if type(row.get('frame_index')) is not int or row['frame_index'] != index:
            raise ValueError('Original native frame indices required')
        if not finite(stamp) or not last < stamp < duration:
            raise ValueError('Strict finite native PTS within media required')
        last = stamp
    for row in classification:
        if not finite(row.get('source_pts_seconds')):
            raise ValueError('Finite original classification PTS required')
    return geometry_tracks(rows, classification)


def exclusion(row, track):
    if track is None:
        return 'track_or_face_unavailable'
    mouth = row.get('hand_mouth') or {}
    if mouth.get('status') == 'hand_overlap_risk':
        return 'projected_hand_mouth_overlap'
    if mouth.get('status') != 'no_positive_overlap':
        return 'mouth_geometry_unavailable'
    aperture = mouth.get('aperture')
    if not finite(aperture) or aperture < 0:
        return 'invalid_mouth_aperture'
    return None


def partition_windows(rows, classification, windows, duration, config=CONFIG):
    """Partition by native validity/track/gap, with no duration-based selection.

The right edge is the next observed native PTS, capped at the parent/media end.
Any excessive edge gap is still held by the inherited mouth geometry gate.
All runs (including short ones) and all excluded samples remain in the output.
"""
    validate_windows(windows, duration)
    tracks, _ = validate_timeline(rows, classification, duration)
    parents = []
    for window in windows:
        left, right = window['interval_seconds']
        selected = [i for i, row in enumerate(rows) if left <= row['frame_time_s'] < right]
        excluded, runs, active = [], [], []

        def finish():
            if not active:
                return
            first, last = active[0], active[-1]
            edge = rows[last + 1]['frame_time_s'] if last + 1 < len(rows) else duration
            runs.append({'parent_segment_index': window['segment_index'], 'local_index': len(runs),
                         'parent_interval_seconds': list(window['interval_seconds']),
                         'interval_seconds': [max(left, rows[first]['frame_time_s']), min(right, edge)],
                         'track_id': tracks[first], 'native_frame_indices': list(active)})
            active.clear()

        for index in selected:
            reason = exclusion(rows[index], tracks[index])
            if reason:
                finish()
                excluded.append({'frame_index': index, 'frame_time_s': rows[index]['frame_time_s'], 'reason': reason})
                continue
            if active and (index != active[-1] + 1 or tracks[index] != tracks[active[-1]]
                           or rows[index]['frame_time_s'] - rows[active[-1]]['frame_time_s'] > config['maximum_frame_gap_seconds']):
                finish()
            active.append(index)
        finish()
        parents.append({'segment_index': window['segment_index'], 'interval_seconds': list(window['interval_seconds']),
                        'selected_native_frames': len(selected), 'local_runs': runs,
                        'excluded_samples': excluded,
                        'exclusion_counts': dict(Counter(r['reason'] for r in excluded)),
                        'parent_speech_decision': 'unknown', 'speaker_verified': False,
                        'driver_state_usable': False, 'interpolated_samples': 0})
    return parents


def chunk_indices(run, eligible, mapping, model_tracks, rows, config=CONFIG):
    """Every sampled frame of a five-frame chunk must stay in one native run."""
    left, right = run['interval_seconds']
    indices = set(run['native_frame_indices'])
    selected = []
    for index in eligible:
        if type(index) is not int or not 0 <= index <= len(mapping) - config['video_window_frames']:
            raise ValueError('Invalid cached eligible chunk index')
        end = index + config['video_window_frames']
        if index/config['fps'] < left - 1e-9 or end/config['fps'] > right + 1e-9:
            continue
        if all(mapping[k] in indices and model_tracks[k] == run['track_id']
               and left <= rows[mapping[k]]['frame_time_s'] < right for k in range(index, end)):
            selected.append(index)
    return selected


def local_support(run, diagnostic, record):
    """Reuse the 80ms gate but expose only local support, never an ASR answer."""
    parent, interval = run['parent_interval_seconds'], run['interval_seconds']
    if (type(run['parent_segment_index']) is not int or run['parent_segment_index'] < 0
            or type(run['local_index']) is not int or run['local_index'] < 0
            or not same_interval(parent, parent) or not same_interval(interval, interval)
            or interval[0] < parent[0] or interval[1] > parent[1]
            or type(run['track_id']) is not int or run['track_id'] < 0):
        raise ValueError('Local interval must stay inside its machine ASR parent')
    gate = assess(run['parent_segment_index'], interval, diagnostic, record)
    reasons = list(gate['hold_reasons'])
    if isinstance(record, dict) and record.get('geometry_track_id') != run['track_id']:
        reasons.append('local_track_mismatch')
    return {'schema': SCHEMA, 'parent_segment_index': run['parent_segment_index'],
            'local_index': run['local_index'], 'parent_interval_seconds': list(parent),
            'interval_seconds': list(interval),
            'status': 'local_av_support_candidate' if not reasons else 'unknown',
            'candidate_track_id': run['track_id'] if not reasons else None,
            'hold_reasons': sorted(set(reasons)),
            'reliable_correspondence_lag_ms': gate['reliable_correspondence_lag_ms'],
            'maximum_media_lag_ms': gate['maximum_media_lag_ms'], 'accepted_offset_ms': None,
            'whole_parent_supported': False, 'visible_person_speaks': 'unknown',
            'current_frame_speech': 'unknown', 'screen_status': 'unknown', 'speech_source': 'unknown',
            'speaker_verified': False, 'mouth_visibility_verified': False,
            'absolute_sync_measured': False, 'driver_state_usable': False,
            'independent_accuracy': None}
