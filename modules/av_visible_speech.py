"""Versioned, machine-only visible-speech candidates; never speaker identity.

The 80 ms band is two 25 fps scan steps, a development exclusion rule, not
measured absolute synchronization accuracy. No lag is accepted or corrected.
"""
from copy import deepcopy
import math

from modules.av_sync_backend import CONFIG

CONTRACT = 'av-visible-v1'
LEGACY_CONTRACT = 'legacy-v3'
MAX_MEDIA_LAG_MS = 2 * 1000 / CONFIG['fps']


def machine_record(record):
    """Retain gate inputs; the full distance curve stays in raw evidence."""
    names = ('schema', 'segment_index', 'interval_seconds', 'correspondence_status',
             'candidate_track_id', 'geometry_track_id', 'hold_reasons',
             'speaker_verified', 'driver_state_usable', 'screen_status')
    result = {name: deepcopy(record.get(name)) for name in names}
    scan = record.get('scan') or {}
    result['scan'] = {name: deepcopy(scan.get(name)) for name in (
        'status', 'peak_at_search_boundary', 'pairs', 'confidence_margin',
        'min_distance', 'candidate_offset_ms', 'accepted_offset_ms')}
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same_interval(first, second):
    return (isinstance(first, list) and isinstance(second, list) and len(first) == len(second) == 2
            and all(finite(v) for v in first + second) and 0 <= first[0] < first[1]
            and all(abs(a - b) <= 1e-6 for a, b in zip(first, second)))


def assess(index, interval, diagnostic, record):
    """Accept neutral ASR index/time, geometry and AV evidence only."""
    reasons = []
    if type(index) is not int or index < 0 or not same_interval(interval, interval):
        reasons.append('invalid_machine_asr_window')
    if (not isinstance(diagnostic, dict) or diagnostic.get('status') != 'audio_with_mouth_motion_candidate'
            or diagnostic.get('geometry_coverage_complete') is not True
            or diagnostic.get('geometry_blockers') != []
            or diagnostic.get('excluded_samples_by_reason') != {}
            or not finite(diagnostic.get('usable_mouth_samples'))
            or diagnostic['usable_mouth_samples'] < CONFIG['mouth_minimum_samples']
            or not finite(diagnostic.get('aperture_range'))
            or diagnostic['aperture_range'] < CONFIG['mouth_aperture_range']):
        reasons.append('continuous_visible_mouth_motion_not_supported')
    lag, track = None, None
    if not isinstance(record, dict):
        reasons.append('av_observation_missing_or_failed')
    else:
        if (record.get('schema') != 'ara-av-correspondence-v1'
                or type(record.get('segment_index')) is not int or record['segment_index'] != index
                or not same_interval(interval, record.get('interval_seconds'))):
            reasons.append('asr_av_interval_or_index_mismatch')
        holds = record.get('hold_reasons')
        if not isinstance(holds, list) or any(not isinstance(v, str) for v in holds):
            reasons.append('invalid_av_hold_reasons')
        elif holds:
            reasons += holds
        if record.get('correspondence_status') != 'in_frame_candidate':
            reasons.append('av_correspondence_held')
        track = record.get('candidate_track_id')
        if type(track) is not int or track < 0 or type(record.get('geometry_track_id')) is not int or track != record['geometry_track_id']:
            reasons.append('track_changed_or_ambiguous')
        scan = record.get('scan')
        if not isinstance(scan, dict):
            scan = {}
        raw_lag = scan.get('candidate_offset_ms')
        reliable = (scan.get('status') == 'diagnostic_candidate'
            and scan.get('peak_at_search_boundary') is False
            and type(scan.get('pairs')) is int and scan['pairs'] >= CONFIG['minimum_pairs']
            and finite(scan.get('confidence_margin')) and scan['confidence_margin'] >= CONFIG['sync_confidence_min']
            and finite(scan.get('min_distance')) and 0 <= scan['min_distance'] <= CONFIG['sync_distance_max']
            and finite(raw_lag) and abs(raw_lag) < CONFIG['scan_max_shift_frames'] * 1000 / CONFIG['fps'])
        if not reliable:
            reasons.append('no_reliable_nonboundary_av_scan')
        # Do not expose held raw lag as a usable timing estimate.
        if reliable and record.get('correspondence_status') == 'in_frame_candidate' and holds == []:
            lag = raw_lag
            if abs(lag) > MAX_MEDIA_LAG_MS:
                reasons.append('av_candidate_outside_media_time_band')
        if (record.get('speaker_verified') is not False or record.get('driver_state_usable') is not False
                or record.get('screen_status') != 'unknown' or scan.get('accepted_offset_ms') is not None):
            reasons.append('unsupported_attribution_or_time_correction')
    return {'schema': 'ara-visible-speech-gate-v1', 'segment_index': index,
        'status': 'near_media_time_candidate' if not reasons else 'unknown',
        'candidate_track_id': track if not reasons else None,
        'hold_reasons': sorted(set(reasons)), 'reliable_correspondence_lag_ms': lag,
        'maximum_media_lag_ms': MAX_MEDIA_LAG_MS, 'accepted_offset_ms': None,
        'speaker_verified': False, 'absolute_sync_measured': False,
        'mouth_visibility_verified': False, 'thresholds_independently_calibrated': False,
        'speech_source': 'unknown', 'screen_status': 'unknown', 'driver_state_usable': False,
        'unresolved_alternatives': ['audio_playback', 'asr_false_positive', 'uncalibrated_av_timing']}


def groups(compact):
    """Recompute support from observations, never trust a supplied gate label."""
    features = {f['evidence_id']: f for f in compact['features']}
    if len(features) != len(compact['features']):
        raise ValueError('duplicate_feature_id')
    result = []
    for ident, asr in features.items():
        if asr.get('kind') != 'asr' or not ident.startswith('E_asr') or not ident[5:].isdigit():
            continue
        suffix = ident[5:]
        mouth, diagnostic, av = [features.get(prefix + suffix) for prefix in ('E_mouth', 'E_speech_mouth', 'E_av_sync')]
        if (not mouth or not diagnostic or not av or mouth.get('kind') != 'mouth_asr_coincidence'
                or diagnostic.get('kind') != 'speech_mouth_diagnostic' or av.get('kind') != 'av_correspondence'
                or not all(same_interval(asr.get('interval_seconds'), f.get('interval_seconds')) for f in (mouth, diagnostic, av))):
            continue
        gate = assess(int(suffix), asr['interval_seconds'], diagnostic.get('value'), av.get('value', {}).get('record'))
        gate_feature = features.get('E_visible_speech' + suffix, {})
        if (gate['status'] == 'near_media_time_candidate' and gate_feature.get('kind') == 'visible_speech_gate'
                and same_interval(asr.get('interval_seconds'), gate_feature.get('interval_seconds'))
                and gate_feature.get('value') == gate):
            result.append([ident, 'E_mouth' + suffix, 'E_av_sync' + suffix, 'E_visible_speech' + suffix])
    return result


def apply_contract(compact):
    """New copy; preserve all legacy features/candidates for side-by-side replay."""
    value = deepcopy(compact)
    value['recognition_contract'] = CONTRACT
    value['experimental_candidates']['visible_person_speaks'] = []
    features = {f['evidence_id']: f for f in value['features']}
    if len(features) != len(value['features']):
        raise ValueError('duplicate_feature_id')
    for asr in list(value['features']):
        ident = asr['evidence_id']
        if asr.get('kind') != 'asr' or not ident.startswith('E_asr') or not ident[5:].isdigit():
            continue
        suffix = ident[5:]
        diagnostic = features.get('E_speech_mouth' + suffix, {})
        av = features.get('E_av_sync' + suffix, {})
        record = av.get('value', {}).get('record')
        if not same_interval(asr.get('interval_seconds'), diagnostic.get('interval_seconds')):
            diagnostic = {}
        if not same_interval(asr.get('interval_seconds'), av.get('interval_seconds')):
            record = None
        gate = assess(int(suffix), asr['interval_seconds'], diagnostic.get('value'), record)
        value['features'].append({'evidence_id': 'E_visible_speech' + suffix, 'kind': 'visible_speech_gate',
            'interval_seconds': asr['interval_seconds'], 'value': gate})
    value['experimental_candidates']['visible_person_speaks'] = [i for group in groups(value) for i in group]
    return value
