"""Compact machine observations and experimental event candidates; no labels."""
from collections import Counter
import math
from pathlib import Path

from modules.observation_integration import QUESTIONS, read_json, read_lines, validate_envelope
from modules.observation_worker_client import digest
from modules.speech_mouth_diagnostic import speech_mouth_diagnostic
from modules.av_visible_speech import CONTRACT, LEGACY_CONTRACT, apply_contract, groups, machine_record
from modules.head_motion_observability import head_observation_diagnostic, head_excursion_return_candidates

STATE_KEYS = ('drowsiness_signs', 'irritation_candidate', 'tension_candidate')


def compact_observations(envelope_path, thresholds, modalities=('visual', 'audio'), *, recognition_contract=LEGACY_CONTRACT):
    if recognition_contract not in (LEGACY_CONTRACT, CONTRACT):
        raise ValueError('Unknown recognition contract')
    path = Path(envelope_path)
    envelope = read_json(path)
    validate_envelope(envelope, path.parent)
    registry = {e['evidence_id']: e for e in envelope['evidence_registry']}
    observations = envelope['observations']
    duration = envelope['input']['interval_seconds'][1]
    features, mapping = [], {}
    candidates = {q: [] for q in QUESTIONS}

    def add(ident, kind, value, ids, interval=None):
        features.append({'evidence_id': ident, 'kind': kind, 'interval_seconds': interval or [0, duration], 'value': value})
        mapping[ident] = {'raw_evidence_ids': sorted(set(ids)), 'source_envelope_sha256': digest(path)}
        if any(i not in registry for i in ids):
            raise ValueError('Unregistered raw evidence')
        return ident

    if 'visual' in modalities:
        face_obs = [o for o in observations if o['kind'] == 'face_classification']
        add('E_face', 'face_summary', {'samples': len(face_obs), 'single_face_samples': sum(len(o['value']) == 1 for o in face_obs),
            'raw_class_counts': dict(Counter(f['raw_emotion'] for o in face_obs for f in o['value'])),
            'scores_calibrated': False}, [e for o in face_obs for e in o['evidence_ids']])
        raw_ids = sorted(k for k in registry if k.startswith('geometry.'))
        rows = []
        if raw_ids:
            raw_path = path.parent / registry[raw_ids[0]]['relative_path']
            rows = read_lines(raw_path)
        add('E_visibility', 'visibility_summary', {'frame_count': len(rows), 'expected_frames': len(face_obs),
            'mouth_visibility_verified': False, 'hand_absence_verified': False,
            'single_face_frames': sum(r['face_count'] == 1 for r in rows),
            'missing_geometry_frames': sum(r['status'] != 'ok' for r in rows),
            'eye_unavailable_frames': sum(o['status'] == 'unavailable' for o in observations if o['kind'] == 'eye_geometry')}, raw_ids)
        risk_runs = []
        active = []
        for index, row in enumerate(rows + [None]):
            gap = row is not None and active and (
                row['frame_index'] != active[-1]['frame_index'] + 1 or
                row['frame_time_s'] - active[-1]['frame_time_s'] > thresholds['maximum_frame_gap_seconds'])
            if gap:
                if len(active) >= thresholds['hand_mouth_risk_minimum_consecutive_frames']:
                    risk_runs.append({'start_s': active[0]['frame_time_s'], 'end_s': active[-1]['frame_time_s'], 'frames': len(active)})
                active = []
            if row is not None and row['hand_mouth']['status'] == 'hand_overlap_risk':
                active.append(row)
            else:
                if len(active) >= thresholds['hand_mouth_risk_minimum_consecutive_frames']:
                    risk_runs.append({'start_s': active[0]['frame_time_s'], 'end_s': row['frame_time_s'] if row else duration,
                                      'frames': len(active)})
                active = []
        ident = add('E_hand_mouth', 'projected_hand_mouth', {'hand_detected_frames': sum(bool(r['hands']) for r in rows),
            'risk_runs': risk_runs, 'risk_is_visibility_proof': False, 'calibrated': False}, raw_ids)
        if risk_runs:
            candidates['hand_over_mouth'].append(ident)
        angles = []
        for row in rows:
            matrix = row.get('head_transform')
            if row['status'] == 'ok' and matrix:
                if len(matrix) == 16:
                    matrix = [matrix[i:i+4] for i in range(0, 16, 4)]
                pitch = math.degrees(math.atan2(matrix[2][1], matrix[2][2]))
                angles.append((row['frame_time_s'], pitch, row['frame_index']))
        # Geometry candidates, not calibrated nod truth; keep missing samples as gaps.
        head_candidates, head_traces = head_excursion_return_candidates(angles, thresholds)
        ident = add('E_head', 'head_pitch_geometry', {'valid_samples': len(angles),
            'pitch_range_degrees': round(max(a[1] for a in angles) - min(a[1] for a in angles), 3) if angles else None,
            'excursion_return_candidates': head_candidates[:8], 'calibrated': False,
            'observation_diagnostic': head_observation_diagnostic(rows, angles, head_traces, thresholds, len(face_obs))}, raw_ids)
        if head_candidates:
            candidates['nod'].append(ident)
        eye = [o for o in observations if o['kind'] == 'eye_geometry']
        add('E_eye', 'eye_summary', {'observed_frames': sum(o['status'] == 'observed' for o in eye),
            'unavailable_frames': sum(o['status'] != 'observed' for o in eye), 'drowsiness_state_validated': False}, raw_ids)
    else:
        rows = []
    if 'audio' in modalities or 'asr' in modalities:
        for index, observation in enumerate(o for o in observations if o['kind'] == 'asr'):
            segment = observation['value']
            if observation['status'] != 'observed' or not segment or not segment.get('text', '').strip():
                continue
            left = segment['start'] + envelope['input']['common_audio_interval_seconds'][0]
            right = segment['end'] + envelope['input']['common_audio_interval_seconds'][0]
            aid = add(f'E_asr{index}', 'asr', {'text': segment['text'], 'speaker_verified': False,
                'time_precision': segment.get('time_precision', 'backend_segment_times')}, observation['evidence_ids'], [left, right])
            available = [r for r in rows if left <= r['frame_time_s'] < right and r['status'] == 'ok'
                         and r['face_count'] == 1 and r['hand_mouth']['status'] == 'no_positive_overlap'
                         and r['hand_mouth'].get('aperture') is not None]
            apertures = [r['hand_mouth']['aperture'] for r in available]
            variation = max(apertures) - min(apertures) if apertures else None
            mid = add(f'E_mouth{index}', 'mouth_asr_coincidence', {'samples': len(available), 'aperture_range': variation,
                'speaker_verified': False, 'absolute_sync_measured': False},
                [f"geometry.{r['frame_index']}" for r in available] + observation['evidence_ids'], [left, right])
            if len(available) >= thresholds['mouth_minimum_samples'] and variation >= thresholds['mouth_aperture_range']:
                candidates['visible_person_speaks'] += [aid, mid]
            diagnostic = speech_mouth_diagnostic(rows, [left, right], duration, thresholds)
            # Include excluded rows as evidence: they explain why a gap/occlusion
            # must not be reinterpreted as a silent visible person.
            ids = [f'geometry.{i}' for i in diagnostic.pop('source_geometry_frame_indices')]
            add(f'E_speech_mouth{index}', 'speech_mouth_diagnostic', diagnostic,
                ids + observation['evidence_ids'], [left, right])
            # No offscreen candidate is created: this diagnostic does not
            # distinguish a person outside the image from replayed speech.
        for observation in (o for o in observations if o['kind'] == 'acoustic_emotion' and 'audio' in modalities):
            value = observation['value']
            native = value[0] if value else {}
            add('E_ser_' + observation['backend_id'], 'acoustic_emotion', {
                'status': observation['status'], 'raw_emotion_tags': native.get('emotion_labels'),
                'top_label': native.get('top_label'), 'scores_calibrated': False,
                'speaker_verified': False, 'driver_state_usable': False}, observation['evidence_ids'])
        if 'visual' in modalities:
            for index, observation in enumerate(o for o in observations if o['kind'] == 'av_correspondence'):
                record = observation['value']
                if observation['status'] != 'observed' or record is None:
                    add(f'E_av_sync_failure{index}', 'av_correspondence',
                        {'status': observation['status'], 'failure_reason': observation.get('failure_reason'),
                         'speaker_verified': False}, observation['evidence_ids'])
                    continue
                scan = record['scan']
                add(f'E_av_sync{record["segment_index"]}', 'av_correspondence', {
                    'correspondence_status': record['correspondence_status'],
                    'candidate_track_id': record['candidate_track_id'], 'geometry_track_id': record['geometry_track_id'],
                    'hold_reasons': record['hold_reasons'],
                    'scan_status': scan['status'], 'candidate_offset_ms': scan.get('candidate_offset_ms'),
                    'min_distance': scan.get('min_distance'), 'confidence_margin': scan.get('confidence_margin'),
                    'pairs': scan.get('pairs', 0), 'peak_at_search_boundary': scan.get('peak_at_search_boundary'),
                    'speaker_verified': False, 'absolute_sync_measured': False,
                    'mouth_visibility_verified': False, 'thresholds_independently_calibrated': False,
                    'driver_state_usable': False, 'offscreen_source_identified': False},
                    observation['evidence_ids'], record['interval_seconds'])
                if recognition_contract == CONTRACT:
                    features[-1]['value']['record'] = machine_record(record)
                # Raw AV correspondence remains separate from event candidate
                # decisions until speaker/visibility/sync calibration is evaluated.
    compact = {'schema': 'ara-compact-observations-v1', 'duration_seconds': duration, 'modalities': list(modalities),
               'features': features, 'experimental_candidates': candidates,
               'limitations': ['speaker_unverified', 'absolute_av_sync_unmeasured', 'uncalibrated_geometry',
                               'negative_events_unverified', 'driver_state_not_evaluated'],
               'self_report': None, 'human_reference_used': False}
    if recognition_contract == CONTRACT:
        compact = apply_contract(compact)
        for feature in compact['features']:
            if feature['kind'] == 'visible_speech_gate':
                suffix = feature['evidence_id'][len('E_visible_speech'):]
                ids = [i for prefix in ('E_asr', 'E_mouth', 'E_speech_mouth', 'E_av_sync')
                       for i in mapping.get(prefix + suffix, {}).get('raw_evidence_ids', [])]
                mapping[feature['evidence_id']] = {'raw_evidence_ids': sorted(set(ids)),
                    'source_envelope_sha256': digest(path)}
    return compact, {'source_envelope': str(path.resolve()), 'source_sha256': digest(path),
                     'feature_evidence': mapping, 'thresholds': thresholds}


def rule_recognition(compact):
    events = {}
    for name in QUESTIONS:
        ids = sorted(set(compact['experimental_candidates'][name]))
        if name == 'visible_person_speaks' and compact.get('recognition_contract') == CONTRACT:
            supported = groups(compact)
            ids = supported[0] if supported else []
        uncertainty = ['uncalibrated_geometry', 'speaker_unverified'] if ids else ['insufficient_observation']
        if name == 'offscreen_voice' and not ids and any(
                f.get('kind') == 'speech_mouth_diagnostic' for f in compact['features']):
            uncertainty = ['speaker_unverified', 'absolute_av_sync_unmeasured', 'visibility_unverified']
        if name == 'visible_person_speaks' and compact.get('recognition_contract') == CONTRACT:
            uncertainty = ['uncalibrated_geometry', 'speaker_unverified', 'absolute_av_sync_unmeasured', 'visibility_unverified']
        events[name] = {'value': 'yes' if ids else 'unknown', 'evidence_ids': ids,
                        'uncertainty': uncertainty}
    return {'schema': 'ara-recognition-v1', 'events': events,
            'states': {k: {'value': 'unknown', 'evidence_ids': [], 'uncertainty': ['state_not_evaluated']} for k in STATE_KEYS}}
