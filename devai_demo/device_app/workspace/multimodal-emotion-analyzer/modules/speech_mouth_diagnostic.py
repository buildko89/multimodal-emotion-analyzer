"""Machine-only ASR/mouth consistency diagnostics, never speaker attribution.

Low aperture variation does not prove closed lips, silence, or an offscreen
human. Even continuous single-face geometry does not verify mouth visibility.
Use inherited development thresholds; no labels, identities or scene metadata.
"""
import math


def speech_mouth_diagnostic(rows, interval, duration, thresholds):
    left, right = interval
    maximum_gap = thresholds['maximum_frame_gap_seconds']
    selected = [r for r in rows if left <= r['frame_time_s'] < right]
    usable, exclusions = [], {}
    for row in selected:
        reason = None
        mouth = row.get('hand_mouth', {})
        aperture = mouth.get('aperture')
        if row.get('status') != 'ok':
            reason = 'geometry_failure'
        elif row.get('face_count') != 1:
            reason = 'single_face_not_observed'
        elif mouth.get('status') == 'hand_overlap_risk':
            reason = 'projected_hand_mouth_overlap'
        elif mouth.get('status') != 'no_positive_overlap':
            reason = 'mouth_geometry_unavailable'
        elif isinstance(aperture, bool) or not isinstance(aperture, (int, float)) or not math.isfinite(aperture) or aperture < 0:
            reason = 'invalid_mouth_aperture'
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
        else:
            usable.append(row)
    values = [r['hand_mouth']['aperture'] for r in usable]
    variation = max(values) - min(values) if values else None
    blockers = []
    if not 0 <= left < right <= duration:
        blockers.append('asr_interval_outside_media')
    if not selected:
        blockers.append('no_visual_samples_in_asr_interval')
    if len(usable) < thresholds['mouth_minimum_samples']:
        blockers.append('too_few_usable_mouth_samples')
    if exclusions:
        blockers.append('excluded_visual_samples')
    gap = bool(selected) and (
        selected[0]['frame_time_s'] - left > maximum_gap or
        right - selected[-1]['frame_time_s'] > maximum_gap or
        any(b['frame_index'] != a['frame_index'] + 1 or
            not 0 < b['frame_time_s'] - a['frame_time_s'] <= maximum_gap
            for a, b in zip(selected, selected[1:])))
    if gap:
        blockers.append('visual_time_or_frame_gap')
    if blockers:
        status = 'insufficient_geometry'
    elif variation < thresholds['mouth_aperture_range']:
        status = 'audio_with_low_mouth_motion_candidate'
    else:
        status = 'audio_with_mouth_motion_candidate'
    return {
        'schema': 'ara-speech-mouth-diagnostic-v1', 'status': status,
        'window_frames': len(selected), 'usable_mouth_samples': len(usable),
        'excluded_samples_by_reason': exclusions, 'geometry_blockers': blockers,
        'aperture_min': min(values) if values else None,
        'aperture_max': max(values) if values else None, 'aperture_range': variation,
        'comparison_threshold': thresholds['mouth_aperture_range'],
        'thresholds_calibrated': False, 'geometry_coverage_complete': not blockers,
        'mouth_closed_verified': False, 'mouth_visibility_verified': False,
        'speaker_verified': False, 'absolute_sync_measured': False,
        'speech_source': 'unknown', 'offscreen_voice_decision': 'unknown',
        'attribution_hold_reasons': ['speaker_association_not_implemented', 'speech_source_unresolved',
                                     'mouth_visibility_unverified', 'absolute_av_sync_unmeasured'],
        'unresolved_alternatives': ['offscreen_person', 'audio_playback',
                                    'visible_person_with_subtle_articulation',
                                    'audio_video_misalignment', 'asr_false_positive'],
        'source_geometry_frame_indices': [r['frame_index'] for r in selected],
    }
