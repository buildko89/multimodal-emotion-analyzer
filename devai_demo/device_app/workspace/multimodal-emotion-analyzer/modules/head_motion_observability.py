"""Uncalibrated geometry candidates and coverage; never create nod truth."""


def head_excursion_return_candidates(angles, thresholds):
    """Accept a return band or its crossing between adjacent valid samples.

    No missing samples are interpolated. The same continuity, duration,
    excursion and return-band thresholds apply. A crossing requires an already
    observed excursion and motion back from its side of the starting angle.
    """
    candidates, traces = [], []
    for i, (start, initial, frame) in enumerate(angles):
        if candidates and start < candidates[-1]['end_s']:
            continue
        peak, last_t = 0.0, start
        previous_delta, previous_frame, previous_t = 0.0, frame, start
        excursion_side = None
        outcome, stopped_at = 'window_end', None
        for t, angle, index in angles[i+1:]:
            if t-start > thresholds['head_event_maximum_seconds'] or t-last_t > thresholds['maximum_frame_gap_seconds']:
                outcome = ('valid_sample_gap' if t-last_t > thresholds['maximum_frame_gap_seconds']
                           else 'event_duration_limit')
                stopped_at = t
                break
            last_t = t
            delta = angle-initial
            peak = max(peak, abs(delta))
            if excursion_side is None and abs(delta) >= thresholds['head_pitch_excursion_degrees']:
                excursion_side = 1 if delta > 0 else -1
            within_band = abs(delta) <= thresholds['head_return_degrees']
            crossed_initial = (excursion_side is not None
                               and excursion_side*previous_delta > 0
                               and excursion_side*delta <= 0)
            if (peak >= thresholds['head_pitch_excursion_degrees']
                    and (within_band or crossed_initial)
                    and t-start >= thresholds['head_event_minimum_seconds']):
                candidates.append(dict(start_s=start, end_s=t, excursion_degrees=round(peak, 3),
                    start_frame=frame, end_frame=index,
                    return_kind='within_band' if within_band else 'crossed_initial_between_valid_samples',
                    return_bracket_frames=[previous_frame, index],
                    return_bracket_seconds=[previous_t, t]))
                outcome = 'candidate'
                break
            previous_delta, previous_frame, previous_t = delta, index, t
        traces.append(dict(start_frame=frame, start_s=start, initial_pitch_degrees=initial,
                           peak_degrees=peak, last_contiguous_s=last_t,
                           stopped_at_s=stopped_at, outcome=outcome))
    return candidates, traces


def head_observation_diagnostic(rows, angles, traces, thresholds, expected_samples):
    valid_frames = {item[2] for item in angles}
    missing_runs = []
    active = []
    for row in rows + [None]:
        if row is not None and row['frame_index'] not in valid_frames:
            active.append(row)
        else:
            if active:
                missing_runs.append(dict(first_frame=active[0]['frame_index'],
                    last_frame=active[-1]['frame_index'], samples=len(active),
                    start_s=active[0]['frame_time_s'], last_observed_s=active[-1]['frame_time_s'],
                    reasons=sorted({reason for item in active for reason in item.get('uncertainty', [])
                                    if reason not in ('state_not_evaluated', 'driver_identity_unverified')})))
                active = []
    gaps = [dict(previous_frame=left[2], next_frame=right[2],
                 previous_s=left[0], next_s=right[0], gap_seconds=right[0]-left[0])
            for left, right in zip(angles, angles[1:])
            if right[0]-left[0] > thresholds['maximum_frame_gap_seconds']]
    missing = max(0, expected_samples-len(angles))
    no_rows = not angles
    interrupted = any(t['peak_degrees'] >= thresholds['head_pitch_excursion_degrees']
                      and t['outcome'] in ('valid_sample_gap', 'event_duration_limit', 'window_end')
                      for t in traces)
    complete = not no_rows and not missing and not gaps and len(rows) == expected_samples
    return dict(schema='head-motion-observability-v1', expected_samples=expected_samples,
                valid_samples=len(angles), missing_samples=missing,
                coverage_complete=complete, missing_runs=missing_runs, valid_sample_gaps=gaps,
                excursion_without_confirmed_return=interrupted,
                scan_traces=traces, thresholds=dict(thresholds),
                interpretation='Observation coverage and existing scan outcomes, not nod truth',
                missing_samples_interpolated=False)
