"""Offline bilateral eye-motion candidates; thresholds are experimental."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class EyeEventConfig:
    close_score: float = 0.5
    reopen_score: float = 0.3
    sustained_duration_s: float = 0.5
    minimum_blink_s: float = 0.05
    maximum_frame_gap_s: float = 0.15

    def validate(self):
        values=asdict(self)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values.values()):
            raise ValueError('Finite numeric configuration required')
        if not 0<=self.reopen_score<self.close_score<=1:
            raise ValueError('Require 0 <= reopen_score < close_score <= 1')
        if not 0<self.minimum_blink_s<self.sustained_duration_s or self.maximum_frame_gap_s<=0:
            raise ValueError('Invalid duration settings')


def scores(row):
    if row.get('status')!='ok' or row.get('face_count')!=1:
        return None
    observations=row.get('observations',[])
    if len(observations)!=1:
        return None
    values={b['label']:b.get('score') for b in observations[0].get('blendshapes',[]) if b.get('label') in ('eyeBlinkLeft','eyeBlinkRight')}
    pair=[values.get(k) for k in ('eyeBlinkLeft','eyeBlinkRight')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in pair):
        return None
    return pair


def detect_eye_events(rows, *, duration_s, config=EyeEventConfig()):
    """Uses native PTS; never joins events across missing or widely spaced frames."""
    config.validate()
    if not rows or not isinstance(duration_s,(int,float)) or isinstance(duration_s,bool) or not math.isfinite(duration_s) or duration_s<=0:
        raise ValueError('Nonempty frames and positive finite media duration required')
    stamps=[r.get('frame_time_s') for r in rows]
    if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<duration_s for t in stamps):
        raise ValueError('Native frame PTS must lie inside media')
    if any(b<=a for a,b in zip(stamps,stamps[1:])):
        raise ValueError('Strictly increasing native PTS required')
    events=[]; missing=[]; active=None; open_seen=False

    def finish(end, reason=None):
        nonlocal active
        elapsed=end-active['start_s']
        if elapsed<=0:
            active=None
            return
        censored=reason is not None or not active['preceded_by_observed_open']
        label='indeterminate_eye_closure' if censored else (
            'sustained_eye_closure_candidate' if elapsed>=config.sustained_duration_s else
            'blink_candidate' if elapsed>=config.minimum_blink_s else 'short_eye_motion_candidate')
        uncertainty=[]
        if not active['preceded_by_observed_open']:
            uncertainty.append('start_censored')
        if reason:
            uncertainty.append(reason)
        events.append({**active,'end_s':end,'duration_s':elapsed,'label':label,'censored':censored,
                       'uncertainty':uncertainty,'finalized_at_s':end,'uses_future_reopening_frame':reason is None,
                       'calibrated':False,'state_candidates':[]})
        active=None

    for index,(row,stamp) in enumerate(zip(rows,stamps)):
        if index and stamp-stamps[index-1]>config.maximum_frame_gap_s:
            # End at last observed timestamp; do not count the unknown gap as closure.
            if active:
                finish(stamps[index-1],'frame_gap')
            missing.append({'start_s':stamps[index-1],'end_s':stamp,'reason':'frame_gap'})
            open_seen=False
        pair=scores(row)
        if pair is None:
            if active:
                finish(stamp,'observation_unavailable')
            end=stamps[index+1] if index+1<len(stamps) else duration_s
            missing.append({'start_s':stamp,'end_s':end,'reason':'eye_scores_unavailable'})
            open_seen=False
            continue
        bilateral_open=max(pair)<=config.reopen_score
        bilateral_closed=min(pair)>=config.close_score
        if active:
            if bilateral_open:
                finish(stamp)
                open_seen=True
            else:
                active['observed_frame_count']+=1
        elif bilateral_closed:
            active={'start_s':stamp,'start_frame_index':index,'preceded_by_observed_open':open_seen,
                    'observed_frame_count':1,'side':'both'}
        elif bilateral_open:
            open_seen=True
    if active:
        finish(duration_s,'end_censored')
    return {'schema':'driver-eye-event-candidates-v0.1','mode':'offline','config':asdict(config),
            'calibrated':False,'duration_s':duration_s,'frame_count':len(rows),'events':events,
            'unavailable_intervals':missing,'accuracy_measured':False,'state_candidates':[],'ready':False}
