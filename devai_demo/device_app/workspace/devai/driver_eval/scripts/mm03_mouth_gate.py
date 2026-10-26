"""Development association gate: geometry and learned sync, never human labels.

No positive hand overlap is not proof of visibility; outputs remain candidates.
"""
import math
import numpy as np

CONFIG = {'mouth_motion_range_min':0.02,'hand_overlap_fraction_min':0.05,
          'mouth_region_height_to_width':0.35,'sync_confidence_min':3.0,
          'sync_distance_max':10.0,'minimum_native_frames':8,
          'automatic_offscreen_identification':False,'thresholds_independently_calibrated':False}


def mouth_risk(face,hands,width=1280,height=720):
    import cv2
    result={'native_time_s':face['frame_time_s'],'status':'unknown','hand_count':len(hands),
            'overlap_fraction':None,'aperture':None,'region_xyxy':None}
    observations=[o for o in face.get('observations',[]) if o.get('kind')=='face_landmarks']
    if face.get('status')!='ok' or face.get('face_count')!=1 or len(observations)!=1:return result
    p=np.asarray(observations[0]['landmarks'],np.float32)
    if p.shape!=(478,3) or not np.isfinite(p).all():return result
    xy=p[:,:2]*[width,height];left,right=xy[61],xy[291]
    w=float(np.linalg.norm(left-right))
    if w<2:return result
    center=(left+right)/2;half=np.array([w/2,w*CONFIG['mouth_region_height_to_width']/2])
    lo,hi=center-half,center+half
    rect=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],np.float32)
    covered=0.0
    for hand in hands:
        h=np.asarray(hand,np.float32)
        if h.shape!=(21,3) or not np.isfinite(h).all():return result
        hull=cv2.convexHull((h[:,:2]*[width,height]).astype(np.float32))
        covered+=max(0.0,float(cv2.intersectConvexConvex(rect,hull)[0]))
    fraction=min(1.0,covered/(w*w*CONFIG['mouth_region_height_to_width']))
    result.update(status='hand_overlap_risk' if fraction>=CONFIG['hand_overlap_fraction_min'] else 'no_positive_overlap',
                  overlap_fraction=fraction,aperture=float(np.linalg.norm(xy[13]-xy[14])/w),
                  region_xyxy=[*lo.tolist(),*hi.tolist()])
    return result


def associate(risks,interval,sync):
    a,b=interval;selected=[r for r in risks if a<=r['native_time_s']<b]
    reasons=[]
    if len(selected)<CONFIG['minimum_native_frames']:reasons.append('insufficient_native_frames')
    if any(r['status']=='unknown' for r in selected):reasons.append('mouth_geometry_unknown')
    if any(r['status']=='hand_overlap_risk' for r in selected):reasons.append('projected_hand_mouth_overlap')
    values=[r['aperture'] for r in selected if r['aperture'] is not None]
    variation=max(values)-min(values) if values else None
    if variation is None or variation<CONFIG['mouth_motion_range_min']:reasons.append('insufficient_visible_mouth_motion')
    if sync.get('candidate_offset_ms') is None:reasons.append('no_sync_candidate')
    if sync.get('peak_at_search_boundary'):reasons.append('sync_peak_at_search_boundary')
    score=sync.get('confidence_margin');distance=sync.get('min_distance')
    if score is None or not math.isfinite(score) or score<CONFIG['sync_confidence_min']:reasons.append('weak_sync_distance_margin')
    if distance is None or not math.isfinite(distance) or distance>CONFIG['sync_distance_max']:reasons.append('high_sync_distance')
    return {'screen_status':'unknown' if reasons else 'in_frame',
            'person_id':None if reasons else 'visible_person_1','hold_reasons':reasons,
            'mouth_motion_range':variation,'native_frames':len(selected),
            'hand_risk_frames':sum(r['status']=='hand_overlap_risk' for r in selected),
            'candidate_offset_ms':None if reasons else sync['candidate_offset_ms'],
            'visibility_verified':False,'ready':False}
