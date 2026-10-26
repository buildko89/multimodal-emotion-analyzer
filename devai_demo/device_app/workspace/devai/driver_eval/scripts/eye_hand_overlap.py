"""Experimental 2D hand-overlap risk filter; absence of overlap is not visibility proof.

Inference uses face/hand landmarks only. No label, asset identity or blink score
is used to find an overlap. Convex hulls approximate hands and include gaps.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import cv2
import numpy as np

@dataclass(frozen=True)
class OverlapConfig:
    eye_height_to_width: float = 0.5
    minimum_overlap_fraction: float = 0.05

    def validate(self):
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Finite numeric settings required')
        if not 0 < self.minimum_overlap_fraction <= 1 or not 0 < self.eye_height_to_width <= 2:
            raise ValueError('Invalid overlap settings')

def classify(face_row,hands,width,height,config=OverlapConfig()):
    config.validate()
    result={'frame_time_s':face_row['frame_time_s'],'hand_count':len(hands),
            'status':'unknown','reason':'face_geometry_unavailable','eye_rects':[],
            'overlap_fractions':None,'hold_eye_scores':True,
            'visibility_confirmed':False,'calibrated':False}
    obs=face_row.get('observations',[])
    if face_row.get('status')!='ok' or face_row.get('face_count')!=1 or len(obs)!=1: return result
    points=np.asarray(obs[0].get('landmarks',[]),dtype=np.float32)
    if points.shape!=(478,3) or not np.isfinite(points).all(): return result
    polygons=[]
    for hand in hands:
        p=np.asarray(hand,dtype=np.float32)
        if p.shape!=(21,3) or not np.isfinite(p).all():
            result['reason']='invalid_hand_geometry'; return result
        polygons.append(cv2.convexHull((p[:,:2]*[width,height]).astype(np.float32)).reshape(-1,2))
    fractions=[]; rects=[]
    for outer,inner in [(33,133),(263,362)]:
        a=points[outer,:2]*[width,height]; b=points[inner,:2]*[width,height]
        eye_width=float(np.linalg.norm(a-b)); center=(a+b)/2
        if eye_width<2: return result
        half=np.asarray([eye_width/2,eye_width*config.eye_height_to_width/2])
        x1,y1=center-half; x2,y2=center+half
        rect=np.asarray([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
        area=eye_width*eye_width*config.eye_height_to_width
        # Sum overlaps conservatively; double-covered pixels can raise the risk.
        coverage=min(1.0,sum(max(0.0,float(cv2.intersectConvexConvex(rect,h)[0])) for h in polygons)/area)
        fractions.append(coverage); rects.append([float(x1),float(y1),float(x2),float(y2)])
    overlapping=any(f>=config.minimum_overlap_fraction for f in fractions)
    result.update(status='hand_overlap_candidate' if overlapping else 'no_hand_overlap_detected',
                  reason='projected_hand_hull_intersects_eye_region' if overlapping else 'no_positive_hand_overlap_evidence',
                  eye_rects=rects,overlap_fractions=fractions,hold_eye_scores=overlapping)
    return result

def gate(rows,risks):
    if len(rows)!=len(risks): raise ValueError('One risk per native face observation required')
    output=[]
    for row,risk in zip(rows,risks):
        if row['frame_time_s']!=risk['frame_time_s']: raise ValueError('Native frame PTS mismatch')
        if risk.get('status') not in {'unknown','hand_overlap_candidate','no_hand_overlap_detected'} or not isinstance(risk.get('hold_eye_scores'),bool):
            raise ValueError('Invalid risk record')
        if risk['hold_eye_scores'] != (risk['status']!='no_hand_overlap_detected'):
            raise ValueError('Unsafe overlap decision')
        value=deepcopy(row)
        if risk['hold_eye_scores']:
            value['status']='eye_visibility_unavailable'
            value['observations']=[]
            value['eye_geometry']={'left_ratio':None,'right_ratio':None,'eye_closed':None,'calibrated':False}
            value['uncertainty']=list(value.get('uncertainty',[]))+[risk['reason']]
        output.append(value)
    return output
