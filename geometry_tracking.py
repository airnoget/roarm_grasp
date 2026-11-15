"""Camera-relative consistency only; never authorizes a robot movement."""
from collections import deque
import numpy as np


class GeometryTracker:
    def __init__(self,frames=5):
        self.frames=frames;self.tracks={};self.next_id=1;self.last_time=None

    def update(self,proposals,timestamp,invalid=False):
        if (invalid or self.last_time is None or timestamp<=self.last_time
                or timestamp-self.last_time>1.5):self.tracks={}
        self.last_time=timestamp
        if invalid:return
        points=[np.asarray(p['camera_xyz_mm'],float) for p in proposals]
        # Associate only one-to-one, unambiguous close neighbours. Unmatched
        # tracks expire immediately; detector order is not treated as identity.
        candidates={i:[k for k,t in self.tracks.items() if np.linalg.norm(x-t[-1][0])<12]
                    for i,x in enumerate(points)}
        usage={k:sum(k in c for c in candidates.values()) for k in self.tracks}
        active={}
        for i,(p,x) in enumerate(zip(proposals,points)):
            c=candidates[i]
            if len(c)==1 and usage[c[0]]==1:
                identity=c[0];history=self.tracks[identity]
            else:
                identity=self.next_id;self.next_id+=1;history=deque(maxlen=self.frames)
            g=p['geometry'];axis=np.asarray(g['camera_closing_axis'],float)
            width=float(g['observed_opening_mm'])
            history.append((x,axis,width));active[identity]=history
            xyz=np.array([t[0] for t in history]);axes=np.array([t[1] for t in history]);widths=np.array([t[2] for t in history])
            spread=float(np.max(np.linalg.norm(xyz-xyz.mean(0),axis=1)))
            angle=float(np.degrees(np.max(np.arccos(np.clip(np.abs(axes@axes[-1]),0,1)))))
            width_span=float(np.ptp(widths))
            stable=len(history)==self.frames and spread<=3 and angle<=5 and width_span<=4
            p['tracking']=dict(id=identity,frames=len(history),required_frames=self.frames,
                              camera_relative_stable=stable,center_spread_mm=spread,axis_spread_deg=angle,
                              opening_span_mm=width_span,motion_authorized=False,
                              scope='Consistency in camera coordinates; unknown camera motion and calibration error remain')
        self.tracks=active
