"""Load an explicitly selected taught route; never auto-import later W samples."""
import json
from pathlib import Path
import numpy as np

LIMITS=np.array([[-179.9,179.9],[-90,90],[0,2047*360/4096],[-90,90],[180-4095*360/4096,180]])


def load_route(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not data.get('enabled') or not data.get('points'):raise ValueError('Unload route is disabled or empty')
    for point in data['points']:
        q=np.asarray(point['joints_deg'],float)
        if q.shape!=(5,) or not np.isfinite(q).all() or np.any(q<LIMITS[:,0]) or np.any(q>LIMITS[:,1]):
            raise ValueError('Invalid taught joint target: '+point.get('name','unknown'))
    if not 0<=float(data['release_deg'])<=4095*360/4096:raise ValueError('Invalid release angle')
    return data
