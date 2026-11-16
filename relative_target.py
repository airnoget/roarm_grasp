"""Publish selected target in arm-base coordinates; metres, x forward/y left."""
import json
import math
import time
from pathlib import Path
from target_selection import selected_proposal


def make_relative_target(result, stamp_unix_s=None):
    proposal=selected_proposal(result)
    identity=result.get('selected_target_id')
    if identity is None:raise ValueError('Persistent selected_target_id required')
    xyz=proposal.get('candidate_base_xyz_mm')
    if result.get('pose_error') or xyz is None or len(xyz)!=3 or not all(math.isfinite(v) for v in xyz):
        raise ValueError('Current stationary base XYZ required')
    stamp=time.time() if stamp_unix_s is None else stamp_unix_s
    return dict(schema=1,target_id=identity,stamp_unix_s=stamp,frame_id='arm_base',units='m',
                relative_xy=[float(xyz[0])/1000,float(xyz[1])/1000],
                valid=True,measurement='top_surface_center',world_xy=None)


def publish_relative_target(path,result,stamp_unix_s=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:payload=make_relative_target(result,stamp_unix_s)
    except ValueError as exc:
        payload=dict(schema=1,target_id=result.get('selected_target_id'),stamp_unix_s=time.time(),
                     frame_id='arm_base',units='m',relative_xy=None,valid=False,reason=str(exc))
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)
    return payload
