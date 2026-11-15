"""User screenshot reference poses; never substitute them for measured joints."""
from pathlib import Path
import json
import numpy as np


def compare(raw_joints,path=None):
    path=Path(path) if path else Path(__file__).resolve().parent/'camera_profiles/observation-poses.json'
    spec=json.loads(path.read_text(encoding='utf-8'));name=spec['selected']
    reference=np.asarray(spec['poses'][name],float)
    actual=np.asarray(raw_joints,float)
    if actual.shape!=(5,) or not np.isfinite(actual).all():raise ValueError('Need five measured joint radians')
    if reference.shape!=(5,) or not np.isfinite(reference).all():raise ValueError('Invalid reference pose')
    actual_deg=np.degrees(actual)
    delta=(actual_deg-reference+180)%360-180
    return dict(reference_name=name,reference_joint_deg=reference.tolist(),measured_joint_deg=actual_deg.tolist(),
                difference_deg=delta.tolist(),max_difference_deg=float(np.max(np.abs(delta))),
                scope='Reference comparison only; actual joint measurements are used for geometry',motion_enabled=False)
