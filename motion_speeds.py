"""Phase-specific servo speeds loaded at application startup."""
import json
from pathlib import Path

SPEEDS=json.loads((Path(__file__).resolve().parent/'camera_profiles/motion-speeds.json').read_text(encoding='utf-8'))
if any(not isinstance(value,(int,float)) or isinstance(value,bool) or value<=0 for key,value in SPEEDS.items() if key!='unit'):
    raise ValueError('Motion speeds and distance thresholds must be positive numbers')


def phase_speed(phase,distance_mm,grasp_distance_mm=None):
    if distance_mm<=SPEEDS['near_target_mm']:
        return SPEEDS['precision']
    if phase=='descent':
        if grasp_distance_mm is None or grasp_distance_mm<=SPEEDS['near_grasp_mm']:
            return SPEEDS['precision']
        return SPEEDS['descent_fast']
    return SPEEDS['retract_fast'] if phase=='retract' else SPEEDS['base_fast'] if phase=='home' else SPEEDS['approach_fast']
