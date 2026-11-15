"""One hand-eye candidate shared by preview and offline coordinate checks."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent

def active_report():
    pointer=json.loads((ROOT/'calibration/active-handeye.json').read_text(encoding='utf-8'))
    path=(ROOT/pointer['report']).resolve()
    path.relative_to((ROOT/'calibration').resolve())
    report=json.loads(path.read_text(encoding='utf-8'))
    if 'wrist_from_depth_mm' not in report or 'joint_mapping' not in report:
        raise ValueError('Active report has no hand-eye solution')
    return path

def fingerprint():return hashlib.sha256(active_report().read_bytes()).hexdigest()
