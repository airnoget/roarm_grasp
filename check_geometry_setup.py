"""Offline preparation check. Never opens a camera or serial port."""
from pathlib import Path
import sys,hashlib,json
ROOT=Path(__file__).resolve().parent
for name in reversed(('.deps','.orbbec-python-deps','.torchdeps','.semdeps')):sys.path.insert(0,str(ROOT/name))


def main():
    import cv2
    import numpy
    import serial
    import pyorbbecsdk
    from preset_preview import load_configuration
    from trained_model import latest_detector
    from target_semantics import TargetDetector
    policy=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))
    preset=ROOT/policy['frozen_preset']
    if hashlib.sha256(preset.read_bytes()).hexdigest()!=policy['sha256']:raise ValueError('Confirmed preset changed')
    load_configuration(preset)
    weights=latest_detector(ROOT);detector=TargetDetector(str(weights),['artificial_object'],'cpu')
    print('PASS: camera SDK, OpenCV, NumPy and serial imports',flush=True)
    print('PASS: confirmed Gemini preset hash and trained YOLO loading',flush=True)
    from unittest import TestLoader,TextTestRunner
    names=['test_rgbd_projection','test_box_geometry','test_geometry_tracking','test_wrist_ik','test_approach_pose','test_plan_hover']
    result=TextTestRunner(verbosity=1).run(TestLoader().loadTestsFromNames(names))
    if not result.wasSuccessful():raise SystemExit(1)
    print('OFFLINE CHECK PASSED. Hardware motion and accuracy have NOT been verified.',flush=True)
    print('On site: python debug.py locate ; S save, Q exit ; python debug.py hover',flush=True)

if __name__=='__main__':main()
