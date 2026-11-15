from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import unittest,tempfile,os
import numpy as np
from wrist_pose import Wrist
from wrist_ik import raw_limits
from hover_step import bounded_step,fresh,claim_capture


class HoverStepTests(unittest.TestCase):
    def test_motion_observation_can_only_be_claimed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);claim_capture(folder,folder/'log')
            with self.assertRaisesRegex(ValueError,'already used'):claim_capture(folder,folder/'other-log')

    def test_step_bounds_and_progress(self):
        w=Wrist(ROOT/'.vendor/roarm_ws',[1]*5,[0]*5);limits=raw_limits(w,ROOT/'.vendor/roarm_ws')
        q=np.mean(limits,axis=1);_,start=w.fk(q);plan=bounded_step(w,q,start+[20,0,20],limits)
        self.assertLessEqual(plan['model_step_mm'],5)
        self.assertLessEqual(plan['max_joint_change_deg'],1)
        self.assertLess(np.linalg.norm(np.array(plan['next_reference_mm'])-start-[20,0,20]),np.linalg.norm([20,0,20]))

    def test_stale_capture_rejected_before_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);f=folder/'result.json';f.write_text('{}');os.utime(f,(1,1))
            with self.assertRaisesRegex(ValueError,'older than'):fresh(folder)

if __name__=='__main__':unittest.main()
