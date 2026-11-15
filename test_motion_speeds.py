import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import numpy as np
from motion_speeds import phase_speed
from live_approach import LiveApproach,FIELDS,TCP_OFFSET


class MotionSpeedTests(unittest.TestCase):
    def test_precision_zone_and_phase_speeds(self):
        self.assertEqual(phase_speed('hover',200),120)
        self.assertEqual(phase_speed('hover',20),40)
        self.assertEqual(phase_speed('descent',60,100),80)
        self.assertEqual(phase_speed('descent',60,39),40)
        self.assertEqual(phase_speed('descent',20,100),40)
        self.assertEqual(phase_speed('descent',60,None),40)
        self.assertEqual(phase_speed('retract',200),120)
        self.assertEqual(phase_speed('home',200),320)

    def test_live_deceleration_preserves_target_and_does_not_overwrite_correction(self):
        class Wrist:
            def fk(self,q):return np.eye(3),np.array([q[0]*1000,0,0])
        class Arm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        for corrected in (False,True):
            arm=Arm();controller=LiveApproach(Wrist(),arm)
            controller.active=True;controller.motion_speed=120
            controller.plan=dict(phase='hover',hover_base_mm=(TCP_OFFSET+[1000,0,0]).tolist(),goal_joints_rad=[1,0,0,0,0])
            controller.report=dict(samples=[],commands=[],tracking_corrections=[{}] if corrected else [])
            state=dict(zip(FIELDS,[.98,0,0,0,0]));state['v']=1235
            with patch.object(controller,'flush'):controller.observe(state)
            self.assertTrue(controller.active)
            if corrected:self.assertEqual(arm.commands,[])
            else:
                self.assertEqual(len(arm.commands),1)
                self.assertEqual(arm.commands[0]['rad'],1)
                self.assertEqual(arm.commands[0]['spd'],40)


if __name__=='__main__':unittest.main()
