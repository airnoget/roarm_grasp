from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import unittest
import numpy as np
from wrist_pose import Wrist
from wrist_ik import solve,raw_limits,solve_position


class WristIKTests(unittest.TestCase):
    def setUp(self):
        self.w=Wrist(ROOT/'.vendor/roarm_ws',[1]*5,[0]*5)
        self.limits=raw_limits(self.w,ROOT/'.vendor/roarm_ws')

    def test_reachable_pose_checked_by_independent_fk(self):
        q=np.mean(self.limits,axis=1)
        r,p=self.w.fk(q);target=np.eye(4);target[:3,:3]=r;target[:3,3]=p
        seed=np.clip(q+[.1,-.1,.1,-.05,.1],self.limits[:,0],self.limits[:,1])
        result=solve(self.w,target,seed,self.limits)
        actual_r,actual_p=self.w.fk(result['joints_rad'])
        self.assertLess(np.linalg.norm(actual_p-p),1)
        self.assertTrue(np.allclose(actual_r,r,atol=.035))
        self.assertFalse(result['motion_enabled'])

    def test_unreachable_pose_is_rejected(self):
        target=np.eye(4);target[:3,3]=[10000,0,10000]
        with self.assertRaises(ValueError):solve(self.w,target,np.mean(self.limits,axis=1),self.limits,iterations=20)

    def test_joint_signs_map_limits(self):
        w=Wrist(ROOT/'.vendor/roarm_ws',[-1,1,1,1,1],[.1,0,0,0,0])
        b=raw_limits(w,ROOT/'.vendor/roarm_ws')
        np.testing.assert_allclose(b[0],np.sort((self.limits[0]-.1)/-1))

    def test_position_hover_accounts_for_explicit_tcp(self):
        q=np.mean(self.limits,axis=1);offset=np.array([10,20,40])
        r,p=self.w.fk(q);target=r@offset+p+[5,0,5]
        result=solve_position(self.w,target,q,self.limits,offset)
        r,p=self.w.fk(result['joints_rad'])
        self.assertLess(np.linalg.norm(r@offset+p-target),1)
        self.assertFalse(result['orientation_constrained'])

if __name__=='__main__':unittest.main()
