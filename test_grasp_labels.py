from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import unittest
import numpy as np
from camera import Frame
from grasp_labels import make_grasp,depth_check


class GraspLabelsTests(unittest.TestCase):
    def test_opposing_contacts_and_angle_convention(self):
        g=make_grasp([[30,50],[70,50],[50,60]],(100,100))
        np.testing.assert_allclose(g['center_xy'],[50,50])
        np.testing.assert_allclose(g['corners_xy'],[[30,40],[70,40],[70,60],[30,60]])
        self.assertAlmostEqual(g['angle_rad_ggcnn'],0)
        r=make_grasp([[50,30],[50,70],[60,50]],(100,100))
        self.assertAlmostEqual(abs(r['angle_rad_ggcnn']),np.pi/2)
        reversed_g=make_grasp([[70,50],[30,50],[50,60]],(100,100))
        self.assertAlmostEqual(reversed_g['angle_rad_ggcnn'],g['angle_rad_ggcnn'])

    def test_bad_geometry_rejected(self):
        with self.assertRaises(ValueError):make_grasp([[30,50],[30,50],[40,60]],(100,100))
        with self.assertRaises(ValueError):make_grasp([[1,1],[90,1],[50,30]],(100,100))

    def test_holes_never_become_measured_depth(self):
        depth=np.ones((100,100),np.float32)*.5
        frame=Frame(depth,np.ones((100,100,3),np.float32))
        g=make_grasp([[30,50],[70,50],[50,60]],depth.shape)
        self.assertTrue(depth_check(frame,g)['usable_for_current_predictor'])
        depth[45:56,45:56]=0
        self.assertFalse(depth_check(frame,g)['usable_for_current_predictor'])
        self.assertEqual(float(depth[50,50]),0)

if __name__=='__main__':unittest.main()
