import unittest
import numpy as np
from approach_pose import preapproach


class ApproachTests(unittest.TestCase):
    def test_tcp_offset_is_removed_after_camera_transform(self):
        camera=np.eye(4);camera[:3,3]=[100,20,10]
        tcp=np.eye(4);tcp[:3,3]=[0,0,60]
        rotation=np.diag([1,-1,-1])
        result=preapproach(dict(camera_surface_xyz_mm=[20,30,40]),camera,tcp,rotation)
        tool=np.array(result['tcp_preapproach_base']);wrist=np.array(result['link5_preapproach_base'])
        np.testing.assert_allclose(tool[:3,3],[120,50,130])
        np.testing.assert_allclose(wrist[:3,3],[120,50,190])
        np.testing.assert_allclose(wrist@tcp,tool)
        self.assertFalse(result['motion_enabled'])

    def test_rotation_and_clearance_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):preapproach(dict(camera_surface_xyz_mm=[0,0,1]),np.eye(4),np.eye(4),np.zeros((3,3)))
        with self.assertRaises(ValueError):preapproach(dict(camera_surface_xyz_mm=[0,0,1]),np.eye(4),np.eye(4),np.eye(3),0)

if __name__=='__main__':unittest.main()
