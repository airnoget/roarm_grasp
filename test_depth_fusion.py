from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import numpy as np
from camera import Frame,median_depth_three
from rgbd_projection import supports_native


class FusionTests(unittest.TestCase):
    def frames(self):
        f=Frame(np.ones((2,3),np.float32),np.ones((2,3,3),np.float32),
                np.zeros((2,3,3),np.uint8),np.zeros((4,6,3),np.uint8),
                np.eye(3),np.zeros(8),np.eye(4),'OBDistortionModel.OB_DISTORTION_BROWN_CONRADY_K6')
        return [replace(f,depth_m=f.depth_m.copy(),raw_bgr=f.raw_bgr.copy()) for _ in range(3)]

    def test_metadata_and_middle_rgb_preserved(self):
        frames=self.frames();f=median_depth_three(frames)
        for name in ('bgr','raw_bgr','raw_K','raw_dist','T_raw_from_depth_mm','raw_dist_model'):
            self.assertIs(getattr(f,name),getattr(frames[1],name))
        self.assertEqual(supports_native(f),supports_native(frames[1]))
        self.assertIsNot(f.depth_m,frames[1].depth_m)

    def test_no_depth_invented_from_one_return(self):
        frames=self.frames()
        frames[0].depth_m[:]=[[1,0,np.nan],[1,1,0]]
        frames[1].depth_m[:]=[[3,2,0],[0,2,0]]
        frames[2].depth_m[:]=[[2,4,3],[0,3,0]]
        np.testing.assert_allclose(median_depth_three(frames).depth_m,[[2,3,0],[0,2,0]])

    def test_changed_camera_profile_rejected(self):
        for name in ('raw_K','raw_dist','T_raw_from_depth_mm','raw_dist_model'):
            frames=self.frames();value=getattr(frames[0],name)
            setattr(frames[0],name,value+'changed' if isinstance(value,str) else value+1)
            with self.assertRaisesRegex(ValueError,'calibration'):median_depth_three(frames)


if __name__=='__main__':unittest.main()
