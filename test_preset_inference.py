from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import unittest
import numpy as np
from preset_inference import native_frame,infer
from perception import Candidate


class Detector:
    allowed={'artificial_object'}
    def __call__(self,frame):
        self.image_source='native_rgb'
        self.last_native_detections=[('artificial_object',np.array([20,20,60,60]))]
        v,u=np.indices(frame.depth_m.shape)
        self.last_depth_to_rgb=np.stack((u,v),axis=-1).astype(float)
        return self.last_native_detections

class Predictor:
    def predict(self,*args):return iter([Candidate(40,40,0,20,.9,model_width_px=20)])

class Wrist:
    def camera_pose(self,joints,hand_eye):return np.eye(3),np.array([100,200,300])

class Tests(unittest.TestCase):
    def frame(self):
        v,u=np.indices((80,80));rays=np.stack(((u-40)/100,(v-40)/100,np.ones((80,80))),axis=-1)
        geometry=(rays,np.array([[100,0,40],[0,100,40],[0,0,1]]),np.zeros(8),np.eye(4),'OBDistortionModel.OB_DISTORTION_BROWN_CONRADY_K6')
        return native_frame(np.ones((80,80),np.float32)*.2,np.full((80,80,3),100,np.uint8),geometry)
    def test_native_rgb_mapping_and_camera_xyz(self):
        f=self.frame()
        np.testing.assert_allclose(f.bgr[10:70,10:70],100)
        result=infer(f,Detector(),Predictor())
        np.testing.assert_allclose(result['proposals'][0]['camera_xyz_mm'],[0,0,200],atol=1e-4)
        self.assertNotIn('candidate_base_xyz_mm',result['proposals'][0])
        self.assertFalse(result['motion_enabled'])
    def test_base_xyz_requires_synchronized_joints(self):
        result=infer(self.frame(),Detector(),Predictor(),Wrist(),np.zeros(5),(np.eye(3),np.zeros(3)))
        np.testing.assert_allclose(result['proposals'][0]['candidate_base_xyz_mm'],[100,200,500],atol=1e-4)
    def test_missing_center_depth_never_creates_xyz(self):
        f=self.frame();f.depth_m[40,40]=0
        result=infer(f,Detector(),Predictor())
        self.assertNotIn('camera_xyz_mm',result['proposals'][0])
        self.assertIn('localization_error',result['proposals'][0])

if __name__=='__main__':unittest.main()
