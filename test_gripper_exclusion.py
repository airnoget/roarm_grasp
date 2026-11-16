import sys,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import numpy as np
from gripper_exclusion import GripperExclusion,project_exclusion
from target_semantics import TargetDetector


class Array:
    def __init__(self,array):self.array=np.asarray(array)
    def cpu(self):return self
    def numpy(self):return self.array


class GripperExclusionTests(unittest.TestCase):
    def setUp(self):self.filter=GripperExclusion()

    def test_corner_jaws_rejected_but_green_scene_target_and_edge_target_kept(self):
        # Color is deliberately not an input: a green cube in the scene survives.
        native=[('jaw_left',[132,57,338,262]),('jaw_right',[1045,17,1280,251]),
                ('green_cube',[500,360,650,520]),('edge_cube',[300,260,410,370])]
        kept,ignored,mask=self.filter.filter(native,(720,1280,3))
        self.assertEqual([name for name,_ in kept],['green_cube','edge_cube'])
        self.assertEqual([d['name'] for d in ignored],['jaw_left','jaw_right'])
        self.assertFalse(mask[440,575])

    def test_unmatched_resolution_bypasses_profile(self):
        boxes=[('object',[0,0,100,100])]
        kept,ignored,mask=self.filter.filter(boxes,(480,640,3))
        self.assertEqual(kept,boxes);self.assertEqual(ignored,[]);self.assertIsNone(mask)

    def test_projects_native_mask_not_rotated_display_coordinates(self):
        mask=self.filter.mask((720,1280,3))
        uv=np.array([[[230,120],[600,450],[1150,100],[np.nan,np.nan]]],np.float32)
        np.testing.assert_array_equal(project_exclusion(mask,uv),[[True,False,True,False]])

    def test_detector_filters_before_native_target_count_and_depth_projection(self):
        detector=TargetDetector.__new__(TargetDetector)
        detector.gripper_exclusion=self.filter;detector.device='cpu'
        result=SimpleNamespace(names={0:'artificial_object'},boxes=SimpleNamespace(
            cls=Array([0,0,0]),xyxy=Array([[132,57,338,262],[1045,17,1280,251],[500,360,650,520]])))
        detector.model=SimpleNamespace(predict=lambda *args,**kwargs:[result])
        frame=SimpleNamespace(depth_m=np.ones((10,10)),bgr=np.zeros((10,10,3),np.uint8),raw_bgr=np.zeros((720,1280,3),np.uint8))
        with patch('rgbd_projection.supports_native',return_value=True),patch('rgbd_projection.depth_to_rgb',return_value=np.tile(np.array([575,440],np.float32),(10,10,1))):
            detections=detector(frame)
        self.assertEqual(len(detector.last_native_detections),1)
        self.assertEqual(len(detector.last_ignored_gripper),2)
        self.assertEqual(len(detections),1)
        self.assertEqual(int(frame.raw_bgr.sum()),0)


if __name__=='__main__':unittest.main()
