import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'.deps'))
import unittest
from types import SimpleNamespace
import numpy as np
from camera import Frame, pinhole_rays
from rgbd_projection import depth_to_rgb, boxes_to_depth, rgb_click_to_depth
from target_semantics import TargetDetector


def frame(depth=1.):
    k = np.array([[100., 0, 0], [0, 100, 0], [0, 0, 1]])
    t = np.eye(4); t[0, 3] = 100.
    return Frame(np.full((10, 20), depth), pinhole_rays((10, 20), 100, 100, 0, 0),
                 np.zeros((10, 20, 3), np.uint8), np.full((10, 40, 3), 120, np.uint8),
                 k, np.zeros(8), t, 'OBCameraDistortionModel.NONE')


class Array:
    def __init__(self, value): self.value = np.array(value)
    def cpu(self): return self
    def numpy(self): return self.value


class ProjectionTests(unittest.TestCase):
    def test_baseline_mm_and_depth_dependent_projection(self):
        for z, shift in [(1., 10.), (2., 5.)]:
            f = frame(z); uv = depth_to_rgb(f)
            np.testing.assert_allclose(uv[4, 8], [8+shift, 4], atol=1e-5)
            self.assertEqual(rgb_click_to_depth(uv, [8+shift, 4]), (8, 4))

    def test_missing_depth_is_not_filled(self):
        f = frame(); f.depth_m[:] = 0
        uv = depth_to_rgb(f)
        self.assertTrue(np.isnan(uv).all())
        self.assertEqual(boxes_to_depth([('wood_block', (0, 0, 40, 10))], uv), [])
        with self.assertRaises(ValueError): rgb_click_to_depth(uv, [20, 5])

    def test_rgb_occlusion_keeps_nearest_surface(self):
        f = frame(); f.depth_m[:] = 0
        f.depth_m[4, 0] = 1.; f.depth_m[4, 5] = 2.
        uv = depth_to_rgb(f)
        np.testing.assert_allclose(uv[4, 0], [10., 4.], atol=1e-5)
        self.assertTrue(np.isnan(uv[4, 5]).all())

    def test_native_detector_and_depth_box_coordinates(self):
        f = frame()
        result = SimpleNamespace(names={0: 'wood_block'}, boxes=SimpleNamespace(
            cls=Array([0]), xyxy=Array([[15., 2., 20., 7.]])))
        def predict(image, **kwargs):
            self.assertIs(image, f.raw_bgr)
            return [result]
        detector = TargetDetector.__new__(TargetDetector)
        detector.allowed={'wood_block'}; detector.device='cpu'
        detector.model=SimpleNamespace(predict=predict)
        boxes=detector(f)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][0], 'wood_block')
        x0,y0,x1,y1=boxes[0][1]
        self.assertLessEqual(abs(x0-5), 1);self.assertLessEqual(abs(x1-10), 1)
        self.assertEqual((y0,y1),(2,7))
        self.assertEqual(detector.image_source, 'native_rgb')

    def test_preview_keeps_rgb_when_all_depth_is_missing(self):
        from preview_live import render
        f = frame(); f.depth_m = np.zeros((128,160)); f.rays=pinhole_rays((128,160),100,100,0,0)
        f.bgr=np.zeros((128,160,3),np.uint8);f.raw_bgr=np.full((256,320,3),120,np.uint8)
        result=SimpleNamespace(names={0:'wood_block'},boxes=SimpleNamespace(
            cls=Array([0]),xyxy=Array([[10,10,300,240]])))
        detector=TargetDetector.__new__(TargetDetector);detector.allowed={'wood_block'};detector.device='cpu'
        detector.model=SimpleNamespace(predict=lambda *a,**k:[result])
        class NoPrediction:
            def predict(self,*a,**k): raise AssertionError('Missing depth must not create a grasp')
        canvas,report=render(f,detector,NoPrediction(),.25)
        self.assertGreater(float(canvas[80:100,60:100].mean()),100)
        self.assertEqual(report['detector_input'],'native_rgb')
        self.assertEqual(report['proposals'],[])
        self.assertIn('no usable depth',report['status'])


if __name__ == '__main__': unittest.main()
