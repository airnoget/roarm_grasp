"""No camera/serial opened: collection counts, metadata, and plan banner tests."""
from data_capture import ROOT, save_capture
from capture_plan import CapturePlan, header
from camera import Frame, load_frame
from rgbd_projection import letterbox
import unittest
import json
import time
import numpy as np
import cv2
from contextlib import contextmanager
from unittest.mock import patch
import data_capture
import sys


class CaptureTests(unittest.TestCase):
    def test_three_type_plan_totals_100(self):
        plan = CapturePlan(3)
        self.assertEqual(plan.export()['total_goal'], 100)
        for key, goal in [('1',20),('2',20),('3',20),('m',30),('b',10)]:
            self.assertTrue(plan.select(ord(key)))
            for _ in range(goal): plan.saved()
        self.assertEqual(plan.export()['total_saved'], 100)
        self.assertIn('完成',plan.hint())
        self.assertFalse(plan.select(ord('s')))

    def test_selection_does_not_change_counts(self):
        plan = CapturePlan()
        plan.select(ord('2'));plan.saved();plan.select(ord('M'))
        self.assertEqual(plan.counts['single_2'],1)
        self.assertEqual(plan.export()['total_saved'],1)
        self.assertEqual(plan.active[1],'mixed')

    def test_capture_remains_unlabelled(self):
        f=Frame(np.ones((8,8),np.float32),np.tile([0,0,1.],(8,8,1)),
                np.zeros((8,8,3),np.uint8),np.ones((12,12,3),np.uint8)*100)
        root=ROOT/'.test-artifacts'/f'plan-save-{time.time_ns()}'
        saved=save_capture(root,f,'layout-0001',3,'single_2')
        meta=json.loads((saved/'capture.json').read_text())
        self.assertEqual(meta['collection_stage'],'single_2')
        self.assertEqual(meta['split_group'],'layout-0001')
        self.assertEqual(meta['annotation_status'],'unlabelled')
        self.assertIsNone(meta['contains_target'])
        self.assertFalse(meta['motion_commands'])

    def test_close_button_exits_without_recreating_window(self):
        f=Frame(np.ones((120,160),np.float32),np.tile([0,0,1.],(120,160,1)),
                np.zeros((120,160,3),np.uint8),np.ones((120,160,3),np.uint8)*100)
        reads=[]
        @contextmanager
        def fake_camera(**unused):
            def read(): reads.append(1);return f
            yield read
        dest=ROOT/'.test-artifacts'/f'close-test-{time.time_ns()}'
        with patch.object(sys,'argv',['data_capture.py','--root',str(dest)]), \
             patch.object(data_capture,'orbbec',fake_camera), \
             patch.object(cv2,'namedWindow') as named, patch.object(cv2,'imshow'), \
             patch.object(cv2,'waitKey',return_value=255), \
             patch.object(cv2,'getWindowProperty',return_value=0), patch.object(cv2,'destroyAllWindows'):
            data_capture.main()
        self.assertEqual(len(reads),16)
        named.assert_called_once()

    def test_banner_render(self):
        p=CapturePlan();p.select(ord('2'))
        panel=header(p,1696)
        self.assertEqual(panel.shape,(161,1696,3))
        self.assertEqual(panel.dtype,np.uint8)
        self.assertGreater(len(np.unique(panel.reshape(-1,3),axis=0)),10)
        f=load_frame(ROOT/'calibration/tag-1794451675210315500/fit/000.npz')
        h,w=f.depth_m.shape
        color,_=letterbox(f.raw_bgr,(h,w))
        valid=np.isfinite(f.depth_m)&(f.depth_m>0)
        depth=cv2.applyColorMap((np.clip(np.where(valid,f.depth_m,0)/2,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        depth[~valid]=0
        out=ROOT/'.test-artifacts/capture-plan-preview.png'
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out),np.vstack([panel,np.hstack([color,depth])]))


if __name__=='__main__':unittest.main()
