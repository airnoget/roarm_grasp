from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import unittest
import json
import time
import cv2
import numpy as np
from dataset_labels import normalize_boxes, save_labels
from label_dataset import LabelWindow


class LabelTests(unittest.TestCase):
    def fixture(self):
        session=ROOT/'.test-artifacts'/f'labels-{time.time_ns()}'
        for i in range(2):
            folder=session/str(i);folder.mkdir(parents=True)
            cv2.imwrite(str(folder/'original_rgb.png'),np.full((800,1280,3),100,np.uint8))
            (folder/'capture.json').write_text(json.dumps(dict(annotation_image='original_rgb.png',
                annotation_status='unlabelled',collection_stage='single_1',contains_target=None)))
        return session

    def test_normalized_box_roundtrip(self):
        row=normalize_boxes([[100,50,300,250]],1280,800)[0]
        self.assertEqual(row[0],0)
        np.testing.assert_allclose(row[1:],[200/1280,150/800,200/1280,200/800])
        for box in [[-1,0,10,10],[0,0,1300,10],[0,0,1,5],[float('nan'),0,3,4]]:
            with self.assertRaises(ValueError):normalize_boxes([box],1280,800)

    def test_empty_requires_explicit_review(self):
        folder=self.fixture()/'0'
        with self.assertRaises(ValueError):save_labels(folder,[],[1280,800])
        save_labels(folder,[],[1280,800],confirmed_empty=True)
        self.assertEqual((folder/'original_rgb.txt').read_text(),'')
        meta=json.loads((folder/'capture.json').read_text())
        self.assertFalse(meta['contains_target']);self.assertEqual(meta['annotation_status'],'reviewed')

    def test_ui_mouse_coordinates_and_resume(self):
        session=self.fixture();window=LabelWindow(session)
        def point(x,y):
            return (int(x*window.layout['scale']+window.layout['offset'][0]),
                    int(y*window.layout['scale']+window.layout['offset'][1]+115))
        a,b=point(100,50),point(300,250)
        window.mouse(cv2.EVENT_LBUTTONDOWN,*a,0,None)
        window.mouse(cv2.EVENT_LBUTTONUP,*b,0,None)
        np.testing.assert_allclose(window.boxes,[[100,50,300,250]],atol=1.5)
        window.save();self.assertEqual(window.index,1);self.assertEqual(len(window.reviewed),1)
        resumed=LabelWindow(session);self.assertEqual(resumed.index,1)
        resumed.save(True);self.assertEqual(len(resumed.reviewed),2)
        frame=resumed.draw();self.assertEqual(frame.shape,(755,1280,3))
        cv2.imwrite(str(ROOT/'.test-artifacts/label-ui-preview.png'),frame)


if __name__=='__main__':unittest.main()
