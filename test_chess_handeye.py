"""Synthetic chessboard detection, ordering, and full offline hand-eye regression."""
from pathlib import Path
import json
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.deps'))
import cv2
import numpy as np
from chessboard import detect, orient, object_points
from wrist_pose import Wrist
from solve_tag_handeye import transform, independent_checks


class ChessTests(unittest.TestCase):
    def test_new_9_by_12_square_board_has_8_by_11_inner_corners(self):
        image=np.full((660,540,3),255,np.uint8)
        for row in range(12):
            for col in range(9):
                if (row+col)%2==0:image[60+row*40:100+row*40,60+col*40:100+col*40]=0
        detected=orient(detect(image,(8,11)),[100,100],(8,11))
        expected=object_points((8,11),15)[:,:2]*(40/15)+100
        self.assertEqual(len(detected),88)
        self.assertLess(np.max(np.linalg.norm(detected-expected,axis=1)),1)

    def test_duplicate_check_excluded_without_changing_inputs(self):
        fit = [dict(name='fit/019', joints=np.zeros(5))]
        check = [dict(name='check/000', joints=np.zeros(5)),
                 dict(name='check/001', joints=np.ones(5)*.1),
                 dict(name='check/002', joints=np.ones(5)*.101)]
        kept, excluded = independent_checks(fit, check)
        self.assertEqual([x['name'] for x in kept], ['check/001'])
        self.assertEqual(excluded[0]['duplicate_of'], 'fit/019')
        self.assertEqual(excluded[1]['duplicate_of'], 'check/001')
        self.assertEqual(len(check), 3)

    def test_detector_on_printed_pattern(self):
        image = np.full((420, 520, 3), 255, dtype=np.uint8)
        for y in range(6):
            for x in range(8):
                if (x+y) % 2 == 0:
                    image[60+y*50:60+(y+1)*50, 60+x*50:60+(x+1)*50] = 0
        detected = orient(detect(image), [110, 110])
        expected = object_points()[:, :2]*2.5+110
        self.assertLess(np.max(np.linalg.norm(detected-expected, axis=1)), 1.)

    def test_origin_reversal_and_missing_click(self):
        corners = (object_points()[:, :2]+100).reshape(5, 7, 2)
        for grid in (corners, corners[::-1], corners[:, ::-1], corners[::-1, ::-1]):
            np.testing.assert_allclose(orient(grid.reshape(-1, 2), [100, 100]), corners.reshape(-1, 2))
        with self.assertRaises(ValueError):
            orient(corners, None)
        with self.assertRaises(ValueError):
            orient(corners, [-100, -100])

    def test_fit_only_reprojection_refinement(self):
        from handeye_refine import refine, pack, pixel_residuals
        wrist = Wrist(ROOT/'.vendor/roarm_ws', [1]*5, [0]*5)
        q0 = np.array([0., 0., np.pi/2, 0., 0.])
        known = transform(cv2.Rodrigues(np.array([.3, .2, 0.]))[0], [20, -35, 75])
        target = transform(*wrist.fk(q0)) @ known @ transform(np.eye(3), [-60, -40, 500])
        k = np.array([[615., 0, 640], [0, 615, 400], [0, 0, 1]])
        rng = np.random.default_rng(92)
        views = []
        for _ in range(20):
            g = transform(*wrist.fk(q0+rng.uniform(-.2, .2, 5)))
            pose = np.linalg.inv(known) @ np.linalg.inv(g) @ target
            corners = cv2.projectPoints(object_points(), cv2.Rodrigues(pose[:3, :3])[0],
                                        pose[:3, 3], k, np.zeros(8))[0].reshape(-1, 2)
            views.append(dict(base_from_wrist=g, object_points=object_points(),
                              corners_px=corners, K=k, distortion=np.zeros(8)))
        initial_x = known @ transform(cv2.Rodrigues(np.array([.02, -.01, .01]))[0], [2, -3, 4])
        initial_y = target @ transform(cv2.Rodrigues(np.array([-.01, .02, -.02]))[0], [3, 4, -2])
        x, y, info = refine(views[:15], initial_x, initial_y)
        self.assertTrue(info['fit_only'])
        np.testing.assert_allclose(x, known, atol=1e-4)
        np.testing.assert_allclose(y, target, atol=1e-4)
        self.assertLess(np.max(np.abs(pixel_residuals(pack(x, y), views[15:]))), 1e-5)

    def test_complete_pipeline_known_extrinsic(self):
        session = ROOT/'.test-artifacts'/f'chess-synthetic-{time.time_ns()}'
        session.mkdir(parents=True)
        setup = dict(schema=1, family='chessboard', pattern_size=[7, 5], square_size_mm=20.,
                     camera_pose_frame='raw_rgb', camera='synthetic')
        (session/'session.json').write_text(json.dumps(setup))
        wrist = Wrist(ROOT/'.vendor/roarm_ws', [1]*5, [0]*5)
        known = transform(cv2.Rodrigues(np.array([.3, .2, 0.]))[0], [20, -35, 75])
        q0 = np.array([0., 0., np.pi/2, 0., 0.])
        world_target = transform(*wrist.fk(q0)) @ known @ transform(np.eye(3), [-60, -40, 500])
        k = np.array([[615., 0, 640], [0, 615, 400], [0, 0, 1]])
        rng = np.random.default_rng(781)
        for group, count in [('fit', 15), ('check', 5)]:
            (session/group).mkdir()
            for i in range(count):
                q = q0+rng.uniform(-.15, .15, 5)
                camera_target = np.linalg.inv(known) @ np.linalg.inv(transform(*wrist.fk(q))) @ world_target
                rv = cv2.Rodrigues(camera_target[:3, :3])[0]
                corners = cv2.projectPoints(object_points(), rv, camera_target[:3, 3], k, np.zeros(8))[0].reshape(-1, 2)
                name = f'{i:03d}'
                record = dict(setup, snapshot=name+'.npz', joints_rad=q.tolist(), corners_raw_px=corners.tolist())
                (session/group/(name+'.json')).write_text(json.dumps(record))
                np.savez_compressed(session/group/(name+'.npz'), joints_rad=q, raw_K=k,
                                    raw_dist=np.zeros(8), raw_dist_model='OBCameraDistortionModel.BROWN_CONRADY_K6',
                                    T_raw_from_depth_mm=np.eye(4), raw_bgr=np.zeros((800, 1280, 3), dtype=np.uint8))
        result = subprocess.run([sys.executable, str(ROOT/'solve_chess_handeye.py'), str(session)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        report = json.loads(next(session.glob('solve-*/report.json')).read_text())
        self.assertTrue(report['passed'])
        self.assertFalse(report['physical_absolute_accuracy_verified'])
        np.testing.assert_allclose(report['wrist_from_rgb_mm'], known, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
