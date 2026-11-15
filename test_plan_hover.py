"""CLI integration with synthetic observations, not physical acceptance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import contextlib,io,json,tempfile,unittest
from unittest.mock import patch
import numpy as np
import plan_hover
from wrist_pose import Wrist
from wrist_ik import raw_limits


class HoverCLITests(unittest.TestCase):
    def test_latest_without_joints_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'No saved observation'):plan_hover.latest_capture(root)
            for name,joints in [('000',True),('001',False)]:
                folder=root/'preset-test'/name;folder.mkdir(parents=True)
                (folder/'result.json').write_text(json.dumps(dict(backend='box_geometry',proposals=[{}])))
                np.savez(folder/'rgbd.npz',**({'joints_rad':np.zeros(5)} if joints else {'depth_m':np.zeros((2,2))}))
            with self.assertRaisesRegex(ValueError,'missing measured joints'):plan_hover.latest_capture(root)

    def test_newest_failed_fit_never_falls_back(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,stamp,proposals in [('000',100,[{}]),('001',200,[])]:
                folder=root/'preset-test'/name;folder.mkdir(parents=True)
                path=folder/'result.json';path.write_text(json.dumps(dict(backend='box_geometry',proposals=proposals,rejected=['fragmented surface'])))
                os.utime(path,(stamp,stamp));np.savez(folder/'rgbd.npz',joints_rad=np.zeros(5))
            with self.assertRaisesRegex(ValueError,'fragmented surface'):plan_hover.latest_capture(root)

    def test_transform_to_ik_and_output_report(self):
        calibration=json.loads(plan_hover.active_report().read_text())
        mapping=calibration['joint_mapping'];w=Wrist(ROOT/'.vendor/roarm_ws',mapping['joint_signs'],mapping['joint_offsets_rad'])
        limits=raw_limits(w,ROOT/'.vendor/roarm_ws');q=np.mean(limits,axis=1)
        _,expected=w.fk(q+[.02,-.02,.02,0,0])
        hand=np.array(calibration['wrist_from_depth_mm']);r,t=w.camera_pose(q,(hand[:3,:3],hand[:3,3]))
        surface=expected-[0,0,80];camera_point=np.linalg.solve(r,surface-t)
        sha=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))['sha256']
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);capture=folder/'synthetic-capture';capture.mkdir()
            np.savez(capture/'rgbd.npz',joints_rad=q)
            data=dict(preset_sha256=sha,backend='box_geometry',pose_error=None,
                      proposals=[dict(geometry=dict(camera_surface_xyz_mm=camera_point.tolist()))])
            (capture/'result.json').write_text(json.dumps(data))
            output=folder/'synthetic-plans'
            argv=['plan_hover.py',str(capture),'--link5','--output-root',str(output)]
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):plan_hover.main()
            report=json.loads(next(output.glob('*/plan.json')).read_text())
            np.testing.assert_allclose(report['surface_base_mm'],surface,atol=1e-6)
            _,actual=w.fk(report['solution']['joints_rad'])
            self.assertLess(np.linalg.norm(actual-expected),1)
            self.assertFalse(report['motion_enabled'])
            self.assertFalse(report['solution']['orientation_constrained'])
            self.assertTrue(report['blockers'])

    def test_no_joint_recording_cannot_generate_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            sha=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))['sha256']
            (folder/'result.json').write_text(json.dumps(dict(preset_sha256=sha,backend='box_geometry')))
            np.savez(folder/'rgbd.npz',depth_m=np.zeros((2,2)))
            with patch.object(sys,'argv',['plan_hover.py',str(folder),'--link5']):
                with self.assertRaisesRegex(ValueError,'No measured joints'):plan_hover.main()

if __name__=='__main__':unittest.main()
