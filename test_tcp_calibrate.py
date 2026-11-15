import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import numpy as np
from wrist_pose import rotate
from tcp_calibrate import fit_face,solve_session,apply_session


def samples(tip,reference,seed=1,count=11):
    rng=np.random.default_rng(seed);rows=[]
    for _ in range(count):
        angles=rng.uniform(-.8,.8,3)
        r=rotate(np.array([0.,0.,1.]),angles[0])@rotate(np.array([0.,1.,0.]),angles[1])@rotate(np.array([1.,0.,0.]),angles[2])
        rows.append(dict(rotation=r.tolist(),position_mm=(reference-r@tip).tolist(),
                         joints_rad=[*angles,0.,0.],gripper_rad=2.35))
    return rows


class TcpCalibrationTests(unittest.TestCase):
    def test_single_contact_point_requires_only_one_group_and_adds_no_cube_offset(self):
        point=np.array([9.,-7.,108.]);rows=samples(point,np.array([310.,-20.,40.]))
        data=dict(reference='contact_point',faces={'1':dict(fit=rows[:8],check=rows[8:])},model={},cube_side_mm=50)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.json';path.write_text(json.dumps(data))
            report=solve_session(path)
        self.assertTrue(report['passed'])
        self.assertEqual(report['reference'],'user_touchable_grasp_point')
        np.testing.assert_allclose(report['link5_tcp_position_mm'],point,atol=1e-8)
        self.assertNotIn('fitted_cube_side_mm',report)

    def test_apply_updates_TCP_and_clears_both_temporary_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);profiles=root/'camera_profiles';profiles.mkdir()
            model=root/'model.xacro';model.write_text('test model identity')
            active=root/'handeye.json';active.write_text(json.dumps(dict(joint_mapping={})))
            bias=profiles/'temporary-grasp-bias.json'
            bias.write_text(json.dumps(dict(enabled=True,display_left_mm=20,descent_extra_down_mm=10)))
            data=dict(model=dict(file='model.xacro',sha256=hashlib.sha256(model.read_bytes()).hexdigest(),joint_mapping={}),cube_side_mm=50,faces={})
            for index,z in enumerate((75.,125.),1):
                rows=samples(np.array([0.,0.,z]),np.array([300.,0.,0.]),index)
                data['faces'][str(index)]=dict(fit=rows[:8],check=rows[8:])
            path=root/'session.json';path.write_text(json.dumps(data))
            with patch('tcp_calibrate.ROOT',root),patch('active_handeye.active_report',return_value=active):apply_session(path)
            np.testing.assert_allclose(json.loads((profiles/'gripper-tcp.json').read_text())['link5_tcp_position_mm'],[0,0,100],atol=1e-8)
            applied_bias=json.loads(bias.read_text())
            self.assertFalse(applied_bias['enabled'])
            self.assertEqual(applied_bias['display_left_mm'],0)
            self.assertEqual(applied_bias['descent_extra_down_mm'],0)

    def test_recovers_contact_point_and_reference_from_independent_poses(self):
        tip=np.array([12.,-8.,115.]);reference=np.array([300.,30.,40.])
        rows=samples(tip,reference)
        report=fit_face(rows[:8],rows[8:])
        np.testing.assert_allclose(report['face_centre_link5_mm'],tip,atol=1e-8)
        np.testing.assert_allclose(report['reference_base_mm'],reference,atol=1e-8)
        self.assertTrue(report['passed'])

    def test_independent_bad_contact_is_rejected(self):
        rows=samples(np.array([0.,0.,100.]),np.array([300.,0.,0.]))
        rows[-1]['position_mm'][0]+=15
        self.assertFalse(fit_face(rows[:8],rows[8:])['passed'])

    def test_single_axis_rotation_cannot_identify_TCP(self):
        rows=samples(np.array([0.,0.,100.]),np.array([300.,0.,0.]))
        for i,row in enumerate(rows):
            angle=i*.12;r=rotate(np.array([0.,0.,1.]),angle)
            row.update(rotation=r.tolist(),position_mm=(np.array([300.,0.,0.])-r@np.array([0.,0.,100.])).tolist(),joints_rad=[angle,0,0,0,0])
        with self.assertRaisesRegex(ValueError,'orientation diversity'):fit_face(rows[:8],rows[8:])

    def test_two_opposite_faces_midpoint_handles_arbitrary_cube_orientation(self):
        centre=np.array([8.,-12.,103.]);half=np.array([15.,20.,0.])
        faces={}
        for index,tip in enumerate((centre+half,centre-half),1):
            rows=samples(tip,np.array([300.,20.,10.]),index)
            faces[str(index)]=dict(fit=rows[:8],check=rows[8:])
        data=dict(faces=faces,model={},cube_side_mm=50)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.json';path.write_text(json.dumps(data))
            report=solve_session(path)
        np.testing.assert_allclose(report['link5_tcp_position_mm'],centre,atol=1e-8)
        self.assertAlmostEqual(report['fitted_cube_side_mm'],50)
        self.assertTrue(report['passed'])
        data['faces']['2']['check'][-1]['gripper_rad']+=.1
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.json';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Gripper angle changed'):solve_session(path)


if __name__=='__main__':unittest.main()
