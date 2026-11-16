import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import numpy as np
from live_approach import LiveApproach,FIELDS
from wrist_pose import Wrist
from unload_route import load_route


class Arm:
    def __init__(self):self.commands=[]
    def send(self,command):self.commands.append(command)


class UnloadRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);(root/'camera_profiles').mkdir()
        self.path=root/'camera_profiles/unload-route.json'
        self.path.write_text(json.dumps(dict(enabled=True,release_deg=110,points=[
            dict(name='middle',joints_deg=[80,0,150,-38,3],speed_steps_s=80),
            dict(name='placement',joints_deg=[175,-5,179.9,-39,3],speed_steps_s=40)])))
        self.patcher=patch('live_approach.ROOT',root);self.patcher.start();self.addCleanup(self.patcher.stop)
        self.arm=Arm();self.c=LiveApproach(Wrist(ROOT/'.vendor/roarm_ws',[1]*5,[0]*5),self.arm)
        self.state=dict(zip(FIELDS,np.radians([35,0,150,-38,3])));self.state.update(v=1235,g=np.radians(134.6))

    def arrive(self):
        self.state.update(zip(FIELDS,self.c.plan['goal_joints_rad']))
        self.c.observe(self.state)

    def test_step_mode_preserves_grip_and_waits_for_next_N(self):
        self.c.next_unload_point(self.state)
        self.c.observe(self.state)
        self.assertEqual(self.c.unload_index,0)
        self.arrive()
        self.assertIsNone(self.c.auto_phase)
        self.assertEqual(self.c.unload_index,1)
        self.assertFalse(self.c.active)
        self.assertNotIn(6,[c['joint'] for c in self.arm.commands])
        self.c.next_unload_point(self.state);self.arrive()
        self.c.next_unload_point(self.state)
        self.assertEqual(self.c.unload_index,2)
        self.assertNotIn(6,[c['joint'] for c in self.arm.commands])

    def test_continuous_route_releases_only_after_last_point_feedback(self):
        self.c.start_unload(self.state)
        self.c.observe(self.state)
        self.assertNotIn(6,[c['joint'] for c in self.arm.commands])
        self.arrive()
        self.assertEqual(self.c.unload_index,1)
        self.assertNotIn(6,[c['joint'] for c in self.arm.commands])
        self.arrive()
        self.assertEqual(self.c.auto_phase,'unload_release')
        self.assertEqual([c['joint'] for c in self.arm.commands].count(6),1)
        self.assertAlmostEqual(np.degrees(self.arm.commands[-1]['rad']),110)
        self.state['g']=np.radians(110)
        for _ in range(3):self.c.observe(self.state)
        self.assertEqual(self.c.auto_phase,'unload_return')
        self.assertEqual(self.c.unload_index,0)
        self.assertEqual(self.c.plan['route_point'],'middle')
        self.arrive()
        self.assertIsNone(self.c.auto_phase)
        self.assertIsNone(self.c.plan)
        self.assertEqual([c['joint'] for c in self.arm.commands].count(6),1)

    def test_manual_reverse_requires_open_jaw_and_does_not_close_it(self):
        self.c.start_unload(self.state,continuous=False);self.arrive()
        self.c.next_unload_point(self.state);self.arrive()
        with self.assertRaisesRegex(ValueError,'open/release'):self.c.next_return_point(self.state)
        self.state['g']=np.radians(110)
        self.c.next_return_point(self.state)
        self.assertEqual(self.c.plan['route_point'],'middle')
        self.arrive()
        self.assertIsNone(self.c.auto_phase)
        self.assertNotIn(6,[c['joint'] for c in self.arm.commands])

    def test_reverse_does_not_jump_to_placement_from_unknown_pose(self):
        self.state['g']=np.radians(110)
        with self.assertRaisesRegex(ValueError,'not near'):self.c.next_return_point(self.state)
        self.assertEqual(self.arm.commands,[])

    def test_reverse_visits_real_route_in_exact_reverse_order(self):
        self.c.unload_steps=load_route(ROOT/'camera_profiles/unload-route.json')
        points=self.c.unload_steps['points']
        self.state.update(zip(FIELDS,np.radians(points[-1]['joints_deg'])));self.state['g']=np.radians(self.c.unload_steps['release_deg'])
        self.c.start_unload_return(self.state)
        visited=[]
        while self.c.active:
            visited.append(self.c.plan['route_point']);self.arrive()
        self.assertEqual(visited,[p['name'] for p in points[-2::-1]])
        self.assertNotIn(6,[c['joint'] for c in self.arm.commands])

    def test_X_cancels_reverse_and_prevents_remaining_commands(self):
        self.c.unload_steps=load_route(self.path)
        self.state.update(zip(FIELDS,np.radians(self.c.unload_steps['points'][-1]['joints_deg'])))
        self.state['g']=np.radians(110);self.c.start_unload_return(self.state)
        self.c.hold(self.state);count=len(self.arm.commands);self.c.observe(self.state)
        self.assertEqual(len(self.arm.commands),count)

    def test_X_discards_remaining_route_and_prevents_release(self):
        self.c.start_unload(self.state);self.c.hold(self.state)
        count=len(self.arm.commands)
        self.c.observe(self.state)
        self.assertEqual(len(self.arm.commands),count)
        self.assertIsNone(self.c.unload_steps)
        self.assertNotIn(6,[c['joint'] for c in self.arm.commands])

    def test_retract_completion_uses_route_instead_of_fixed_base_half_turn(self):
        self.c.auto_phase='retract'
        with patch.object(self.c.rear,'start') as old:
            self.c.advance_auto(self.state);old.assert_not_called()
        self.assertEqual(self.c.auto_phase,'unload')
        self.assertEqual(self.c.plan['route_point'],'middle')

    def test_out_of_range_record_is_rejected_before_motion(self):
        data=json.loads(self.path.read_text());data['points'][0]['joints_deg'][2]=200
        self.path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):self.c.start_unload(self.state)
        self.assertEqual(self.arm.commands,[])

    def test_selected_real_route_is_explicit_six_points_with_clamped_encoder_endpoint(self):
        data=load_route(ROOT/'camera_profiles/unload-route.json')
        self.assertEqual(len(data['points']),6)
        self.assertEqual(data['points'][-1]['name'],'placement')
        self.assertAlmostEqual(data['points'][3]['joints_deg'][2],2047*360/4096)
        self.assertGreater(data['points'][3]['recorded_joints_deg'][2],180)


if __name__=='__main__':unittest.main()
