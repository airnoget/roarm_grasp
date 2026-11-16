from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import numpy as np
from wrist_pose import Wrist
from live_approach import hover_plan,LiveApproach,FIELDS,tcp_position,TCP_OFFSET
from wrist_ik import downward_tilt_deg


class LiveApproachTests(unittest.TestCase):
    def test_new_gripper_295_open_335_close_and_A_gate(self):
        class Arm:
            def __init__(self):self.commands=[]
            def send(self,c):self.commands.append(c)
        with patch('live_approach.CONTROL_PROFILE',ROOT/'camera_profiles/gripper-control.json'):
            arm=Arm();controller=LiveApproach(self.w,arm)
            state=dict(zip(FIELDS,self.q));state.update(v=1235,g=float(np.radians(295)))
            controller.set_gripper(state,True);controller.set_gripper(state,False)
            np.testing.assert_allclose([np.degrees(c['rad']) for c in arm.commands],[335,295])
            with patch.object(controller,'prepare'),patch.object(controller,'execute') as execute:
                controller.start_auto(dict(native_detections=[{}],proposals=[{}]),self.q,state)
                execute.assert_called_once()
            self.assertEqual(controller.auto_closed_deg,335)
            self.assertEqual(controller.rear.open_deg,295)
            controller.auto_phase=None;state['g']=float(np.radians(110))
            with patch.object(controller,'prepare'),patch.object(controller,'execute') as execute:
                controller.start_auto(dict(native_detections=[{}],proposals=[{}]),self.q,state)
                execute.assert_not_called()
            self.assertEqual(controller.auto_phase,'opening')
            self.assertAlmostEqual(np.degrees(arm.commands[-1]['rad']),295)

    def test_A_opens_then_waits_for_measured_angle_before_arm_movement(self):
        class Arm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        for angle in (30,90,135,170):
            arm=Arm();controller=LiveApproach(self.w,arm)
            result=dict(native_detections=[{}],proposals=[{}])
            state=dict(zip(FIELDS,self.q));state.update(v=1235,g=float(np.radians(angle)))
            with patch.object(controller,'prepare') as prepare,patch.object(controller,'execute') as execute:
                controller.start_auto(result,self.q,state)
                prepare.assert_called_once();execute.assert_not_called()
                self.assertEqual(controller.auto_phase,'opening')
                self.assertEqual([c['joint'] for c in arm.commands],[6])
                self.assertAlmostEqual(np.degrees(arm.commands[0]['rad']),110)
                for _ in range(4):controller.advance_auto(state)
                execute.assert_not_called()
                state['g']=float(np.radians(110))
                for _ in range(3):controller.advance_auto(state)
                execute.assert_called_once_with(state)
            self.assertEqual(controller.auto_phase,'hover')

    def test_planning_can_keep_rear_base_pose_beyond_old_90_degree_limit(self):
        q=self.q.copy();q[0]=np.radians(140)
        target=tcp_position(self.w,q)
        plan=hover_plan(self.w,q,target-[0,0,80])
        self.assertGreater(np.degrees(plan['goal_joints_rad'][0]),90)
        self.assertLess(np.linalg.norm(tcp_position(self.w,plan['goal_joints_rad'])-target),1)

    def test_tracking_correction_reduces_static_servo_offset_without_changing_IK_goal(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        goal=self.q.copy();bias=np.radians([0,2.5,3.4,0,0]);actual=goal+bias
        controller.plan=dict(phase='descent',goal_joints_rad=goal.tolist(),hover_base_mm=tcp_position(self.w,goal).tolist())
        controller.report={};commanded=goal.copy()
        start_error=np.linalg.norm(tcp_position(self.w,actual)-tcp_position(self.w,goal))
        with patch.object(controller,'flush'):
            for _ in range(2):
                state=dict(zip(FIELDS,actual));state['v']=1235
                controller.correct_tracking(state)
                for command in arm.commands:commanded[command['joint']-1]=command['rad']
                actual=commanded+bias
        np.testing.assert_allclose(actual,goal,atol=1e-9)
        np.testing.assert_allclose(controller.plan['goal_joints_rad'],goal)
        self.assertGreater(start_error,10)
        self.assertEqual(set(c['joint'] for c in arm.commands),{2,3})

    def test_B_and_U_take_over_active_auto_descent_and_discard_queue(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        for key in ('b','u'):
            arm=FakeArm();controller=LiveApproach(self.w,arm)
            controller.active=True;controller.auto_phase='descent'
            controller.plan=dict(phase='descent');controller.report={}
            controller.descent_queue=[dict(stale=True)]
            controller.gripper_moving=True
            state=dict(zip(FIELDS,self.q));state.update(v=1235,g=np.radians(135.1))
            with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)),patch.object(controller,'flush'):
                if key=='b':controller.transfer_rear(state)
                else:controller.retract(state)
            self.assertEqual(controller.descent_queue,[])
            self.assertEqual(controller.plan['phase'],'retract')
            self.assertEqual(controller.auto_phase,'retract' if key=='b' else None)
            self.assertNotIn(6,[c['joint'] for c in arm.commands])
            # The only base command so far is a finite hold at its actual angle.
            for c in arm.commands:
                if c['joint']==1:self.assertEqual(c['rad'],state['b'])

    def test_home_retracts_before_returning_base_and_never_opens_jaw(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        state=dict(zip(FIELDS,self.q));state.update(b=np.radians(-179),v=1235,g=np.radians(135.1))
        with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)):
            controller.return_home(state)
            self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5])
            controller.observe(state)
            self.assertEqual(controller.auto_phase,'home_retract')
            state.update(zip(FIELDS,controller.plan['goal_joints_rad']))
            controller.observe(state)
            self.assertEqual(controller.auto_phase,'home')
            self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5,1])
            self.assertAlmostEqual(np.degrees(arm.commands[-1]['rad']),.44)
            state['b']=arm.commands[-1]['rad'];controller.observe(state)
            self.assertIsNone(controller.auto_phase)
            self.assertIsNone(controller.plan)
            self.assertIsNone(controller.rear.phase)
            self.assertNotIn(6,[c['joint'] for c in arm.commands])

    def test_waypoint_is_measured_feedback_and_never_sends_motion(self):
        import json
        class NoMotionArm:
            def send(self,command):raise AssertionError('Recording must never move robot')
        controller=LiveApproach(self.w,NoMotionArm())
        state=dict(zip(FIELDS,self.q));state.update(g=np.radians(135),v=1235)
        with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)):
            path=controller.save_waypoint(state);record=json.loads(path.read_text())
        np.testing.assert_allclose(record['joints_rad'],self.q)
        self.assertFalse(record['execution_enabled'])
        self.assertFalse(record['stationary_verified'])

    def test_D_extra_down_is_20mm_beyond_cube_centre(self):
        from live_approach import grasp_clearance_mm
        self.assertEqual(grasp_clearance_mm(50,20),-45)
        self.assertEqual(grasp_clearance_mm(50,0),-25)
        self.assertEqual(grasp_clearance_mm(50,10),-35)

    def test_stationary_25mm_residual_corrects_instead_of_advancing_auto(self):
        class StaticWrist:
            def fk(self,q):return np.eye(3),np.zeros(3)
        controller=LiveApproach(StaticWrist(),None)
        controller.plan=dict(hover_base_mm=(TCP_OFFSET+[0,0,25]).tolist(),goal_joints_rad=[0,np.radians(2.2),0,0,0],phase='hover')
        controller.active=True;controller.auto_phase='hover';controller.report=dict(samples=[])
        state=dict(b=0,s=0,e=0,t=0,r=0,v=1235)
        with patch.object(controller,'flush'),patch.object(controller,'correct_tracking') as correct,patch.object(controller,'descend') as descend:
            for i in range(4):
                with patch('live_approach.time.monotonic',return_value=i*.3):controller.observe(state)
            descend.assert_not_called()
            correct.assert_called_once_with(state)
        self.assertEqual(controller.auto_phase,'hover')
        self.assertTrue(controller.active)

    def test_stalled_large_joint_error_does_not_advance_auto(self):
        class StaticWrist:
            def fk(self,q):return np.eye(3),np.zeros(3)
        controller=LiveApproach(StaticWrist(),None)
        controller.plan=dict(hover_base_mm=[0,0,125],goal_joints_rad=[0,np.radians(12),0,0,0],phase='hover')
        controller.active=True;controller.auto_phase='hover';controller.report=dict(samples=[])
        state=dict(b=0,s=0,e=0,t=0,r=0,v=1235)
        with patch.object(controller,'flush'),patch.object(controller,'correct_tracking'),patch.object(controller,'descend') as descend:
            for i in range(4):
                with patch('live_approach.time.monotonic',return_value=i*.3):controller.observe(state)
            descend.assert_not_called()
        self.assertTrue(controller.active)

    def setUp(self):
        import json
        self.control_temp=tempfile.TemporaryDirectory();self.addCleanup(self.control_temp.cleanup)
        control=json.loads((ROOT/'camera_profiles/gripper-control.json').read_text())
        control.update(open_reference_deg=110,closed_angle_deg=138,contact_angle_deg=135.1)
        profile=Path(self.control_temp.name)/'gripper.json';profile.write_text(json.dumps(control))
        p=patch('live_approach.CONTROL_PROFILE',profile);p.start();self.addCleanup(p.stop)
        self.w=Wrist(ROOT/'.vendor/roarm_ws',[1]*5,[0]*5)
        self.q=np.radians([.7,9.8,130.8,-2.1,47.8])

    def test_full_hover_is_not_a_tiny_step(self):
        plan=hover_plan(self.w,self.q,[391.5,-13.6,-50.7])
        position=tcp_position(self.w,plan['goal_joints_rad'])
        self.assertLess(np.linalg.norm(position-[391.5,-13.6,29.3]),1)
        self.assertGreater(np.max(np.abs(np.degrees(np.array(plan['goal_joints_rad'])-self.q))),10)
        self.assertFalse(plan['collision_checked'])
        self.assertGreaterEqual(min(plan['speed_steps_s']),40)

    def test_posture_selection_and_vertical_descent_reference(self):
        seed=self.q.copy();last=None
        for height in (80.,70.,60.,50.,40.):
            plan=hover_plan(self.w,seed,[391.5,-13.6,-50.7],height)
            seed=np.array(plan['goal_joints_rad']);position=tcp_position(self.w,seed)
            self.assertIn('no forced vertical',plan['posture_method'])
            self.assertLess(np.linalg.norm(position-plan['hover_base_mm']),1)
            self.assertFalse(plan['tcp_calibrated'])
            if last is not None:
                self.assertLess(np.linalg.norm(position[:2]-last[:2]),1)
                self.assertAlmostEqual(position[2]-last[2],-10,delta=1)
            last=position

    def test_reaching_current_tcp_does_not_force_vertical(self):
        current=tcp_position(self.w,self.q)
        plan=hover_plan(self.w,self.q,current-[0,0,80])
        self.assertLess(np.max(np.abs(np.array(plan['goal_joints_rad'])-self.q)),.05)

    def test_grasp_axis_alignment_can_change_roll_and_respects_line_symmetry(self):
        current=tcp_position(self.w,self.q);rotation,_=self.w.fk(self.q)
        desired=rotation@np.array([0.,1.,0.])
        results=[]
        for direction in (desired,-desired):
            plan=hover_plan(self.w,self.q,current-[0,0,80],closing_axis_base=direction,closing_axis_local=[1.,0.,0.])
            self.assertLess(plan['closing_axis_error_deg'],5)
            self.assertGreater(abs(plan['goal_joints_rad'][4]-self.q[4]),.2)
            self.assertLess(np.linalg.norm(tcp_position(self.w,plan['goal_joints_rad'])-current),1)
            results.append(plan)
        self.assertAlmostEqual(results[0]['posture_score'],results[1]['posture_score'],places=5)

    def test_invalid_new_observation_discards_previous_plan(self):
        controller=LiveApproach(self.w,None);controller.plan={'old':True}
        with self.assertRaises(ValueError):controller.prepare(dict(native_detections=[],proposals=[]),self.q)
        self.assertIsNone(controller.plan)

    def test_auto_without_measured_close_angle_sends_no_commands(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        import json
        with tempfile.TemporaryDirectory() as directory:
            profile=Path(directory)/'gripper.json'
            profile.write_text(json.dumps(dict(open_reference_deg=110,closed_angle_deg=None)))
            with patch('live_approach.CONTROL_PROFILE',profile):
                with self.assertRaisesRegex(ValueError,'closed_angle_deg'):
                    controller.start_auto({},self.q,dict(g=np.radians(110),v=1235))
        self.assertEqual(arm.commands,[])

    def test_auto_lift_completion_does_not_claim_grasp_success(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm);controller.auto_phase='lift'
        controller.report={}
        state=dict(zip(FIELDS,self.q));state.update(v=1235,g=np.radians(135.1))
        with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)),patch.object(controller,'flush'):
            controller.advance_auto(state)
            self.assertEqual(controller.auto_phase,'retract')
            self.assertTrue(controller.active)
            self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5])
            np.testing.assert_allclose(np.degrees(controller.plan['goal_joints_rad'])[1:],[0,160,-38,3])
            self.assertAlmostEqual(controller.plan['goal_joints_rad'][0],state['b'])
            # Being active must not trigger turning, even with repeated ticks.
            controller.advance_auto(state)
            self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5])
            controller.active=False
            controller.advance_auto(state)
            self.assertEqual(controller.auto_phase,'rear_transfer')
            self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5,1])
            controller.rear.phase=None
            controller.advance_auto(dict(v=1235,b=0))
        self.assertIsNone(controller.auto_phase)
        self.assertEqual(controller.report['outcome'],'auto_rear_release_finished_grasp_unverified')

    def test_one_press_C_O_commands_endpoints_not_three_degree_steps(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        controller.set_gripper(dict(v=1235,g=np.radians(110)),True)
        controller.set_gripper(dict(v=1235,g=np.radians(135)),False)
        np.testing.assert_allclose([np.degrees(c['rad']) for c in arm.commands],[138,110])
        self.assertEqual([c['joint'] for c in arm.commands],[6,6])

    def test_manual_B_retracts_before_base_turn_and_keeps_jaw_held(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        state=dict(zip(FIELDS,self.q));state.update(v=1235,g=np.radians(135.1))
        with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)):
            controller.transfer_rear(state)
            controller.observe(state)
            self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5])
            state.update(zip(FIELDS,controller.plan['goal_joints_rad']))
            controller.observe(state)
            self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5,1])
            self.assertEqual(controller.rear.phase,'turning')
            self.assertEqual(controller.auto_phase,'rear_transfer')
            self.assertNotIn(6,[c['joint'] for c in arm.commands])

    def test_U_retract_only_and_X_cancels_pending_turn(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        for cancel in (False,True):
            arm=FakeArm();controller=LiveApproach(self.w,arm)
            state=dict(zip(FIELDS,self.q));state.update(v=1235,g=np.radians(135.1))
            with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)):
                if cancel:
                    controller.transfer_rear(state);controller.hold(state)
                    count=len(arm.commands);controller.observe(state)
                    self.assertEqual(len(arm.commands),count)
                else:
                    controller.retract(state)
                    state.update(zip(FIELDS,controller.plan['goal_joints_rad']))
                    controller.observe(state)
                    self.assertEqual([c['joint'] for c in arm.commands],[2,3,4,5])
                self.assertIsNone(controller.auto_phase)
                self.assertIsNone(controller.rear.phase)

    def test_auto_contact_uses_measured_angle_instead_of_commanded_angle(self):
        controller=LiveApproach(self.w,None)
        controller.plan=hover_plan(self.w,self.q,[391.5,-13.6,-50.7])
        controller.auto_phase='closing';controller.auto_closed_deg=138;controller.auto_contact_deg=135.1
        state=dict(zip(FIELDS,self.q));state.update(v=1235,g=np.radians(135.1))
        with patch.object(controller,'_retract') as retract:
            for _ in range(3):controller.advance_auto(state)
            retract.assert_called_once_with(state)
        self.assertEqual(controller.auto_phase,'retract')

    def test_descent_executes_preplanned_joint_branch_without_resolving(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        controller.plan=hover_plan(self.w,self.q,[391.5,-13.6,-50.7])
        expected=controller.plan['goal_joints_rad'].copy()
        state=dict(zip(FIELDS,self.q));state['v']=1235
        with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)),patch('live_approach.hover_plan',side_effect=AssertionError('Must preserve queued IK branch')):
            controller.execute(state,descent=True)
        np.testing.assert_allclose(controller.plan['goal_joints_rad'],expected)
        for command in arm.commands:self.assertAlmostEqual(command['rad'],expected[command['joint']-1])

    def test_D_replans_even_if_previous_descent_flag_is_active(self):
        controller=LiveApproach(self.w,None)
        controller.plan=hover_plan(self.w,self.q,[391.5,-13.6,-50.7])
        state=dict(zip(FIELDS,controller.plan['goal_joints_rad']));state['v']=1235
        controller.plan['phase']='descent';controller.active=True
        controller.grasp_clearance=-25;controller.report={}
        with patch.object(controller,'flush'),patch.object(controller,'execute') as execute:
            controller.descend(state)
            execute.assert_called_once()
        self.assertTrue(controller.descent_queue)

    def test_record_closure_saves_measured_angle_without_motion(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            profile=Path(directory)/'gripper.json'
            profile.write_text(json.dumps(dict(open_reference_deg=110,closed_angle_deg=None)))
            controller=LiveApproach(self.w,None)
            with patch('live_approach.CONTROL_PROFILE',profile):
                with self.assertRaises(ValueError):controller.remember_closed(dict(g=np.radians(110)))
                controller.remember_closed(dict(g=np.radians(125)))
            self.assertAlmostEqual(json.loads(profile.read_text())['closed_angle_deg'],125)

    def test_manual_gripper_uses_only_id17_and_increase_closes(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        controller.jog_gripper(dict(g=np.radians(110),v=1235),True)
        self.assertEqual(arm.commands[-1]['joint'],6)
        self.assertAlmostEqual(np.degrees(arm.commands[-1]['rad']),113)
        controller.jog_gripper(dict(g=np.radians(112),v=1235),False)
        self.assertAlmostEqual(np.degrees(arm.commands[-1]['rad']),110)
        with self.assertRaises(ValueError):controller.jog_gripper(dict(g=0,v=1235),True)
        controller.hold(dict(g=np.radians(111)))
        self.assertAlmostEqual(np.degrees(arm.commands[-1]['rad']),111)

    def test_descent_ends_at_half_height_below_top_not_old_40mm_hover(self):
        controller=LiveApproach(self.w,None)
        controller.plan=hover_plan(self.w,self.q,[391.5,-13.6,-50.7])
        controller.hover_reached=True;controller.grasp_clearance=-25
        state=dict(zip(FIELDS,controller.plan['goal_joints_rad']));state['v']=1235
        with patch.object(controller,'execute') as execute:
            controller.descend(state)
            execute.assert_called_once()
        self.assertAlmostEqual(controller.descent_queue[-1]['clearance_mm'],-25)
        self.assertAlmostEqual(controller.descent_queue[-1]['hover_base_mm'][2],-75.7)

    def test_hold_commands_only_five_installed_joints(self):
        class FakeArm:
            def __init__(self):self.commands=[]
            def send(self,cmd):self.commands.append(cmd)
        arm=FakeArm();controller=LiveApproach(self.w,arm)
        state=dict(zip(FIELDS,self.q));state['v']=1235
        controller.plan=hover_plan(self.w,self.q,[391.5,-13.6,-50.7])
        with tempfile.TemporaryDirectory() as directory,patch('live_approach.ROOT',Path(directory)):
            # Execution re-plans using the real vendor model root.
            with patch('live_approach.hover_plan',return_value=controller.plan):controller.execute(state)
            self.assertTrue(controller.active)
            controller.hold(state)
            self.assertFalse(controller.active)
            self.assertEqual([c['joint'] for c in arm.commands[-5:]],[1,2,3,4,5])
            np.testing.assert_allclose([c['rad'] for c in arm.commands[-5:]],self.q)

if __name__=='__main__':unittest.main()
