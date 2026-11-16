"""Window-driven hover commissioning; owns no serial connection or camera."""
import json
import time
from pathlib import Path
import numpy as np
from wrist_ik import raw_limits, solve_grasp_posture, downward_tilt_deg
from active_handeye import active_report, fingerprint
from rear_transfer import RearTransfer
from motion_speeds import SPEEDS,phase_speed
from unload_route import load_route,LIMITS as ROUTE_LIMITS

ROOT=Path(__file__).resolve().parent
FIELDS=('b','s','e','t','r')
TCP_SPEC=json.loads((ROOT/'camera_profiles/gripper-tcp.json').read_text(encoding='utf-8'))
TCP_OFFSET=np.array(TCP_SPEC['link5_tcp_position_mm'],float)
CONTROL_PROFILE=Path(__file__).resolve().parent/'camera_profiles/gripper-control.json'
REAR_PROFILE=ROOT/'camera_profiles/rear-placement.json'


def tcp_position(wrist,q):
    rotation,position=wrist.fk(q)
    return position+rotation@TCP_OFFSET


def grasp_clearance_mm(height_mm,extra_down_mm):
    if not np.isfinite([height_mm,extra_down_mm]).all() or height_mm<=0 or extra_down_mm<0:
        raise ValueError('Invalid measured object height or downward adjustment')
    return -float(height_mm)/2-float(extra_down_mm)


def hover_plan(wrist,q,surface,clearance=80.,closing_axis_base=None,closing_axis_local=None):
    q=np.asarray(q,float);surface=np.asarray(surface,float)
    if surface.shape!=(3,) or not np.isfinite(surface).all():
        raise ValueError('Invalid surface XYZ')
    limits=raw_limits(wrist,ROOT/'.vendor/roarm_ws')
    firmware=np.radians([[-179.9,179.9],[-90,90],[0,179.9],[-90,90],[-179.9,180]])
    limits[:,0]=np.maximum(limits[:,0],firmware[:,0])
    limits[:,1]=np.minimum(limits[:,1],firmware[:,1])
    target=surface+[0,0,clearance]
    solution=solve_grasp_posture(wrist,target,q,limits,TCP_OFFSET,closing_axis_base,closing_axis_local)
    goal=np.array(solution['joints_rad'])
    # Common nominal duration; servo speeds are encoder steps/s, not degrees/s.
    duration=max(2.,float(np.max(np.abs(np.degrees(goal-q))))/5.)
    speeds=np.maximum(40,np.rint(np.abs(goal-q)*4096/(2*np.pi)/duration)).astype(int)
    return dict(surface_base_mm=surface.tolist(),hover_base_mm=target.tolist(),
                start_joints_rad=q.tolist(),goal_joints_rad=goal.tolist(),
                nominal_duration_s=duration,speed_steps_s=speeds.tolist(),
                reference=f'Provisional TCP link5 XYZ mm: {TCP_OFFSET.tolist()}; user approximate horn measurement',
                tcp_link5_mm=TCP_OFFSET.tolist(),
                collision_checked=False,orientation_constrained=False,
                downward_tilt_deg=solution['downward_tilt_deg'],clearance_mm=clearance,
                closing_axis_base=None if closing_axis_base is None else np.asarray(closing_axis_base).tolist(),
                closing_axis_local=None if closing_axis_local is None else np.asarray(closing_axis_local).tolist(),
                closing_axis_error_deg=solution['closing_axis_error_deg'],posture_score=solution['posture_score'],
                posture_method=solution['orientation_preference'],
                tcp_calibrated=False,phase='hover',
                model_residual_mm=solution['position_residual_mm'])


class LiveApproach:
    def __init__(self,wrist,arm,hover_mm=80.):
        self.wrist=wrist;self.arm=arm;self.plan=None;self.active=False
        self.status='P plan | G grasp-aligned hover | D descend | X hold | Q exit'
        self.report=None;self.directory=None;self.last_log=0
        self.hover_reached=False;self.descent_queue=[]
        self.hover_mm=float(hover_mm);self.grasp_clearance=None
        control=json.loads(CONTROL_PROFILE.read_text(encoding='utf-8'))
        self.gripper_status=f'C close {control.get("closed_angle_deg")}deg | O open {control["open_reference_deg"]}deg | [/] fine jog'
        self.rear=RearTransfer(arm,open_deg=control['open_reference_deg'])
        self.gripper_moving=False
        self.manual_offset_base=np.zeros(3);self.display_left_base=None
        self.auto_phase=None;self.auto_closed_deg=None;self.auto_contact_deg=None;self.auto_close_samples=0
        self.gripper_last_target=None
        self.notice=''
        self.settle_samples=[];self.extra_down_mm=0.
        self.unload_steps=None;self.unload_index=0;self.unload_continuous=False;self.unload_returning=False

    def set_gripper(self,state,closing):
        config=json.loads(CONTROL_PROFILE.read_text(encoding='utf-8'))
        target=config.get('closed_angle_deg') if closing else config['open_reference_deg']
        if not isinstance(target,(int,float)) or not np.isfinite(target):raise ValueError('Record closed angle with R first')
        if state['v']<700 or state.get('torswitchG')==0:raise ValueError('ID17 power/torque unavailable')
        command=dict(T=101,joint=6,rad=float(np.radians(target)),spd=SPEEDS['gripper_close'] if closing else SPEEDS['gripper_release'],acc=5)
        self.arm.send(command);self.gripper_last_target=float(target);self.gripper_moving=True
        self.gripper_status=f'ID17 one-shot {"CLOSE" if closing else "OPEN"} to {target:g}deg | C close O open'
        if self.report is not None:self.report.setdefault('gripper_commands',[]).append(command);self.flush()

    def transfer_rear(self,state):
        self.retract(state)
        self.auto_phase='retract'

    def start_unload(self,state,continuous=True):
        self.unload_steps=load_route(ROOT/'camera_profiles/unload-route.json')
        self.unload_index=0;self.unload_continuous=continuous;self.unload_returning=False
        self.execute_unload_point(state)

    def start_unload_return(self,state,continuous=True):
        if self.unload_steps is None:self.unload_steps=load_route(ROOT/'camera_profiles/unload-route.json')
        if abs(np.degrees(state.get('g',np.nan))-self.unload_steps['release_deg'])>=3 or not np.isfinite(state.get('g',np.nan)):
            raise ValueError('Return: O open/release first; waiting for measured open angle')
        q=np.degrees([state[k] for k in FIELDS])
        distances=[np.max(np.abs(q-np.asarray(point['joints_deg']))) for point in self.unload_steps['points']]
        nearest=int(np.argmin(distances))
        if distances[nearest]>5:raise ValueError('Return: current pose is not near a taught route point; will not jump to placement')
        self.unload_returning=True;self.unload_continuous=continuous;self.unload_index=nearest-1
        if self.unload_index>=0:self.execute_unload_point(state)
        else:self.finish_unload_return()

    def next_return_point(self,state):
        if self.active or self.auto_phase or self.rear.phase:raise ValueError('K: wait for current movement; X cancels')
        if not self.unload_returning:self.start_unload_return(state,continuous=False)
        elif self.unload_index>=0:self.execute_unload_point(state)
        else:self.status='RETURN finished at taught carry_start | P plan new target'

    def finish_unload_return(self):
        self.auto_phase=None;self.gripper_moving=False;self.plan=None
        if self.report is not None:
            self.report['outcome']='taught_route_return_finished_release_unverified';self.flush()
        self.status='RETURN finished at taught carry_start | gripper open | P plan new target'

    def next_unload_point(self,state):
        if self.active or self.auto_phase or self.rear.phase:
            raise ValueError('N: wait for current point; X cancels, B starts full placement')
        if self.unload_steps is None or self.unload_returning:self.start_unload(state,continuous=False)
        elif self.unload_index<len(self.unload_steps['points']):self.execute_unload_point(state)
        else:self.status='All taught points reached | O releases | B restarts route'

    def execute_unload_point(self,state):
        if state['v']<700:raise ValueError('Servo power unavailable')
        point=self.unload_steps['points'][self.unload_index]
        q=np.array([state[k] for k in FIELDS]);goal=np.radians(point['joints_deg'])
        self.descent_queue=[];self.hover_reached=False;self.settle_samples=[]
        self.plan=dict(phase='unload',route_return=self.unload_returning,route_point=point['name'],route_step=self.unload_index+1,goal_joints_rad=goal.tolist(),
                       hover_base_mm=tcp_position(self.wrist,goal).tolist())
        self.directory=ROOT/'approach_steps'/f'unload-{time.time_ns()}';self.directory.mkdir(parents=True)
        self.report=dict(plan=self.plan,commands=[],samples=[],outcome='moving',physical_accuracy_verified=False)
        self.active=True;self.auto_phase='unload_return' if self.unload_returning else 'unload';self.motion_speed=int(point.get('speed_steps_s',80))
        self.plan['route_speed']=self.motion_speed
        try:
            for i in range(5):
                if abs(goal[i]-q[i])<np.radians(.05):continue
                command=dict(T=101,joint=i+1,rad=float(goal[i]),spd=self.motion_speed,acc=5)
                self.arm.send(command);self.report['commands'].append(command)
            label='RETURN' if self.unload_returning else 'UNLOAD'
            grip='gripper open' if self.unload_returning else 'gripper held'
            self.status=f'{label} {self.unload_index+1}/{len(self.unload_steps["points"])} {point["name"]} | {grip} | X hold'
            self.flush()
        except Exception:
            self.hold(state);raise

    def retract(self,state):
        """Return shoulder/elbow/wrists near boot; retain base and jaw targets."""
        if self.active or self.auto_phase or self.rear.phase:
            # Manual takeover cancels queued descent/auto before retracting.
            # Retain the jaw holding command rather than stopping closure.
            self.gripper_moving=False
            self.hold(state)
        self._retract(state)

    def return_home(self,state):
        if self.active or self.auto_phase or self.rear.phase or self.gripper_moving:self.hold(state)
        self._retract(state)
        self.auto_phase='home_retract'
        self.status='HOME: retract arm first, then return base | gripper held | X hold'

    def _retract(self,state,return_base=False):
        if state['v']<700:raise ValueError('Servo power unavailable')
        profile=json.loads(REAR_PROFILE.read_text(encoding='utf-8'))
        actual=np.asarray([state[k] for k in FIELDS],float)
        goal=actual.copy()
        if return_base:goal[0]=np.radians(profile['home_base_deg'])
        else:goal[1:]=np.radians(profile['retract_joints_deg'])
        if not np.isfinite(goal).all():raise ValueError('Invalid retract joint angles')
        self.descent_queue=[];self.hover_reached=False;self.settle_samples=[]
        self.plan=dict(phase='home' if return_base else 'retract',goal_joints_rad=goal.tolist(),
                       hover_base_mm=tcp_position(self.wrist,goal).tolist(),
                       start_joints_rad=actual.tolist(),base_and_gripper_preserved=not return_base,
                       gripper_preserved=True)
        self.directory=ROOT/'approach_steps'/f'retract-{time.time_ns()}'
        self.directory.mkdir(parents=True)
        self.report=dict(plan=self.plan,commands=[],samples=[],outcome='moving',physical_accuracy_verified=False)
        self.active=True
        self.motion_speed=None
        try:
            self.motion_speed=self.select_speed(actual)
            for i in ([0] if return_base else range(1,5)):
                if abs(goal[i]-actual[i])<np.radians(.05):continue
                command=dict(T=101,joint=i+1,rad=float(goal[i]),spd=self.motion_speed,acc=5)
                self.arm.send(command);self.report['commands'].append(command)
            self.status='HOME base returning | gripper held | X hold' if return_base else 'RETRACT near startup | base fixed, gripper held | X hold'
            self.flush()
        except Exception:
            self.hold(state)
            raise

    def save_waypoint(self,state):
        q=np.asarray([state[k] for k in FIELDS],float)
        if not np.isfinite(q).all() or state['v']<700:raise ValueError('Need valid powered joint feedback to record a waypoint')
        record=dict(recorded_ns=time.time_ns(),joint_order=list(FIELDS),joints_rad=q.tolist(),
                    joints_deg=np.degrees(q).tolist(),gripper_deg=float(np.degrees(state['g'])) if 'g' in state else None,
                    tcp_base_mm=tcp_position(self.wrist,q).tolist(),tcp_offset_mm=TCP_OFFSET.tolist(),
                    handeye_report=str(active_report()),handeye_sha256=fingerprint(),
                    temporary_bias_base_mm=self.manual_offset_base.tolist(),extra_down_mm=self.extra_down_mm,
                    controller_active=bool(self.active or self.rear.phase),stationary_verified=False,
                    execution_enabled=False,scope='Measured teach-point candidate only; no replay or motion command')
        folder=ROOT/'waypoints';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'pose-{record["recorded_ns"]}.json'
        path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        self.notice='W saved '+path.name+' (measured joints; no motion)'
        print('WAYPOINT',path,flush=True)
        return path

    def jog_gripper(self,state,closing):
        config=json.loads(CONTROL_PROFILE.read_text(encoding='utf-8'))
        if not isinstance(state.get('g'),(int,float)) or not np.isfinite(state['g']):raise ValueError('ID17 angle feedback unavailable')
        if state.get('torswitchG')==0:raise ValueError('ID17 torque off; check servo power')
        if state['v']<700:raise ValueError('Servo power unavailable')
        angle=float(np.degrees(state['g']))
        if angle<config['open_reference_deg']-2 or angle>4095*360/4096+2:raise ValueError('ID17 angle outside expected range; check feedback before jogging')
        target=float(np.clip(angle+(3 if closing else -3)*config['closing_angle_direction'],config['open_reference_deg'],4095*360/4096))
        command=dict(T=101,joint=6,rad=float(np.radians(target)),spd=40,acc=5)
        self.arm.send(command)
        self.gripper_last_target=target
        self.gripper_moving=True
        self.gripper_status=f'ID17 {angle:.1f} -> {target:.1f}deg | C close +3 | O open -3'
        if self.report is not None:
            self.report.setdefault('gripper_commands',[]).append(command);self.flush()
        print('GRIPPER',self.gripper_status,flush=True)

    def remember_closed(self,state):
        if self.auto_phase or self.active:raise ValueError('Record closure only while arm is stationary')
        angle=float(np.degrees(state.get('g',np.nan)))
        config=json.loads(CONTROL_PROFILE.read_text(encoding='utf-8'))
        if not np.isfinite(angle) or not config['open_reference_deg']<angle<=4095*360/4096:
            raise ValueError('Close onto the block with C first, then R records the feedback angle')
        commanded=self.gripper_last_target
        if commanded is None or not angle<=commanded<=angle+5:commanded=angle
        config.update(closed_angle_deg=commanded,contact_angle_deg=angle,closed_angle_source='User pressed R with block held; command and measured contact stored separately',
                      automatic_closure_enabled=True,closed_angle_recorded_ns=time.time_ns())
        temporary=CONTROL_PROFILE.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(config,indent=2),encoding='utf-8');temporary.replace(CONTROL_PROFILE)
        self.gripper_status=f'RECORDED close angle {angle:.1f}deg | O open near{config["open_reference_deg"]:g}; A auto'
        print(self.gripper_status,flush=True)

    def prepare(self,result,joints):
        if self.active:raise ValueError('Press X to hold before planning again')
        # Failed planning must invalidate an older target, never reuse it.
        self.plan=None
        self.hover_reached=False;self.descent_queue=[]
        if joints is None or result.get('pose_error'):raise ValueError('Release Wi-Fi controls and wait for a stationary pose')
        from target_selection import selected_proposal
        proposal=selected_proposal(result)
        surface=proposal.get('candidate_base_xyz_mm')
        if surface is None:raise ValueError('Base XYZ missing')
        config=json.loads(CONTROL_PROFILE.read_text(encoding='utf-8'))
        reference=np.radians(config['reference_joints_deg'])
        reference_r,_=self.wrist.fk(reference)
        horizontal=np.cross([0.,0.,1.],reference_r[:,2])
        if np.linalg.norm(horizontal)<.05:raise ValueError('Boot tool axis vertical: horizontal jaw direction needs explicit measurement')
        horizontal/=np.linalg.norm(horizontal)
        local=reference_r.T@horizontal
        calibration=json.loads(active_report().read_text(encoding='utf-8'))
        hand=np.asarray(calibration['wrist_from_depth_mm'])
        camera_r,_=self.wrist.camera_pose(joints,(hand[:3,:3],hand[:3,3]))
        # Display is rotated 180deg: display-left is native depth camera +X.
        self.display_left_base=camera_r[:,0].copy()
        bias=json.loads((Path(__file__).resolve().parent/'camera_profiles/temporary-grasp-bias.json').read_text(encoding='utf-8'))
        if bias['enabled'] and TCP_SPEC.get('physically_validated'):
            raise ValueError('TCP calibrated: disable temporary-grasp-bias.json before using calibrated TCP')
        self.manual_offset_base=self.display_left_base*(float(bias['display_left_mm']) if bias['enabled'] else 0.)
        self.extra_down_mm=float(bias.get('descent_extra_down_mm',0)) if bias['enabled'] else 0.
        geometry=proposal['geometry']
        closing=camera_r@np.asarray(geometry['camera_closing_axis'])
        # Cube centre is half the measured height below its top face.
        # Current isolated-box model assumes support is horizontal in base.
        self.grasp_clearance=grasp_clearance_mm(geometry['height_above_support_mm'],self.extra_down_mm)
        self.plan=hover_plan(self.wrist,joints,np.asarray(surface)+self.manual_offset_base,self.hover_mm,closing_axis_base=closing,closing_axis_local=local)
        self.plan['uncompensated_surface_base_mm']=np.asarray(surface).tolist()
        self.plan['selected_target_id']=result.get('selected_target_id')
        self.plan['manual_offset_base_mm']=self.manual_offset_base.tolist()
        self.plan['grasp_center_base_mm']=(np.asarray(self.plan['surface_base_mm'])+[0,0,self.grasp_clearance]).tolist()
        self.status='PLANNED: G move to hover | X cancel'
        print('HOVER PLAN',json.dumps(self.plan,indent=2),flush=True)

    def start_auto(self,result,joints,state):
        if self.auto_phase:raise ValueError('AUTO is already running; X cancels before another A')
        config=json.loads(CONTROL_PROFILE.read_text(encoding='utf-8'))
        closed=config.get('closed_angle_deg')
        if not isinstance(closed,(float,int)) or isinstance(closed,bool) or not np.isfinite(closed):
            raise ValueError('Auto grasp needs measured closed_angle_deg in camera_profiles/gripper-control.json; C/O can measure it first')
        if not config['open_reference_deg']<closed<=4095*360/4096:raise ValueError('Invalid measured closing angle')
        if not isinstance(state.get('g'),(float,int)) or not np.isfinite(state['g']):raise ValueError('ID17 feedback unavailable')
        if joints is None or result.get('pose_error'):raise ValueError('A: wait for stable joint feedback')
        from target_selection import selected_proposal
        selected_proposal(result)
        if self.active:self.hold(state)
        self.prepare(result,joints)
        contact=config.get('contact_angle_deg',closed)
        if not isinstance(contact,(int,float)) or not np.isfinite(contact) or not config['open_reference_deg']<contact<=closed:
            raise ValueError('Invalid recorded contact angle')
        self.auto_closed_deg=float(closed);self.auto_contact_deg=float(contact);self.auto_phase='hover'
        try:
            if abs(np.degrees(state['g'])-config['open_reference_deg'])>5:
                self.auto_open_deg=float(config['open_reference_deg']);self.auto_close_samples=0
                self.set_gripper(state,False);self.auto_phase='opening'
                self.status=f'AUTO opening: measured {np.degrees(state["g"]):.1f} -> {self.auto_open_deg:g}deg; arm waiting'
            else:self.execute(state)
        except Exception:self.auto_phase=None;raise

    def advance_auto(self,state):
        if not self.auto_phase or self.active:return
        if state['v']<700:
            self.auto_phase=None;self.status='AUTO stopped: servo power unavailable';return
        if self.auto_phase=='opening':
            angle=float(np.degrees(state.get('g',np.nan)))
            self.auto_close_samples=self.auto_close_samples+1 if np.isfinite(angle) and abs(angle-self.auto_open_deg)<=5 else 0
            self.status=f'AUTO opening: measured {angle:.1f} / target {self.auto_open_deg:g}deg | arm waiting | X cancels'
            if self.auto_close_samples>=3:
                self.gripper_moving=False;self.auto_phase='hover';self.execute(state)
        elif self.auto_phase=='home_retract':
            self._retract(state,return_base=True);self.auto_phase='home'
        elif self.auto_phase=='home':
            self.auto_phase=None;self.status='HOME reached near startup | gripper held | P plan again'
            self.report['outcome']='home_finished_gripper_preserved';self.flush()
            self.plan=None
        elif self.auto_phase=='hover' and self.hover_reached:
            self.auto_phase='descent';self.descend(state)
        elif self.auto_phase=='descent' and self.plan.get('phase')=='descent' and not self.descent_queue:
            command=dict(T=101,joint=6,rad=float(np.radians(self.auto_closed_deg)),spd=40,acc=5)
            self.arm.send(command);self.gripper_moving=True
            self.report.setdefault('gripper_commands',[]).append(command);self.flush()
            self.auto_phase='closing';self.auto_close_samples=0
            self.status=f'AUTO closing to measured {self.auto_closed_deg:g}deg | X cancel'
        elif self.auto_phase=='closing':
            if not isinstance(state.get('g'),(int,float)):raise ValueError('ID17 feedback lost during automatic close')
            self.auto_close_samples=self.auto_close_samples+1 if abs(np.degrees(state['g'])-self.auto_contact_deg)<2 else 0
            if self.auto_close_samples>=3:
                # Returning to the carrying posture performs the lift itself.
                # A separate Cartesian lift could stall on servo following error
                # and prevent reaching the retract/base-transfer stages at all.
                self.gripper_moving=False
                self._retract(state);self.auto_phase='retract'
        elif self.auto_phase=='lift':
            self._retract(state);self.auto_phase='retract'
        elif self.auto_phase=='retract':
            if (ROOT/'camera_profiles/unload-route.json').is_file():self.start_unload(state)
            else:
                self.rear.start(state);self.auto_phase='rear_transfer'
                self.status=self.rear.status
        elif self.auto_phase=='unload':
            self.unload_index+=1
            if not self.unload_continuous:
                self.auto_phase=None
                self.status=f'UNLOAD point {self.unload_index} reached | N next | O release manually'
            elif self.unload_index<len(self.unload_steps['points']):self.execute_unload_point(state)
            else:
                target=float(self.unload_steps['release_deg'])
                command=dict(T=101,joint=6,rad=float(np.radians(target)),spd=SPEEDS['gripper_release'],acc=5)
                self.arm.send(command);self.gripper_moving=True;self.gripper_last_target=target
                self.report.setdefault('gripper_commands',[]).append(command);self.flush()
                self.auto_phase='unload_release';self.auto_close_samples=0
                self.status='UNLOAD final point reached; releasing'
        elif self.auto_phase=='unload_release':
            self.auto_close_samples=self.auto_close_samples+1 if abs(np.degrees(state.get('g',np.nan))-self.unload_steps['release_deg'])<2 else 0
            if self.auto_close_samples>=3:
                self.gripper_moving=False
                self.report['outcome']='taught_route_release_finished_grasp_unverified';self.flush()
                self.start_unload_return(state,continuous=True)
        elif self.auto_phase=='unload_return':
            self.unload_index-=1
            if self.unload_index<0:self.finish_unload_return()
            elif self.unload_continuous:self.execute_unload_point(state)
            else:
                self.auto_phase=None
                self.status=f'RETURN point reached | K next (point {self.unload_index+1}) | gripper open'
        elif self.auto_phase=='rear_transfer' and not self.rear.phase:
            self.auto_phase=None
            self.report['rear_transfer_events']=self.rear.events
            self.report['outcome']='auto_rear_release_finished_grasp_unverified';self.flush()
            self.status=self.rear.status

    def execute(self,state,descent=False):
        if self.active or self.plan is None:raise ValueError('Press P to plan first')
        if state['v']<700:raise ValueError('Servo power unavailable')
        actual=np.array([state[k] for k in FIELDS])
        # Descent was solved as one continuous chain from measured feedback.
        # Re-solving each queued segment can switch to a different IK branch.
        uncompensated=self.plan.get('uncompensated_surface_base_mm',self.plan['surface_base_mm'])
        if not descent:
            self.plan=hover_plan(self.wrist,actual,self.plan['surface_base_mm'],self.plan.get('clearance_mm',80.),
                                 self.plan.get('closing_axis_base'),self.plan.get('closing_axis_local'))
        self.plan['uncompensated_surface_base_mm']=uncompensated
        self.plan['manual_offset_base_mm']=self.manual_offset_base.tolist()
        self.plan['phase']='descent' if descent else 'hover'
        if not descent:self.hover_reached=False
        self.directory=ROOT/'approach_steps'/f'live-{time.time_ns()}'
        self.directory.mkdir(parents=True)
        self.report=dict(plan=self.plan,commands=[],samples=[],outcome='moving',physical_accuracy_verified=False)
        self.active=True
        self.motion_speed=self.select_speed(actual)
        self.plan['speed_mode_steps_s']=self.motion_speed
        self.settle_samples=[]
        self.flush()
        try:
            for i,(angle,speed) in enumerate(zip(self.plan['goal_joints_rad'],self.plan['speed_steps_s'])):
                if abs(angle-actual[i])<np.radians(.05):continue
                scaled_speed=max(SPEEDS['precision'],round(speed/max(self.plan['speed_steps_s'])*self.motion_speed))
                command=dict(T=101,joint=i+1,rad=angle,spd=scaled_speed,acc=5)
                self.arm.send(command);self.report['commands'].append(command)
            self.status='MOVING '+self.plan['phase']+' | X hold | Q hold + exit'
            self.flush()
        except Exception:
            self.hold(state)
            raise

    def descend(self,state):
        if self.plan is None:
            raise ValueError('P plan a target first; D uses the current measured pose')
        # Pre-solve the entire vertical reference path before sending commands.
        # Approximate TCP and box centre; no automatic jaw closure.
        seed=np.array([state[k] for k in FIELDS])
        raw_surface=np.asarray(self.plan.get('uncompensated_surface_base_mm',self.plan['surface_base_mm']))
        surface=(raw_surface+self.manual_offset_base).tolist()
        start=tcp_position(self.wrist,seed)
        queue=[]
        if self.grasp_clearance is None:raise ValueError('Cube height missing; P plan again')
        current_clearance=float(start[2]-surface[2])
        if np.linalg.norm(start[:2]-np.asarray(surface)[:2])>10:
            current_clearance=max(self.hover_mm,current_clearance)
            alignment=hover_plan(self.wrist,seed,surface,current_clearance,self.plan.get('closing_axis_base'),self.plan.get('closing_axis_local'))
            queue.append(alignment);seed=np.array(alignment['goal_joints_rad'])
        count=max(1,int(np.ceil(abs(current_clearance-self.grasp_clearance)/15)))
        for clearance in np.linspace(current_clearance,self.grasp_clearance,count+1)[1:]:
            plan=hover_plan(self.wrist,seed,surface,clearance,self.plan.get('closing_axis_base'),self.plan.get('closing_axis_local'))
            queue.append(plan);seed=np.array(plan['goal_joints_rad'])
        if self.active:
            self.report['outcome']='superseded_by_user_D_from_live_pose';self.active=False;self.flush()
        for plan in queue:plan['uncompensated_surface_base_mm']=raw_surface.tolist()
        self.descent_queue=queue
        self.plan=self.descent_queue.pop(0);self.hover_reached=False
        self.execute(state,descent=True)

    def nudge_lateral(self,state,left=True):
        if self.plan is None or self.display_left_base is None:raise ValueError('P plan first; J/L adjust in the locked camera view')
        if self.active and self.plan.get('phase')=='descent':raise ValueError('X hold and P replan before lateral adjustment during descent')
        offset=self.manual_offset_base+(5 if left else -5)*self.display_left_base
        raw=np.asarray(self.plan.get('uncompensated_surface_base_mm',self.plan['surface_base_mm']))
        seed=np.array([state[k] for k in FIELDS]);position=tcp_position(self.wrist,seed)
        # Preserve current height relative to the compensated target; only lateral
        # position is nudged in the locked view, not a new perception estimate.
        clearance=float(position[2]-(raw+offset)[2])
        plan=hover_plan(self.wrist,seed,raw+offset,clearance,self.plan.get('closing_axis_base'),self.plan.get('closing_axis_local'))
        if self.active:
            self.report['outcome']='superseded_by_manual_lateral';self.active=False;self.flush()
        self.manual_offset_base=offset;plan['uncompensated_surface_base_mm']=raw.tolist()
        self.plan=plan;self.descent_queue=[];self.execute(state)
        print('MANUAL view-left correction mm',float(offset@self.display_left_base),'; temporary for this target, NOT TCP calibration',flush=True)

    def select_speed(self,q):
        position=tcp_position(self.wrist,q)
        distance=float(np.linalg.norm(position-np.asarray(self.plan['hover_base_mm'])))
        grasp_distance=None
        if self.plan['phase']=='descent' and self.grasp_clearance is not None:
            centre=np.asarray(self.plan['surface_base_mm'])+[0,0,self.grasp_clearance]
            # Both ends of this segment must be outside the precision zone.
            grasp_distance=min(np.linalg.norm(position-centre),np.linalg.norm(np.asarray(self.plan['hover_base_mm'])-centre))
        if self.plan['phase']=='unload':return SPEEDS['precision'] if distance<=30 else self.plan.get('route_speed',80)
        return phase_speed(self.plan['phase'],distance,grasp_distance)

    def observe(self,state):
        q=np.array([state[k] for k in FIELDS]);position=tcp_position(self.wrist,q)
        if self.active:
            error=float(np.linalg.norm(position-np.array(self.plan['hover_base_mm'])))
            speed=self.select_speed(q)
            # Decelerate once near the endpoint; never accelerate mid-segment
            # or overwrite a corrected servo command with the original goal.
            current_speed=getattr(self,'motion_speed',None)
            if current_speed is not None and speed<current_speed and not self.report.get('tracking_corrections'):
                indices=[0] if self.plan['phase']=='home' else range(1,5) if self.plan['phase']=='retract' else range(5)
                for i in indices:
                    if abs(self.plan['goal_joints_rad'][i]-q[i])<np.radians(.05):continue
                    payload=dict(T=101,joint=i+1,rad=float(self.plan['goal_joints_rad'][i]),spd=speed,acc=5)
                    self.arm.send(payload);self.report['commands'].append(payload)
                self.motion_speed=speed
                self.report.setdefault('speed_changes',[]).append(dict(speed_steps_s=speed,distance_mm=error))
            angle_error=float(np.max(np.abs(np.degrees(q-np.array(self.plan['goal_joints_rad'])))))
            self.settle_samples.append((time.monotonic(),q.copy()))
            self.settle_samples=self.settle_samples[-4:]
            stationary=(len(self.settle_samples)==4 and
                        self.settle_samples[-1][0]-self.settle_samples[0][0]>=.6 and
                        np.max(np.degrees(np.ptp([sample[1] for sample in self.settle_samples],axis=0)))<.3)
            delta=position-np.asarray(self.plan['hover_base_mm'])
            self.report['samples'].append(dict(time_ns=time.time_ns(),feedback=state,
                                              reference_base_mm=position.tolist(),position_error_mm=error,
                                              error_xyz_mm=(position-np.array(self.plan['hover_base_mm'])).tolist()))
            self.status=f'MOVING | model remaining {error:.1f}mm | joint error {angle_error:.1f}deg | speed {getattr(self,"motion_speed",None)} | X hold'
            if state['v']<700:
                self.report['outcome']='supply_unavailable';self.active=False
                self.status='POWER LOST: no more motion commands'
            elif error<10 and angle_error<2:
                completion='feedback_near'
                self.report['outcome']=completion+'_'+self.plan['phase'];self.active=False
                self.report['phase_completion']=dict(reason=completion,position_error_mm=error,joint_error_deg=angle_error,
                                                     error_xyz_mm=delta.tolist(),physical_accuracy_verified=False)
                print('PHASE',self.plan['phase'],completion,'residual XYZ mm',np.round(delta,1).tolist(),flush=True)
                self.hover_reached=self.plan['phase']=='hover'
                if self.plan['phase']=='unload':
                    self.status='UNLOAD point reached'
                elif self.plan['phase']=='retract':
                    self.status='RETRACT finished | B rear turn + release | gripper held'
                else:
                    self.status='HOVER reached | D cube-centre approach | C close O open' if self.hover_reached else 'NEAR cube centre by model | C close O open | X hold'
                self.flush()
                if self.descent_queue:
                    self.plan=self.descent_queue.pop(0)
                    self.execute(state,descent=True)
            elif self.report.get('tracking_blocked'):
                self.status=f'NOT REACHED | residual {error:.1f}mm, {angle_error:.1f}deg | B/U/H take over | X hold'
            elif stationary:
                self.correct_tracking(state)
            if not self.active or time.monotonic()-self.last_log>1:self.flush()
        self.advance_auto(state)
        if self.rear.phase:
            self.rear.update(state);self.status=self.rear.status
        return position

    def correct_tracking(self,state):
        """Finite joint feedback correction; keep original IK goal unchanged."""
        q=np.asarray([state[k] for k in FIELDS],float)
        goal=np.asarray(self.plan['goal_joints_rad'],float)
        corrections=self.report.setdefault('tracking_corrections',[])
        if len(corrections)>=6:
            self.report['tracking_blocked']=True
            self.status='NOT REACHED after correction | B/U/H take over | X hold'
            self.flush();return
        bounds=np.radians(ROUTE_LIMITS).copy() if self.plan['phase']=='unload' else raw_limits(self.wrist,ROOT/'.vendor/roarm_ws')
        firmware=np.radians([[-180,179.9],[-90,90],[0,179.9],[-90,90],[-179.9,180]])
        bounds[:,0]=np.maximum(bounds[:,0],firmware[:,0]);bounds[:,1]=np.minimum(bounds[:,1],firmware[:,1])
        previous=np.asarray(corrections[-1]['command_joints_rad'] if corrections else goal,float)
        # Estimate static servo following offset using the measured angle.
        # Each settled correction adds at most 2deg, total at most 8deg.
        command=np.clip(previous+np.clip(goal-q,-np.radians(2),np.radians(2)),
                        np.maximum(bounds[:,0],goal-np.radians(8)),
                        np.minimum(bounds[:,1],goal+np.radians(8)))
        indices=[0] if self.plan['phase']=='home' else range(1,5) if self.plan['phase']=='retract' else range(5)
        sent=[]
        for i in indices:
            if abs(goal[i]-q[i])<np.radians(.3) or abs(command[i]-previous[i])<np.radians(.05):continue
            payload=dict(T=101,joint=i+1,rad=float(command[i]),spd=SPEEDS['precision'],acc=5)
            self.arm.send(payload);sent.append(payload)
        corrections.append(dict(feedback_joints_rad=q.tolist(),goal_joints_rad=goal.tolist(),
                                command_joints_rad=command.tolist(),commands=sent))
        self.settle_samples=[]
        if not sent:self.report['tracking_blocked']=True
        self.status=f'CORRECTING {self.plan["phase"]} | pass {len(corrections)} | original target retained | X hold'
        self.flush()

    def hold(self,state):
        self.auto_phase=None
        self.unload_steps=None;self.unload_index=0;self.unload_continuous=False;self.unload_returning=False
        self.rear.stop(state)
        if self.gripper_moving and isinstance(state.get('g'),(int,float)):
            self.arm.send(dict(T=101,joint=6,rad=float(state['g']),spd=40,acc=5))
            self.gripper_moving=False
        if self.active:
            for i,key in enumerate(FIELDS):
                self.arm.send(dict(T=101,joint=i+1,rad=float(state[key]),spd=80,acc=5))
            self.report['outcome']='user_hold';self.active=False;self.flush()
        self.plan=None;self.hover_reached=False;self.descent_queue=[];self.status='HELD at feedback angles | P plan again'

    def flush(self):
        if self.report is not None:
            (self.directory/'report.json').write_text(json.dumps(self.report,indent=2),encoding='utf-8')
            self.last_log=time.monotonic()
