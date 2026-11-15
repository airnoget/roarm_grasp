from active_handeye import active_report, fingerprint
"""Single bounded commissioning step, link5 wrist-axis reference, no gripper."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import argparse
import numpy as np
from plan_hover import latest_capture
from wrist_pose import Wrist
from wrist_ik import raw_limits,solve_position


def bounded_step(wrist,q,target,limits):
    q=np.asarray(q,float);target=np.asarray(target,float)
    _,start=wrist.fk(q);distance=np.linalg.norm(target-start)
    if distance<2:raise ValueError('Already near hover reference; no step requested')
    direction=(target-start)/distance;step=min(5.,distance)
    for _ in range(8):
        solution=solve_position(wrist,start+direction*step,q,limits,tolerance_mm=.15)
        goal=np.asarray(solution['joints_rad']);_,end=wrist.fk(goal)
        if np.max(np.abs(goal-q))<=np.radians(1) and np.linalg.norm(end-start)<=5.:
            return dict(start_reference_mm=start.tolist(),target_hover_mm=target.tolist(),
                        next_reference_mm=end.tolist(),model_step_mm=float(np.linalg.norm(end-start)),
                        start_joints_rad=q.tolist(),goal_joints_rad=goal.tolist(),
                        max_joint_change_deg=float(np.degrees(np.max(np.abs(goal-q)))))
        step*=.5
    raise ValueError('No bounded commissioning step found')


def fresh(capture):
    age=time.time()-(capture/'result.json').stat().st_mtime
    if not 0<=age<=120:raise ValueError('Observation older than 120s. Run python debug.py locate, S save while stationary, Q exit, then retry.')


def claim_capture(capture,log_directory):
    """Atomic one-shot claim just before sending any motion command."""
    marker=capture/'hover-step-used.json'
    try:
        with marker.open('x',encoding='utf-8') as stream:
            json.dump(dict(time_ns=time.time_ns(),report=str(log_directory/'report.json')),stream)
    except FileExistsError:
        raise ValueError('This observation was already used for a motion attempt. Capture a NEW stationary frame before another step.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--port',default='COM6')
    a=p.parse_args();capture=latest_capture(ROOT/'field_recordings');fresh(capture)
    if a.execute and (capture/'hover-step-used.json').exists():raise ValueError('Observation already used; save a NEW stationary frame before another step')
    result=json.loads((capture/'result.json').read_text(encoding='utf-8'))
    if len(result.get('native_detections',[]))!=1 or len(result['proposals'])!=1:raise ValueError('Need exactly one target and geometric fit')
    if result.get('pose_error'):raise ValueError('Source pose was not stationary')
    track=result['proposals'][0].get('tracking',{})
    if not track.get('camera_relative_stable'):raise ValueError('Need 5 consistent frames before saving')
    policy=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))
    if result.get('preset_sha256')!=policy['sha256']:raise ValueError('Wrong camera preset')
    with np.load(capture/'rgbd.npz',allow_pickle=False) as saved:q=saved['joints_rad'].copy()
    calibration=json.loads((active_report()).read_text())
    mapping=calibration['joint_mapping'];w=Wrist(ROOT/'.vendor/roarm_ws',mapping['joint_signs'],mapping['joint_offsets_rad'])
    hand=np.array(calibration['wrist_from_depth_mm']);r,t=w.camera_pose(q,(hand[:3,:3],hand[:3,3]))
    surface=r@np.array(result['proposals'][0]['geometry']['camera_surface_xyz_mm'])+t
    limits=raw_limits(w,ROOT/'.vendor/roarm_ws')
    firmware_limits=np.radians([[-90,90],[-90,90],[0,179.9],[-90,90],[-179.9,180]])
    limits[:,0]=np.maximum(limits[:,0],firmware_limits[:,0]);limits[:,1]=np.minimum(limits[:,1],firmware_limits[:,1])
    plan=bounded_step(w,q,surface+[0,0,80],limits)
    print(json.dumps(plan,indent=2),flush=True)
    if not a.execute:print('PREVIEW ONLY. --execute performs ONE bounded wrist-axis step, not a full approach.');return
    from arm import Arm
    fields=('b','s','e','t','r','v');report=dict(plan=plan,source=str(capture),commands=[],feedback_samples=[],outcome='not_started',physical_accuracy_verified=False)
    out=ROOT/'approach_steps'/str(time.time_ns());out.mkdir(parents=True)
    try:
        print('Opening serial can reset/home the arm. Checking measured start; no gripper commands.',flush=True)
        with Arm(a.port) as arm:
            state=arm.feedback(timeout=12,required_fields=fields)
            now=np.array([state[k] for k in fields[:5]])
            if state['v']<700:raise ValueError('Servo supply unavailable')
            if np.max(np.abs((now-q+np.pi)%(2*np.pi)-np.pi))>np.radians(.7):raise ValueError('Current joints differ from capture, possibly reset/home; save a new stationary observation')
            fresh(capture)
            # A second sample prevents starting while feedback is still changing.
            again=arm.feedback(required_fields=fields)
            q2=np.array([again[k] for k in fields[:5]])
            if np.max(np.abs(q2-now))>.003:raise ValueError('Arm moving before step')
            if again['v']<700:raise ValueError('Servo supply unavailable')
            if np.max(np.abs((q2-q+np.pi)%(2*np.pi)-np.pi))>np.radians(.7):raise ValueError('Start pose departed from observation during checks')
            # Recompute the bounded move from the latest actual joint reading,
            # while keeping the target in the base frame of its capture.
            plan=bounded_step(w,q2,surface+[0,0,80],limits)
            report['plan']=plan;report['measured_start_feedback']=again
            q=q2
            goal=np.array(plan['goal_joints_rad'])
            fresh(capture);claim_capture(capture,out)
            for index,angle in enumerate(goal):
                if abs(angle-q[index])<np.radians(.05):continue
                cmd=dict(T=101,joint=index+1,rad=float(angle),spd=30,acc=5)
                arm.send(cmd);report['commands'].append(cmd);time.sleep(.08)
            started=time.monotonic();deadline=started+10;settled=0;last_print=-2
            while time.monotonic()<deadline:
                state=arm.feedback(required_fields=fields);actual=np.array([state[k] for k in fields[:5]])
                errors=np.degrees(actual-goal);elapsed=time.monotonic()-started
                report['feedback_samples'].append(dict(elapsed_s=elapsed,feedback=state,error_deg=errors.tolist()))
                report['feedback']=state
                if elapsed-last_print>=2:
                    print('Actual deg',np.round(np.degrees(actual),2).tolist(),'goal error deg',np.round(errors,2).tolist(),flush=True)
                    last_print=elapsed
                if state['v']<700:raise RuntimeError('Supply dropped during step')
                if np.max(np.abs(actual-q))>np.radians(3):raise RuntimeError('Measured joint departure exceeds commissioning bound')
                settled=settled+1 if np.max(np.abs(actual-goal))<np.radians(.4) else 0
                if settled>=3:
                    report.update(outcome='joint_feedback_reached',feedback=state)
                    print('Joint feedback reached. Independently inspect wrist movement; this does NOT prove physical XYZ accuracy.',flush=True);break
            else:
                worst=int(np.argmax(np.abs(errors)));names=('base','shoulder','elbow','wrist pitch','wrist roll')
                raise TimeoutError(f'Step did not reach tolerance: {names[worst]} actual={np.degrees(actual[worst]):.2f} deg, goal={np.degrees(goal[worst]):.2f} deg, error={errors[worst]:+.2f} deg. Feedback saved; no further targets sent')
    except Exception as exc:
        report.update(outcome='aborted',error=str(exc));raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('LOG',out/'report.json',flush=True)

if __name__=='__main__':
    try:main()
    except (ValueError,TimeoutError) as exc:raise SystemExit(str(exc))
