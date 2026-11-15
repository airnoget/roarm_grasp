from active_handeye import active_report, fingerprint
"""Offline hover plan from a saved Gemini geometry observation; never sends commands."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import argparse,hashlib,json,time
import numpy as np
from wrist_pose import Wrist
from wrist_ik import raw_limits,solve_position


def latest_capture(root):
    candidates=list(Path(root).glob('preset-*/*/result.json'))
    if not candidates:raise ValueError('No saved observation. Run python debug.py locate ; S save, Q exit.')
    path=max(candidates,key=lambda p:p.stat().st_mtime_ns)
    # A failed new observation invalidates use of an earlier target. Never
    # silently fall back to another scene, pose, object or stale observation.
    result=json.loads(path.read_text(encoding='utf-8'))
    prefix=f'Latest capture {path.parent.name}: '
    if result.get('backend')!='box_geometry':raise ValueError(prefix+'not a geometry observation; run python debug.py locate')
    if result.get('pose_error'):raise ValueError(prefix+str(result['pose_error']))
    if not result.get('proposals'):
        reason='; '.join(result.get('rejected',[])) or 'no detected target or usable geometric fit'
        raise ValueError(prefix+reason+'. No movement requested; no older capture selected.')
    try:
        with np.load(path.parent/'rgbd.npz',allow_pickle=False) as recording:q=recording['joints_rad']
    except KeyError:raise ValueError(prefix+'missing measured joints; capture with python debug.py locate')
    if q.shape!=(5,) or not np.isfinite(q).all():raise ValueError(prefix+'invalid measured joints')
    return path.parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path,nargs='?',help='Optional saved directory; default: latest geometry observation with joints')
    p.add_argument('--target',type=int,default=1,help='One-based geometric proposal index')
    p.add_argument('--clearance',type=float,default=80)
    p.add_argument('--output-root',type=Path,default=ROOT/'approach_plans')
    reference=p.add_mutually_exclusive_group(required=True)
    reference.add_argument('--link5',action='store_true',help='Explicit wrist-axis reference; NOT gripper centre')
    reference.add_argument('--tcp',type=Path,help='JSON with measured link5_tcp_position_mm')
    a=p.parse_args()
    if not 50<=a.clearance<=150:p.error('Hover clearance must be 50..150 mm')
    try:capture=(a.capture.resolve() if a.capture else latest_capture(ROOT/'field_recordings'))
    except ValueError as exc:p.exit(2,str(exc)+'\n')
    if not (capture/'result.json').is_file() or not (capture/'rgbd.npz').is_file():
        p.exit(2,'Invalid capture directory. Omit the directory to select the latest saved observation: python plan_hover.py --link5\n')
    result=json.loads((capture/'result.json').read_text(encoding='utf-8'))
    policy=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))
    if result.get('preset_sha256')!=policy['sha256']:raise ValueError('Observation camera preset differs from the calibrated Gemini configuration')
    if result.get('backend')!='box_geometry' or result.get('pose_error'):raise ValueError('Need a stationary geometric observation')
    with np.load(capture/'rgbd.npz',allow_pickle=False) as recording:
        if 'joints_rad' not in recording:raise ValueError('No measured joints; capture with --geometry --port COM6 and S')
        q=recording['joints_rad'].copy()
    if not 1<=a.target<=len(result['proposals']):raise ValueError('Target index outside proposal list')
    proposal=result['proposals'][a.target-1]
    handeye=active_report()
    calibration=json.loads(handeye.read_text());mapping=calibration['joint_mapping']
    wrist=Wrist(ROOT/'.vendor/roarm_ws',mapping['joint_signs'],mapping['joint_offsets_rad'])
    transform=np.array(calibration['wrist_from_depth_mm']);r,t=wrist.camera_pose(q,(transform[:3,:3],transform[:3,3]))
    surface=r@np.array(proposal['geometry']['camera_surface_xyz_mm'])+t
    above=surface+[0,0,a.clearance]
    offset=np.zeros(3) if a.link5 else np.array(json.loads(a.tcp.read_text())['link5_tcp_position_mm'],float)
    solution=solve_position(wrist,above,q,raw_limits(wrist,ROOT/'.vendor/roarm_ws'),offset)
    delta=np.max(np.abs(np.array(solution['joints_rad'])-q))
    count=max(2,int(np.ceil(delta/.02))+1)
    joint_path=np.linspace(q,solution['joints_rad'],count).tolist()
    report=dict(source=str(capture),surface_base_mm=surface.tolist(),hover_reference_base_mm=above.tolist(),
                reference='link5 wrist axis; not gripper centre' if a.link5 else 'user-specified TCP',
                tcp_link5_mm=offset.tolist(),start_joints_rad=q.tolist(),solution=solution,
                proposed_joint_path=joint_path,handeye_sha256=hashlib.sha256(handeye.read_bytes()).hexdigest(),
                blockers=['Camera hand-eye absolute accuracy not accepted','Physical joint mapping not verified',
                          'Current collision geometry not commissioned','Tool orientation unconstrained',
                          'No current-pose/freshness check against the live robot'],
                motion_enabled=False,scope='Offline plan only, no serial/camera connection or motion commands')
    out=a.output_root/str(time.time_ns());out.mkdir(parents=True)
    (out/'plan.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('PLAN',out/'plan.json');print('Reference',report['reference'])
    print('Hover XYZ mm',above.tolist(),'model residual mm',solution['position_residual_mm'])
    print('NOT EXECUTED. Physical positioning and collision checks remain unverified.')

if __name__=='__main__':main()
