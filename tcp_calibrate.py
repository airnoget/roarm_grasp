"""Read-only single-contact-point TCP calibration; optional two-face legacy mode."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import numpy as np

FIELDS=('b','s','e','t','r')


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def fit_face(fit,check):
    if len(fit)<8 or len(check)<3:raise ValueError('Each face needs >=8 fit poses (S), >=3 independent check poses (V)')
    rows=fit+check
    rotations=np.asarray([row['rotation'] for row in rows],float)
    positions=np.asarray([row['position_mm'] for row in rows],float)
    joints=np.asarray([row['joints_rad'] for row in rows],float)
    if rotations.shape!=(len(rows),3,3) or positions.shape!=(len(rows),3) or joints.shape!=(len(rows),5):
        raise ValueError('Malformed pose samples')
    if not all(np.isfinite(value).all() for value in (rotations,positions,joints)):raise ValueError('Nonfinite pose samples')
    for i in range(len(rows)):
        if i and np.min(np.max(np.abs(joints[i]-joints[:i]),axis=1))<np.radians(2):
            raise ValueError('Repeated/near-identical pose; change wrist pitch AND roll, keep contact on same mark')
    n=len(fit);r=rotations[:n];p=positions[:n]
    matrix=np.concatenate((r,-np.broadcast_to(np.eye(3),r.shape)),axis=2).reshape(-1,6)
    solution,_,rank,singular=np.linalg.lstsq(matrix,-p.reshape(-1),rcond=None)
    ratio=float(singular[-1]/singular[0])
    if rank!=6 or ratio<.04:raise ValueError('Not enough orientation diversity: vary pitch and roll, not just one rotation axis')
    tip,reference=solution[:3],solution[3:]
    errors=np.linalg.norm(rotations@tip+positions-reference,axis=1)
    return dict(face_centre_link5_mm=tip.tolist(),reference_base_mm=reference.tolist(),
                singular_ratio=ratio,fit_errors_mm=errors[:n].tolist(),check_errors_mm=errors[n:].tolist(),
                fit_max_mm=float(errors[:n].max()),check_max_mm=float(errors[n:].max()),
                passed=bool(errors.max()<=5))


def solve_session(path):
    data=json.loads(path.read_text(encoding='utf-8'))
    all_rows=[row for face in data['faces'].values() for split in ('fit','check') for row in face[split]]
    if not all_rows:raise ValueError('No samples saved')
    angles=[row['gripper_rad'] for row in all_rows]
    if not np.isfinite(angles).all() or np.ptp(angles)>np.radians(1.5):
        raise ValueError('Gripper angle changed; keep cube firmly held at the same opening throughout calibration')
    faces={name:fit_face(face['fit'],face['check']) for name,face in data['faces'].items()}
    if data.get('reference')=='contact_point':
        if set(faces)!={'1'}:raise ValueError('Single-point session must contain exactly one contact point')
        face=faces['1']
        report=dict(schema=1,method='single fixed contact-point pivot fit',faces=faces,
                    reference='user_touchable_grasp_point',
                    session_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model=data['model'],
                    link5_tcp_position_mm=face['face_centre_link5_mm'],passed=face['passed'],
                    assumptions=['Same user-defined grasp point touched the same fixed reference tip',
                                 'Cube did not slip relative to jaws'],physical_absolute_accuracy_verified=False)
        output=path.parent/'report.json';write_json(output,report)
        print('TCP link5 XYZ mm:',np.round(report['link5_tcp_position_mm'],2).tolist())
        print('Fit max / check max mm:',round(face['fit_max_mm'],2),round(face['check_max_mm'],2))
        print('PASSED',report['passed'],'REPORT',output,flush=True)
        print('Single contact point; no cube half-height or bottom-face offset added. Not applied yet.')
        return report
    first=np.asarray(faces['1']['face_centre_link5_mm']);second=np.asarray(faces['2']['face_centre_link5_mm'])
    separation=float(np.linalg.norm(first-second))
    if separation<5:raise ValueError('Face centres almost coincide: use two OPPOSITE faces, not the same face twice')
    expected=data.get('cube_side_mm')
    size_error=None if expected is None else abs(separation-expected)
    report=dict(schema=1,method='midpoint of two opposite face-centre pivot fits',faces=faces,
                session_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model=data['model'],link5_tcp_position_mm=((first+second)/2).tolist(),
                fitted_cube_side_mm=separation,measured_cube_side_mm=expected,size_error_mm=size_error,
                passed=all(face['passed'] for face in faces.values()) and (size_error is None or size_error<=5),
                assumptions=['Marks are centres of two opposite faces',
                             'Cube did not slip relative to jaws',
                             'Reference tip remained fixed for all poses within each face'],
                physical_absolute_accuracy_verified=False)
    output=path.parent/'report.json';write_json(output,report)
    print('TCP link5 XYZ mm:',np.round(report['link5_tcp_position_mm'],2).tolist())
    print('Fitted face separation / cube side mm:',round(separation,2))
    for name,face in faces.items():print('FACE',name,'fit max',round(face['fit_max_mm'],2),'check max',round(face['check_max_mm'],2),'mm')
    print('PASSED',report['passed'],'REPORT',output,flush=True)
    print('Report only; active TCP and temporary bias have not changed.')
    return report


def apply_session(path):
    report=solve_session(path)
    if not report['passed']:raise ValueError('Fit/check failed; collect better contact poses before applying')
    from active_handeye import active_report
    current=json.loads(active_report().read_text(encoding='utf-8'))['joint_mapping']
    if current!=report['model']['joint_mapping']:raise ValueError('Joint mapping changed since capture')
    model=ROOT/report['model']['file']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=report['model']['sha256']:raise ValueError('Robot FK model changed since capture')
    spec=dict(link5_tcp_position_mm=report['link5_tcp_position_mm'],source=str(path.parent/'report.json'),
              method=report['method'],contact_calibrated=True,physically_validated=False,
              reference=report.get('reference','cube_centre'),
              validation_scope='Independent repeated-contact check passed; absolute positioning accuracy not verified')
    bias_path=ROOT/'camera_profiles/temporary-grasp-bias.json'
    bias=json.loads(bias_path.read_text(encoding='utf-8'))
    bias.update(enabled=False,display_left_mm=0,descent_extra_down_mm=0)
    # Clear the old provisional compensation before enabling the new TCP.
    write_json(bias_path,bias);write_json(ROOT/'camera_profiles/gripper-tcp.json',spec)
    print('APPLIED new TCP; temporary left/down bias cleared. Restart approach to use it.')


def latest_session():
    paths=list((ROOT/'calibration').glob('tcp-*/session.json'))
    if not paths:raise ValueError('No TCP session; run python debug.py tcp first')
    return max(paths,key=lambda path:path.parent.name)


def collect(port,side,reference='contact_point'):
    from active_handeye import active_report
    from wrist_pose import Wrist
    from arm import Arm
    mapping=json.loads(active_report().read_text(encoding='utf-8'))['joint_mapping']
    model=ROOT/'.vendor/roarm_ws/src/roarm_main/roarm_description/urdf/roarm_m3/roarm_m3.xacro'
    wrist=Wrist(ROOT/'.vendor/roarm_ws',mapping['joint_signs'],mapping['joint_offsets_rad'])
    path=ROOT/'calibration'/f'tcp-{time.time_ns()}'/'session.json'
    single=reference=='contact_point'
    data=dict(schema=1,reference=reference,cube_side_mm=side,faces={name:dict(fit=[],check=[]) for name in (('1',) if single else ('1','2'))},
              model=dict(file=model.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(model.read_bytes()).hexdigest(),joint_mapping=mapping))
    write_json(path,data)
    print('SESSION',path)
    print('Close approach/locate first. Opening serial may reset/home: support cube; re-grip after startup.')
    print('READ ONLY: manually align via Wi-Fi. Keep cube/gripper opening fixed. No motion commands.')
    if single:
        print('SINGLE POINT: touch your chosen grasp point to the SAME fixed reference tip in every pose.')
        print('Commands + Enter: S fit sample (8 poses) | V check sample (3 NEW poses) | F solve | Q exit')
        print('No second face or cube offset needed. Vary pitch AND roll. Keep cube fixed in jaws.')
    else:
        print('Two OPPOSITE exposed faces: mark each centre. Use a fixed pointed reference, not a flat table cross.')
        print('Commands + Enter: 1/2 select face | S fit sample | V check sample | F solve | Q exit')
        print('Per face: >=8 S poses, then >=3 NEW V poses. Vary wrist pitch and roll; do not slide cube.')
    print('Existing temporary bias is not used by this raw-joint calibration.')
    face='1'
    with Arm(port) as arm:
        arm.feedback(timeout=12,required_fields=(*FIELDS,'g','v'))
        while True:
            command=input(f'Face {face} [S {len(data["faces"][face]["fit"])}/8 V {len(data["faces"][face]["check"])}/3]> ').strip().lower()
            if command=='q':break
            if command in data['faces']:face=command;continue
            try:
                if command=='f':solve_session(path);continue
                if command not in ('s','v'):continue
                states=[arm.feedback(required_fields=(*FIELDS,'g','v')) for _ in range(3)]
                values=np.asarray([[state[k] for k in FIELDS] for state in states])
                if min(state['v'] for state in states)<700:raise ValueError('Servo supply unavailable')
                if np.max(np.ptp(values,axis=0))>np.radians(.3):raise ValueError('Arm still moving: release Wi-Fi button and re-align before saving')
                if np.ptp([state['g'] for state in states])>np.radians(.5):raise ValueError('Gripper is still moving')
                q=np.mean(values,axis=0);r,p=wrist.fk(q)
                existing=data['faces'][face]['fit']+data['faces'][face]['check']
                if existing and min(np.max(np.abs(q-row['joints_rad'])) for row in existing)<np.radians(2):
                    raise ValueError('Pose repeated: change orientation, touch the same face-centre mark again')
                g=float(np.mean([state['g'] for state in states]))
                all_rows=[row for group in data['faces'].values() for split in ('fit','check') for row in group[split]]
                if all_rows and abs(g-all_rows[0]['gripper_rad'])>np.radians(1.5):raise ValueError('Gripper opening changed; cube must not slip')
                row=dict(time_ns=time.time_ns(),joints_rad=q.tolist(),rotation=r.tolist(),position_mm=p.tolist(),gripper_rad=g)
                data['faces'][face]['fit' if command=='s' else 'check'].append(row)
                write_json(path,data);print('SAVED face',face,'fit' if command=='s' else 'check','joint deg',np.round(np.degrees(q),1).tolist())
            except (ValueError,TimeoutError) as error:print('NOT SAVED / NOT SOLVED:',error)
    print('Saved session:',path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=('collect','solve','apply'),default='collect')
    parser.add_argument('--session',type=Path,help='session.json; defaults to latest TCP collection')
    parser.add_argument('--port',default='COM6')
    parser.add_argument('--reference',choices=('contact_point','cube_centre'),default='contact_point',help='Default: directly calibrate the one touchable grasp point; cube_centre retains old two-face mode')
    parser.add_argument('--cube-side-mm',type=float,help='Optional measured side length to cross-check opposite-face separation')
    args=parser.parse_args()
    if args.cube_side_mm is not None and (not np.isfinite(args.cube_side_mm) or args.cube_side_mm<=0):parser.error('Cube side must be positive mm')
    if args.action=='collect':collect(args.port,args.cube_side_mm,args.reference)
    elif args.action=='solve':solve_session(args.session or latest_session())
    else:apply_session(args.session or latest_session())


if __name__=='__main__':main()
