from active_handeye import active_report, fingerprint
"""Offline camera/base coordinate audit; consistency is not absolute accuracy."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import argparse,json,time
import numpy as np
from wrist_pose import Wrist


def summarize(rows,reference=None):
    xyz=np.asarray([r['base_surface_mm'] for r in rows],float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():raise ValueError('No finite base positions')
    mean=xyz.mean(0)
    maximum=float(np.max(np.linalg.norm(xyz[:,None]-xyz[None,:],axis=2)))
    distinct=[]
    for row in rows:
        q=np.asarray(row['joints_rad'],float)
        if q.shape!=(5,) or not np.isfinite(q).all():raise ValueError('Need five measured joint angles')
        if all(np.max(np.abs((q-p+np.pi)%(2*np.pi)-np.pi))>=.05 for p in distinct):distinct.append(q)
    result=dict(samples=len(rows),distinct_joint_poses=len(distinct),mean_base_surface_mm=mean.tolist(),
                max_pair_distance_mm=maximum,absolute_accuracy_verified=False,
                independent_pose_check_available=len(distinct)>=3,
                motion_enabled=False,scope='Fixed-object surface-centre consistency, not a grasp-error bound')
    if reference is not None:
        reference=np.asarray(reference,float)
        if reference.shape!=(3,) or not np.isfinite(reference).all():raise ValueError('Invalid surveyed base reference')
        errors=np.linalg.norm(xyz-reference,axis=1)
        result.update(user_reference_mm=reference.tolist(),reference_errors_mm=errors.tolist(),
                      max_reference_error_mm=float(errors.max()),
                      reference_note='User must survey the SAME top-face centre in the robot model base frame')
    return result


def recording_paths(root,session=None,all_sessions=False):
    if session is not None:
        return sorted(Path(session).glob('*/result.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    paths=sorted(Path(root).glob('preset-*/*/result.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    if not all_sessions and paths:
        latest_session=paths[0].parent.parent
        paths=[p for p in paths if p.parent.parent==latest_session]
    return paths


def observations(root,policy_sha,session=None,all_sessions=False):
    calibration=json.loads((active_report()).read_text())
    mapping=calibration['joint_mapping'];wrist=Wrist(ROOT/'.vendor/roarm_ws',mapping['joint_signs'],mapping['joint_offsets_rad'])
    hand=np.asarray(calibration['wrist_from_depth_mm']);rows=[];excluded=[]
    for path in recording_paths(root,session,all_sessions):
        try:
            result=json.loads(path.read_text(encoding='utf-8'))
            if result.get('backend')!='box_geometry' or result.get('preset_sha256')!=policy_sha:continue
            if result.get('handeye_sha256')!=fingerprint():
                raise ValueError('Observation uses older/unspecified hand-eye; capture a new view with locate')
            if result.get('pose_error'):raise ValueError('Arm moving or feedback power invalid')
            targets=len(result.get('native_detections',[]));fits=len(result.get('proposals',[]))
            if targets!=1 or fits!=1:
                reason='; '.join(result.get('rejected',[])) or ('YOLO did not detect a target' if not targets else 'Need one unambiguous target and geometric fit')
                raise ValueError(f'targets={targets}, fits={fits}: {reason}')
            with np.load(path.parent/'rgbd.npz',allow_pickle=False) as saved:q=saved['joints_rad'].copy()
            point=np.asarray(result['proposals'][0]['geometry']['camera_surface_xyz_mm'],float)
            if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Invalid fitted camera point')
            r,t=wrist.camera_pose(q,(hand[:3,:3],hand[:3,3]));base=r@point+t
            rows.append(dict(capture=str(path.parent),saved_time_ns=path.stat().st_mtime_ns,
                             camera_surface_mm=point.tolist(),base_surface_mm=base.tolist(),
                             joints_rad=q.tolist(),joints_deg=np.degrees(q).tolist(),
                             center_source='Fitted box top-face centre; not a measured jaw TCP',
                             observation_reference=result.get('observation_pose')))
        except (OSError,ValueError,KeyError) as exc:excluded.append(dict(capture=str(path.parent),reason=str(exc)))
    return rows,excluded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed-target',action='store_true',help='Confirm that the selected recent observations contain the SAME unmoved block and fixed base')
    p.add_argument('--count',type=int,default=3)
    p.add_argument('--reference-mm',type=float,nargs=3,metavar=('X','Y','Z'))
    p.add_argument('--session',type=Path,help='Select one recording session; default: newest session')
    p.add_argument('--all-sessions',action='store_true',help='Explicitly compare across sessions only if target and base remained fixed')
    a=p.parse_args()
    if a.count<1:p.error('--count must be positive')
    if a.reference_mm and not a.fixed_target:p.error('--reference-mm requires --fixed-target')
    if a.all_sessions and (not a.fixed_target or a.session):p.error('--all-sessions requires --fixed-target and no --session')
    sha=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))['sha256']
    rows,excluded=observations(ROOT/'field_recordings',sha,a.session,a.all_sessions)
    for entry in excluded[:5]:
        print('EXCLUDED',entry['capture'],':',entry['reason'])
    if not rows:p.exit(2,'No valid single-target observation. Run python debug.py locate ; keep one block visible, wait for stationary joints and S save.\n')
    chosen=rows[:a.count if a.fixed_target else 1]
    if a.fixed_target and len(chosen)<a.count:
        print(f'Only {len(chosen)}/{a.count} valid observations; saved captures can be excluded for missing geometry or joint feedback.')
    report=dict(observations=chosen,excluded=excluded,fixed_target_confirmed_by_user=a.fixed_target,
                handeye_report=str(active_report()),handeye_sha256=fingerprint(),
                cross_session_comparison=a.all_sessions,
                handeye_acceptance='Active hand-eye candidate: consistency residual is not a physical grasp-error bound',
                motion_enabled=False)
    if a.fixed_target:report['comparison']=summarize(chosen,a.reference_mm)
    for row in chosen:
        print('CAPTURE',row['capture']);print('JOINT deg',row['joints_deg'])
        print('CAMERA top centre mm',row['camera_surface_mm']);print('BASE top centre mm',row['base_surface_mm'])
    if a.fixed_target:
        summary=report['comparison'];print('PAIR SPREAD mm',summary['max_pair_distance_mm'],'DISTINCT POSES',summary['distinct_joint_poses'])
        if not summary['independent_pose_check_available']:print('Need at least 3 distinct observation poses; repeated S at one pose is not an independent pose check.')
    out=ROOT/'localization_checks'/str(time.time_ns());out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('REPORT',out/'report.json');print('No motion; no physical accuracy acceptance claimed.')

if __name__=='__main__':main()
