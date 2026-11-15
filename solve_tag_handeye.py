"""Offline AprilTag/chessboard eye-in-hand solve; no camera or robot connection.

Transforms use column vectors, translations in mm. Reference:
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
Only fit views select the method/reference. Check views are held out.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.deps'))
import cv2
import numpy as np
from wrist_pose import Wrist


def transform(r, t):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, np.asarray(t).reshape(3)
    if (not np.isfinite(out).all() or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    return out


def mean_pose(poses):
    u, _, vt = np.linalg.svd(sum(t[:3, :3] for t in poses))
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return transform(r, np.mean([t[:3, 3] for t in poses], axis=0))


def residuals(poses, reference):
    return [dict(center_error_mm=float(np.linalg.norm(t[:3, 3]-reference[:3, 3])),
                 angle_error_deg=float(np.linalg.norm(cv2.Rodrigues(
                     reference[:3, :3].T @ t[:3, :3])[0])*180/np.pi)) for t in poses]


def independent_checks(fit, check, threshold=.005):
    """Keep raw captures intact; explicitly exclude held-out views that repeat a pose."""
    kept, excluded = [], []
    for view in check:
        references = fit+kept
        distances = [float(np.max(np.abs((view['joints']-x['joints']+np.pi)
                                         % (2*np.pi)-np.pi))) for x in references]
        nearest = int(np.argmin(distances))
        if distances[nearest] < threshold:
            excluded.append(dict(name=view['name'], reason='repeated_joint_pose',
                                 duplicate_of=references[nearest]['name'],
                                 max_joint_difference_deg=distances[nearest]*180/np.pi))
        else:
            kept.append(view)
    return kept, excluded


def read_views(session, group, wrist, settings):
    rows = []
    for file in sorted((session / group).glob('*.json')):
        meta = json.loads(file.read_text(encoding='utf-8-sig'))
        keys=('family','camera_pose_frame','pattern_size','square_size_mm') if settings['family']=='chessboard' else ('tag_id','tag_size_mm','family','camera_pose_frame')
        if any(meta.get(k) != settings[k] for k in keys):
            raise ValueError(f'Inconsistent session metadata: {file}')
        snapshot = (file.parent / meta['snapshot']).resolve()
        if snapshot.parent != file.parent.resolve():
            raise ValueError('Snapshot must be inside observation directory')
        with np.load(snapshot, allow_pickle=False) as data:
            q = data['joints_rad'].astype(float)
            k, dist = data['raw_K'].astype(float), data['raw_dist'].astype(float)
            model = str(data['raw_dist_model'].item())
            extrinsic = data['T_raw_from_depth_mm'].astype(float)
            height, width = data['raw_bgr'].shape[:2]
        if not np.allclose(q, meta['joints_rad'], atol=1e-8):
            raise ValueError(f'Joint metadata mismatch: {file}')
        if not all(np.isfinite(x).all() for x in (q, k, dist, extrinsic)):
            raise ValueError(f'Nonfinite calibration inputs: {file}')
        if 'BROWN_CONRADY' not in model and not model.endswith('.NONE'):
            raise ValueError(f'Unsupported distortion model: {model}')
        if model.endswith('.NONE'):
            dist = np.zeros(8)
        if k.shape != (3, 3) or min(k[0, 0], k[1, 1]) <= 0 or extrinsic.shape != (4, 4):
            raise ValueError('Invalid factory calibration shape')
        transform(extrinsic[:3, :3], extrinsic[:3, 3])
        if settings['family']=='chessboard':
            from chessboard import object_points
            obj=object_points(tuple(settings['pattern_size']),settings['square_size_mm'])
            pnp_flag=cv2.SOLVEPNP_IPPE
        else:
            h=settings['tag_size_mm']/2
            obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],dtype=float)
            pnp_flag=cv2.SOLVEPNP_IPPE_SQUARE
        corners = np.asarray(meta['corners_raw_px'], dtype=float)
        if (corners.shape != (len(obj), 2) or not np.isfinite(corners).all() or
                (corners < 0).any() or (corners[:, 0] >= width).any() or
                (corners[:, 1] >= height).any()):
            raise ValueError(f'Invalid tag corners: {file}')
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, corners, k, dist, flags=pnp_flag)
        choices = []
        if ok:
            for rv, tv in zip(rvecs, tvecs):
                r = cv2.Rodrigues(rv)[0]
                if np.min((obj @ r.T + tv.reshape(3))[:, 2]) <= 0:
                    continue
                predicted = cv2.projectPoints(obj, rv, tv, k, dist)[0].reshape(-1, 2)
                error = float(np.sqrt(np.mean(np.sum((predicted-corners)**2, axis=1))))
                choices.append((error, transform(r, tv)))
        if not choices:
            raise ValueError(f'Target PnP failed: {file}')
        choices.sort(key=lambda x: x[0])
        rows.append(dict(name=f'{group}/{file.stem}', joints=q,
                         base_from_wrist=transform(*wrist.fk(q)),
                         rgb_from_tag=choices[0][1], rgb_from_depth=extrinsic,
                         reprojection_px=choices[0][0],
                         alternate_reprojection_px=choices[1][0] if len(choices)>1 else None,
                         object_points=obj, corners_px=corners, K=k, distortion=dist,
                         profile=np.r_[k.ravel(), dist.ravel(), extrinsic.ravel()],
                         input_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    return rows


def solve_ax_xb(fit):
    """Linear AX=XB with column-major rotation vectorization and SO(3) projection."""
    pairs = []
    for i, left in enumerate(fit):
        for right in fit[i+1:]:
            a = np.linalg.inv(left['base_from_wrist']) @ right['base_from_wrist']
            b = left['rgb_from_tag'] @ np.linalg.inv(right['rgb_from_tag'])
            pairs.append((a, b))
    matrix = np.vstack([np.kron(np.eye(3), a[:3, :3]) -
                        np.kron(b[:3, :3].T, np.eye(3)) for a, b in pairs])
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    approximate = vt[-1].reshape(3, 3, order='F')
    if np.linalg.det(approximate) < 0:
        approximate = -approximate
    u, _, vt = np.linalg.svd(approximate)
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    lhs = np.vstack([a[:3, :3]-np.eye(3) for a, _ in pairs])
    rhs = np.concatenate([r @ b[:3, 3]-a[:3, 3] for a, b in pairs])
    t, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if rank < 3:
        raise ValueError('Hand-eye translation is not observable')
    return r, t


def solve(fit):
    motions = [cv2.Rodrigues(a['base_from_wrist'][:3, :3].T @
                            b['base_from_wrist'][:3, :3])[0].ravel()
               for i, a in enumerate(fit) for b in fit[i+1:]]
    singular = np.linalg.svd(motions, compute_uv=False)
    if singular[1] < .05 or singular[1]/max(singular[0], 1e-12) < .05:
        raise ValueError('Insufficient independent rotation axes; collect more varied poses')
    candidates = []
    methods = ('PARK', 'TSAI', 'HORAUD') if hasattr(cv2, 'calibrateHandEye') else ('AX_XB_SVD',)
    for name in methods:
        try:
            if name == 'AX_XB_SVD':
                r, t = solve_ax_xb(fit)
            else:
                r, t = cv2.calibrateHandEye(
                [x['base_from_wrist'][:3, :3] for x in fit],
                [x['base_from_wrist'][:3, 3] for x in fit],
                [x['rgb_from_tag'][:3, :3] for x in fit],
                [x['rgb_from_tag'][:3, 3] for x in fit],
                method=getattr(cv2, 'CALIB_HAND_EYE_'+name))
            hand_eye = transform(r, t)
            poses = [x['base_from_wrist'] @ hand_eye @ x['rgb_from_tag'] for x in fit]
            reference = mean_pose(poses)
            errors = residuals(poses, reference)
            score = np.mean([e['center_error_mm']**2 +
                             (.5*e['angle_error_deg'])**2 for e in errors])
            candidates.append((float(score), name, hand_eye, reference, errors))
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
    if not candidates:
        raise ValueError('All hand-eye methods failed')
    candidates.sort(key=lambda x: x[0])
    return candidates[0], singular, {x[1]: x[0] for x in candidates}


def main(expected_family=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    p.add_argument('--joint-config', type=Path, help='Optional confirmed signs/offsets/vendor JSON')
    p.add_argument('--max-mm', type=float, default=5.)
    p.add_argument('--max-deg', type=float, default=3.)
    a = p.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (a.max_mm, a.max_deg)):
        p.error('Thresholds must be positive and finite')
    session = a.session.resolve()
    output = session / f'solve-{time.time_ns()}'
    output.mkdir()
    report = dict(schema=1, passed=False, commissioned=False, session=str(session),
                  translation_unit='mm', physical_absolute_accuracy_verified=False,
                  position_error_reference='fixed target coordinate origin; not ruler-measured grasp error',
                  note='Consistency checks are not physical localization acceptance')
    try:
        settings = json.loads((session/'session.json').read_text(encoding='utf-8-sig'))
        family=settings['family']
        if expected_family and family!=expected_family:
            raise ValueError(f'Expected {expected_family} session, got {family}')
        if family not in ('tag36h11','chessboard') or settings['camera_pose_frame']!='raw_rgb':
            raise ValueError('Expected supported target and raw RGB observations')
        size=settings['square_size_mm'] if family=='chessboard' else settings['tag_size_mm']
        if not np.isfinite(size) or size<=0:
            raise ValueError('Use the measured printed edge length in mm')
        report['target_family']=family
        cfg = dict(vendor=str(ROOT/'.vendor/roarm_ws'), joint_signs=[1]*5,
                   joint_offsets_rad=[0]*5)
        if a.joint_config:
            cfg.update(json.loads(a.joint_config.read_text(encoding='utf-8-sig')))
        report['joint_mapping'] = {k: cfg[k] for k in ('joint_signs', 'joint_offsets_rad')}
        report['joint_mapping_physically_verified'] = False
        wrist = Wrist(cfg['vendor'], cfg['joint_signs'], cfg['joint_offsets_rad'])
        fit, check = [read_views(session, group, wrist, settings) for group in ('fit', 'check')]
        if len(fit) < 10 or len(check) < 5:
            raise ValueError('Need at least 10 fit and 5 held-out check views')
        if any(not np.allclose(x['profile'], fit[0]['profile'], atol=1e-6, rtol=0)
               for x in fit+check):
            raise ValueError('Camera calibration/profile changed within session')
        report['raw_check_count'] = len(check)
        check, excluded = independent_checks(fit, check)
        report['excluded_check_views'] = excluded
        report['minimum_independent_check_views'] = 3
        for view in excluded:
            print(f'EXCLUDED {view["name"]}: repeats {view["duplicate_of"]}; original files retained')
        if len(check) < 3:
            raise ValueError(f'Only {len(check)} independent check views remain; collect more new poses')
        (_, method, hand_eye, reference, fit_errors), axes, scores = solve(fit)
        check_errors = residuals([x['base_from_wrist'] @ hand_eye @ x['rgb_from_tag']
                                 for x in check], reference)
        all_errors = fit_errors+check_errors
        passed = (max(x['center_error_mm'] for x in all_errors) <= a.max_mm and
                  max(x['angle_error_deg'] for x in all_errors) <= a.max_deg and
                  max(x['reprojection_px'] for x in fit+check) <= 2.)
        report.update(passed=bool(passed), method=method, fit_count=len(fit),
                      check_count=len(check), rotation_motion_singular_values=axes.tolist(),
                      fit_only_method_scores=scores, max_mm=a.max_mm, max_deg=a.max_deg,
                      wrist_from_rgb_mm=hand_eye.tolist(),
                      wrist_from_depth_mm=(hand_eye @ fit[0]['rgb_from_depth']).tolist(),
                      base_from_target_fit_reference_mm=reference.tolist(),
                      views=[dict(name=x['name'], reprojection_px=x['reprojection_px'],
                                  alternate_reprojection_px=x['alternate_reprojection_px'],
                                  input_json_sha256=x['input_sha256'], **e)
                             for x, e in zip(fit+check, all_errors)])
        # Candidate files are never installed into the live robot configuration.
        np.save(output/'camera_depth_to_link5_candidate.npy', hand_eye @ fit[0]['rgb_from_depth'])
        np.save(output/'camera_rgb_to_link5_candidate.npy', hand_eye)
        print(f'Fit: {len(fit)}, check: {len(check)}, method: {method}')
        for name, errors in [('fit', fit_errors), ('check', check_errors)]:
            print(f'{name}: max target-origin error {max(e["center_error_mm"] for e in errors):.2f} mm, '
                  f'max angle error {max(e["angle_error_deg"] for e in errors):.2f} deg')
        print('PASS (consistency only)' if passed else 'FAIL: do not use for robot motion')
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError, cv2.error) as e:
        report['failure'] = str(e)
        print('FAIL:', e)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('REPORT', output/'report.json')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
