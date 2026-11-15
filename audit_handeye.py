"""Offline diagnostic only: compare linear and fit-only reprojection hand-eye estimates."""
import argparse
import json
from pathlib import Path
import time
from solve_tag_handeye import ROOT, Wrist, read_views, independent_checks, solve, residuals
from handeye_refine import refine, pack, pixel_residuals
import numpy as np
import cv2


def relative_angle_audit(views):
    pairs = []
    for i, left in enumerate(views):
        for right in views[i+1:]:
            robot = float(np.linalg.norm(cv2.Rodrigues(
                left['base_from_wrist'][:3, :3].T @ right['base_from_wrist'][:3, :3])[0])*180/np.pi)
            vision = float(np.linalg.norm(cv2.Rodrigues(
                left['rgb_from_tag'][:3, :3] @ right['rgb_from_tag'][:3, :3].T)[0])*180/np.pi)
            pairs.append(dict(left=left['name'], right=right['name'], robot_deg=robot,
                              vision_deg=vision, mismatch_deg=abs(robot-vision)))
    return dict(median_mismatch_deg=float(np.median([x['mismatch_deg'] for x in pairs])),
                maximum_mismatch_deg=max(x['mismatch_deg'] for x in pairs),
                worst_pairs=sorted(pairs, key=lambda x: x['mismatch_deg'], reverse=True)[:8])


def metrics(views, x, y):
    errors = residuals([v['base_from_wrist'] @ x @ v['rgb_from_tag'] for v in views], y)
    return dict(max_origin_mm=max(e['center_error_mm'] for e in errors),
                max_angle_deg=max(e['angle_error_deg'] for e in errors),
                pixel_component_rms=float(np.sqrt(np.mean(pixel_residuals(pack(x, y), views)**2))),
                views=[dict(name=v['name'], **e) for v, e in zip(views, errors)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    a = p.parse_args()
    settings = json.loads((a.session/'session.json').read_text())
    wrist = Wrist(ROOT/'.vendor/roarm_ws', [1]*5, [0]*5)
    fit = read_views(a.session, 'fit', wrist, settings)
    raw_check = read_views(a.session, 'check', wrist, settings)
    check, excluded = independent_checks(fit, raw_check)
    (_, method, x, y, _), axes, _ = solve(fit)
    nx, ny, info = refine(fit, x, y)
    report = dict(schema=1, diagnostic_only=True, commissioned=False,
                  reference='fixed target coordinate origin, not measured physical grasp accuracy',
                  original_method=method, refinement=info, excluded_checks=excluded,
                  relative_rotation_invariants=relative_angle_audit(fit),
                  rotation_axis_singular_values=axes.tolist(),
                  linear=dict(fit=metrics(fit, x, y), check=metrics(check, x, y)),
                  refined=dict(fit=metrics(fit, nx, ny), check=metrics(check, nx, ny)),
                  wrist_from_rgb_refined_mm=nx.tolist(), base_from_target_refined_mm=ny.tolist())
    output = a.session/f'audit-{time.time_ns()}.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name in ('linear', 'refined'):
        print(name, {group: {key: round(val, 3) for key, val in report[name][group].items()
                            if key != 'views'} for group in ('fit', 'check')})
    print('DIAGNOSTIC REPORT', output)


if __name__ == '__main__':
    main()
