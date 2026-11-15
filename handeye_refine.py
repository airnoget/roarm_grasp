"""Fit-only joint hand-eye/target reprojection refinement, with held-out diagnostics.

Never changes joint signs, offsets, camera intrinsics, board size, or robot geometry.
"""
import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'.calibdeps'))
from scipy.optimize import least_squares


def pack(x, y):
    return np.r_[cv2.Rodrigues(x[:3, :3])[0].ravel(), x[:3, 3],
                 cv2.Rodrigues(y[:3, :3])[0].ravel(), y[:3, 3]]


def unpack(params):
    poses = []
    for v in (params[:6], params[6:]):
        t = np.eye(4)
        t[:3, :3] = cv2.Rodrigues(v[:3])[0]
        t[:3, 3] = v[3:]
        poses.append(t)
    return poses


def pixel_residuals(params, views):
    x, y = unpack(params)
    residuals = []
    for view in views:
        pose = np.linalg.inv(x) @ np.linalg.inv(view['base_from_wrist']) @ y
        projected = cv2.projectPoints(view['object_points'], cv2.Rodrigues(pose[:3, :3])[0],
                                      pose[:3, 3], view['K'], view['distortion'])[0].reshape(-1, 2)
        residuals.extend((projected-view['corners_px']).ravel())
        depth = (view['object_points'] @ pose[:3, :3].T + pose[:3, 3])[:, 2]
        residuals.extend(np.minimum(depth-1., 0.)*10.)
    return np.asarray(residuals)


def refine(views, x, y):
    initial = pack(x, y)
    result = least_squares(pixel_residuals, initial, args=(views,), x_scale='jac',
                           loss='soft_l1', f_scale=1., max_nfev=250,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    new_x, new_y = unpack(result.x)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Reprojection refinement did not converge: '+result.message)
    info = dict(success=True, fit_only=True, evaluations=result.nfev,
                initial_pixel_component_rms=float(np.sqrt(np.mean(pixel_residuals(initial, views)**2))),
                final_pixel_component_rms=float(np.sqrt(np.mean(pixel_residuals(result.x, views)**2))))
    return new_x, new_y, info
