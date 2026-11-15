import numpy as np


def sample(frame, u, v, radius=2):
    h, w = frame.depth_m.shape
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("Pixel outside frame")
    patch = frame.depth_m[max(0, v-radius):v+radius+1, max(0, u-radius):u+radius+1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if not np.isfinite(frame.depth_m[v, u]) or frame.depth_m[v, u] <= 0 or valid.size < 5:
        raise ValueError("Insufficient depth at pixel")
    depth = np.median(valid)
    if np.max(np.abs(valid - depth)) > 0.015:
        raise ValueError("Depth discontinuity at pixel")
    return frame.rays[v, u] * depth


def fit_transform(camera_m, base_mm):
    a, b = np.asarray(camera_m) * 1000, np.asarray(base_mm)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 4:
        raise ValueError("Provide at least four corresponding 3D points")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Calibration coordinates must be finite")
    centered_a, centered_b = a - a.mean(0), b - b.mean(0)
    if min(np.linalg.matrix_rank(centered_a), np.linalg.matrix_rank(centered_b)) < 2:
        raise ValueError("Calibration points must span at least two directions")
    u, _, vt = np.linalg.svd(centered_a.T @ centered_b)
    correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = b.mean(0) - rotation @ a.mean(0)
    errors = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
    return rotation, translation, errors


def load_transform(path):
    transform = np.load(path, allow_pickle=False)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError("Expected a homogeneous rigid transform")
    r = transform[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Invalid rotation in calibration")
    return r, transform[:3, 3]


def base_points(frame, rotation, translation):
    return (frame.rays * frame.depth_m[..., None] * 1000) @ rotation.T + translation



def grasp_geometry(frame, candidate, rotation, translation, component=None, horizontal_surface=False):
    """Map center and projected closing axis to base coordinates in millimeters."""
    point = sample(frame, candidate.u, candidate.v)
    surface = rotation @ (point * 1000) + translation
    u, v = candidate.u, candidate.v
    lo_u, hi_u = max(0, u - 1), min(frame.depth_m.shape[1] - 1, u + 1)
    lo_v, hi_v = max(0, v - 1), min(frame.depth_m.shape[0] - 1, v + 1)
    ray_u = (frame.rays[v, hi_u] - frame.rays[v, lo_u]) / (hi_u - lo_u)
    ray_v = (frame.rays[hi_v, u] - frame.rays[lo_v, u]) / (hi_v - lo_v)
    image_axis = np.cos(candidate.angle) * ray_u + np.sin(candidate.angle) * ray_v
    base_axis = rotation @ image_axis
    if horizontal_surface:
        base_ray = rotation @ frame.rays[v, u]
        if abs(base_ray[2]) < .2 * np.linalg.norm(base_ray):
            raise ValueError("Viewing ray is too shallow to recover a horizontal grasp direction")
        # Neighboring pixels intersect the same horizontal surface at different depths.
        base_axis -= base_ray * (base_axis[2] / base_ray[2])
    axis_mm = base_axis * point[2] * 1000
    width_mm = candidate.width_px * np.linalg.norm(axis_mm[:2])
    heading = np.arctan2(axis_mm[1], axis_mm[0])
    if component is not None:
        width_mm = max(width_mm, component_opening(frame, component, rotation, heading))
    return surface, float(heading), float(width_mm)


def component_opening(frame, component, rotation, heading):
    depth = frame.depth_m[component]
    valid = np.isfinite(depth) & (depth > 0)
    if np.count_nonzero(valid) < 5:
        raise ValueError("Insufficient target depth for opening check")
    closing_axis = np.array([np.cos(heading), np.sin(heading), 0.])
    projection = (frame.rays[component][valid] @ (rotation.T @ closing_axis)) * depth[valid] * 1000
    return float(np.ptp(projection))
