"""Select a collision-screened wrist view in the mapped sampling region."""
import mujoco
import numpy as np

from geometry import base_points
from perception import blocks
from sim_camera import wrist_frame
from support_plane import estimate_support
from target_semantics import support_pixels


# Ordered pose library. Each pose is independent of simulated sample truth.
VIEWS = (
    ((.235, .055, .095), 0.),
    ((.18, 0., .24), .7),
    ((.18, 0., .20), .95),
    ((.22, .04, .16), .25),
    ((.19, -.04, .17), .45),
)


class ObservationRejected(RuntimeError):
    def __init__(self, attempts):
        super().__init__("No safe wrist observation with sufficient visible depth")
        self.attempts = attempts


def read_support(frame, rotation, translation, ledge=False, detections=()):
    points = base_points(frame, rotation, translation)
    valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
    search = [[225, -80, -80], [350, 80, 430 if ledge else 190]]
    background = support_pixels(valid, detections)
    estimate = estimate_support(np.where(background[..., None], points, np.nan), search)
    return points, valid, estimate


def select_view(motion, seed, ledge, sensor_observation=None, rays=None):
    m, d = motion.m, motion.d
    attempts = []
    scratch = motion.scratch
    start = d.qpos[motion.q].copy()
    for xyz, pitch in VIEWS:
        entry = dict(tcp_m=list(xyz), pitch_rad=pitch)
        attempts.append(entry)
        try:
            q = motion.solve(np.asarray(xyz), 0., seed, pitch)
            path = np.linspace(start, q, 100)
            scratch.qpos[:] = d.qpos
            scratch.qpos[motion.grip_q] = motion.opening
            for waypoint in path:
                scratch.qpos[motion.q] = waypoint
                mujoco.mj_fwdPosition(m, scratch)
                motion.contacts(scratch, allow_grip=False)
            frame, rotation, translation = wrist_frame(m, scratch, rays=rays)
            if sensor_observation is not None:
                frame, rotation, translation = sensor_observation(frame, rotation, translation)
            points, valid, support = read_support(frame, rotation, translation, ledge)
            height = points[..., 2] - support.height_mm
            mask = valid & (height > 10) & (height < 70)
            if ledge:
                mask &= (points[..., 0] > 225) & (points[..., 0] < 350) & (np.abs(points[..., 1]) < 90)
            components = list(blocks(mask, 30))
            if not components:
                raise ValueError("No foreground component in wrist view")
            # Check visibility from the observation itself, without sample truth.
            # A component touching the raster border has unknown full extent.
            complete = [component for component in components
                        if not (component[:2].any() or component[-2:].any()
                                or component[:, :2].any() or component[:, -2:].any())]
            entry['complete_foreground_components'] = len(complete)
            entry['edge_foreground_components'] = len(components)-len(complete)
            if not complete:
                raise ValueError('Foreground truncated by image border; try another wrist view')
            components = complete
            if ledge:
                upper_height = max(float(np.percentile(height[component], 85)) for component in components)
                entry["upper_foreground_mm"] = upper_height
                if upper_height < 20:
                    raise ValueError("Visible foreground lacks a sufficiently exposed upper surface")
        except (ValueError, RuntimeError) as error:
            entry.update(accepted=False, reason=str(error))
            continue
        entry.update(accepted=True, support_mm=support.height_mm)
        return q, path, attempts
    raise ObservationRejected(attempts)

