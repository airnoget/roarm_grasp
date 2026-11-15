"""Generate and screen a tabletop grasp with the real jaw's swept geometry."""
import numpy as np


def tilted_lift_extension(motion, pitch, rise=.015):
    data = motion.d
    start = data.site_xpos[motion.tcp].copy()
    x_axis, _, z_axis = data.xmat[motion.wrist].reshape(3, 3).T
    horizontal = x_axis - z_axis * (x_axis[2] / z_axis[2])
    heading = np.arctan2(horizontal[1], horizontal[0])
    points = start + np.linspace(0, rise, 11)[1:, None] * np.array([0., 0., 1.])
    path = motion.waypoint_path(points, heading, data.qpos[motion.q].copy(), pitch)
    motion.check_carry(path)
    return path


def plan_path(motion, surface_mm, heading, width_mm, seed, gripper, approach_from_seed, support_z_mm, tilted=False):
    surface = np.asarray(surface_mm) / 1000
    floor_z = support_z_mm / 1000
    # Leave room for the finger tip below TCP and for tracking error.
    max_depth = min(.016, surface[2] - floor_z - .010)
    if max_depth < .003:
        raise ValueError("Insufficient support clearance for vertical grasp")
    offset = width_mm / 2000 - .010 if gripper != "parallel-pads" else 0
    failures = []
    pitches = (0., .65, .9, 1.15) if tilted else (0.,)
    for pitch in pitches:
        # Tilted tips need extra clearance during the final tracking transient.
        pitch_depth = max_depth - (.004 if pitch else 0.)
        if pitch_depth < .003:
            continue
        for depth in sorted({pitch_depth, max(.003, pitch_depth - .004), max(.003, pitch_depth - .008)}, reverse=True):
            headings = (heading, heading + np.pi) if gripper != "parallel-pads" else (heading,)
            for direction in headings:
                clearances = (0, .003, .006) if gripper != "parallel-pads" else (0,)
                for clearance in clearances:
                    lateral = offset + clearance
                    target = surface - [lateral * np.cos(direction), lateral * np.sin(direction), depth]
                    radial = target[:2] / np.linalg.norm(target[:2])
                    axis = np.r_[radial * np.sin(pitch), -np.cos(pitch)]
                    above = target - .07 * axis
                    try:
                        path = motion.waypoint_path(np.linspace(above, target, 41), direction, seed, pitch)
                        transit = np.linspace(seed, path[0], 70)[1:] if approach_from_seed else []
                        motion.check_sweep(transit, path)
                    except (RuntimeError, ValueError) as error:
                        failures.append(str(error))
                        continue
                    return target, path, transit, depth, pitch
    raise ValueError("No swept path with support clearance: " + "; ".join(dict.fromkeys(failures)))




