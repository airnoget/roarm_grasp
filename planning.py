import numpy as np

from geometry import grasp_geometry


def inside(point, limits):
    xyz = np.asarray(point)
    low, high = np.asarray(limits)
    return bool(np.all(xyz >= low) and np.all(xyz <= high))


def plan(candidate, frame, rotation, translation, config, component=None):
    surface, heading, width = grasp_geometry(
        frame, candidate, rotation, translation, component,
        horizontal_surface=config.get("test_horizontal_grasp_projection") is True,
    )
    if not 0 < width <= config["max_width_mm"]:
        raise ValueError("Candidate exceeds calibrated opening")
    target = surface.copy()
    target[2] += config["surface_to_tcp_mm"]
    if config.get("support_search_mm") is not None:
        clearance = config["tcp_support_clearance_mm"]
        if clearance is None or not np.isfinite(clearance) or clearance <= 0:
            raise ValueError("Measured TCP support clearance must be positive")
        target[2] = max(target[2], config["table_z_mm"] + clearance)
        if surface[2] - target[2] < 3:
            raise ValueError("Insufficient downstroke after applying TCP support clearance")
    above = target + [0, 0, config["approach_mm"]]
    if (target[2] <= config["table_z_mm"] or
            not all(inside(p, config["workspace_mm"]) for p in (target, above))):
        raise ValueError("Target or approach is outside the configured workspace")
    bearing = np.arctan2(target[1], target[0])
    roll = config["roll_sign"] * (heading - bearing) + config["roll_zero_rad"]
    low, high = config["roll_limits_rad"]
    equivalents = [roll + k*np.pi for k in range(-3, 4)]
    allowed = [float(r) for r in equivalents if low <= r <= high]
    if not allowed:
        raise ValueError("No equivalent roll within measured joint limits")
    roll = min(allowed, key=lambda r: abs(r - (low + high) / 2))

    def move(point, grip):
        return dict(T=104, x=float(point[0]), y=float(point[1]), z=float(point[2]),
                    t=config["pitch_rad"], r=roll, g=grip, spd=config["speed"])

    opened, closed = config["open_rad"], config["close_rad"]
    return [
        move(above, opened), move(target, opened),
        dict(T=106, cmd=closed, spd=10, acc=10),
        move(above, closed),
    ], surface, float(width)



