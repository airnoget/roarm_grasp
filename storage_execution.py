"""Validate taught transfer waypoints and execute a monitored release."""
import math

import numpy as np

from planning import inside


def plan_storage(config, lift):
    storage = config["storage"]
    if not isinstance(storage, dict) or storage.get("commissioned") is not True:
        raise ValueError("Storage route requires independent commissioning")
    corridor = np.asarray(storage["lift_entry_workspace_mm"], float)
    bin_xy = np.asarray(storage["bin_xy_mm"], float)
    rim = float(storage["rim_z_mm"])
    clearance = float(storage["tcp_rim_clearance_mm"])
    margin = float(storage["release_margin_mm"])
    tolerance = config["position_tolerance_mm"]
    if (corridor.shape != (2, 3) or bin_xy.shape != (2, 2)
            or not np.isfinite(corridor).all() or not np.isfinite(bin_xy).all()
            or np.any(corridor[0] >= corridor[1]) or np.any(bin_xy[0] >= bin_xy[1])
            or not all(math.isfinite(v) for v in (rim, clearance, margin, tolerance))
            or clearance <= tolerance or margin <= tolerance):
        raise ValueError("Invalid storage clearance, bin, or lift entry geometry")
    if not inside([lift[a] for a in "xyz"], corridor):
        raise ValueError("Grasp lift is outside the commissioned storage entry")
    if lift["z"] < rim + clearance:
        raise ValueError("Lift height below storage transfer clearance")
    def waypoint(pose):
        if not isinstance(pose, list) or len(pose) != 5 or not all(math.isfinite(v) for v in pose):
            raise ValueError("Storage waypoint must be finite [x,y,z,pitch,roll]")
        x, y, z, pitch, roll = pose
        if (not inside([x, y, z], config["workspace_mm"]) or z < rim + clearance
                or not config["roll_limits_rad"][0] <= roll <= config["roll_limits_rad"][1]
                or abs(pitch) > math.pi):
            raise ValueError("Storage waypoint outside workspace or configured clearance")
        return dict(T=104, x=x, y=y, z=z, t=pitch, r=roll, g=lift["g"],
                    spd=min(lift["spd"], .10))

    waypoints = [waypoint(pose) for pose in storage["waypoints"]]
    view = storage.get("verification_waypoint") if storage.get("verify_depth") else None
    view = waypoint(view) if view is not None else None
    if not waypoints:
        raise ValueError("Storage route has no waypoints")
    opening = np.array([bin_xy[0] + margin, bin_xy[1] - margin])
    if not inside([waypoints[-1][a] for a in "xy"], opening):
        raise ValueError("Release waypoint outside the inner bin opening")
    return waypoints, opening, rim + clearance, view


def release(arm, config, opening, minimum_z):
    state = arm.feedback()
    # Account for endpoint tolerance when permitting release.
    if not inside([state[a] for a in "xy"], opening) or state["z"] < minimum_z:
        raise RuntimeError("Measured release position outside the bin clearance")
    command = dict(T=106, cmd=config["open_rad"], spd=10, acc=10)
    return command, state

