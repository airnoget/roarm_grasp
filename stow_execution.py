"""Commissioned retreat routes and stationary transport-pose evidence."""
import math
import time

import numpy as np

from planning import inside


JOINTS = ("b", "s", "e", "t", "r", "g")


class TransportStow:
    def __init__(self, config):
        storage = config["storage"]
        spec = storage["stow"]
        if spec.get("commissioned") is not True:
            raise ValueError("Transport stow route requires independent commissioning")
        self.bounds = np.asarray(spec["joint_bounds_rad"], float)
        if (self.bounds.shape != (2, 6) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[0] >= self.bounds[1])
                or np.any(self.bounds[1]-self.bounds[0] > .25)
                or not self.bounds[0, 5] <= config["open_rad"] <= self.bounds[1, 5]):
            raise ValueError("Expected a measured narrow transport envelope for six raw joints")
        self.routes = {}
        origins = ["release"]
        if storage.get("verify_depth") and storage.get("verification_waypoint") is not None:
            origins.append("view")
        for origin in origins:
            poses = spec[f"from_{origin}"]
            if not poses:
                raise ValueError(f"Missing stow path from {origin}")
            route = []
            for pose in poses:
                if len(pose) != 5 or not all(math.isfinite(v) for v in pose):
                    raise ValueError("Stow waypoint requires finite XYZ/Pitch/Roll")
                x, y, z, pitch, roll = pose
                if (not inside([x, y, z], config["workspace_mm"])
                        or abs(pitch) > math.pi
                        or not config["roll_limits_rad"][0] <= roll <= config["roll_limits_rad"][1]):
                    raise ValueError("Stow waypoint outside the commissioned workspace")
                route.append(dict(T=104, x=x, y=y, z=z, t=pitch, r=roll, g=config["open_rad"],
                                  spd=min(config["speed"], .10)))
            self.routes[origin] = route

    def check(self, arm, on_feedback):
        samples = []
        for i in range(3):
            if i:
                time.sleep(.1)
            state = arm.feedback()
            on_feedback(state)
            q = np.asarray([state.get(key, np.nan) for key in JOINTS], dtype=float)
            if not np.isfinite(q).all() or not inside(q, self.bounds):
                raise RuntimeError("Measured arm joints are outside the transport envelope")
            samples.append(q)
        if np.max(np.ptp(samples, axis=0)) > .015:
            raise RuntimeError("Arm joints not stationary in the transport envelope")
