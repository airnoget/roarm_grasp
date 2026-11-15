import json
from pathlib import Path

import numpy as np

from camera import pinhole_rays


def main():
    folder = Path("demo_output")
    folder.mkdir(exist_ok=True)
    depth = np.full((480, 640), 0.6, dtype=np.float32)
    depth[218:250, 303:350] = 0.57
    rays = pinhole_rays(depth.shape, 600, 600, 320, 240)
    np.savez_compressed(folder / "frame.npz", depth_m=depth, rays=rays)
    transform = np.diag([1., -1., -1., 1.])
    transform[:3, 3] = [245, 0, 600]
    np.save(folder / "transform.npy", transform)
    config = json.loads(Path("config.example.json").read_text())
    config["transform"] = str(folder / "transform.npy")
    config["roll_limits_rad"] = [-1.65, 1.65]
    (folder / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

