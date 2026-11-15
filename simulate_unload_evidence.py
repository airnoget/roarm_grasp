"""Reproducible physical drop and depth-evidence acceptance cases."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from camera import pinhole_rays
from sim_camera import raycast
from unload_depth import UnloadDepth


def trial(positions, expected):
    blocks = "".join(
        f'<body pos="{x} {y} .12"><freejoint/>'
        '<geom type="box" size=".02 .017 .014" mass=".02"/></body>'
        for x, y in positions
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
        + blocks + '</worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    eye = np.array([.25, .25, .8])
    rotation = np.diag([1., -1., -1.])
    rays = pinhole_rays((240, 240), 295, 295, 119.5, 119.5)
    verifier = UnloadDepth([0, 0, .5, .5], 0)
    empty = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )

    def grid(m, d):
        mujoco.mj_forward(m, d)
        return verifier.grid(*raycast(m, d, eye, rotation, rays, [1,0,0,0,0,0]))

    before = grid(empty, mujoco.MjData(empty))
    mujoco.mj_step(model, data, nstep=450)
    after = []
    for _ in range(3):
        mujoco.mj_step(model, data, nstep=50)
        after.append(grid(model, data))
    return verifier.compare(before, after, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("sim_output/unload-depth.json"))
    args = parser.parse_args()
    scenarios = [
        ("two_inside", [( .16,.18), (.33,.31)], 2, True),
        ("crossing_boundary", [(.495,.30)], 1, False),
        ("missing_sample", [(.20,.20)], 2, False),
    ]
    results = []
    for name, positions, expected, should_confirm in scenarios:
        report = trial(positions, expected)
        results.append(dict(case=name, expected_confirmation=should_confirm, **report))
    result = dict(passed=all(r["confirmed"] == r["expected_confirmation"] for r in results),
                  trials=results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
