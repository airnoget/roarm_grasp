"""Storage-bin geometry and physical deposit verification."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from bin_depth import BinDepth


CENTER = np.array([.22, -.14])
INNER = np.array([.055, .055])
FLOOR_Z = -.060
WALL_TOP = -.022


def add_bin(world):
    def box(name, xyz, half_size):
        ET.SubElement(world, "geom", name=f"bin_{name}", type="box",
                      pos=" ".join(map(str, xyz)), size=" ".join(map(str, half_size)),
                      group="0", rgba=".25 .50 .35 1")

    x, y = CENTER
    hx, hy = INNER
    thickness = .004
    half_height = (WALL_TOP - FLOOR_Z) / 2
    box("floor", (x, y, FLOOR_Z - .005), (hx + thickness, hy + thickness, .005))
    for side in (-1, 1):
        box(f"x{side}", (x + side * (hx + thickness / 2), y, FLOOR_Z + half_height),
            (thickness / 2, hy + thickness, half_height))
        box(f"y{side}", (x, y + side * (hy + thickness / 2), FLOOR_Z + half_height),
            (hx, thickness / 2, half_height))


def bin_result(motion):
    m, d = motion.m, motion.d
    geom = motion.block_geom
    rotation = d.geom_xmat[geom].reshape(3, 3)
    center = d.geom_xpos[geom]
    if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        local_center, half_size = m.geom_aabb[geom].reshape(2, 3)
        center = center + rotation @ local_center
    else:
        half_size = m.geom_size[geom]
    extent = np.abs(rotation) @ half_size
    floor = m.geom("bin_floor").id
    on_floor = any(
        {c.geom1, c.geom2} == {floor, geom} and c.dist <= 0
        for c in d.contact
    )
    speed = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, motion.block, speed, 0)
    inside = bool(np.all(np.abs(center[:2] - CENTER) + extent[:2] <= INNER)
                  and center[2] + extent[2] <= WALL_TOP + .001)
    settled = bool(np.linalg.norm(speed[3:]) < .01 and np.linalg.norm(speed[:3]) < .1)
    return dict(deposited=inside and settled and on_floor, inside=inside, settled=settled,
                on_floor=on_floor, final_center_mm=(center * 1000).tolist())


def transfer_path(motion, end, carrying, headings=None):
    d = motion.d
    heading_axis = d.xmat[motion.wrist].reshape(3, 3)[:, 0]
    heading = np.arctan2(heading_axis[1], heading_axis[0])
    start = d.site_xpos[motion.tcp].copy()
    errors = []
    fractions = np.linspace(0, 1, 45)[1:]
    for final_heading in headings or (heading, 0., np.pi / 2, -np.pi / 2):
        delta = (final_heading - heading + np.pi) % (2 * np.pi) - np.pi
        q = d.qpos[motion.q].copy()
        path = []
        try:
            for progress in fractions:
                q = motion.solve(start + progress * (end - start), heading + progress * delta, q)
                path.append(q)
            if carrying:
                motion.check_carry(path)
            else:
                motion.check_sweep([], path, close=False)
        except (ValueError, RuntimeError) as error:
            errors.append(str(error))
            continue
        return path
    raise ValueError("No valid bin path: " + " | ".join(errors))


def travel(motion, path, carrying, steps):
    grip = 0 if carrying else motion.opening
    for q in path:
        motion.step(q, grip, steps)
        if carrying and len(motion.contacts()) < 2:
            raise RuntimeError("Lost bilateral contact during bin transfer")
    motion.step(path[-1], grip, 125)
    if carrying and len(motion.contacts()) < 2:
        raise RuntimeError("Lost bilateral contact while settling at bin")


def deposit(motion, lift_steps, verify=False, sensor=None, sensor_seed=0):
    from sim_camera import wrist_frame
    from sensor_profile import perturb

    def observe(index):
        observation = wrist_frame(motion.m, motion.d)
        if sensor is None:
            return observation
        # A different depth realization per frame, but one hand-eye error for the trial.
        return perturb(*observation, sensor, [sensor_seed, index], mount_seed=sensor_seed)

    d = motion.d
    start_z = d.site_xpos[motion.tcp, 2]
    end = np.r_[CENTER, max(start_z, WALL_TOP + .038)]
    if verify:
        verifier = BinDepth(dict(bin_xy_mm=[(CENTER-INNER)*1000, (CENTER+INNER)*1000],
                                 rim_z_mm=WALL_TOP*1000, floor_z_mm=FLOOR_Z*1000))
        view = np.array([.22, -.075, .095])
        path = transfer_path(motion, view, carrying=True, headings=(0.,))
        travel(motion, path, carrying=True, steps=lift_steps)
        before = verifier.reference(observe(1))

    path = transfer_path(motion, end, carrying=True)
    if verify:
        travel(motion, path, carrying=True, steps=lift_steps)
    else:
        for q in path:
            motion.step(q, 0, lift_steps)
            if len(motion.contacts()) < 2:
                return dict(deposited=False, reason="lost_during_transfer")
    motion.check_release()
    motion.step(path[-1], motion.opening, 450)
    result = bin_result(motion)
    if verify:
        path = transfer_path(motion, view, carrying=False, headings=(0.,))
        travel(motion, path, carrying=False, steps=lift_steps)
        after = []
        for index in range(2, 5):
            motion.step(path[-1], motion.opening, 50)
            after.append(verifier.height_map(observe(index)))
        result = bin_result(motion)
        result["depth_verification"] = verifier.compare(before, after)
    return result


