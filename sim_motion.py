"""Model-based waypoint IK, contact checks, and physics-driven grip/lift."""
import mujoco
import numpy as np
from scipy.optimize import least_squares


class Motion:
    def __init__(self, model, data):
        self.m, self.d = model, data
        self.trace = None
        self.trace_steps = 0
        self.scratch = mujoco.MjData(model)
        self.joints = np.array([model.actuator_trnid[i, 0] for i in range(5)])
        self.q = model.jnt_qposadr[self.joints]
        self.tcp = model.site("tcp").id
        self.wrist = model.body("link5").id
        self.block = model.body("block").id
        names = ("pad_-1", "pad_1") if model.nu == 7 else ("link5", "gripper_link")
        self.fingers = {model.geom(name).id: name for name in names}
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link5_tip") >= 0:
            self.fingers = {model.geom(f"{jaw}_{part}").id: jaw
                            for jaw in names for part in ("tip", "section_1")}
        self.grip_q = model.jnt_qposadr[model.actuator_trnid[5:, 0]]
        self.opening = .027 if model.nu == 7 else 1.1
        self.block_geom = model.geom("block_geom").id
        self.bounds = model.jnt_range[self.joints].T

    def solve(self, xyz, heading, seed, viewing_pitch=0.):
        bearing = np.arctan2(xyz[1], xyz[0])
        desired_z = np.array([np.sin(viewing_pitch) * np.cos(bearing),
                              np.sin(viewing_pitch) * np.sin(bearing),
                              -np.cos(viewing_pitch)])
        desired_x = np.array([np.cos(heading), np.sin(heading), 0.])
        desired_x -= desired_z * np.dot(desired_x, desired_z)
        desired_x /= np.linalg.norm(desired_x)

        def residual(q):
            self.scratch.qpos[self.q] = q
            mujoco.mj_kinematics(self.m, self.scratch)
            axes = self.scratch.xmat[self.wrist].reshape(3, 3)
            return np.r_[self.scratch.site_xpos[self.tcp] - xyz,
                         0.15 * (axes[:, 2] - desired_z),
                         0.15 * (axes[:, 0] - desired_x)]

        q = least_squares(residual, np.clip(seed, *self.bounds),
                          bounds=self.bounds, max_nfev=100,
                          ftol=1e-8, xtol=1e-8, gtol=1e-8).x
        error = residual(q)
        if np.linalg.norm(error[:3]) > .002 or np.linalg.norm(error[3:]) > .006:
            raise ValueError(f"Unreachable TCP waypoint {xyz}; residual {error.round(4)}")
        return q

    def contacts(self, data=None, allow_grip=True):
        data = self.d if data is None else data
        touched = set()
        for contact in data.contact:
            if contact.dist > 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            groups = self.m.geom_group[[a, b]]
            if not np.any((groups == 1) | (groups == 4)):
                continue
            pair = {a, b}
            if allow_grip and self.block_geom in pair and pair & self.fingers.keys():
                touched.update(self.fingers[i] for i in pair & self.fingers.keys())
            else:
                names = [self.m.geom(i).name for i in (a, b)]
                raise RuntimeError(f"Unexpected simulated contact: {names}")
        return touched

    def check_sweep(self, transit, path, close=True):
        scratch = self.scratch
        scratch.qpos[:] = self.d.qpos
        scratch.qvel[:] = 0

        def check(q, grip, allow_grip):
            scratch.qpos[self.q] = q
            scratch.qpos[self.grip_q] = grip
            mujoco.mj_fwdPosition(self.m, scratch)
            self.contacts(scratch, allow_grip=allow_grip)

        for q in (*transit, *path):
            check(q, self.opening, False)
        if close:
            for grip in np.linspace(self.opening, 0, 41):
                check(path[-1], grip, True)

    def check_carry(self, path):
        # Use a rigidly carried proxy only for geometric screening.
        # The execution model still relies on actual jaw contact and friction.
        scratch = self.scratch
        scratch.qpos[:] = self.d.qpos
        scratch.qvel[:] = 0
        rotation = self.d.xmat[self.wrist].reshape(3, 3)
        relative = rotation.T @ (self.d.xpos[self.block] - self.d.xpos[self.wrist])
        relative_rotation = rotation.T @ self.d.xmat[self.block].reshape(3, 3)
        block_q = self.m.jnt_qposadr[self.m.body_jntadr[self.block]]
        for q in path:
            scratch.qpos[self.q] = q
            mujoco.mj_kinematics(self.m, scratch)
            wrist_rotation = scratch.xmat[self.wrist].reshape(3, 3)
            scratch.qpos[block_q:block_q+3] = scratch.xpos[self.wrist] + wrist_rotation @ relative
            mujoco.mju_mat2Quat(scratch.qpos[block_q+3:block_q+7],
                               (wrist_rotation @ relative_rotation).ravel())
            mujoco.mj_fwdPosition(self.m, scratch)
            self.contacts(scratch)
            for contact in scratch.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                if self.block_geom in (a, b) and contact.dist <= 0:
                    other = b if a == self.block_geom else a
                    if other not in self.fingers:
                        raise RuntimeError(f"Carried block intersects {self.m.geom(other).name}")


    def check_release(self):
        scratch = self.scratch
        scratch.qpos[:] = self.d.qpos
        scratch.qvel[:] = 0
        start = self.d.qpos[self.grip_q].copy()
        for grip in np.linspace(start, np.full_like(start, self.opening), 41):
            scratch.qpos[self.grip_q] = grip
            mujoco.mj_fwdPosition(self.m, scratch)
            self.contacts(scratch)

    def grip_contacts(self):
        result = []
        force = np.zeros(6)
        for index, contact in enumerate(self.d.contact):
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.block_geom not in pair or contact.dist > 0:
                continue
            other = next(iter(pair - {self.block_geom}))
            if other in self.fingers:
                mujoco.mj_contactForce(self.m, self.d, index, force)
                result.append({"finger": self.m.geom(other).name,
                               "jaw": self.fingers[other],
                               "normal_force_n": float(force[0]),
                               "tangential_force_n": float(np.linalg.norm(force[1:3])),
                               "point_mm": (contact.pos * 1000).round(2).tolist(),
                               "normal": contact.frame[:3].round(3).tolist()})
        return result

    def grip_state(self):
        contacts = self.grip_contacts()
        normal = {}
        tangential = {}
        for item in contacts:
            jaw = item["jaw"]
            normal[jaw] = normal.get(jaw, 0) + item["normal_force_n"]
            tangential[jaw] = tangential.get(jaw, 0) + item["tangential_force_n"]
        wrist_axes = self.d.xmat[self.wrist].reshape(3, 3)
        relative_center = wrist_axes.T @ (self.d.xpos[self.block] - self.d.xpos[self.wrist])
        return {
            "time_s": round(float(self.d.time), 4),
            "block_z_mm": float(self.d.xpos[self.block, 2] * 1000),
            "tcp_z_mm": float(self.d.site_xpos[self.tcp, 2] * 1000),
            "block_center_wrist_mm": (relative_center * 1000).tolist(),
            "jaw_position": self.d.qpos[self.grip_q].tolist(),
            "jaw_actuator_force": self.d.actuator_force[5:].tolist(),
            "normal_n_by_jaw": normal,
            "tangential_n_by_jaw": tangential,
        }

    def step(self, q, grip, count):
        self.d.ctrl[:5] = q
        self.d.ctrl[5:] = grip
        for _ in range(count):
            mujoco.mj_step(self.m, self.d)
            self.contacts()
            if self.trace is not None:
                if self.trace_steps % 10 == 0:
                    self.sample_trace()
                self.trace_steps += 1

    def sample_trace(self):
        if not self.trace or self.d.time > self.trace[-1][0]:
            self.trace.append((self.d.time, self.d.qpos.copy(), self.d.ctrl[:5].copy()))

    def waypoint_path(self, points, heading, seed, pitch=0.):
        targets = []
        for point in points:
            seed = self.solve(point, heading, seed, pitch)
            targets.append(seed)
        return targets










