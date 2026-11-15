"""Compute RoArm-M3 link5 pose from measured joints and the official model."""
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def rotate(axis, angle):
    return cv2.Rodrigues(axis * angle)[0]


class Wrist:
    def __init__(self, vendor, signs, offsets):
        source = Path(vendor) / "src/roarm_main/roarm_description/urdf/roarm_m3/roarm_m3.xacro"
        joints = {j.find("child").attrib["link"]: j for j in ET.parse(source).getroot().findall("joint")}
        self.chain = []
        x, y, z = np.eye(3)
        for i in range(1, 6):
            joint = joints[f"link{i}"]
            origin = joint.find("origin").attrib
            roll, pitch, yaw = np.fromstring(origin["rpy"], sep=" ")
            rotation = rotate(z, yaw) @ rotate(y, pitch) @ rotate(x, roll)
            translation = np.fromstring(origin["xyz"], sep=" ") * 1000
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            self.chain.append((rotation, translation, axis))
        self.signs, self.offsets = np.asarray(signs, float), np.asarray(offsets, float)
        if (self.signs.shape != (5,) or self.offsets.shape != (5,)
                or not np.all(np.isin(self.signs, [-1, 1])) or not np.isfinite(self.offsets).all()):
            raise ValueError("Expected five joint signs and offsets")

    def fk(self, raw):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (5,) or not np.isfinite(raw).all():
            raise ValueError("Expected five finite joint angles")
        r, p = np.eye(3), np.zeros(3)
        for (origin_r, origin_p, axis), angle in zip(self.chain, raw * self.signs + self.offsets):
            p += r @ origin_p
            r = r @ origin_r @ rotate(axis, angle)
        return r, p

    def camera_pose(self, raw, hand_eye):
        r, p = self.fk(raw)
        camera_r, camera_p = hand_eye
        return r @ camera_r, r @ camera_p + p


def stationary_frame(arm, read):
    def joints():
        state = arm.feedback()
        q = np.asarray([state[key] for key in ("b", "s", "e", "t", "r")], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError("Invalid joint feedback")
        return q

    samples = [joints()]
    for _ in range(5):
        frame = read()
        samples.append(joints())
    # Endpoints alone miss an arm moving away and returning during acquisition.
    # Unwrap relative to the first sample so +/-pi is not treated as a full turn.
    samples = np.asarray(samples)
    delta = (samples - samples[0] + np.pi) % (2*np.pi) - np.pi
    if np.max(np.ptp(delta, axis=0)) > .003:
        raise RuntimeError("Arm moved during wrist observation")
    # The returned frame lies between the final two feedback samples.
    return frame, samples[0] + (delta[-2] + delta[-1]) / 2
