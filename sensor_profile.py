"""Seeded perturbations for depth and hand-eye robustness tests."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from camera import Frame


@dataclass(frozen=True)
class SensorProfile:
    min_range_mm: float = 0.
    max_range_mm: float = 10000.
    depth_bias_mm: float = 0.
    depth_sigma_mm: float = 0.
    dropout: float = 0.
    hole_count: int = 0
    hole_radius_px: int = 0
    mount_translation_sigma_mm: float = 0.
    mount_rotation_sigma_deg: float = 0.

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not np.isfinite(values).all():
            raise ValueError("Sensor profile values must be finite")
        if (not 0 <= self.min_range_mm < self.max_range_mm
                or not 0 <= self.dropout < 1
                or min(self.depth_sigma_mm, self.mount_translation_sigma_mm,
                       self.mount_rotation_sigma_deg) < 0
                or type(self.hole_count) is not int or not 0 <= self.hole_count <= 100
                or type(self.hole_radius_px) is not int or not 0 <= self.hole_radius_px <= 80):
            raise ValueError("Invalid range, noise, or hole parameters")


def perturb(frame, rotation, translation, profile, seed, mount_seed=None):
    rng = np.random.default_rng(seed)
    original = frame.depth_m * 1000
    measured = original + profile.depth_bias_mm + rng.normal(0, profile.depth_sigma_mm, original.shape)
    valid = (np.isfinite(original) & (original > 0) & (original >= profile.min_range_mm)
             & (original <= profile.max_range_mm) & (measured > 0))
    valid &= rng.random(original.shape) >= profile.dropout
    v, u = np.ogrid[:original.shape[0], :original.shape[1]]
    for _ in range(profile.hole_count):
        y, x = rng.integers([0, 0], original.shape)
        valid &= (u - x)**2 + (v - y)**2 > profile.hole_radius_px**2
    depth = np.where(valid, measured / 1000, 0).astype(np.float32)
    # Apply calibration error only to the estimated transform. Rays were
    # rendered from the true simulated pose.
    mount_rng = rng if mount_seed is None else np.random.default_rng(mount_seed)
    translation_error = mount_rng.normal(0, profile.mount_translation_sigma_mm, 3)
    angular_error = np.deg2rad(mount_rng.normal(0, profile.mount_rotation_sigma_deg, 3))
    estimated_rotation = rotation @ Rotation.from_rotvec(angular_error).as_matrix()
    estimated_translation = translation + rotation @ translation_error
    return Frame(depth, frame.rays, frame.bgr), estimated_rotation, estimated_translation

