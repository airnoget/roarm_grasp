"""Estimate a visible, near-horizontal support in a bounded base-frame region."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Support:
    height_mm: float
    inliers: int
    rms_mm: float


def estimate_support(points, bounds_mm, min_points=80):
    bounds = np.asarray(bounds_mm, dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1])):
        raise ValueError("Support bounds require finite increasing XYZ limits")
    points = np.asarray(points)
    region = np.isfinite(points).all(axis=-1)
    region &= np.all((points >= bounds[0]) & (points <= bounds[1]), axis=-1)
    samples = points[region]
    if len(samples) < min_points:
        raise ValueError("Insufficient depth points in support search region")
    heights = samples[:, 2]
    # Evaluate overlapping 4 mm bands so points on a bin edge remain together.
    ordered = np.sort(heights)
    centers = np.arange(ordered[0], ordered[-1] + 1, 1.0)
    counts = (np.searchsorted(ordered, centers + 2, side="right")
              - np.searchsorted(ordered, centers - 2))
    best = int(np.argmax(counts))
    inliers = samples[np.abs(heights - centers[best]) <= 2]
    competitors = counts[np.abs(centers - centers[best]) >= 7]
    if len(inliers) < min_points or (competitors.size and competitors.max() >= .8 * len(inliers)):
        raise ValueError("Support height is ambiguous or insufficiently visible")
    xy = inliers[:, :2] - inliers[:, :2].mean(axis=0)
    span = np.linalg.svd(xy, compute_uv=False)[-1] / np.sqrt(len(xy))
    if span < 7:
        raise ValueError("Support samples do not span a usable area")
    height = float(np.median(inliers[:, 2]))
    slope = np.linalg.lstsq(xy, inliers[:, 2] - inliers[:, 2].mean(), rcond=None)[0]
    if np.linalg.norm(slope) > np.tan(np.deg2rad(6)):
        raise ValueError("Support is too steep for the vertical grasp planner")
    rms = float(np.sqrt(np.mean((inliers[:, 2] - height) ** 2)))
    return Support(height, len(inliers), rms)
