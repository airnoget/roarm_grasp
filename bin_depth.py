"""Conservative before/after depth evidence for a sample deposited in the bin."""
import cv2
import numpy as np

from geometry import base_points


class BinDepth:
    cell_mm = 5.
    minimum_rise_mm = 8.
    minimum_visibility = .55

    def __init__(self, storage):
        self.bounds = np.asarray(storage["bin_xy_mm"], float)
        self.rim = float(storage["rim_z_mm"])
        self.floor = float(storage["floor_z_mm"])
        if (self.bounds.shape != (2, 2) or not np.isfinite(self.bounds).all()
                or not np.isfinite([self.floor, self.rim]).all()
                or not self.floor + 12 < self.rim):
            raise ValueError("Invalid visual bin geometry")
        size = self.bounds[1] - self.bounds[0]
        if np.any(size < 30) or np.any(size > 500):
            raise ValueError("Bin visual verification expects a 30–500 mm opening")
        self.shape = tuple(np.ceil(size / self.cell_mm).astype(int))

    def height_map(self, observation):
        frame, rotation, translation = observation
        points = base_points(frame, rotation, translation)
        valid = np.isfinite(points).all(axis=-1) & (frame.depth_m > 0)
        xy = points[..., :2]
        valid &= np.all((xy >= self.bounds[0]) & (xy < self.bounds[1]), axis=-1)
        valid &= (points[..., 2] >= self.floor - 4)
        points = points[valid]
        cells = np.floor((points[:, :2] - self.bounds[0]) / self.cell_mm).astype(int)
        result = np.full(self.shape, -np.inf)
        np.maximum.at(result, tuple(cells.T), points[:, 2])
        return result

    def reference(self, observation):
        result = self.height_map(observation)
        if np.isfinite(result).mean() < self.minimum_visibility:
            raise ValueError("Insufficient bin visibility before release")
        return result

    def compare(self, before, after):
        before = np.asarray(before)
        after = np.asarray(after)
        if before.shape != self.shape or after.shape != (3, *self.shape):
            raise ValueError("Expected one reference and three settled bin depth frames")
        # A previous surface and a new surface must both be visible. Above-rim
        # points are retained so a held/occluding object cannot masquerade as floor.
        known = np.isfinite(before) & np.isfinite(after).all(axis=0)
        if known.mean() < self.minimum_visibility:
            return dict(confirmed=False, reason="insufficient_bin_visibility")
        if np.any(np.count_nonzero(after > self.rim + 3, axis=(1, 2)) >= 4):
            return dict(confirmed=False, reason="above_rim_occlusion")
        stable = np.zeros(self.shape, dtype=bool)
        stable[known] = np.ptp(after[:, known], axis=0) <= 4
        rises = np.zeros(self.shape, dtype=bool)
        rises[known] = after[:, known].min(axis=0) - before[known] >= self.minimum_rise_mm
        candidate = stable & rises & (after.max(axis=0) <= self.rim - 2)
        _, _, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
        components = [row for row in stats[1:] if row[cv2.CC_STAT_AREA] >= 4]
        if len(components) != 1:
            return dict(confirmed=False, reason="no_unique_stable_deposit")
        x, y, w, h, area = map(int, components[0])
        if x < 1 or y < 1 or x+w >= self.shape[1] or y+h >= self.shape[0]:
            return dict(confirmed=False, reason="deposit_touches_bin_boundary")
        return dict(confirmed=True, evidence="stable_new_depth_inside_bin",
                    cells=area, shared_coverage=float(known.mean()))

