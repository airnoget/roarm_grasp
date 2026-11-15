"""Compare surveyed unload-zone depth before and after release without commanding hardware."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from camera import load_frame
from geometry import base_points, load_transform


CELL = 5.  # millimeters
BORDER = 50.


class UnloadDepth:
    def __init__(self, zone_m, floor_mm):
        zone = np.asarray(zone_m, float)
        if (zone.shape != (4,) or not np.isfinite(zone).all()
                or not np.allclose(zone[2:] - zone[:2], .5) or not np.isfinite(floor_mm)):
            raise ValueError("Expected a surveyed 0.5 m × 0.5 m zone and finite floor height")
        self.zone = zone * 1000
        self.origin = self.zone[:2] - BORDER
        self.floor = floor_mm
        self.size = int((500 + 2*BORDER) / CELL)

    def grid(self, frame, rotation, translation):
        points = base_points(frame, rotation, translation)
        valid = (frame.depth_m > 0) & np.isfinite(points).all(axis=-1)
        valid &= np.all((points[..., :2] >= self.origin)
                        & (points[..., :2] < self.origin + CELL*self.size), axis=-1)
        valid &= points[..., 2] >= self.floor - 5
        points = points[valid]
        u, v = np.floor((points[:, :2] - self.origin) / CELL).astype(int).T
        height = np.full((self.size, self.size), -np.inf)
        np.maximum.at(height, (v, u), points[:, 2])
        return height

    def compare(self, reference, after, expected):
        after = np.asarray(after)
        if reference.shape != (self.size, self.size) or after.shape != (3, self.size, self.size):
            raise ValueError("Expected one reference and three post-unload depth grids")
        known = np.isfinite(reference) & np.isfinite(after).all(axis=0)
        coverage = float(known.mean())
        if coverage < .8:
            return dict(confirmed=False, reason="insufficient_shared_visibility", coverage=coverage)
        changes = np.zeros(after.shape, bool)
        changes[:, known] = after[:, known] - reference[known] >= 8
        _, labels, stats, _ = cv2.connectedComponentsWithStats(changes.any(axis=0).astype(np.uint8), connectivity=8)
        found = []
        for i, row in enumerate(stats[1:], 1):
            x, y, w, h, area = map(int, row)
            if area < 4:
                continue
            cells = labels == i
            values = after[:, cells]
            stable = bool(changes[:, cells].all() and np.max(np.ptp(values, axis=0)) <= 4)
            neighborhood = cv2.dilate(cells.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            boundary = neighborhood & ~cells
            border_coverage = float(known[boundary].mean()) if boundary.any() else 0.
            bordered = bool(border_coverage >= .80 and x > 0 and y > 0
                            and x + w < self.size and y + h < self.size)
            # Include two cells of uncertainty at every side.
            lower = self.origin + CELL*np.array([x-2, y-2])
            upper = self.origin + CELL*np.array([x+w+2, y+h+2])
            inside = bool(np.all(lower >= self.zone[:2]) and np.all(upper <= self.zone[2:]))
            plausible = bool(max(w,h)*CELL <= 60 and np.max(values) <= self.floor + 60)
            found.append(dict(footprint_m=(np.r_[lower, upper]/1000).tolist(), cells=area,
                              settled=stable, inside=inside, plausible=plausible,
                              visible_boundary=bordered, boundary_coverage=border_coverage))
        confirmed = len(found) == expected and all(
            x["settled"] and x["inside"] and x["plausible"] and x["visible_boundary"] for x in found
        )
        return dict(confirmed=confirmed, expected=expected, coverage=coverage, visible_objects=found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    parser.add_argument("transform", type=Path, help="Surveyed optical-camera to map transform")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path, nargs=3)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    files = [args.before, *args.after]
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if not 1 <= args.expected <= 10 or len(set(digests)) != 4:
        parser.error("Use four independent recordings and 1–10 expected samples")
    scenario = json.loads(args.scenario.read_text(encoding="utf-8-sig"))
    verifier = UnloadDepth(scenario["unload_zone_m"], scenario["floor_z_mm"])
    r, t = load_transform(args.transform)
    frames = [load_frame(p) for p in files]
    if any(not np.array_equal(f.rays, frames[0].rays) for f in frames[1:]):
        parser.error("Use one fixed surveyed camera and depth profile")
    before, *after = [verifier.grid(f, r, t) for f in frames]
    result = verifier.compare(before, after, args.expected)
    result["frame_sha256"] = digests
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    if not result["confirmed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


