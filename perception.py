from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    u: int
    v: int
    angle: float  # closing direction, image +u toward +v
    width_px: float
    score: float
    model_width_px: float | None = None
    target_class: str | None = None
    support_region: int | None = None


def draw_grasp(view, candidate, zoom=1):
    center = (np.array([candidate.u, candidate.v]) + .5) * zoom - .5
    axis = np.array([np.cos(candidate.angle), np.sin(candidate.angle)])
    normal = np.array([-axis[1], axis[0]])
    width = candidate.model_width_px if candidate.model_width_px is not None else candidate.width_px
    half = width * zoom / 2
    corners = [center + half * (x * axis + .5 * y * normal)
               for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    cv2.polylines(view, [np.rint(corners).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    ends = np.rint([center - half * axis, center + half * axis]).astype(int)
    cv2.line(view, tuple(ends[0]), tuple(ends[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(view, tuple(np.rint(center).astype(int)), max(2, zoom), (255, 255, 255), -1)


def blocks(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            yield labels == i


def rectangle_candidates(components):
    for mask in components:
        y, x = np.nonzero(mask)
        points = np.column_stack((x, y)).astype(np.float32)
        rect = cv2.minAreaRect(points)
        corners = cv2.boxPoints(rect)
        edges = corners[1:3] - corners[:2]
        lengths = np.linalg.norm(edges, axis=1)
        i = int(np.argmin(lengths))
        if lengths[i] < 2:
            continue
        u, v = map(round, rect[0])
        if mask[v, u]:
            dx, dy = edges[i]
            yield Candidate(u, v, np.arctan2(dy, dx), lengths[i], 1.0)



def projected_span(mask, angle):
    """Conservative image span of the segmented block along the closing axis."""
    v, u = np.nonzero(mask)
    projection = u * np.cos(angle) + v * np.sin(angle)
    return float(np.ptp(projection) + 1)
