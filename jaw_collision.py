"""Collision pieces cut from official M3 meshes, preserving the hinge and visuals."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull


def add_jaw_collision(asset, body, path, link):
    # Split across the wrist/finger transition so the convex hull cannot bridge it.
    axis, cuts = (2, [.061, .087]) if link == "link5" else (0, [.026, .047])
    data = path.read_bytes()
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(data, dtype=record, offset=84)["vertices"].astype(float) * .001
    vertices = triangles.reshape(-1, 3)
    edges = triangles[:, [1, 2, 0]] - triangles
    sections = []
    for cut in cuts:
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = (cut - triangles[..., axis]) / edges[..., axis]
        valid = (alpha > 0) & (alpha < 1)
        sections.append(triangles[valid] + alpha[valid, None] * edges[valid])
    bounds = [-np.inf, *cuts, np.inf]
    for i, (low, high) in enumerate(zip(bounds, bounds[1:])):
        in_band = vertices[(vertices[:, axis] >= low) & (vertices[:, axis] <= high)]
        boundary = sections[max(i - 1, 0):min(i + 1, len(sections))]
        points = np.unique(np.concatenate([in_band, *boundary]).round(7), axis=0)
        hull = points[ConvexHull(points).vertices]
        name = f"{link}_{'tip' if i == 2 else 'section_' + str(i)}"
        ET.SubElement(asset, "mesh", name=name, vertex=" ".join(map(str, hull.flat)))
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, mass="0",
                      group="1", friction="1.3 .02 .001", rgba="0 0 0 0")
