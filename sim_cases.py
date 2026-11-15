"""Load measured cuboid sample cases for physics simulation."""
import json
import math
import struct

import numpy as np


def validate_sample_mesh(path, size_m):
    data = path.read_bytes()
    if path.suffix.lower() != ".stl" or len(data) < 84:
        raise ValueError("Sample mesh must be a binary STL in millimeters")
    count = struct.unpack_from("<I", data, 80)[0]
    if count < 4 or len(data) != 84 + 50 * count:
        raise ValueError("Invalid binary STL triangle count")
    records = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("flags", "<u2")])
    vertices = np.frombuffer(data, dtype=records, offset=84)["vertices"].reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError("Sample STL contains nonfinite coordinates")
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    if (not np.allclose(high - low, np.asarray(size_m) * 1000, atol=1, rtol=0)
            or not np.allclose((low + high) / 2, 0, atol=.5, rtol=0)):
        raise ValueError("Sample STL must be centered and match size_mm within 1 mm")


def load_cases(path):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Case manifest requires a nonempty JSON array")
    cases = []
    names = set()
    for row in rows:
        name = row["id"]
        xy = tuple(float(value) / 1000 for value in row["xy_mm"])
        size = tuple(float(value) / 1000 for value in row["size_mm"])
        yaw = math.radians(float(row["yaw_deg"]))
        mass = float(row["mass_g"]) / 1000
        height = float(row.get("support_height_mm", 0)) / 1000
        clutter = row.get("clutter", False)
        if (not isinstance(name, str) or not name.strip() or name in names
                or len(xy) != 2 or len(size) != 3
                or not all(math.isfinite(value) for value in (*xy, *size, yaw, mass, height))
                or min(size) <= 0 or mass <= 0 or not 0 <= height < .5 or type(clutter) is not bool):
            raise ValueError(f"Invalid or duplicate measured case: {name!r}")
        names.add(name)
        mesh = (path.parent / row["mesh_stl"]).resolve() if row.get("mesh_stl") else None
        if mesh is not None:
            validate_sample_mesh(mesh, size)
        ledge = None
        if "ledge_box_mm" in row:
            ledge = tuple(float(value) / 1000 for value in row["ledge_box_mm"])
            if (len(ledge) != 5 or not all(math.isfinite(value) for value in ledge)
                    or min(ledge[2:]) <= 0 or height <= 0 or ledge[4] > height):
                raise ValueError("ledge_box_mm needs [center_x, center_y, length, width, thickness] and a positive support height")
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            extent = np.abs(rotation) @ (np.asarray(size[:2]) / 2)
            if np.any(np.abs(np.asarray(xy) - ledge[:2]) + extent > np.asarray(ledge[2:4]) / 2):
                raise ValueError("Sample footprint lies outside the measured ledge")
        cases.append((xy, yaw, size, mass, height or None, clutter, name, mesh, ledge))
    return cases
