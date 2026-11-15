"""Validate stationary wrist localization and bind evidence to calibration inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from camera import load_frame
from geometry import load_transform, sample
from wrist_pose import Wrist


def calibration_id(config):
    vendor = Path(config["vendor"])
    model = vendor / "src/roarm_main/roarm_description/urdf/roarm_m3/roarm_m3.xacro"
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (model, Path(config["transform"]))]
    hashes.extend([config["joint_signs"], config["joint_offsets_rad"]])
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def ray_id(rays):
    array = np.ascontiguousarray(rays, dtype="<f4")
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(array)
    return digest.hexdigest()


def require_acceptance(config, camera, frame):
    path = config.get("wrist_acceptance")
    if not path:
        raise ValueError("Wrist motion requires a localization acceptance report")
    report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (report.get("schema") != 1 or report.get("passed") is not True
            or report.get("calibration_id") != calibration_id(config)
            or report.get("camera") != camera or report.get("ray_id") != ray_id(frame.rays)):
        raise ValueError("Wrist acceptance does not match current calibration or depth stream")


def validate(config, manifest, camera, max_error, max_spread):
    hand_eye = load_transform(config["transform"])
    wrist = Wrist(config["vendor"], config["joint_signs"], config["joint_offsets_rad"])
    groups = {}
    profiles = set()
    captures = set()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            path = (manifest.parent / row["frame"]).resolve()
            target = row["target_id"].strip()
            if not target or (target, path) in captures:
                raise ValueError("Each target needs a unique capture per observation")
            captures.add((target, path))
            frame = load_frame(path)
            profiles.add(ray_id(frame.rays))
            with np.load(path, allow_pickle=False) as snapshot:
                q = snapshot["joints_rad"]
            rotation, translation = wrist.camera_pose(q, hand_eye)
            point = rotation @ (sample(frame, int(row["u"]), int(row["v"])) * 1000) + translation
            truth = np.array([float(row[f"{axis}_mm"]) for axis in "xyz"])
            if not np.isfinite(truth).all():
                raise ValueError("Measured reference coordinates must be finite")
            groups.setdefault(target, []).append((q, point, truth))
    if len(groups) < 2 or len(profiles) != 1:
        raise ValueError("Use at least two targets and one consistent depth stream")
    targets = []
    references = []
    for name, rows in groups.items():
        angles, points, truths = map(np.asarray, zip(*rows))
        if not np.allclose(truths, truths[0], rtol=0, atol=.01):
            raise ValueError(f"Inconsistent measured coordinates for {name}")
        if len(rows) < 3 or np.max(np.ptp(angles, axis=0)) < .12:
            raise ValueError(f"{name} requires at least three views with 0.12 rad joint span")
        pose_distances = np.max(np.abs(angles[:, None] - angles), axis=2)
        if np.any(pose_distances[np.triu_indices(len(rows), k=1)] < .02):
            raise ValueError(f"{name} contains duplicate or nearly identical views")
        references.append(truths[0])
        error = float(np.linalg.norm(points - truths, axis=1).max())
        spread = float(np.linalg.norm(points[:, None] - points, axis=2).max())
        targets.append(dict(target_id=name, views=len(rows), maximum_error_mm=error,
                            maximum_spread_mm=spread, estimated_base_mm=points.tolist(),
                            measured_base_mm=truths[0].tolist(),
                            passed=error <= max_error and spread <= max_spread))
    if np.linalg.norm(np.ptp(references, axis=0)) < 20:
        raise ValueError("Measured targets must span at least 20 mm in the workspace")
    return dict(schema=1, passed=all(item["passed"] for item in targets),
                calibration_id=calibration_id(config), camera=camera, ray_id=profiles.pop(),
                max_error_mm=max_error, max_spread_mm=max_spread, targets=targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--camera", choices=["realsense", "orbbec"], required=True)
    parser.add_argument("--max-error-mm", type=float, default=5)
    parser.add_argument("--max-spread-mm", type=float, default=4)
    args = parser.parse_args()
    if not (0 < args.max_error_mm <= 5 and 0 < args.max_spread_mm <= 4):
        parser.error("Maximum acceptance limits are 5 mm absolute and 4 mm spread")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    report = validate(cfg, args.manifest, args.camera, args.max_error_mm, args.max_spread_mm)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        parser.exit(2, "Wrist localization exceeded acceptance limits\n")


if __name__ == "__main__":
    main()

