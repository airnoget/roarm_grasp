"""Cross-check firmware Cartesian feedback against the pinned M3-S model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from wrist_pose import Wrist


def model_file(config):
    return Path(config["vendor"]) / "src/roarm_main/roarm_description/urdf/roarm_m3/roarm_m3.xacro"


def model_id(config):
    identity = [hashlib.sha256(model_file(config).read_bytes()).hexdigest(),
                config["joint_signs"], config["joint_offsets_rad"]]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def read_feedback(path, wrist):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = np.asarray([[float(row[k]) for k in ("b", "s", "e", "t", "r", "x", "y", "z")]
                           for row in csv.DictReader(stream)])
    if rows.ndim != 2 or len(rows) < 6 or not np.isfinite(rows).all():
        raise ValueError("Each dataset needs six or more finite poses with Cartesian feedback")
    selected = []
    for row in rows:
        if not selected or min(np.max(np.abs(row[:5]-previous[:5])) for previous in selected) >= .05:
            selected.append(row)
    if len(selected) < 6:
        raise ValueError("Record at least six distinct wrist poses per dataset")
    values = np.asarray(selected)
    rotation, translation = map(np.asarray, zip(*(wrist.fk(row[:5]) for row in values)))
    return rotation, translation, values[:, 5:], values[:, :5]


def check_tcp(config, fit, check, tolerance):
    if hashlib.sha256(fit.read_bytes()).digest() == hashlib.sha256(check.read_bytes()).digest():
        raise ValueError("Fit and check recordings must be independent")
    wrist = Wrist(config["vendor"], config["joint_signs"], config["joint_offsets_rad"])
    rotation, translation, firmware, fit_joints = read_feedback(fit, wrist)
    matrix = np.concatenate((rotation, np.broadcast_to(np.eye(3), rotation.shape)), axis=2).reshape(-1, 6)
    solution, _, rank, singular = np.linalg.lstsq(matrix, (firmware-translation).ravel(), rcond=None)
    if rank != 6 or singular[-1] / singular[0] < .06:
        raise ValueError("Insufficient orientation diversity to identify wrist TCP and base offset")
    tool, base_shift = solution[:3], solution[3:]
    joint = ET.parse(model_file(config)).find("joint[@name='link5_to_hand_tcp']/origin")
    nominal = np.fromstring(joint.attrib["xyz"], sep=" ") * 1000
    groups = {}
    for name, capture in [("fit", (rotation, translation, firmware, fit_joints)),
                          ("check", read_feedback(check, wrist))]:
        r, p, xyz, joints = capture
        if name == "check":
            separation = np.max(np.abs(joints[:, None] - fit_joints), axis=2)
            if np.count_nonzero(separation.min(axis=1) >= .05) < 3:
                raise ValueError("Check needs at least three poses absent from fit")
        nominal_errors = np.linalg.norm(r @ nominal + p - xyz, axis=1)
        fitted_errors = np.linalg.norm(r @ tool + p + base_shift - xyz, axis=1)
        groups[name] = dict(poses=len(xyz), nominal_max_mm=float(nominal_errors.max()),
                            fitted_max_mm=float(fitted_errors.max()),
                            nominal_errors_mm=nominal_errors.tolist(), fitted_errors_mm=fitted_errors.tolist())
    return dict(schema=1, model_id=model_id(config),
                passed=all(g["nominal_max_mm"] <= tolerance for g in groups.values()),
                offset_model_passed=all(g["fitted_max_mm"] <= tolerance for g in groups.values()),
                tolerance_mm=tolerance, groups=groups, nominal_tcp_link5_mm=nominal.tolist(),
                fitted_tcp_link5_mm=tool.tolist(), firmware_base_shift_mm=base_shift.tolist())


def check_probe(config, path, nominal, tolerance):
    """Compare firmware and URDF predictions with independently surveyed touches."""
    wrist = Wrist(config["vendor"], config["joint_signs"], config["joint_offsets_rad"])
    groups = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            reference = row["target_id"].strip()
            measured = np.array([float(row[f"tip_{axis}_mm"]) for axis in "xyz"])
            firmware = np.array([float(row[axis]) for axis in "xyz"])
            q = np.array([float(row[key]) for key in ("b", "s", "e", "t", "r")])
            if not reference or not np.isfinite([measured, firmware]).all():
                raise ValueError("Probe references and feedback must be finite")
            rotation, translation = wrist.fk(q)
            predicted = rotation @ nominal + translation
            groups.setdefault(reference, []).append((measured, firmware, predicted))
    if len(groups) < 3:
        raise ValueError("Measure at least three independent probe references")
    reports, positions = [], []
    for name, samples in groups.items():
        if len(samples) < 2:
            raise ValueError(f"Probe reference {name} needs at least two separate touches")
        truth, firmware, predicted = map(np.asarray, zip(*samples))
        if not np.allclose(truth, truth[0], rtol=0, atol=.01):
            raise ValueError(f"Probe reference {name} has conflicting coordinates")
        positions.append(truth[0])
        fw_error = float(np.linalg.norm(firmware-truth, axis=1).max())
        urdf_error = float(np.linalg.norm(predicted-truth, axis=1).max())
        reports.append(dict(target_id=name, touches=len(samples), reference_mm=truth[0].tolist(),
                            firmware_max_error_mm=fw_error, model_max_error_mm=urdf_error,
                            passed=max(fw_error, urdf_error) <= tolerance))
    points = np.asarray(positions)
    if np.linalg.svd(points - points.mean(axis=0), compute_uv=False)[1] < 10:
        raise ValueError("Probe references must span the workspace in at least two directions")
    return dict(passed=all(item["passed"] for item in reports), targets=reports,
                measurements_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require_compatible_tcp(config):
    path = config.get("tcp_compatibility_report")
    require_probe = config.get("require_physical_tcp", False)
    if require_probe and not path:
        raise ValueError("Physical TCP acceptance requires a compatibility report")
    if path:
        report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if (report.get("schema") != 1 or report.get("passed") is not True
                or report.get("model_id") != model_id(config)):
            raise ValueError("Firmware TCP report did not pass or no longer matches the model")
        probe = report.get("physical_probe")
        if (require_probe or probe is not None) and (not probe or probe.get("passed") is not True):
            raise ValueError("Physical probe acceptance is missing or failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("fit", type=Path)
    parser.add_argument("check", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--probe", type=Path, help="Independent physical probe touch CSV")
    parser.add_argument("--max-error-mm", type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.max_error_mm <= 5:
        parser.error("Max error must be in (0,5] mm")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    result = check_tcp(config, args.fit, args.check, args.max_error_mm)
    if args.probe:
        result["physical_probe"] = check_probe(config, args.probe,
                                               result["nominal_tcp_link5_mm"], args.max_error_mm)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"] or (args.probe and not result["physical_probe"]["passed"]):
        parser.exit(2, "TCP acceptance failed. See diagnostic report.\n")


if __name__ == "__main__":
    main()

