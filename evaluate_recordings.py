"""Read-only acceptance of grasp planning on independently labeled recordings."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from geometry import load_transform
from run import NoGrasp, find_plan, replay_observation
from wrist_pose import Wrist


def evaluate(config, manifest, transform, wrist=None, predictor=None):
    entries = json.loads(manifest.read_text(encoding="utf-8-sig"))
    files = [(manifest.parent / case["frame"]).resolve() for case in entries]
    if len(files) != len(set(files)) or not any(case["targets"] for case in entries) or not any(
        not case["targets"] for case in entries
    ):
        raise ValueError("Use unique captures with both pickable and no-target cases")

    results = []
    for case, file in zip(entries, files):
        boxes = np.asarray(case["targets"], float)
        expected = len(boxes) > 0
        result = dict(frame=case["frame"], condition=case["condition"], expected_target=expected)
        try:
            frame, r, t = replay_observation(file, transform, wrist)
            h, w = frame.depth_m.shape
            if expected and (boxes.ndim != 2 or boxes.shape[1] != 4
                             or not np.isfinite(boxes).all()
                             or np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > [w,h])
                             or np.any(boxes[:, :2] >= boxes[:, 2:])):
                raise ValueError("Target boxes must be valid half-open depth pixel bounds")
            axes = case.get("target_axes_deg")
            if axes is not None:
                axes = [np.asarray(options, float) for options in axes]
                if len(axes) != len(boxes) or any(
                    options.ndim != 1 or not options.size or not np.isfinite(options).all() for options in axes
                ):
                    raise ValueError("Provide a nonempty list of finite closing axes for each target box")
            positions = case.get("target_surface_mm")
            if positions is not None:
                positions = np.asarray(positions, float)
                if positions.shape != (len(boxes), 3) or not np.isfinite(positions).all():
                    raise ValueError("Provide one finite XYZ surface point per target box in base millimeters")
                tolerance = float(case.get("surface_tolerance_mm", 5.))
                if not np.isfinite(tolerance) or tolerance <= 0:
                    raise ValueError("Surface tolerance must be finite and positive")
            candidate, _, surface, width, _ = find_plan(frame, config, r, t, predictor)
            pixel = [candidate.u, candidate.v]
            matches = np.flatnonzero(np.all((pixel >= boxes[:, :2]) & (pixel < boxes[:, 2:]), axis=1)) if expected else []
            matched = len(matches) > 0
            status = "correct_target" if matched else "wrong_target" if expected else "false_pick"
            angle_deg = math.degrees(candidate.angle)
            if matched and (axes is not None or positions is not None) and len(matches) != 1:
                raise ValueError("Pose-annotated target boxes overlap at the grasp center")
            if matched and axes is not None:
                error_deg = min(abs(math.remainder(angle_deg - reference, 180.)) for reference in axes[matches[0]])
                result["closing_axis_error_deg"] = error_deg
                if error_deg > 10:
                    status = "wrong_grasp_axis"
            if matched and positions is not None:
                delta = surface - positions[matches[0]]
                error_mm = float(np.linalg.norm(delta))
                result.update(surface_delta_mm=delta.tolist(), surface_error_mm=error_mm,
                              surface_tolerance_mm=tolerance)
                if error_mm > tolerance and status == "correct_target":
                    status = "wrong_grasp_position"
            result.update(outcome=status, pixel=pixel, surface_mm=surface.tolist(), width_mm=width,
                          closing_axis_deg=angle_deg, target_class=candidate.target_class)
        except NoGrasp as error:
            result.update(outcome="miss" if expected else "correct_rejection", reason=str(error))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            result.update(outcome="invalid_observation", reason=str(error))
        results.append(result)
    names = ("correct_target", "wrong_target", "wrong_grasp_axis", "wrong_grasp_position",
             "false_pick", "miss", "correct_rejection",
             "invalid_observation")
    counts = {name: sum(row["outcome"] == name for row in results) for name in names}
    return dict(passed=counts["correct_target"] + counts["correct_rejection"] == len(results),
                counts=counts, trials=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--target-weights", required=True)
    parser.add_argument("--target-class", action="append", required=True)
    parser.add_argument("--upstream")
    parser.add_argument("--trusted-checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if bool(args.upstream) != bool(args.trusted_checkpoint):
        parser.error("Provide both GG-CNN arguments or neither")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    transform = load_transform(config["transform"])
    wrist = None
    if config.get("camera_mount", "fixed") == "wrist":
        wrist = Wrist(config["vendor"], config["joint_signs"], config["joint_offsets_rad"])
    from target_semantics import TargetDetector
    config["_detector"] = TargetDetector(args.target_weights, args.target_class, args.device)
    predictor = None
    if args.upstream:
        from ggcnn import Predictor
        predictor = Predictor(args.upstream, args.trusted_checkpoint, args.device)
    report = evaluate(config, args.manifest, transform, wrist, predictor)
    report.update(classes=args.target_class, target_weights=args.target_weights,
                  target_weights_sha256=hashlib.sha256(Path(args.target_weights).read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  grasp_checkpoint_sha256=(hashlib.sha256(Path(args.trusted_checkpoint).read_bytes()).hexdigest()
                                           if predictor else None),
                  backend="ggcnn" if predictor else "rectangle")
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"]))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()



