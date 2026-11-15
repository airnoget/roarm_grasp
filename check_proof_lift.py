"""Measure object rise from two wrist frames with manually matched points."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from geometry import load_transform, sample
from run import replay_observation
from wrist_pose import Wrist


def measure(before, after, object_pixels, reference_pixels, minimum_rise_mm, max_reference_error_mm):
    if not all(math.isfinite(value) and value > 0 for value in
               (minimum_rise_mm, max_reference_error_mm)):
        raise ValueError("Rise and reference tolerances must be finite and positive")
    if not np.array_equal(before[0].rays, after[0].rays):
        raise ValueError("Proof lift observations require the same calibrated depth rays")

    def point(observation, pixel):
        frame, rotation, translation = observation
        return rotation @ (sample(frame, *pixel) * 1000) + translation

    def displacement(pixels):
        return point(after, pixels[2:]) - point(before, pixels[:2])

    object_delta = displacement(object_pixels)
    reference_delta = displacement(reference_pixels)
    reference_error = float(np.linalg.norm(reference_delta))
    return dict(
        passed=bool(object_delta[2] >= minimum_rise_mm and reference_error <= max_reference_error_mm),
        object_delta_mm=object_delta.tolist(), reference_delta_mm=reference_delta.tolist(),
        minimum_rise_mm=minimum_rise_mm, max_reference_error_mm=max_reference_error_mm,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--object-pixels", nargs=4, type=int, required=True)
    parser.add_argument("--reference-pixels", nargs=4, type=int, required=True)
    parser.add_argument("--minimum-rise-mm", type=float, default=7.)
    parser.add_argument("--max-reference-error-mm", type=float, default=3.)
    args = parser.parse_args()
    try:
        if args.before.resolve() == args.after.resolve():
            raise ValueError("Use two different captures")
        timestamps = []
        for path in (args.before, args.after):
            with np.load(path, allow_pickle=False) as recording:
                timestamp = recording["capture_time_ns"]
                if timestamp.shape != () or not np.issubdtype(timestamp.dtype, np.integer):
                    raise ValueError("Each frame needs an integer capture_time_ns")
                timestamps.append(int(timestamp))
        if not 0 < timestamps[0] < timestamps[1]:
            raise ValueError("Proof lift capture must be newer than the before capture")
        cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if cfg.get("camera_mount") != "wrist":
            raise ValueError("Use a calibrated wrist configuration")
        transform = load_transform(cfg["transform"])
        wrist = Wrist(cfg["vendor"], cfg["joint_signs"], cfg["joint_offsets_rad"])
        before, after = [replay_observation(path, transform, wrist) for path in (args.before, args.after)]
        report = measure(before, after, args.object_pixels, args.reference_pixels,
                         args.minimum_rise_mm, args.max_reference_error_mm)
        report["capture_interval_s"] = (timestamps[1] - timestamps[0]) / 1e9
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
