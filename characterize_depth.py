"""Measure a conservative depth stress profile against known camera-frame planes."""
import argparse
import json
from pathlib import Path

import numpy as np

from sensor_profile import SensorProfile


def measure(folder, entry):
    plane = np.asarray(entry["plane"], dtype=float)
    if (plane.shape != (4,) or not np.isfinite(plane).all()
            or not np.isclose(np.linalg.norm(plane[:3]), 1, atol=1e-4) or plane[3] <= 0):
        raise ValueError("Expected unit normal and positive distance in plane [nx,ny,nz,d_mm]")
    with np.load(folder / entry["file"], allow_pickle=False) as data:
        frames, rays = data["depth_stack_m"], data["rays"]
    if (frames.ndim != 3 or len(frames) < 3 or rays.shape != (*frames.shape[1:], 3)
            or not np.isfinite(rays).all()):
        raise ValueError("Expected at least 3 depth frames with a common ray map")
    x, y, w, h = entry["roi"]
    if min(x, y) < 0 or min(w, h) < 8 or y+h > frames.shape[1] or x+w > frames.shape[2]:
        raise ValueError("ROI must be inside the depth frame and at least 8x8")
    cosines = rays[y:y+h, x:x+w] @ plane[:3]
    if np.any(cosines <= .05):
        raise ValueError("Calibration plane does not face the requested camera rays")
    expected = plane[3] / cosines
    measured = frames[:, y:y+h, x:x+w] * 1000
    valid = np.isfinite(measured) & (measured > 0)
    if not valid.any():
        raise ValueError(f"No valid target depth in {entry['file']}")
    residual = (measured - expected)[valid]
    bias = float(np.median(residual))
    return dict(file=entry["file"], min_mm=float(expected.min()), max_mm=float(expected.max()),
                valid_samples=int(valid.sum()), dropout=float(1 - valid.mean()),
                bias_mm=bias, sigma_mm=float(1.4826 * np.median(np.abs(residual - bias))),
                abs_error_p95_mm=float(np.quantile(np.abs(residual), .95)))


def fit_profile(entries, checks, max_dropout, max_error):
    bias = float(np.median([row["bias_mm"] for row in entries]))
    sigma = max(np.hypot(row["sigma_mm"], row["bias_mm"] - bias) for row in entries)
    low, high = min(row["min_mm"] for row in entries), max(row["max_mm"] for row in entries)
    profile = SensorProfile(min_range_mm=low, max_range_mm=high, depth_bias_mm=bias,
                            depth_sigma_mm=float(sigma), dropout=max(row["dropout"] for row in entries))
    reasons = []
    for row in [*entries, *checks]:
        if row["dropout"] > max_dropout or row["abs_error_p95_mm"] > max_error:
            reasons.append(f"{row['file']}: depth quality exceeded limits")
    for row in checks:
        if row["min_mm"] < low - .001 or row["max_mm"] > high + .001:
            reasons.append(f"{row['file']}: outside the characterized interval")
        if abs(row["bias_mm"] - bias) > max(2., 3 * sigma):
            reasons.append(f"{row['file']}: systematic bias differs from fit")
        if row["sigma_mm"] > max(sigma + .5, 1.5 * sigma):
            reasons.append(f"{row['file']}: depth noise exceeds fitted profile")
        if row["dropout"] > profile.dropout + .03:
            reasons.append(f"{row['file']}: dropout exceeds fitted profile")
    return profile, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-dropout", type=float, default=.2)
    parser.add_argument("--max-error-mm", type=float, default=8)
    args = parser.parse_args()
    if not 0 <= args.max_dropout < 1 or not 0 < args.max_error_mm < np.inf:
        parser.error("Invalid validation thresholds")
    if args.output.resolve() == args.report.resolve():
        parser.error("Profile and report paths must differ")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    fit, check = manifest["fit"], manifest["check"]
    files = [(args.manifest.parent / row["file"]).resolve() for row in [*fit, *check]]
    if len(fit) < 2 or not check or len(set(files)) != len(files):
        parser.error("Use at least two fit captures and one independent check capture")
    entries, checks = ([measure(args.manifest.parent, row) for row in group] for group in (fit, check))
    profile, reasons = fit_profile(entries, checks, args.max_dropout, args.max_error_mm)
    report = dict(passed=not reasons, reasons=reasons, fit=entries, check=checks,
                  profile=vars(profile), max_dropout=args.max_dropout, max_error_mm=args.max_error_mm)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if reasons:
        parser.exit(2, "Depth characterization failed. See report.\n")
    args.output.write_text(json.dumps(vars(profile), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

