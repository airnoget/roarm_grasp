"""Compare independently marked simulation and hardware motion intervals."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


JOINTS = ("b", "s", "e", "t", "r")


def interpolate(times, joints, query):
    return np.column_stack([np.interp(query, times, values) for values in joints.T])


def segment(times, joints, bounds, label, max_gap):
    start, end = bounds
    if (times.ndim != 1 or joints.shape != (len(times), len(JOINTS))
            or len(times) < 3 or not np.isfinite(times).all()
            or not np.isfinite(joints).all() or np.any(np.diff(times) <= 0)):
        raise ValueError(f"{label}: expected finite joint samples with increasing timestamps")
    if not times[0] <= start < end <= times[-1]:
        raise ValueError(f"{label}: window must lie inside recorded timestamps")
    intervals = np.diff(times)[(times[:-1] < end) & (times[1:] > start)]
    if np.max(intervals) > max_gap:
        raise ValueError(f"{label}: sample gap exceeds {max_gap:.3f} s inside the window")
    window = np.r_[start, times[(times > start) & (times < end)], end]
    return window - start, interpolate(times, joints, window), float(np.median(intervals))


def compare(sim, real, config, sim_window, real_window, max_gap=.15):
    signs = np.asarray(config["joint_signs"], float)
    offsets = np.asarray(config["joint_offsets_rad"], float)
    if (signs.shape != (5,) or offsets.shape != (5,) or not np.isin(signs, [-1, 1]).all()
            or not np.isfinite(offsets).all()):
        raise ValueError("Expected five joint signs and finite radian offsets")
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("Maximum sample gap must be finite and positive")
    ts, qs, ds = segment(*sim, sim_window, "Simulation", max_gap)
    tr, qr, dr = segment(*real, real_window, "Hardware", max_gap)
    qr = qr * signs + offsets
    overlap = min(ts[-1], tr[-1])
    steps = int(np.floor(overlap / max(ds, dr) + 1e-9))
    if steps < 2:
        raise ValueError("Common motion interval is too short for comparison")
    grid = np.linspace(0, overlap, steps + 1)
    error = interpolate(tr, qr, grid) - interpolate(ts, qs, grid)
    # Finite differences compare average velocity on each shared interval.
    velocity_error = np.diff(error, axis=0) / (overlap / steps)
    return dict(
        sim_window_s=list(sim_window), hardware_window_s=list(real_window),
        comparison_s=float(overlap), hardware_minus_sim_duration_s=float(tr[-1] - ts[-1]),
        comparison_step_s=float(overlap / steps),
        joints={
            name: dict(angle_rmse_rad=float(np.sqrt(np.mean(error[:, i] ** 2))),
                       max_angle_error_rad=float(np.max(np.abs(error[:, i]))),
                       velocity_rmse_rad_s=float(np.sqrt(np.mean(velocity_error[:, i] ** 2))),
                       end_pose_delta_rad=float(qr[-1, i] - qs[-1, i]))
            for i, name in enumerate(JOINTS)
        },
    )


def load_hardware(path, stage=None):
    with path.open(encoding="utf-8-sig", newline="") as file:
        if path.suffix.lower() == ".jsonl":
            rows = []
            completed = 0
            for line in file:
                record = json.loads(line)
                if stage is not None and record.get("stage") != stage:
                    continue
                if record.get("motion_complete") is True:
                    completed += 1
                if record.get("motion_sample") is True:
                    rows.append([float(record["elapsed_s"]),
                                 *(float(record["feedback"][key]) for key in JOINTS)])
            if stage is not None and completed != 1:
                raise ValueError(f"Hardware log needs exactly one completed {stage} stage")
        else:
            reader = csv.DictReader(file)
            if not {"time_s", *JOINTS} <= set(reader.fieldnames or []):
                raise ValueError("Hardware CSV needs time_s,b,s,e,t,r")
            rows = [[float(row[key]) for key in ("time_s", *JOINTS)] for row in reader]
    samples = np.asarray(rows, float).reshape(-1, 6)
    if len(samples) < 3:
        raise ValueError("Hardware log needs at least three motion samples")
    return samples[:, 0], samples[:, 1:]


def stage_bounds(trace, stage):
    names = list(trace["stage_names"])
    times = trace["stage_time_s"]
    if names.count(stage) != 1 or len(times) != len(names):
        raise ValueError(f"Simulation trace needs exactly one {stage} stage")
    index = names.index(stage)
    if index + 1 >= len(times) or not times[index] < times[index + 1]:
        raise ValueError(f"Simulation stage {stage} has no valid end boundary")
    return float(times[index]), float(times[index + 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("sim_trace", type=Path)
    parser.add_argument("hardware_log", type=Path, help="Motion JSONL or joint feedback CSV")
    parser.add_argument("--sim-window", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--hardware-window", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--stage", choices=["descend", "proof_lift", "full_lift"])
    parser.add_argument("--max-gap", type=float, default=.15)
    args = parser.parse_args()
    if args.stage:
        if args.sim_window or args.hardware_window or args.hardware_log.suffix.lower() != ".jsonl":
            parser.error("--stage requires motion JSONL and replaces the two explicit windows")
    elif not (args.sim_window and args.hardware_window):
        parser.error("Provide both explicit windows or --stage")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        with np.load(args.sim_trace, allow_pickle=False) as trace:
            if not {"time_s", "arm_q_rad"} <= set(trace.files):
                raise ValueError("Simulation trace needs timestamps and arm joints; regenerate with --trace")
            sim = trace["time_s"], trace["arm_q_rad"]
            sim_window = stage_bounds(trace, args.stage) if args.stage else args.sim_window
        hardware = load_hardware(args.hardware_log, args.stage)
        hardware_window = (float(hardware[0][0]), float(hardware[0][-1])) if args.stage else args.hardware_window
        report = compare(sim, hardware, config, sim_window, hardware_window, args.max_gap)
        if args.stage:
            report["stage"] = args.stage
            report["hardware_duration_basis"] = "first_to_last_feedback"
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
