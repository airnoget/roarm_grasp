"""Collect read-only gripper telemetry and qualify a conservative held-state rule."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


LABELS = {"held", "empty", "jam"}


def held(state, limits, min_load):
    return limits[0] <= state["g"] <= limits[1] and abs(state["tG"]) >= min_load


def read_trials(manifest):
    spec = json.loads(manifest.read_text(encoding="utf-8-sig"))
    used = set()
    groups = {}
    for split in ("fit", "check"):
        trials = []
        for row in spec[split]:
            path = manifest.parent / row["file"]
            data = path.read_bytes()
            identity = hashlib.sha256(data).hexdigest()
            capture = json.loads(data)
            label, samples = capture["label"], capture["samples"]
            if identity in used or label not in LABELS or len(samples) < 3:
                raise ValueError("Use independent captures with at least three samples per trial")
            if not all(math.isfinite(s[key]) for s in samples for key in ("g", "tG")):
                raise ValueError("Nonfinite grip feedback")
            used.add(identity)
            trials.append((row["file"], label, samples))
        if {label for _, label, _ in trials} != LABELS:
            raise ValueError(f"{split} requires independently labeled held, empty and jam trials")
        if sum(label == "held" for _, label, _ in trials) < 2:
            raise ValueError(f"{split} needs two separately acquired held trials")
        groups[split] = trials
    return groups


def qualify(groups, config):
    positives = [sample for _, label, samples in groups["fit"]
                 if label == "held" for sample in samples]
    low = min(s["g"] for s in positives) - .03
    high = max(s["g"] for s in positives) + .03
    min_load = .9 * min(abs(s["tG"]) for s in positives)
    limits = [low, high]
    verdicts = []
    for split, trials in groups.items():
        for file, label, samples in trials:
            matches = sum(held(s, limits, min_load) for s in samples)
            passed = matches == len(samples) if label == "held" else matches == 0
            verdicts.append(dict(file=file, split=split, label=label, passed=passed,
                                 matching_samples=matches, samples=len(samples)))
    return dict(schema=1, passed=min_load > 0 and all(t["passed"] for t in verdicts),
                held_angle_rad=limits, held_min_load=min_load,
                open_rad=config["open_rad"], close_rad=config["close_rad"], trials=verdicts)


def apply_report(config):
    path = config.get("grip_acceptance")
    if not path:
        return config
    report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (report.get("schema") != 1 or report.get("passed") is not True
            or any(report.get(key) != config[key] for key in ("open_rad", "close_rad"))):
        raise ValueError("Grip report failed or does not match configured gripper angles")
    return dict(config, held_angle_rad=report["held_angle_rad"], held_min_load=report["held_min_load"])


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    capture = actions.add_parser("capture")
    capture.add_argument("port")
    capture.add_argument("label", choices=sorted(LABELS))
    capture.add_argument("output", type=Path)
    capture.add_argument("--count", type=int, default=10)
    fit = actions.add_parser("fit")
    fit.add_argument("config", type=Path)
    fit.add_argument("manifest", type=Path)
    fit.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        from arm import Arm

        if not 3 <= args.count <= 100:
            parser.error("--count must be between 3 and 100")
        if args.output.exists():
            parser.error("Capture path already exists")
        with Arm(args.port) as arm:
            samples = []
            for _ in range(args.count):
                samples.append(arm.feedback())
                time.sleep(.1)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(dict(label=args.label, samples=samples), file, indent=2)
        return
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    report = qualify(read_trials(args.manifest), config)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        parser.exit(2, "Held, empty and jam feedback failed separation. See report.\n")


if __name__ == "__main__":
    main()
