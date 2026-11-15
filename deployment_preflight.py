"""Check deployment evidence without opening a camera or serial port."""
import argparse
import hashlib
import json
from pathlib import Path

from camera import load_frame
from grip_calibration import apply_report
from target_semantics import COMPETITION_TARGET
from validate_tcp import require_compatible_tcp
from validate_wrist import require_acceptance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def inspect(args):
    config = json.loads(read_json(args.config))
    config_id = digest(args.config)
    checks = []

    def check(name, fn):
        try:
            fn()
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks.append(dict(name=name, passed=False, reason=str(error)))
        else:
            checks.append(dict(name=name, passed=True))

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def mode():
        require(config.get("commissioned") is True and config.get("camera_mount") == "wrist"
                and bool(config.get("port")), "Set commissioned wrist mode and the real feedback port")

    def wrist():
        require_acceptance(config, args.camera, load_frame(args.rgbd))

    def tcp():
        require_compatible_tcp(dict(config, require_physical_tcp=True))

    def grip():
        require(bool(config.get("grip_acceptance")), "No independent grip acceptance report configured")
        updated = apply_report(config)
        limits = updated["held_angle_rad"]
        require(len(limits) == 2 and limits[0] < limits[1] and updated["held_min_load"] > 0,
                "Invalid accepted grip thresholds")

    def semantics():
        require(set(args.target_class) == {COMPETITION_TARGET},
                "Automatic pick only deploys the artificial_object class")
        frame = load_frame(args.rgbd)
        require(frame.bgr is not None, "Deployment recording lacks aligned color")
        report = json.loads(read_json(args.target_report))
        require(report.get("passed") is True, "Target recording evaluation did not pass")
        require(report.get("config_sha256") == config_id, "Target evaluation used another configuration")
        require(report.get("target_weights_sha256") == digest(args.weights), "Target weights changed")
        require(set(report.get("classes", [])) == set(args.target_class), "Target classes changed")
        trials = report.get("trials", [])
        require(len({row["frame"] for row in trials}) == len(trials),
                "Target evidence repeats a capture")
        require(any(row.get("expected_target") is True for row in trials)
                and any(row.get("expected_target") is False for row in trials),
                "Target evidence must contain both target-present and no-target captures")
        require(all(
            (row.get("expected_target") is True and row.get("outcome") == "correct_target"
             and row.get("target_class") == COMPETITION_TARGET)
            or (row.get("expected_target") is False and row.get("outcome") == "correct_rejection")
            for row in trials
        ), "Target evidence contains a failed capture or another selected class")
        require(report.get("backend") == "ggcnn", "Automatic pick requires GG-CNN evaluation")
        require(report.get("grasp_checkpoint_sha256") == digest(args.grasp_checkpoint),
                "GG-CNN checkpoint changed")

    def storage():
        spec = config.get("storage") or {}
        require(spec.get("commissioned") is True and spec.get("verify_depth") is True
                and bool(spec.get("waypoints")), "Storage route and depth verification must be configured")
        from bin_depth import BinDepth

        BinDepth(spec)

    def stow():
        from stow_execution import TransportStow

        TransportStow(config)

    def sweep():
        report = json.loads(read_json(args.sweep_report))
        require(report.get("passed") is True and report.get("payload_checked") is True,
                "No passing carried-object sweep report")
        inputs = report["inputs_sha256"]
        require(inputs.get("config") == config_id, "Sweep used another configuration")
        require(inputs.get("workspace") == digest(args.workspace), "Measured workspace changed")
        require(inputs.get("trace") == digest(args.trace), "Recorded sweep trace changed")
        if Path(args.trace).suffix.lower() == ".jsonl":
            require(getattr(args, "grasp_only", False),
                    "Motion JSONL lift stages cover bench grasp only; mission needs a carry route trace")
            require(set(report.get("motion_stages", {})) == {"proof_lift", "full_lift"},
                    "Bench motion JSONL evidence must cover proof_lift and full_lift")

    grasp_only = getattr(args, "grasp_only", False)
    required = [("deployment_mode", mode), ("wrist_localization", wrist),
                ("physical_tcp", tcp), ("grip_feedback", grip), ("target_selection", semantics),
                ("recorded_carry_sweep", sweep)]
    if not grasp_only:
        required.extend([("storage_configuration", storage), ("transport_stow_configuration", stow)])
    for name, fn in required:
        check(name, fn)
    return dict(scope="offline_evidence_check", passed=all(row["passed"] for row in checks),
                profile="bench_grasp_only" if grasp_only else "mission_grasp_and_stow",
                checks=checks, config_sha256=config_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--rgbd", required=True, type=Path)
    parser.add_argument("--camera", choices=["orbbec", "realsense"], required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--target-class", action="append", choices=[COMPETITION_TARGET])
    parser.add_argument("--backend", choices=["ggcnn"], default="ggcnn")
    parser.add_argument("--grasp-checkpoint", type=Path, required=True)
    parser.add_argument("--target-report", required=True, type=Path)
    parser.add_argument("--sweep-report", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--grasp-only", action="store_true",
                        help="Validate bench pick/lift without commissioning storage and transport stow")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.target_class:
        args.target_class = [COMPETITION_TARGET]
    report = inspect(args)
    result = json.dumps(report, indent=2)
    print(result)
    if args.report:
        args.report.write_text(result, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


