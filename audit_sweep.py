"""Offline collision screening of sampled M3-S joint feedback against measured boxes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim_model import add_wrist_envelopes, scene


CHANNELS = ("b", "s", "e", "t", "r", "g")


def audit_model(config, workspace):
    root = ET.fromstring(scene(config["vendor"], (.31, 0), 0, "vendor-split"))
    world = root.find("worldbody")
    world.remove(world.find("geom[@name='table']"))
    world.remove(world.find("body[@name='block']"))
    wrist = world.find(".//body[@name='link5']")
    for geom in list(wrist.findall("geom")):
        if geom.attrib["name"].startswith("camera_"):
            wrist.remove(geom)

    padding = float(workspace["padding_mm"])
    if not 0 <= padding <= 30:
        raise ValueError("Workspace padding must be between 0 and 30 mm")

    def boxes(parent, items, prefix, group, contype="1", conaffinity="1"):
        if not items:
            raise ValueError(f"At least one measured {prefix} box is required")
        names = set()
        for item in items:
            name = item["name"]
            center, size = np.asarray(item["center_mm"], float), np.asarray(item["size_mm"], float)
            if (not name or name in names or center.shape != (3,) or size.shape != (3,)
                    or not np.isfinite([center, size]).all() or np.any(size <= 0)):
                raise ValueError(f"Invalid {prefix} box")
            names.add(name)
            ET.SubElement(parent, "geom", name=f"{prefix}_{name}", type="box",
                          pos=" ".join(map(str, center / 1000)),
                          size=" ".join(map(str, (size / 2 + padding) / 1000)),
                          group=group, mass="0", contype=contype, conaffinity=conaffinity)

    boxes(world, workspace["obstacles"], "fixture", "0", conaffinity="3")
    add_wrist_envelopes(wrist, workspace)
    payload = workspace.get("payload_envelope")
    if payload is not None:
        # Use a separate collision channel: fixture contact is relevant, grasp contact is intended.
        boxes(wrist, [payload], "payload", "1", contype="2", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def audit(config, workspace, rows):
    samples = np.asarray([[float(row[key]) for key in ("time_s", *CHANNELS)] for row in rows])
    if samples.ndim != 2 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("At least two finite feedback samples are required")
    gaps = np.diff(samples[:, 0])
    signs = np.asarray([*config["joint_signs"], workspace["jaw_sign"]], float)
    offsets = np.asarray([*config["joint_offsets_rad"], workspace["jaw_offset_rad"]], float)
    if signs.shape != (6,) or offsets.shape != (6,) or not np.isin(signs, [-1, 1]).all() or not np.isfinite(offsets).all():
        raise ValueError("Expected five calibrated arm mappings and one jaw mapping")
    q = samples[:, 1:] * signs + offsets
    jumps = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    model = audit_model(config, workspace)
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    address = model.jnt_qposadr[joints]
    low, high = model.jnt_range[joints].T
    errors = []
    tested = 0
    for index, (start, end) in enumerate(zip(q, q[1:])):
        if not 0 < gaps[index] <= .15 or jumps[index] > .10:
            errors.append(dict(segment=index, reason="undersampled", gap_s=float(gaps[index]),
                               joint_jump_rad=float(jumps[index])))
            continue
        subdivisions = max(1, int(np.ceil(jumps[index] / .01)))
        for alpha in np.linspace(0, 1, subdivisions + 1):
            angles = start + alpha * (end - start)
            tested += 1
            if np.any(angles < low) or np.any(angles > high):
                errors.append(dict(segment=index, fraction=float(alpha), reason="joint_limit"))
                break
            data.qpos[address] = angles
            mujoco.mj_fwdPosition(model, data)
            collision = next((c for c in data.contact if c.dist <= 0), None)
            if collision is not None:
                names = [model.geom(i).name for i in (collision.geom1, collision.geom2)]
                reason = "payload_collision" if any(name.startswith("payload_") for name in names) else "collision"
                errors.append(dict(segment=index, fraction=float(alpha), reason=reason, geoms=names))
                break
    return dict(passed=not errors, samples=len(samples), checked_states=tested,
                payload_checked=workspace.get("payload_envelope") is not None, failures=errors)


def load_motion_stage(stream, stage):
    rows = []
    command_at = None
    complete_at = None
    for line in stream:
        record = json.loads(line)
        if record.get("stage") != stage:
            continue
        if "command" in record:
            if command_at is not None or complete_at is not None or rows:
                raise ValueError(f"{stage}: expected one command before the motion samples")
            command_at = float(record["elapsed_s"])
        if record.get("motion_sample") is True:
            if command_at is None or complete_at is not None:
                raise ValueError(f"{stage}: motion sample lies outside the command interval")
            rows.append(dict(time_s=record["elapsed_s"], **record["feedback"]))
        if record.get("motion_complete") is True:
            if not rows or complete_at is not None:
                raise ValueError(f"{stage}: invalid motion completion marker")
            complete_at = float(record["elapsed_s"])
    if command_at is None or complete_at is None or len(rows) < 2:
        raise ValueError(f"{stage}: need a completed command and at least two motion samples")
    if not (0 <= float(rows[0]["time_s"]) - command_at <= .15
            and 0 <= complete_at - float(rows[-1]["time_s"]) <= .15):
        raise ValueError(f"{stage}: motion samples do not cover the command boundaries")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--motion-stage", action="append", choices=["proof_lift", "full_lift"],
                        help="Audit one or both sampled lift stages from an execution JSONL")
    args = parser.parse_args()
    config, workspace = [json.loads(path.read_text(encoding="utf-8-sig"))
                         for path in (args.config, args.workspace)]
    is_motion_log = args.trace.suffix.lower() == ".jsonl"
    if is_motion_log != bool(args.motion_stage):
        parser.error("--motion-stage is required for motion JSONL and cannot be used with CSV")
    with args.trace.open(newline="", encoding="utf-8-sig") as stream:
        if is_motion_log:
            reports = {}
            for stage in dict.fromkeys(args.motion_stage):
                stream.seek(0)
                reports[stage] = audit(config, workspace, load_motion_stage(stream, stage))
            report = dict(passed=all(result["passed"] for result in reports.values()),
                          payload_checked=all(result["payload_checked"] for result in reports.values()),
                          motion_stages=reports)
        else:
            report = audit(config, workspace, list(csv.DictReader(stream)))
    report["inputs_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("config", args.config), ("workspace", args.workspace), ("trace", args.trace))
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


