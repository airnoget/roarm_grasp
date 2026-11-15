import argparse
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
import json
import math
import sys

import cv2
import numpy as np

from camera import load_frame, orbbec, real_sense, save_frame
from geometry import base_points, load_transform
from perception import blocks, draw_grasp, rectangle_candidates
from planning import plan
from support_plane import estimate_support
from target_semantics import COMPETITION_TARGET, confirmed_components, support_pixels
from validate_wrist import require_acceptance


class NoGrasp(ValueError):
    """A valid observation produced no executable grasp."""

class SupportUnavailable(ValueError):
    """The requested station has no usable support in this observation."""


def pose_delta(first, second):
    distance = math.dist([first[axis] for axis in "xyz"], [second[axis] for axis in "xyz"])
    roll = abs(math.remainder(first["r"] - second["r"], 2 * math.pi))
    return distance, roll


def require_stable_hover(observations):
    for (surface_a, pose_a), (surface_b, pose_b) in combinations(observations, 2):
        distance, roll = pose_delta(pose_a, pose_b)
        if (math.dist(surface_a, surface_b) > 3 or distance > 3 or roll > math.radians(5)):
            raise RuntimeError("Target or grasp pose unstable at hover; stopped before descent")


def replay_observation(path, transform, wrist=None):
    frame = load_frame(path)
    if wrist is None:
        return frame, *transform
    with np.load(path, allow_pickle=False) as recording:
        if "joints_rad" not in recording:
            raise ValueError("Wrist recording requires joints_rad captured with --arm-port")
        return frame, *wrist.camera_pose(recording["joints_rad"], transform)


def find_plan(frame, cfg, rotation, translation, predictor, locked=None, locked_region=None):
    detector = cfg.get("_detector")
    detections = detector(frame) if detector is not None else []
    regions = cfg.get("support_regions_mm")
    if regions is None:
        stations = [(None, cfg)]
    else:
        regions = np.asarray(regions, dtype=float)
        if (regions.ndim != 3 or regions.shape[1:] != (2, 3) or not len(regions)
                or not np.isfinite(regions).all() or np.any(regions[:, 0] >= regions[:, 1])):
            raise ValueError("Support regions require finite increasing XYZ bounds")
        if any(np.all(np.maximum(a[0, :2], b[0, :2]) <= np.minimum(a[1, :2], b[1, :2]))
               for a, b in combinations(regions, 2)):
            raise ValueError("Support regions must have separate XY footprints")
        stations = [(index, dict(cfg, support_search_mm=bounds)) for index, bounds in enumerate(regions)]
    plans, failures = [], []
    segmented = []
    for index, station in stations:
        try:
            local_cfg, mask, components = segment_station(frame, station, rotation, translation, detections)
        except SupportUnavailable as error:
            if regions is None:
                raise
            failures.append(f"region {index}: {error}")
            continue
        segmented.append((index, local_cfg, mask, components))
    labels = None
    eligible_components = [component for _, _, _, components in segmented for component in components]
    if detector is not None:
        confirmed = list(confirmed_components(eligible_components, detections, detector.allowed))
        labels = {id(component): label for component, label in confirmed}
        eligible_components = [component for component, _ in confirmed]
    candidates = []
    if eligible_components:
        proposals = (predictor.predict(frame.depth_m, cfg["roi"], eligible_components, cfg["min_score"])
                     if predictor else rectangle_candidates(eligible_components))
        candidates = sorted(proposals, key=lambda candidate: candidate.score, reverse=True)
    for index, local_cfg, mask, components in segmented:
        if locked_region is not None and locked_region != index:
            continue
        try:
            result = select_candidate(frame, local_cfg, rotation, translation, locked,
                                      mask, components, labels, candidates)
        except NoGrasp as error:
            if regions is None:
                raise
            failures.append(f"region {index}: {error}")
            continue
        result[0].support_region = index
        plans.append(result)
    if not plans:
        raise NoGrasp("No executable grasp in configured support regions: " + "; ".join(failures))
    return max(plans, key=lambda result: result[0].score)


def segment_station(frame, cfg, rotation, translation, detections):
    x, y, size = cfg["roi"]
    h, w = frame.depth_m.shape
    if min(x, y) < 0 or size < 32 or x + size > w or y + size > h:
        raise ValueError("ROI must be a square inside the active depth frame")
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[y+2:y+size-2, x+2:x+size-2] = True
    base = base_points(frame, rotation, translation)
    valid = roi_mask & np.isfinite(frame.depth_m) & (frame.depth_m > 0)
    search = cfg.get("support_search_mm")
    if search is not None:
        background = support_pixels(valid, detections)
        try:
            support = estimate_support(np.where(background[..., None], base, np.nan), search)
        except ValueError as error:
            raise SupportUnavailable(str(error)) from error
        cfg = dict(cfg, table_z_mm=support.height_mm)
    height = base[..., 2] - cfg["table_z_mm"]
    minimum, maximum = cfg["block_height_mm"]
    mask = valid & (height >= minimum) & (height <= maximum)
    if search is not None:
        lower, upper = np.asarray(search)
        mask &= np.all((base[..., :2] >= lower[:2]) & (base[..., :2] <= upper[:2]), axis=-1)
    components = list(blocks(mask, cfg["min_area_px"]))
    return cfg, mask, components


def select_candidate(frame, cfg, rotation, translation, locked, mask, components, classes, candidates):
    if classes is not None:
        confirmed = [(component, classes[id(component)]) for component in components if id(component) in classes]
        if not confirmed:
            raise NoGrasp("No depth component confirmed as a requested target")
        components, names = zip(*confirmed)
    labels = np.full(frame.depth_m.shape, -1, dtype=np.int32)
    for i, component in enumerate(components):
        labels[component] = i
    matched = None
    matched_component = None
    for candidate in candidates:
        index = labels[candidate.v, candidate.u]
        if index < 0:
            continue
        if classes is not None:
            candidate.target_class = names[index]
        if locked is not None and candidate.target_class != locked[1]:
            continue
        try:
            commands, surface, width = plan(candidate, frame, rotation, translation, cfg, components[index])
        except ValueError as error:
            print(f"Rejected ({candidate.u}, {candidate.v}): {error}")
            continue
        if locked is not None and np.linalg.norm(surface - locked[0]) > 8:
            continue
        result = candidate, commands, surface, width, mask
        if locked is None or classes is None:
            return result
        if matched is None:
            matched, matched_component = result, index
        elif matched_component != index:
            raise NoGrasp("Multiple depth components match the original target")
    if matched is not None:
        return matched
    raise NoGrasp("No valid grasp candidate")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", choices=["realsense", "orbbec"])
    source.add_argument("--frame")
    parser.add_argument("--backend", choices=["rectangle", "ggcnn"])
    parser.add_argument("--upstream")
    parser.add_argument("--trusted-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-weights", help="Trusted local Ultralytics detection checkpoint")
    parser.add_argument("--target-class", action="append", default=[])
    parser.add_argument("--preview")
    parser.add_argument("--record-observations", type=Path,
                        help="Save live wrist observations to a new directory for offline replay")
    parser.add_argument("--sample-id", help="Mission-assigned sample identity for the evidence log")
    parser.add_argument("--store", action="store_true", help="Use the commissioned wrist storage route")
    parser.add_argument("--proof-lift-only", action="store_true",
                        help="Stop the bench cycle while holding the sample after the short proof lift")
    parser.add_argument("--motion-log", help="Write a new JSONL file for motion feedback")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--hover-only", action="store_true")
    args = parser.parse_args()
    if args.proof_lift_only and (not args.execute or args.store):
        parser.error("--proof-lift-only requires --execute without --store")
    if args.backend is None:
        args.backend = "ggcnn" if args.execute or args.upstream or args.trusted_checkpoint else "rectangle"
    if args.execute and args.backend != "ggcnn":
        parser.error("Automatic pick requires the GG-CNN grasp backend")
    if args.target_weights and not args.target_class:
        args.target_class = [COMPETITION_TARGET]
    if args.execute and (not args.target_weights or set(args.target_class) != {COMPETITION_TARGET}):
        parser.error("Automatic pick requires detector weights and only the artificial_object class")
    if args.backend == "ggcnn" and not (args.upstream and args.trusted_checkpoint):
        parser.error("GG-CNN requires --upstream and --trusted-checkpoint")
    with open(args.config, encoding="utf-8-sig") as file:
        cfg = json.load(file)
    if (args.execute or args.hover_only) and (not args.camera or cfg["commissioned"] is not True or not cfg["port"]):
        parser.error("Motion requires live camera, commissioned config and serial port")
    if args.sample_id is not None and (not args.sample_id.strip() or not args.store
                                       or not args.motion_log or not args.target_weights):
        parser.error("--sample-id needs --store, --motion-log and semantic gating")
    if args.store and (not args.execute or cfg.get("camera_mount") != "wrist"):
        parser.error("--store requires --execute and wrist camera mode")
    if args.execute:
        from grip_calibration import apply_report

        try:
            cfg = apply_report(cfg)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        limits = cfg.get("held_angle_rad")
        if (not limits or len(limits) != 2
                or not all(math.isfinite(x) for x in limits) or not limits[0] < limits[1]
                or not 0 < cfg.get("held_min_load", 0) < math.inf):
            parser.error("Full motion requires measured held_angle_rad and held_min_load")
    if (not 0 < cfg["speed"] <= 0.25 or cfg["roll_sign"] not in (-1, 1)
            or cfg["approach_mm"] <= 0):
        parser.error("Invalid arm parameters")
    if bool(args.target_weights) != bool(args.target_class):
        parser.error("Provide both --target-weights and --target-class for semantic gating")
    if args.target_weights:
        from target_semantics import TargetDetector

        cfg["_detector"] = TargetDetector(args.target_weights, args.target_class, args.device)
    transform = load_transform(cfg["transform"])
    wrist_mode = cfg.get("camera_mount", "fixed") == "wrist"
    if args.record_observations is not None and not (wrist_mode and args.camera):
        parser.error("--record-observations requires a live wrist camera")
    if wrist_mode and (args.execute or args.hover_only):
        from validate_tcp import require_compatible_tcp

        try:
            require_compatible_tcp(cfg)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if wrist_mode and args.camera and not cfg["port"]:
        parser.error("Wrist mode requires the robot feedback port")
    rotation, translation = transform
    predictor = None
    if args.backend == "ggcnn":
        from ggcnn import Predictor
        predictor = Predictor(args.upstream, args.trusted_checkpoint, args.device)

    wrist = None
    if wrist_mode:
        from wrist_pose import Wrist

        wrist = Wrist(cfg["vendor"], cfg["joint_signs"], cfg["joint_offsets_rad"])

    if args.frame:
        try:
            frame, rotation, translation = replay_observation(args.frame, transform, wrist)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        run_cycle(args, cfg, predictor, rotation, translation, frame)
    else:
        try:
            run_live(args, cfg, predictor, transform, wrist)
        except (OSError, RuntimeError, ValueError, SystemExit) as error:
            if args.motion_log and (not isinstance(error, SystemExit) or isinstance(error.code, str)):
                try:
                    with open(args.motion_log, "x", encoding="utf-8") as file:
                        record = dict(stage="before_pick", outcome="aborted", reason=str(error),
                                      sample_id=args.sample_id, arm_stowed=False)
                        file.write(json.dumps(record, allow_nan=False) + "\n")
                except FileExistsError:
                    pass  # The execution path already owns this log.
                except OSError as log_error:
                    print(f"Could not write motion failure log: {log_error}", file=sys.stderr)
            raise


def run_live(args, cfg, predictor, transform, wrist):
    camera = {"realsense": real_sense, "orbbec": orbbec}[args.camera]
    with_color = bool(args.target_weights)
    if wrist is not None:
        from arm import Arm
        from wrist_pose import stationary_frame
        with camera(with_color=with_color) as read, Arm(cfg["port"]) as arm:
            recorded = 0
            if args.record_observations is not None:
                args.record_observations.mkdir()

            def observe(record_name=None):
                nonlocal recorded
                frame, joints = stationary_frame(arm, read)
                if args.record_observations is not None:
                    output = args.record_observations / (record_name or f"{recorded:03d}.npz")
                    with output.open("xb") as file:
                        save_frame(file, frame, joints)
                    recorded += 1
                return frame, *wrist.camera_pose(joints, transform)

            frame, r, t = observe()
            if args.execute or args.hover_only:
                require_acceptance(cfg, args.camera, frame)
            run_cycle(args, cfg, predictor, r, t, frame, read, arm, observe)
        return
    with camera(with_color=with_color) as read:
        for _ in range(15):
            frame = read()
        run_cycle(args, cfg, predictor, *transform, frame, read)


def run_cycle(args, cfg, predictor, rotation, translation, frame, read=None, connected_arm=None, observe=None):
    try:
        candidate, commands, surface, width, mask = find_plan(frame, cfg, rotation, translation, predictor)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(dict(pixel=[candidate.u, candidate.v], score=candidate.score,
                          surface_mm=surface.tolist(), width_mm=width,
                          model_width_px=candidate.model_width_px, target_class=candidate.target_class,
                          support_region=candidate.support_region,
                          commands=commands),
                     indent=2, allow_nan=False))
    if args.preview:
        depth = np.where(np.isfinite(frame.depth_m), frame.depth_m, 0)
        view = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                                 .astype(np.uint8), cv2.COLORMAP_TURBO)
        view[depth <= 0] = 0
        draw_grasp(view, candidate)
        if not cv2.imwrite(args.preview, view):
            raise OSError("Cannot write preview")
    if getattr(args, "store", False):
        from storage_execution import plan_storage

        plan_storage(cfg, commands[-1])
    if args.execute or args.hover_only:
        from arm import Arm

        connection = nullcontext(connected_arm) if connected_arm else Arm(cfg["port"])
        with connection as arm:
            lock_token = None
            if cfg.get("require_base_interlock") is True:
                from base_handshake import lock_base

                lock_token = lock_base(cfg)
            arm.check_start(commands[0], cfg)
            arm.move(commands[0], cfg)
            if args.hover_only:
                print("Reached observation pose")
                return
            for _ in range(5):
                read()
            verified = []
            for _ in range(3):
                if observe:
                    frame, r, t = observe()
                    require_acceptance(cfg, args.camera, frame)
                else:
                    frame, r, t = read(), rotation, translation
                try:
                    _, new_commands, new_surface, _, _ = find_plan(
                        frame, cfg, r, t, predictor, locked=(surface, candidate.target_class),
                        locked_region=candidate.support_region,
                    )
                except NoGrasp as error:
                    raise RuntimeError("Original target was not recovered at hover") from error
                distance, angle = pose_delta(new_commands[1], commands[1])
                if math.dist(new_surface, surface) > 8 or distance > 8 or angle > math.radians(10):
                    raise RuntimeError("Target changed or occluded at hover; stopped before descent")
                verified.append((new_surface, new_commands[1]))
            require_stable_hover(verified)
            from pick_execution import execute_pick

            def observe_bin():
                observation = observe()
                require_acceptance(cfg, args.camera, observation[0])
                return observation

            outcome = execute_pick(arm, cfg, new_commands, getattr(args, "motion_log", None),
                                   store=getattr(args, "store", False),
                                   proof_only=getattr(args, "proof_lift_only", False),
                                   observe_proof=(lambda: observe("proof-lift.npz"))
                                   if observe and getattr(args, "record_observations", None) else None,
                                   observe_bin=observe_bin if observe else None,
                                   sample_id=getattr(args, "sample_id", None),
                                   target_class=candidate.target_class, base_lock_token=lock_token)
        print(outcome)


if __name__ == "__main__":
    main()


















