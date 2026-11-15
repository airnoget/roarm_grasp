import argparse
import json
import time
from pathlib import Path
from dataclasses import asdict

import cv2
import mujoco
import numpy as np

from camera import load_frame, median_depth_three
from geometry import base_points, component_opening, grasp_geometry, load_transform
from perception import blocks, draw_grasp, rectangle_candidates
from sim_camera import overhead_frame, wrist_frame
from sim_cases import load_cases
from sim_model import JAW_RATED_TORQUE_NM, VENDOR_COMMIT, scene
from sim_motion import Motion
from sim_observation import ObservationRejected, read_support, select_view
from sim_planning import plan_path, tilted_lift_extension
from sim_storage import deposit
from sensor_profile import SensorProfile, perturb
from target_semantics import COMPETITION_TARGET, confirmed_components


def trial(vendor, xy, yaw, predictor=None, gripper="parallel-pads", trace=None, camera_mode="wrist",
          block_size=(.046, .030, .030), block_mass=.020, lift_steps=12, jaw_limit_nm=JAW_RATED_TORQUE_NM, storage=False, ledge_height=None, sensor=None, sensor_seed=0, verify_storage=False, clutter=False, ledge_x_shift=0., ledge_y_shift=0., test_tilted_extension=False, test_horizontal_projection=False, detector=None, test_centering=False, test_depth_median=False, grip_friction=None, block_mesh=None, ledge_box=None, camera_transform=None, camera_rays=None, wrist_workspace=None, rgb_renderer='mujoco'):
    xy = (xy[0] + ledge_x_shift, xy[1] + ledge_y_shift)
    model = mujoco.MjModel.from_xml_string(scene(vendor, xy, yaw, gripper, camera_mode, block_size, block_mass, jaw_limit_nm, storage, ledge_height, clutter, ledge_x_shift, ledge_y_shift, grip_friction, block_mesh, ledge_box, camera_transform, wrist_workspace))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    motion = Motion(model, data)
    motion.trace = [] if trace else None
    seed = np.array([0., .6, 1.5, .8, 0.])
    stages = []

    def mark(stage):
        if trace:
            stages.append((stage, data.time))

    def save_trace():
        if trace:
            mark("finished")
            motion.sample_trace()
            times, poses, commands = zip(*motion.trace)
            stage_names, stage_times = zip(*stages)
            poses = np.asarray(poses)
            np.savez_compressed(trace, qpos=poses, time_s=np.asarray(times),
                                arm_q_rad=poses[:, motion.q], arm_command_rad=np.asarray(commands),
                                stage_names=np.asarray(stage_names), stage_time_s=np.asarray(stage_times),
                                xy=xy, yaw=yaw, gripper=gripper,
                                camera_mode=camera_mode, block_size=block_size, block_mass=block_mass,
                                jaw_limit_nm=jaw_limit_nm, storage=storage, clutter=clutter,
                                ledge_x_shift_m=ledge_x_shift, ledge_y_shift_m=ledge_y_shift,
                                grip_friction=np.nan if grip_friction is None else grip_friction,
                                block_mesh="" if block_mesh is None else str(block_mesh),
                                wrist_workspace=json.dumps(wrist_workspace),
                                camera_transform=np.asarray(camera_transform if camera_transform is not None else ()),
                                camera_rays=np.asarray(camera_rays if camera_rays is not None else ()),
                                ledge_box_m=np.asarray(ledge_box if ledge_box is not None else ()),
                                ledge_height=np.nan if ledge_height is None else ledge_height)

    observations = []
    def apply_sensor(frame, rotation, translation):
        if sensor is None:
            return frame, rotation, translation
        if test_depth_median:
            frames = [perturb(frame, rotation, translation, sensor, [sensor_seed, 0, index],
                              mount_seed=sensor_seed) for index in range(3)]
            return median_depth_three([sample[0] for sample in frames]), *frames[1][1:]
        return perturb(frame, rotation, translation, sensor, [sensor_seed, 0], mount_seed=sensor_seed)

    if camera_mode == "wrist":
        mark("observation")
        observe, observation_path, observations = select_view(
            motion, seed, ledge_height is not None, apply_sensor if ledge_height is not None else None, camera_rays
        )
        data.qpos[motion.grip_q] = motion.opening
        mujoco.mj_forward(model, data)
        for waypoint in observation_path:
            motion.step(waypoint, motion.opening, 10)
        motion.step(observe, motion.opening, 200)
        frame, rotation, translation = wrist_frame(model, data, with_color=detector is not None, rays=camera_rays, rgb_renderer=rgb_renderer)
        seed = data.qpos[motion.q].copy()
    else:
        frame, rotation, translation = overhead_frame(model, data)
    frame, rotation, translation = apply_sensor(frame, rotation, translation)
    detected = detector(frame) if detector is not None else []
    points, valid, support_estimate = read_support(frame, rotation, translation,
                                                   ledge_height is not None, detected)
    support = "table" if ledge_height is None else "rock_ledge"
    support_id = model.geom(support).id
    support_z = (data.geom_xpos[support_id, 2] + model.geom_size[support_id, 2]) * 1000
    heights = points[..., 2] - support_estimate.height_mm
    mask = valid & (heights > 10) & (heights < 70)
    if ledge_height is not None:
        # The rock face itself is outside the mapped ledge approach region.
        mask &= (points[..., 0] > 225) & (points[..., 0] < 350) & (np.abs(points[..., 1]) < 90)
    components = list(blocks(mask, 30))
    depth_components = len(components)
    detections = None
    if detector is not None:
        detections = [dict(label=label, xyxy=np.asarray(box).tolist()) for label, box in detected]
        components = [component for component, _ in confirmed_components(components, detected, detector.allowed)]
        if not components:
            save_trace()
            return dict(passed=False, failure="semantic_target_not_confirmed", depth_components=depth_components,
                        detections=detections, camera_axes_base=rotation.tolist(),
                        camera_origin_base_mm=translation.tolist()), frame, None
    if predictor is None:
        candidates = list(rectangle_candidates(components))
    else:
        candidates = list(predictor.predict(frame.depth_m, (0, 0, frame.depth_m.shape[0]), components, .25,
                                            surface_z_mm=points[..., 2] if ledge_height is not None else None))
    rejected = []
    localized = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        try:
            component = next(mask for mask in components if mask[candidate.v, candidate.u])
            surface, heading, width_mm = grasp_geometry(
                frame, candidate, rotation, translation, horizontal_surface=test_horizontal_projection
            )
            correction = np.zeros(2)
            if test_centering and ledge_height is not None:
                samples = points[component]
                upper = samples[:, 2] >= np.percentile(samples[:, 2], 85) - 5
                offset = np.median(samples[upper, :2], axis=0) - surface[:2]
                correction = offset * (4. / max(4., np.linalg.norm(offset)))
                surface[:2] += correction
            opening_mm = max(width_mm, component_opening(frame, component, rotation, heading))
            localized.append(dict(pixel=[candidate.u, candidate.v], surface_base_mm=surface.tolist()))
            if not 0 < opening_mm <= 55:
                raise ValueError("Candidate exceeds 55 mm opening")
            heading = (heading + np.pi/2) % np.pi - np.pi/2
            target, path, transit, depth, pitch = plan_path(
                motion, surface, heading, width_mm, seed, gripper, camera_mode == "wrist", support_estimate.height_mm,
                tilted=ledge_height is not None
            )
        except ValueError as error:
            rejected.append(str(error))
            continue
        break
    else:
        reasons = " | ".join(dict.fromkeys(rejected))
        if ledge_height is None:
            raise ValueError("No executable grasp candidate" + (f": {reasons}" if reasons else ""))
        result = dict(passed=False, failure="grasp_path_rejected",
                      next_action="change_approach_or_reposition_base",
                      error=reasons or "No executable grasp candidate",
                      observation_attempts=observations,
                      support_estimate_mm=support_estimate.height_mm,
                      support_error_mm=support_estimate.height_mm-support_z,
                      camera_axes_base=rotation.tolist(), camera_origin_base_mm=translation.tolist())
        result["localized_candidates"] = localized
        save_trace()
        return result, frame, None
    mark("approach")
    if camera_mode == "wrist":
        for q in transit:
            motion.step(q, motion.opening, 10)
    else:
        data.qpos[motion.q] = path[0]
        data.qpos[motion.grip_q] = motion.opening
        mujoco.mj_forward(model, data)
    motion.step(path[0], motion.opening, 150)
    block_z = float(data.xpos[motion.block, 2])
    mark("descend")
    for q in path[1:]:
        motion.step(q, motion.opening, 12)
    motion.step(path[-1], motion.opening, 160)
    tcp_error = float(np.linalg.norm(data.site_xpos[motion.tcp] - target))
    if tcp_error > .005:
        raise RuntimeError(f"Simulated tracking error {tcp_error*1000:.2f} mm")
    mark("close")
    motion.step(path[-1], 0, 350)
    fingers = len(motion.contacts())
    close_contact_geometry = motion.grip_contacts()
    block_shift_mm = (data.xpos[motion.block, :2] - np.asarray(xy)) * 1000
    grip_history = [motion.grip_state()]
    proof = None
    extension_error = None
    if fingers == 2:
        mark("proof_lift")
        lift_path = path[-2::-1]
        # The planned lift follows the approach axis, so use its vertical component.
        proof_steps = min(len(lift_path), int(np.ceil(.012 / (.07 * np.cos(pitch)) * len(lift_path))))
        closed_z = float(data.site_xpos[motion.tcp, 2])
        for q in lift_path[:proof_steps]:
            motion.step(q, 0, lift_steps)
            grip_history.append(motion.grip_state())
        motion.step(lift_path[proof_steps - 1], 0, 125)
        grip_history.append(motion.grip_state())
        tcp_rise = float(data.site_xpos[motion.tcp, 2] - closed_z)
        block_rise = float(data.xpos[motion.block, 2] - block_z)
        proof_contacts = len(motion.contacts())
        proof = dict(passed=tcp_rise >= .007 and block_rise >= .007 and proof_contacts == 2,
                     tcp_rise_mm=tcp_rise*1000, block_rise_mm=block_rise*1000, contacts=proof_contacts)
        if proof["passed"]:
            mark("full_lift")
            for q in lift_path[proof_steps:]:
                motion.step(q, 0, lift_steps)
                grip_history.append(motion.grip_state())
                if len(grip_history[-1]["normal_n_by_jaw"]) < 2:
                    break
            else:
                motion.step(path[0], 0, 300)
                grip_history.append(motion.grip_state())
                if (test_tilted_extension and pitch and len(grip_history[-1]["normal_n_by_jaw"]) == 2
                        and data.xpos[motion.block, 2] - block_z <= .05):
                    try:
                        extension = tilted_lift_extension(motion, pitch)
                    except (ValueError, RuntimeError) as error:
                        extension_error = str(error)
                    else:
                        mark("lift_extension")
                        for q in extension:
                            motion.step(q, 0, lift_steps)
                            grip_history.append(motion.grip_state())
                            if len(grip_history[-1]["normal_n_by_jaw"]) < 2:
                                break
                        else:
                            motion.step(extension[-1], 0, 125)
                            grip_history.append(motion.grip_state())
    first_loss = next((row for row in grip_history if len(row["normal_n_by_jaw"]) < 2), None)
    lift = float(data.xpos[motion.block, 2] - block_z)
    contacts = len(motion.contacts())
    passed = bool(proof and proof["passed"] and first_loss is None and lift > .05 and contacts == 2)
    failure = None
    if not passed:
        if proof is None:
            failure = "no_bilateral_contact"
        elif not proof["passed"]:
            failure = "proof_lift_failed"
        elif first_loss is not None or contacts != 2:
            failure = "lost_after_close"
        else:
            failure = "insufficient_object_lift"
    storage_result = None
    mark("grasp_end")
    if storage and passed:
        try:
            storage_result = deposit(motion, lift_steps, verify_storage, sensor, sensor_seed)
        except (ValueError, RuntimeError) as error:
            storage_result = dict(deposited=False, reason=str(error))
        if not storage_result["deposited"]:
            passed, failure = False, "deposit_failed"
        elif verify_storage and not storage_result["depth_verification"]["confirmed"]:
            passed, failure = False, "deposit_depth_unverified"
    save_trace()
    return dict(observation_attempts=observations, depth_components=depth_components, detections=detections,
                support_estimate_mm=support_estimate.height_mm,
                support_inliers=support_estimate.inliers, support_rms_mm=support_estimate.rms_mm,
                support_error_mm=support_estimate.height_mm-support_z,
                camera_origin_base_mm=translation.tolist(), camera_axes_base=rotation.tolist(),
                surface_base_mm=surface.tolist(), width_mm=opening_mm, projected_width_mm=width_mm,
                image_closing_angle_deg=float(np.rad2deg(candidate.angle)),
                base_closing_axis_deg=float(np.rad2deg(heading)),
                grasp_center_correction_mm=correction.tolist(),
                model_width_px=candidate.model_width_px, xy_m=xy, yaw_rad=yaw, pixel=[candidate.u, candidate.v],
                center_error_mm=float(np.linalg.norm(surface[:2] - np.asarray(xy)*1000)),
                tcp_error_mm=tcp_error*1000, downstroke_mm=depth*1000, approach_pitch_rad=pitch, lift_mm=lift*1000,
                close_contacts=fingers, hold_contacts=contacts,
                close_contact_geometry=close_contact_geometry,
                grip_history=grip_history, first_non_bilateral_sample=first_loss,
                block_center_drift_in_wrist_mm=float(np.linalg.norm(
                    np.asarray(grip_history[-1]["block_center_wrist_mm"])
                    - grip_history[0]["block_center_wrist_mm"])),
                block_shift_after_close_mm=block_shift_mm.tolist(),
                passed=bool(passed), failure=failure, lift_attempted=fingers == 2, proof_lift=proof,
                lift_extension_rejection=extension_error,
                storage=storage_result), frame, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", default=".vendor/roarm_ws")
    parser.add_argument("--report", default="sim_output/report.json")
    parser.add_argument("--gripper", choices=["parallel-pads", "vendor-mesh", "vendor-split"], default="parallel-pads")
    parser.add_argument("--camera", choices=["wrist", "overhead"], default="wrist")
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument("--suite", choices=["baseline", "envelope", "ledge", "clutter", "mixed-height"], default="baseline")
    scenario.add_argument("--case-manifest", type=Path, help="Simulate named cuboids from measured sample dimensions and poses")
    parser.add_argument("--ledge-x-shift-mm", type=float, default=0.,
                        help="Change the stand-off distance in the ledge suite")
    parser.add_argument("--ledge-y-shift-mm", type=float, default=0.,
                        help="Shift the wall laterally relative to the arm in the ledge suite")
    parser.add_argument("--jaw-limit-nm", type=float, default=JAW_RATED_TORQUE_NM)
    parser.add_argument("--grip-friction", type=float,
                        help="Override sliding friction only for block-to-fingertip contacts")
    parser.add_argument("--lift-steps", type=int, default=12,
                        help="Physics steps per lift waypoint")
    parser.add_argument("--test-tilted-lift-extension", action="store_true",
                        help="Test an extra vertical lift after a tilted retreat")
    parser.add_argument("--test-horizontal-grasp-projection", action="store_true",
                        help="Test projecting the image grasp axis onto a horizontal surface")
    parser.add_argument("--test-ledge-centering", action="store_true",
                        help="Test a bounded shift toward the visible upper surface center")
    parser.add_argument("--test-depth-median", action="store_true",
                        help="Fuse three independent depth noise realizations at each observation")
    parser.add_argument("--verify-storage", action="store_true", help="Require wrist depth deposit confirmation")
    parser.add_argument("--storage", action="store_true", help="Require physical deposit into a bin")
    parser.add_argument("--sensor-profile", type=Path, help="JSON perturbation profile")
    parser.add_argument("--sensor-seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat the selected suite with consecutive sensor seeds")
    parser.add_argument("--trace", action="store_true", help="Save trajectory for graphical replay")
    parser.add_argument("--upstream")
    parser.add_argument("--trusted-checkpoint")
    parser.add_argument("--target-weights", help="Run semantic gating on simulated wrist RGB-D")
    parser.add_argument("--rgb-renderer", choices=['mujoco','flat'], default='mujoco',
                        help="RGB appearance for semantic inference; depth remains geometric raycast")
    parser.add_argument("--wrist-config", type=Path, help="Use the wrist hand-eye transform and ROI from this config")
    parser.add_argument("--wrist-recording", type=Path, help="Use calibrated depth rays from this NPZ")
    parser.add_argument("--wrist-workspace", type=Path, help="Use measured link5 camera boxes from an audit workspace")
    args = parser.parse_args()
    wrist_workspace = None
    if args.wrist_workspace:
        if args.camera != "wrist":
            parser.error("--wrist-workspace requires a wrist camera")
        try:
            workspace = json.loads(args.wrist_workspace.read_text(encoding="utf-8-sig"))
            wrist_workspace = {key: workspace[key] for key in ("padding_mm", "wrist_envelopes")}
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    camera_transform = camera_rays = None
    if args.wrist_config or args.wrist_recording:
        if not (args.wrist_config and args.wrist_recording and args.camera == "wrist") or args.verify_storage:
            parser.error("Calibrated replay requires both wrist inputs and wrist mode, without storage depth verification")
        try:
            config = json.loads(args.wrist_config.read_text(encoding="utf-8-sig"))
            if config.get("camera_mount") != "wrist":
                raise ValueError("Calibration config must describe a wrist camera")
            rotation, translation = load_transform(config["transform"])
            camera_transform = np.eye(4)
            camera_transform[:3, :3], camera_transform[:3, 3] = rotation, translation
            rays = load_frame(args.wrist_recording).rays
            x, y, size = config["roi"]
            h, w = rays.shape[:2]
            if min(x, y) < 0 or size < 32 or x + size > w or y + size > h:
                raise ValueError("Wrist ROI must fit inside the recorded depth raster")
            camera_rays = rays[y:y+size, x:x+size]
            if not np.allclose(camera_rays[..., 2], 1):
                raise ValueError("Recorded rays must have optical Z equal to 1")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    if args.grip_friction is not None and not (args.gripper == "vendor-split" and 0 < args.grip_friction <= 2):
        parser.error("--grip-friction requires vendor-split and a value in (0, 2]")
    if args.test_depth_median and not args.sensor_profile:
        parser.error("--test-depth-median requires --sensor-profile")
    if not -50 <= args.ledge_x_shift_mm <= 50 or (args.ledge_x_shift_mm and args.suite != "ledge"):
        parser.error("--ledge-x-shift-mm requires the ledge suite and a value in [-50,50]")
    if not -50 <= args.ledge_y_shift_mm <= 50 or (args.ledge_y_shift_mm and args.suite != "ledge"):
        parser.error("--ledge-y-shift-mm requires the ledge suite and a value in [-50,50]")
    if not 1 <= args.lift_steps <= 250:
        parser.error("--lift-steps must be between 1 and 250")
    if not 0 < args.jaw_limit_nm <= JAW_RATED_TORQUE_NM:
        parser.error("--jaw-limit-nm must be positive and within the vendor EoAT torque rating")
    if args.verify_storage and (not args.storage or args.camera != "wrist"):
        parser.error("--verify-storage requires --storage and wrist camera")
    if args.sensor_seed < 0:
        parser.error("--sensor-seed must be nonnegative")
    if not 1 <= args.repeats <= 100 or (args.repeats > 1 and not args.sensor_profile):
        parser.error("--repeats must be in [1,100] and requires a sensor profile when greater than 1")
    sensor = None
    if args.sensor_profile:
        try:
            sensor = SensorProfile(**json.loads(args.sensor_profile.read_text(encoding="utf-8-sig")))
        except (ValueError, TypeError) as error:
            parser.error(str(error))
    predictor = None
    if args.trusted_checkpoint:
        if not args.upstream:
            parser.error("--upstream is required for GG-CNN")
        from ggcnn import Predictor
        predictor = Predictor(args.upstream, args.trusted_checkpoint, "cpu")
    detector = None
    if args.target_weights:
        if predictor is None or args.camera != "wrist":
            parser.error("--target-weights requires wrist mode and GG-CNN")
        from target_semantics import TargetDetector
        detector = TargetDetector(args.target_weights, [COMPETITION_TARGET])
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    cases = [(xy, yaw, (.046, .030, .030), .020)
             for xy, yaw in [((.31, 0), 0.), ((.31, .04), .45), ((.29, -.04), -.55)]]
    if args.suite == "envelope":
        cases.extend([((.29, 0), .20, (.050, .050, .050), .300),
                      ((.29, 0), -.30, (.045, .030, .038), .300),
                      ((.29, 0), .70, (.025, .023, .020), .025)])
    cases = [(*case, None) for case in cases]
    if args.suite == "clutter":
        cases = cases[:1]
    if args.suite == "ledge":
        cases = [((.272, 0), 0., (.040, .028, .030), .020, .035),
                 ((.272, .025), .35, (.040, .028, .030), .020, .055),
                 ((.275, 0), 0., (.040, .028, .030), .020, .20),
                 ((.275, 0), 0., (.040, .028, .030), .020, .35),
                 ((.275, 0), 0., (.040, .028, .030), .020, .49)]
    if args.suite == "mixed-height":
        cases = [((.272, .025), .35, (.040, .028, .030), .020, .055)]
    if args.case_manifest:
        try:
            cases = load_cases(args.case_manifest)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    else:
        clutter = args.suite in ("clutter", "mixed-height")
        cases = [(*case, clutter, f"{args.suite}-{index}", None, None) for index, case in enumerate(cases)]
    for i, (xy, yaw, size, mass, ledge_height, clutter, case_id, mesh, ledge_box) in enumerate(cases * args.repeats):
        shifted_xy = (xy[0] + args.ledge_x_shift_mm / 1000, xy[1] + args.ledge_y_shift_mm / 1000)
        try:
            trace = output.parent / f"{output.stem}_{i}.npz" if args.trace else None
            start = time.perf_counter()
            result, frame, candidate = trial(args.vendor, xy, yaw, predictor, args.gripper, trace, args.camera, size, mass, args.lift_steps,
                                             args.jaw_limit_nm, args.storage, ledge_height, sensor,
                                             args.sensor_seed + i, args.verify_storage,
                                             clutter,
                                             args.ledge_x_shift_mm / 1000, args.ledge_y_shift_mm / 1000,
                                             args.test_tilted_lift_extension, args.test_horizontal_grasp_projection,
                                             detector, args.test_ledge_centering, args.test_depth_median,
                                             args.grip_friction, mesh, ledge_box, camera_transform, camera_rays, wrist_workspace, args.rgb_renderer)
            result["wall_time_s"] = round(time.perf_counter() - start, 3)
            depth = cv2.normalize(frame.depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            view = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
            zoom = 4
            view = cv2.resize(view, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            if candidate is not None:
                cv2.drawMarker(view, (view.shape[1] // 2, view.shape[0] // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                draw_grasp(view, candidate, zoom)
            if candidate is not None or frame.bgr is not None:
                color = {"bgr": frame.bgr} if frame.bgr is not None else {}
                np.savez_compressed(output.parent / f"{output.stem}_depth_{i}.npz",
                                    depth_m=frame.depth_m, rays=frame.rays,
                                    rotation_base=np.asarray(result["camera_axes_base"]),
                                    translation_base_mm=np.asarray(result["camera_origin_base_mm"]), **color)
            cv2.imwrite(str(output.parent / f"{output.stem}_depth_{i}.png"), view)
            if frame.bgr is not None:
                cv2.imwrite(str(output.parent / f"{output.stem}_color_{i}.png"), frame.bgr)
        except ObservationRejected as error:
            result = dict(xy_m=shifted_xy, yaw_rad=yaw, passed=False, error=str(error),
                          failure="observation_pose_blocked", observation_attempts=error.attempts,
                          next_action="select_another_observation_or_reposition_base")
        except (ValueError, RuntimeError) as error:
            result = dict(xy_m=shifted_xy, yaw_rad=yaw, passed=False, error=str(error))
        result.update(sensor_seed=args.sensor_seed + i if sensor else None,
                      case_id=case_id, case_index=i % len(cases), repeat_index=i // len(cases),
                      block_mesh=str(mesh) if mesh else None,
                      ledge_box_mm=[value * 1000 for value in ledge_box] if ledge_box else None,
                      gripper_model=args.gripper, block_size_mm=[x*1000 for x in size],
                      block_mass_g=mass*1000,
                      ledge_height_mm=None if ledge_height is None else ledge_height*1000)
        reports.append(result)
        print(json.dumps(result))
    report = dict(sensor_profile=asdict(sensor) if sensor else None,
                  calibrated_wrist_camera=camera_transform is not None, measured_wrist_workspace=wrist_workspace,
                  repeats=args.repeats, passed_trials=sum(row["passed"] for row in reports),
                  total_trials=len(reports),
                  storage=args.storage, verify_storage=args.verify_storage, jaw_limit_nm=args.jaw_limit_nm if args.gripper != "parallel-pads" else None,
                  grip_friction=args.grip_friction,
                  lift_steps=args.lift_steps, suite="manifest" if args.case_manifest else args.suite,
                  camera=args.camera, vendor_reference=VENDOR_COMMIT, mujoco=mujoco.__version__,
                  backend="ggcnn" if predictor else "rectangle", semantic_gating=detector is not None,
                  rgb_renderer=args.rgb_renderer if detector is not None else None,
                  ledge_x_shift_mm=args.ledge_x_shift_mm,
                  ledge_y_shift_mm=args.ledge_y_shift_mm,
                  test_tilted_lift_extension=args.test_tilted_lift_extension,
                  test_ledge_centering=args.test_ledge_centering,
                  test_depth_median=args.test_depth_median,
                  test_horizontal_grasp_projection=args.test_horizontal_grasp_projection, trials=reports)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(r["passed"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()



















