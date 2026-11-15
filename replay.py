"""View a recorded simulation trajectory without controlling any hardware."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import cv2
import numpy as np

from sim_model import scene
from sim_camera import wrist_frame


def depth_preview(depth_m):
    low, high = .10, .45
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth = np.where(valid, depth_m, 0)
    scaled = np.clip((depth - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    view = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    view[~valid] = 0
    view = cv2.resize(view, (480, 480), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(view, 30, 0, 0, 90, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    ramp = cv2.applyColorMap(np.linspace(0, 255, 400, dtype=np.uint8)[:, None],
                             cv2.COLORMAP_TURBO)
    canvas[65:465, 494:510] = ramp
    for z in (100, 200, 300, 400, 450):
        y = 65 + round((z - 100) * 399 / 350)
        cv2.putText(canvas, str(z), (518, y+4), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1)
    cv2.putText(canvas, "Simulated depth (mm) | Black = invalid", (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    return canvas


def inference_image(trace_path):
    trace = Path(trace_path)
    stem, separator, index = trace.stem.rpartition("_")
    if not separator or not index.isdecimal():
        raise ValueError("Inference preview requires a numbered simulate.py trace")
    image = trace.with_name(f"{stem}_depth_{index}.png")
    if not image.is_file():
        raise ValueError(f"Missing saved inference depth image: {image}")
    view = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if view is None:
        raise ValueError(f"Cannot decode saved inference depth image: {image}")
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--vendor", default=".vendor/roarm_ws")
    parser.add_argument("--wrist-view", action="store_true")
    parser.add_argument("--depth-view", action="store_true", help="Open a live wrist depth window")
    parser.add_argument("--inference-view", action="store_true",
                        help="Show the saved depth observation with the selected grasp overlay")
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as record:
        poses = record["qpos"]
        times = record["time_s"] if "time_s" in record else np.arange(len(poses)) * .02
        camera_mode = str(record["camera_mode"]) if "camera_mode" in record else "overhead"
        block_size = record["block_size"] if "block_size" in record else (.046, .030, .030)
        block_mass = float(record["block_mass"]) if "block_mass" in record else .020
        jaw_limit = float(record["jaw_limit_nm"]) if "jaw_limit_nm" in record else 18
        storage = bool(record["storage"]) if "storage" in record else False
        ledge = float(record["ledge_height"]) if "ledge_height" in record else np.nan
        friction = float(record["grip_friction"]) if "grip_friction" in record else np.nan
        block_mesh = str(record["block_mesh"]) if "block_mesh" in record else ""
        ledge_box = tuple(record["ledge_box_m"]) if "ledge_box_m" in record else ()
        camera_transform = record["camera_transform"] if "camera_transform" in record else np.empty(0)
        camera_rays = record["camera_rays"] if "camera_rays" in record else np.empty(0)
        wrist_workspace = json.loads(str(record["wrist_workspace"])) if "wrist_workspace" in record else None
        model = mujoco.MjModel.from_xml_string(
            scene(args.vendor, record["xy"], float(record["yaw"]), str(record["gripper"]), camera_mode,
                  block_size, block_mass, jaw_limit, storage,
                  ledge if np.isfinite(ledge) else None,
                  bool(record["clutter"]) if "clutter" in record else False,
                  float(record["ledge_x_shift_m"]) if "ledge_x_shift_m" in record else 0.,
                  float(record["ledge_y_shift_m"]) if "ledge_y_shift_m" in record else 0.,
                  friction if np.isfinite(friction) else None, block_mesh or None, ledge_box or None,
                  camera_transform if camera_transform.size else None, wrist_workspace)
        )
    if poses.ndim != 2 or not len(poses) or poses.shape[1] != model.nq or not np.isfinite(poses).all():
        raise ValueError("Trace does not match the simulation model")
    if (times.shape != (len(poses),) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Trace timestamps must be finite and strictly increasing")
    elapsed = times - times[0]
    if (args.wrist_view or args.depth_view) and camera_mode != "wrist":
        parser.error("This trace has no wrist camera")
    inference = None
    if args.inference_view:
        try:
            inference = inference_image(args.trace)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if inference is not None:
            cv2.imshow("Saved depth observation", inference)
        if args.wrist_view:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("wrist_depth").id
        while viewer.is_running():
            start = time.monotonic()
            for index, pose in enumerate(poses):
                if not viewer.is_running():
                    break
                delay = start + elapsed[index] - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                with viewer.lock():
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                viewer.sync()
                if args.depth_view and index % 5 == 0:
                    frame, _, _ = wrist_frame(model, data, rays=camera_rays if camera_rays.size else None)
                    cv2.imshow("Live wrist depth (replay)", depth_preview(frame.depth_m))
                if args.depth_view or args.inference_view:
                    if cv2.waitKey(1) & 0xff in (27, ord("q")):
                        cv2.destroyAllWindows()
                        return
            # Keep the terminal pose visible before starting the next loop.
            time.sleep(.02)
    if args.depth_view or args.inference_view:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






