import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from camera import orbbec, real_sense, save_frame
from geometry import sample


POINT_FIELDS = ['frame', 'u', 'v', 'x_mm', 'y_mm', 'z_mm']


def append_point(csv_path, snapshot, u, v, base_mm):
    """Preserve existing rows while recording a measured correspondence."""
    csv_path, snapshot = Path(csv_path), Path(snapshot).resolve()
    xyz = np.asarray(base_mm, float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or not snapshot.is_file():
        raise ValueError('Need a saved snapshot and three finite measured coordinates')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(snapshot, csv_path.parent.resolve())
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    if exists:
        with csv_path.open(newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != POINT_FIELDS:
                raise ValueError('Existing points CSV has an incompatible header')
            if any(row['frame'] == relative for row in reader):
                raise ValueError('This snapshot already has a point in this CSV')
    with csv_path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(POINT_FIELDS)
        writer.writerow([relative, int(u), int(v), *xyz])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera", choices=["realsense", "orbbec"])
    parser.add_argument("output")
    parser.add_argument("--arm-port", help="Record stationary M3 wrist feedback with the depth frame")
    parser.add_argument("--depth-stack", type=int, help="Record 3 to 300 frames for sensor characterization")
    parser.add_argument("--color", action="store_true", help="Save color aligned to native depth")
    parser.add_argument("--no-preview", action="store_true", help="Save the recording without opening windows")
    parser.add_argument("--points-csv", type=Path, help="Append one clicked, measured calibration point")
    parser.add_argument("--point-base-mm", nargs=3, type=float, metavar=('X','Y','Z'))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Snapshot already exists; choose a new output name')
    if bool(args.points_csv) != (args.point_base_mm is not None):
        parser.error('--points-csv and --point-base-mm must be used together')
    if args.points_csv and (args.no_preview or args.depth_stack is not None
                            or not np.isfinite(args.point_base_mm).all()):
        parser.error('Point capture needs a preview, finite coordinates and a single snapshot')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.color and args.depth_stack is not None:
        parser.error("--color requires a single snapshot")
    if args.depth_stack is not None and (args.arm_port or not 3 <= args.depth_stack <= 300):
        parser.error("--depth-stack requires 3 to 300 frames and a stationary camera without --arm-port")
    joints = None
    camera = {"realsense": real_sense, "orbbec": orbbec}[args.camera]
    with camera(with_color=args.color) as read:
        for _ in range(15):
            frame = read()
        if args.depth_stack is not None:
            stack = np.stack([read().depth_m for _ in range(args.depth_stack)])
            with output.open('xb') as file:
                np.savez_compressed(file, depth_stack_m=stack, rays=frame.rays)
            return
        if args.arm_port:
            from arm import Arm
            from wrist_pose import stationary_frame

            with Arm(args.arm_port) as arm:
                frame, joints = stationary_frame(arm, read)
    with output.open('xb') as file:
        save_frame(file, frame, joints)
    if args.no_preview:
        return
    depth = np.nan_to_num(frame.depth_m)
    view = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                             .astype(np.uint8), cv2.COLORMAP_TURBO)

    recorded = False

    def clicked(event, u, v, *_):
        nonlocal recorded
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                print(f"u={u}, v={v}, camera_m={sample(frame, u, v)}", flush=True)
                if args.points_csv and not recorded:
                    append_point(args.points_csv, output, u, v, args.point_base_mm)
                    recorded = True
                    print(f'Calibration point saved to {args.points_csv}', flush=True)
            except (ValueError, OSError) as error:
                print(error, flush=True)

    cv2.imshow("Depth snapshot", view)
    cv2.setMouseCallback("Depth snapshot", clicked)
    if frame.bgr is not None:
        cv2.imshow("Color registered to depth", cv2.addWeighted(frame.bgr, .7, view, .3, 0))
        cv2.setMouseCallback("Color registered to depth", clicked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





