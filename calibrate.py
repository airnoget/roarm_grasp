import argparse
import csv
import json
from pathlib import Path

import numpy as np

from camera import load_frame
from geometry import fit_transform, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", help="CSV columns frame,u,v,x_mm,y_mm,z_mm")
    parser.add_argument("output")
    parser.add_argument("--check", required=True, help="Independent CSV in the same format")
    parser.add_argument("--max-error-mm", type=float, default=5)
    parser.add_argument("--wrist-config", help="Fit link5-from-depth using saved joint feedback")
    args = parser.parse_args()
    wrist = None
    if args.wrist_config:
        from wrist_pose import Wrist

        cfg = json.loads(Path(args.wrist_config).read_text(encoding="utf-8-sig"))
        wrist = Wrist(cfg["vendor"], cfg["joint_signs"], cfg["joint_offsets_rad"])

    def read(path):
        camera, base = [], []
        with open(path, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                snapshot = Path(path).parent / row["frame"]
                frame = load_frame(snapshot)
                camera.append(sample(frame, int(row["u"]), int(row["v"])))
                point = np.array([float(row[f"{axis}_mm"]) for axis in "xyz"])
                if wrist:
                    with np.load(snapshot, allow_pickle=False) as data:
                        rotation, translation = wrist.fk(data["joints_rad"])
                    point = rotation.T @ (point - translation)
                base.append(point)
        return np.asarray(camera), np.asarray(base)

    rotation, translation, errors = fit_transform(*read(args.pairs))
    a, b = read(args.check)
    if len(a) < 2:
        parser.error("Provide at least two independent validation points")
    check = np.linalg.norm((a * 1000) @ rotation.T + translation - b, axis=1)
    print("Fit errors (mm):", errors.round(2), "Validation (mm):", check.round(2))
    if (not np.isfinite(check).all() or not 0 < args.max_error_mm < np.inf
            or max(errors.max(), check.max()) > args.max_error_mm):
        parser.error("Calibration error exceeds threshold")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    np.save(args.output, transform)


if __name__ == "__main__":
    main()

