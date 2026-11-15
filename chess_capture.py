"""Collect chessboard poses; default 8x11 inner corners, 15mm squares."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from arm import Arm
from camera import orbbec, save_frame
from wrist_pose import stationary_frame
from chessboard import PATTERN, detect, orient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', default='COM6')
    p.add_argument('--square-mm', type=float, default=15.)
    p.add_argument('--cols',type=int,default=8,help='Inner corner columns, not squares')
    p.add_argument('--rows',type=int,default=11,help='Inner corner rows, not squares')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    pattern=(a.cols,a.rows)
    if min(pattern)<3:p.error('Need at least 3x3 inner corners')
    if not np.isfinite(a.square_mm) or a.square_mm <= 0:
        p.error('Measure one square edge in mm')
    root = a.output or Path('calibration')/f'chess-{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=False)
    setup = dict(schema=1, family='chessboard', pattern_size=list(pattern),
                 square_size_mm=a.square_mm, camera='orbbec', camera_pose_frame='raw_rgb',
                 target_must_remain_fixed=True, robot_mount='link5', port=a.port,
                 motion_commands=False, origin_selection='same physical endpoint selected each sample')
    (root/'session.json').write_text(json.dumps(setup, indent=2), encoding='utf-8')
    print(f'SESSION {root.resolve()}\nF=fit C=check S=save Q=exit.', flush=True)
    print('Mark ONE board corner on the outside margin. Before EVERY save, click its adjacent outermost INNER corner.\n'
          'Keep base, board and camera mount fixed. Only move robot joints.\n'
          'Opening COM6 may reset the controller and move it to its initial pose.', flush=True)
    group = 'fit'
    counts = dict(fit=0, check=0)
    selected = dict(anchor=None)
    window = 'Chessboard wrist capture - raw RGB'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, context):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected['anchor'] = np.array([x, y], dtype=float)

    cv2.setMouseCallback(window, mouse)
    try:
        with Arm(a.port) as arm, orbbec(with_color=True) as read:
            for _ in range(15):
                read()
            while True:
                frame = read()
                if frame.raw_bgr is None:
                    raise RuntimeError('Raw RGB required')
                view = frame.raw_bgr.copy()
                status = 'Click marked origin, then S'
                try:
                    corners = detect(frame.raw_bgr,pattern)
                    cv2.drawChessboardCorners(view, pattern, corners.astype(np.float32), True)
                    if selected['anchor'] is not None:
                        ordered = orient(corners, selected['anchor'],pattern)
                        origin = tuple(ordered[0].astype(int))
                        cv2.circle(view, origin, 10, (255, 0, 255), 2)
                        cv2.putText(view, 'ORIGIN', origin, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 2)
                        status = f'{pattern[0]*pattern[1]} corners + origin ready; release jog, wait, S'
                except ValueError as e:
                    status = str(e)
                cv2.putText(view, f'{group} {counts} | {status}', (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')):
                    break
                if key in (ord('f'), ord('c')):
                    group = 'fit' if key == ord('f') else 'check'
                    selected['anchor'] = None
                if key == ord('s'):
                    try:
                        if selected['anchor'] is None:
                            raise ValueError('Click the marked origin corner before EACH save')
                        captured, joints = stationary_frame(arm, read)
                        corners = orient(detect(captured.raw_bgr,pattern), selected['anchor'],pattern)
                        if any(getattr(captured, k) is None for k in
                               ('raw_K', 'raw_dist', 'T_raw_from_depth_mm', 'raw_dist_model')):
                            raise ValueError('Factory RGB calibration is missing')
                        dest = root/group/f'{counts[group]:03d}'
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with dest.with_suffix('.npz').open('xb') as out:
                            save_frame(out, captured, joints)
                        record = dict(setup, group=group, corners_raw_px=corners.tolist(),
                                      joints_rad=joints.tolist(), snapshot=dest.with_suffix('.npz').name)
                        dest.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
                        counts[group] += 1
                        selected['anchor'] = None
                        print(f'SAVED {dest} {counts}', flush=True)
                    except (ValueError, RuntimeError, TimeoutError) as e:
                        print(f'NOT SAVED: {e}', flush=True)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
