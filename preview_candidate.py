"""YOLO/GG-CNN and wrist XYZ preview using an explicitly unvalidated candidate.

Only queries arm feedback (T105); no movement/torque commands are sent.
Opening the serial port may still reset the controller and trigger firmware homing.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for folder in reversed(('.deps', '.orbbec-python-deps', '.torchdeps', '.semdeps')):
    sys.path.insert(0, str(ROOT/folder))
import argparse
from contextlib import ExitStack
import hashlib
import json
import time
import cv2
import numpy as np
from camera import orbbec, load_frame, save_frame
from geometry import sample
from wrist_pose import Wrist
from rgbd_projection import rgb_click_to_depth, to_panel

from active_handeye import active_report
DEFAULT_REPORT = active_report()


def project_probe(frame, pixel, wrist, joints, hand_eye, bias):
    rotation, translation = wrist.camera_pose(joints, hand_eye)
    return rotation @ (sample(frame, *pixel)*1000)+translation+np.asarray(bias)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    p.add_argument('--port', default='COM6')
    p.add_argument('--trained', action='store_true', help='Use latest completed artificial_object detector')
    p.add_argument('--depth-only',action='store_true',help='Manual depth point localization without RGB or models')
    p.add_argument('--near',action='store_true',help='Use 1280x800 with disparity 256 / offset 127')
    p.add_argument('--rotate',type=int,choices=[0,180],default=0,help='Display rotation; saved geometry stays native')
    p.add_argument('--target-class', action='append', help='Bench default wood_block; competition class remains unchanged')
    p.add_argument('--frame', type=Path, help='Offline saved RGB-D/joints; never opens serial or camera')
    p.add_argument('--pixel', type=int, nargs=2)
    p.add_argument('--bias-mm', type=float, nargs=3, default=[0., 0., 0.], help='Preview-only base-frame XYZ offset')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'candidate_preview/latest.png')
    args = p.parse_args()
    if args.headless and not args.frame:
        p.error('Use --frame for headless mode')
    if not np.isfinite(args.bias_mm).all():
        p.error('Bias must be finite millimeters')
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    matrix = np.asarray(report['wrist_from_depth_mm'], float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1]) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
        p.error('Invalid candidate transform')
    mapping = report['joint_mapping']
    wrist = Wrist(ROOT/'.vendor/roarm_ws', mapping['joint_signs'], mapping['joint_offsets_rad'])
    hand_eye = matrix[:3, :3], matrix[:3, 3]
    check = [v for v in report['views'] if v['name'].startswith('check/')]
    error_mm = max(v['center_error_mm'] for v in check)
    print(f'UNVALIDATED CANDIDATE: check residual {error_mm:.2f} mm; NOT a grasp-error bound.', flush=True)
    print('NO robot movement commands. Click a surface to inspect candidate base XYZ. Q exits.', flush=True)
    if not args.frame:
        print('Opening COM6 can reset/homing the arm; clear the movement area before running.', flush=True)
    from preview_live import render
    from target_semantics import TargetDetector
    from ggcnn import Predictor
    allowed = args.target_class or (['artificial_object'] if args.trained else ['wood_block'])
    weights = 'D:/Lab/挑战杯/LunarRover_HUST-master/weights/best.pt'
    if args.trained:
        from trained_model import latest_detector
        weights = str(latest_detector(ROOT))
        print('TRAINED DETECTOR',weights,flush=True)
    detector = None if args.depth_only else TargetDetector(weights, allowed, 'cpu')
    predictor = None if args.depth_only else Predictor(ROOT/'.vendor/ggcnn', ROOT/'.vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell', 'cpu')
    pixel = list(args.pixel) if args.pixel else None
    selected_display = None
    window = f'Candidate {error_mm:.2f}mm - XYZ/YOLO/GG-CNN - no movement commands'
    width = 0

    def click(event, x, y, flags, unused):
        nonlocal selected_display
        if event == cv2.EVENT_LBUTTONDOWN and width:
            if args.rotate==180:x,y=2*width-1-x,display_height-1-y
            selected_display = [x, y]

    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, click)
    try:
        with ExitStack() as stack:
            arm = read = None
            if not args.frame:
                from arm import Arm
                arm = stack.enter_context(Arm(args.port))
                camera_status={}
                read = stack.enter_context(orbbec(with_color=not args.depth_only,near=args.near,diagnostics=camera_status))
                expected={'OB_PROP_DEPTH_ROTATE_INT':0,'OB_PROP_DEPTH_MIRROR_BOOL':False,'OB_PROP_DEPTH_FLIP_BOOL':False}
                for name,value in expected.items():
                    row=camera_status['properties'][name]
                    if row.get('status')!='read' or row.get('value')!=value:
                        raise ValueError('Localization needs verified native depth orientation: set Viewer depth rotation=0, mirror/flip off. Use --rotate 180 for display only. '+name)
                for _ in range(15):
                    read()
                print('Waiting up to 12 seconds for five-joint feedback (no gripper fields required)...',flush=True)
                arm.feedback(timeout=12,required_fields=('b','s','e','t','r','v'))
            while True:
                reason = None
                if args.frame:
                    frame = load_frame(args.frame)
                    with np.load(args.frame, allow_pickle=False) as data:
                        joints = data['joints_rad']
                else:
                    keys = ('b', 's', 'e', 't', 'r')
                    before = arm.feedback(required_fields=('b','s','e','t','r','v'))
                    frame = read()
                    after = arm.feedback(required_fields=('b','s','e','t','r','v'))
                    q0, q1 = [np.array([v[k] for k in keys], float) for v in (before, after)]
                    delta = (q1-q0+np.pi) % (2*np.pi)-np.pi
                    joints = q0+delta/2
                    if not np.isfinite(joints).all() or np.max(np.abs(delta)) > .003:
                        reason = 'Arm moving: stop before inspecting XYZ'
                    if after.get('v', 0) < 700:
                        reason = 'No valid servo power; XYZ hidden'
                if args.depth_only:
                    from depth_health import depth_image
                    view=depth_image(frame.depth_m)
                    canvas=np.hstack((view,view))
                    result=dict(color_panel=dict(space='native_depth'),proposals=[],motion_enabled=False,
                                depth_valid_fraction=float((np.isfinite(frame.depth_m)&(frame.depth_m>0)).mean()))
                else:
                    canvas, result = render(frame, detector, predictor, .25)
                height, width = frame.depth_m.shape
                display_height=height
                if pixel is None:
                    pixel = [width//2, height//2]
                layout = result['color_panel']
                probe_mapping_ok = True
                try:
                    if selected_display is not None:
                        x, y = selected_display
                        if x < width and layout['space'] == 'native_rgb':
                            raw_point = (np.array([x, y])-layout['offset'])/layout['scale']
                            pixel = list(rgb_click_to_depth(detector.last_depth_to_rgb, raw_point))
                        else:
                            pixel = [x-width if x >= width else x, y]
                except ValueError as e:
                    reason = str(e)
                    probe_mapping_ok = False
                xyz = None
                if reason is None:
                    try:
                        xyz = project_probe(frame, pixel, wrist, joints, hand_eye, args.bias_mm)
                    except ValueError as e:
                        reason = str(e)
                if probe_mapping_ok and 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                    cv2.drawMarker(canvas, (pixel[0]+width, pixel[1]), (255,255,255), cv2.MARKER_CROSS, 16, 1)
                    if layout['space'] == 'native_rgb':
                        uv = detector.last_depth_to_rgb[pixel[1], pixel[0]]
                        if np.isfinite(uv).all():
                            cv2.drawMarker(canvas, tuple(to_panel(uv, layout).astype(int)), (255,255,255), cv2.MARKER_CROSS, 16, 1)
                    else:
                        cv2.drawMarker(canvas, tuple(pixel), (255,255,255), cv2.MARKER_CROSS, 16, 1)
                text = reason or f'Candidate base XYZ mm: {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}'
                cv2.rectangle(canvas, (0, height-45), (2*width, height), (0,0,0), -1)
                cv2.putText(canvas, text, (8, height-27), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
                cv2.putText(canvas, f'UNVALIDATED {error_mm:.2f}mm consistency residual; NO MOVEMENT COMMANDS',
                            (8, height-9), cv2.FONT_HERSHEY_SIMPLEX, .43, (0,0,255), 1)
                result.update(scope='unvalidated_wrist_candidate_preview', motion_enabled=False,
                              candidate_report=str(args.report), candidate_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                              held_out_consistency_mm=error_mm,
                              robot_joints_rad=joints.tolist() if np.isfinite(joints).all() else None,
                              probe_pixel=pixel, probe_base_mm=None if xyz is None else xyz.tolist(),
                              bias_base_mm=args.bias_mm, localization_error=reason)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.output), canvas):
                    raise OSError('Preview output failed')
                args.output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                if args.frame:
                    print(json.dumps(result, allow_nan=False), flush=True)
                if not args.headless:
                    cv2.imshow(window, cv2.rotate(canvas,cv2.ROTATE_180) if args.rotate==180 else canvas)
                    key=cv2.waitKey(1)&255
                    if key==ord('s'):
                        folder=ROOT/'localization_recordings'/str(time.time_ns());folder.mkdir(parents=True)
                        save_frame(folder/'rgbd.npz',frame,joints)
                        (folder/'localization.json').write_text(json.dumps(result,indent=2,allow_nan=False))
                        print('SAVED LOCALIZATION',folder,flush=True)
                    if key in (27, ord('q')):
                        break
                if args.frame:
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
