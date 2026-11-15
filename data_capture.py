"""Fast real RGB-D dataset capture: camera only, no model inference or arm connection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
for name in reversed(('.deps', '.orbbec-python-deps')):
    sys.path.insert(0, str(ROOT/name))
import argparse
import json
import time
import cv2
import numpy as np
from camera import orbbec, save_frame
from rgbd_projection import letterbox
from capture_plan import CapturePlan, header


def save_capture(session, frame, scene_id, object_types=None, collection_stage=None, camera_profile=None):
    if frame.bgr is None:
        raise ValueError('RGB frame missing')
    dest = Path(session)/str(time.time_ns())
    dest.mkdir(parents=True, exist_ok=False)
    with (dest/'rgbd.npz').open('xb') as stream:
        save_frame(stream, frame)
    for name, image in [('aligned_rgb.png', frame.bgr), ('original_rgb.png', frame.raw_bgr)]:
        if image is not None and not cv2.imwrite(str(dest/name), image):
            raise OSError(f'Cannot save {name}')
    valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
    meta = dict(schema=1, camera='orbbec', session_id=Path(session).name, scene_id=scene_id,
                split_group=scene_id, annotation_status='unlabelled', contains_target=None,
                annotation_image='original_rgb.png' if frame.raw_bgr is not None else 'aligned_rgb.png',
                depth_unit='meters', depth_shape=list(frame.depth_m.shape),
                depth_valid_fraction=float(valid.mean()),
                note='No labels or negatives inferred. RGB boxes do not constitute grasp labels.',
                robot_port_opened=False, motion_commands=False)
    meta['planned_target_types'] = object_types
    meta['collection_stage'] = collection_stage
    if camera_profile is not None:meta['camera_profile']=camera_profile
    # Manifest is written last so incomplete captures are not considered complete.
    (dest/'capture.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'datasets/competition/raw')
    p.add_argument('--scene', help='Reuse scene ID for same setup; keep that group in one train/val/test split')
    p.add_argument('--object-types', type=int, default=3, help='Number of target types; default 3, not automatic labels')
    args = p.parse_args()
    if not 1 <= args.object_types <= 9:
        p.error('object-types must be 1 to 9')
    plan = CapturePlan(args.object_types)
    session = args.root/f'session-{time.time_ns()}'
    session.mkdir(parents=True, exist_ok=False)
    scene = args.scene or f'{session.name}-layout-0000'
    (session/'session.json').write_text(json.dumps(dict(scene_id=scene, camera='orbbec',
        planned_target_types=args.object_types, capture_plan=plan.export(),
        grouping_assumption='Operator changes physical layout before every S; explicit --scene keeps one group',
        annotation_status='unlabelled', robot_port_opened=False), indent=2), encoding='utf-8')
    print(f'DATASET SESSION {session}\n1/2/3=single type M=mixed B=background S=one photo Q=quit. No arm connection.', flush=True)
    print('Each new layout: save ONE photo. Counts are collection intentions, not annotations.', flush=True)
    try:
        with orbbec(with_color=True) as read:
            for _ in range(15): read()
            count = 0
            ready = False
            scene_number = 0
            banner = None
            banner_key = None
            window = 'Dataset RGB-D capture - camera only'
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while True:
                if ready:
                    try:
                        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                frame = read()
                h, w = frame.depth_m.shape
                image = frame.raw_bgr if frame.raw_bgr is not None else frame.bgr
                color, _ = letterbox(image, (h, w))
                valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
                depth = cv2.applyColorMap((np.clip(np.where(valid, frame.depth_m, 0)/2, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
                depth[~valid] = 0
                canvas = np.hstack([color, depth])
                cv2.putText(canvas, f'Original RGB / measured depth | valid {100*valid.mean():.1f}%',
                            (8, h-12), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
                state_key = (plan.active[1], tuple(plan.counts.values()), 2*w)
                if banner_key != state_key:
                    banner, banner_key = header(plan, 2*w), state_key
                canvas = np.vstack([banner, canvas])
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 255
                if not ready:
                    print(f'CAMERA_READY RGB={image.shape[1]}x{image.shape[0]} DEPTH={w}x{h} valid={100*valid.mean():.1f}%', flush=True)
                    ready = True
                if key in (27, ord('q')): break
                if plan.select(key):
                    print('BATCH', plan.active[1], 'goal', plan.active[3], flush=True)
                if key in (ord('n'), ord('N')):
                    scene_number += 1
                    scene = f'{args.scene or session.name}-scene-{scene_number:03d}'
                    print('SCENE', scene, flush=True)
                if key in (ord('s'), ord('S')):
                    print('SAVED', save_capture(session, frame, scene, args.object_types, plan.active[1]), flush=True)
                    count += 1
                    plan.saved()
                    (session/'progress.json').write_text(json.dumps(plan.export(), indent=2), encoding='utf-8')
                    if not args.scene:
                        scene_number += 1
                        scene = f'{session.name}-layout-{scene_number:04d}'
                    print('PROGRESS', json.dumps(plan.export()), flush=True)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__': main()
