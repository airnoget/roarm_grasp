"""RGB-D model commissioning preview; never opens the arm or plans motion."""
import argparse
from contextlib import nullcontext
import json
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from camera import load_frame, real_sense, orbbec, save_frame
from ggcnn import Predictor
from perception import draw_grasp
from target_semantics import TargetDetector, COMPETITION_TARGET
from rgbd_projection import letterbox, to_panel


def target_roi(box, shape):
    """Keep the full detection in a square crop, including at image borders."""
    h, w = shape
    x0, y0, x1, y1 = box
    span = max(x1-x0, y1-y0)
    if span > min(h,w):
        raise ValueError('Target exceeds square GG-CNN field; move camera farther away')
    size = min(min(h,w), max(64, int(np.ceil(span*1.4))))
    x = int(np.clip(round((x0+x1-size)/2), 0, w-size))
    y = int(np.clip(round((y0+y1-size)/2), 0, h-size))
    return x,y,size


def save_observation(directory, frame, canvas, result):
    """Save sensor evidence without promoting model predictions to labels."""
    directory = Path(directory) / str(time.time_ns())
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'rgbd.npz').open('xb') as file:
        save_frame(file, frame)
    images = {'aligned_rgb.png': frame.bgr, 'preview.png': canvas}
    if frame.raw_bgr is not None:
        images['original_rgb.png'] = frame.raw_bgr
    for name, img in images.items():
        if img is not None and not cv2.imwrite(str(directory/name), img):
            raise OSError(f'Cannot save {directory/name}')
    report = dict(result, annotation_status='unreviewed',
                  detection_coordinates='aligned_rgb.png / native depth raster; not original_rgb.png',
                  native_detection_coordinates='original_rgb.png when native_detections are present',
                  note='Predictions are not ground truth. Review images before labeling.')
    (directory/'predictions.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return directory


def render(frame, detector, predictor, min_score):
    h, w = frame.depth_m.shape
    valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0) & (frame.depth_m < 10)
    depth = np.where(valid, frame.depth_m, 0)
    view = cv2.applyColorMap((np.clip(depth/2., 0, 1)*255).astype('uint8'), cv2.COLORMAP_TURBO)
    view[~valid] = 0
    color = frame.bgr.copy()
    detections = detector(frame)
    native = getattr(detector, 'image_source', None) == 'native_rgb'
    layout = dict(space='depth_aligned_rgb', scale=1., offset=[0, 0], raw_shape=[h, w])
    native_boxes = []
    if native:
        color, layout = letterbox(frame.raw_bgr, (h, w))
        for name, box in detector.last_native_detections:
            a, b = to_panel(np.asarray(box).reshape(2, 2), layout).astype(int)
            selected = name in detector.allowed
            tint = (0, 255, 255) if selected else (128, 128, 128)
            cv2.rectangle(color, tuple(a), tuple(b), tint, 2)
            cv2.putText(color, name, (int(a[0]), max(18, int(a[1])-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, tint, 1)
            native_boxes.append(dict(name=name, xyxy=[float(v) for v in box]))
    masks = []
    boxes = []
    for name, box in detections:
        x0, y0 = np.clip(np.floor(box[:2]), 0, [w, h]).astype(int)
        x1, y1 = np.clip(np.ceil(box[2:]), 0, [w, h]).astype(int)
        selected = name in detector.allowed
        for img in ((view,) if native else (color, view)):
            cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,255) if selected else (128,128,128), 2)
            cv2.putText(img, name, (x0,max(18,y0-5)), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255) if selected else (128,128,128),1)
        boxes.append(dict(name=name, xyxy=[int(x0),int(y0),int(x1),int(y1)]))
        if selected and x1>x0 and y1>y0:
            mask = np.zeros((h,w), bool)
            mask[y0:y1,x0:x1] = valid[y0:y1,x0:x1]
            if mask.any(): masks.append((mask,(x0,y0,x1,y1)))
    status = 'No target detected'
    proposals = []
    rejected = []
    for mask,box in masks:
        try:
            roi=target_roi(box,(h,w))
            for img in ((view,) if native else (color,view)):
                cv2.rectangle(img,roi[:2],(roi[0]+roi[2],roi[1]+roi[2]),(255,128,0),1)
            clean = np.where(valid, frame.depth_m, 0)
            candidates = sorted(predictor.predict(clean,roi,[mask],min_score), key=lambda c:-c.score)
            # Boxes localize targets for visualization only. No support-plane,
            # collision, jaw-opening or base-frame acceptance is claimed here.
            candidate = next(iter(candidates),None)
            if candidate is not None:
                for img in ((view,) if native else (color,view)): draw_grasp(img,candidate)
                if native:
                    uv = detector.last_depth_to_rgb[candidate.v, candidate.u]
                    if np.isfinite(uv).all():
                        cv2.drawMarker(color, tuple(to_panel(uv, layout).astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
                proposals.append(dict(pixel=[candidate.u,candidate.v],score=candidate.score,
                                      angle_rad=float(candidate.angle),width_px=candidate.model_width_px,
                                      roi=list(map(int,roi))))
        except ValueError as e:
            rejected.append(str(e))
    if masks:
        status = f'{len(proposals)} GG-CNN image proposals'
        if rejected: status += ' | ' + rejected[0]
    elif native and any(name in detector.allowed for name, _ in detector.last_native_detections):
        status = 'Target seen in RGB; no usable depth for grasp'
    for img in (color,view):
        cv2.putText(img,'PREVIEW ONLY - uncalibrated, no robot commands',(8,h-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
    canvas=np.hstack((color,view))
    mode = 'Native RGB / YOLO | measured depth / GG-CNN' if native else 'RGB is depth-aligned'
    cv2.putText(canvas,f'Depth valid: {100*valid.mean():.1f}% | {mode}',
                (8,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    cv2.putText(canvas,status,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
    return canvas,dict(status=status,detections=boxes,proposals=proposals,rejected=rejected,
                       native_detections=native_boxes, color_panel=layout,
                       detector_input=getattr(detector, 'image_source', 'depth_aligned_rgb'),
                       selected_classes=sorted(detector.allowed), depth_valid_fraction=float(valid.mean()),
                       scope='uncalibrated_image_preview',motion_enabled=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--camera',choices=['realsense','orbbec'])
    src.add_argument('--frame',type=Path)
    p.add_argument('--target-weights',required=True)
    p.add_argument('--target-class', action='append', help='Preview class override; default artificial_object')
    p.add_argument('--upstream',required=True)
    p.add_argument('--trusted-checkpoint',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--near',action='store_true',help='Orbbec 1280x800, disparity256/offset127')
    p.add_argument('--min-score',type=float,default=.25)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--frames',type=int,default=0,help='0 runs until Esc/Q; frame input runs once')
    p.add_argument('--output',type=Path)
    p.add_argument('--record-dir',type=Path,default=Path('field_recordings'),
                   help='Press S to save an unlabelled RGB-D observation here')
    args=p.parse_args()
    if args.headless and not args.frame and args.frames<=0: p.error('headless live mode requires --frames > 0')
    detector=TargetDetector(args.target_weights,args.target_class or [COMPETITION_TARGET],args.device)
    predictor=Predictor(args.upstream,args.trusted_checkpoint,args.device)
    provenance = dict(
        detector_sha256=hashlib.sha256(Path(args.target_weights).read_bytes()).hexdigest(),
        grasp_sha256=hashlib.sha256(Path(args.trusted_checkpoint).read_bytes()).hexdigest(),
        detector_classes=dict(detector.model.names),selected_classes=sorted(detector.allowed),
        camera_backend=args.camera if args.camera else 'saved_frame')
    if args.near and args.camera!='orbbec':p.error('--near requires --camera orbbec')
    context=(nullcontext(None) if args.frame else
             orbbec(with_color=True,near=args.near) if args.camera=='orbbec' else real_sense(with_color=True))
    count=0
    try:
        with context as read:
            if read:
                for _ in range(15): read()
            while True:
                started=time.monotonic()
                frame=load_frame(args.frame) if args.frame else read()
                canvas,result=render(frame,detector,predictor,args.min_score)
                result['inference_seconds']=time.monotonic()-started
                result['models']=provenance
                print(json.dumps(result,allow_nan=False),flush=True)
                if args.output:
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(args.output),canvas): raise OSError('Cannot save preview')
                    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
                count+=1
                if not args.headless:
                    cv2.imshow('Real RGB-D: YOLO + GG-CNN (no motion)',canvas)
                    if frame.raw_bgr is not None:
                        cv2.namedWindow('Original RGB - no depth alignment, no boxes',cv2.WINDOW_NORMAL)
                        cv2.imshow('Original RGB - no depth alignment, no boxes',frame.raw_bgr)
                    key=cv2.waitKey(1)&255
                    if key in (27,ord('q')): break
                    if key in (ord('s'),ord('S')):
                        print(f'SAVED {save_observation(args.record_dir,frame,canvas,result)}',flush=True)
                if args.frame or (args.frames>0 and count>=args.frames): break
    finally:
        if not args.headless: cv2.destroyAllWindows()


if __name__=='__main__': main()
