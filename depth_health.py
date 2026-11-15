"""Measured-depth diagnostics. No interpolation, detector, serial, or motion."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
for name in reversed(('.deps','.orbbec-python-deps','.realsense-deps')):
    sys.path.insert(0,str(ROOT/name))
import cv2
import numpy as np
from camera import orbbec,real_sense,load_frame,save_frame


def statistics(depth, box=None):
    depth=np.asarray(depth)
    if depth.ndim!=2 or not depth.size: raise ValueError('Expected nonempty depth raster')
    if box is not None:
        x0,y0,x1,y1=map(int,box)
        if not (0<=x0<x1<=depth.shape[1] and 0<=y0<y1<=depth.shape[0]):
            raise ValueError('ROI outside native depth raster')
        depth=depth[y0:y1,x0:x1]
    valid=np.isfinite(depth)&(depth>0)
    d=depth[valid]
    return dict(pixels=int(depth.size),valid_pixels=int(valid.sum()),
                valid_fraction=float(valid.mean()),zero_fraction=float((depth==0).mean()),
                invalid_nonzero_fraction=float((~valid & (depth!=0)).mean()),
                depth_mm_percentiles_5_50_95=(np.percentile(d*1000,[5,50,95]).tolist() if d.size else None),
                roi_depth_xyxy=box)


def depth_image(depth):
    valid=np.isfinite(depth)&(depth>0)
    values=depth[valid]
    scaled=np.zeros(depth.shape,np.uint8)
    if values.size:
        lo,hi=np.percentile(values,[2,98]);hi=max(hi,lo+.001)
        scaled[valid]=np.clip((depth[valid]-lo)/(hi-lo)*255,0,255).astype(np.uint8)
    view=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
    view[~valid]=0
    return view


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',choices=['orbbec','realsense'],default='orbbec')
    p.add_argument('--near',action='store_true',help='Orbbec 1280x800, disparity256/offset127; restore settings on normal exit')
    p.add_argument('--depth-only',action='store_true',help='Do not open the color stream or RGB alignment')
    p.add_argument('--rotate',type=int,choices=[0,180],default=180,help='Display rotation only; raw saved geometry stays unchanged')
    src=p.add_mutually_exclusive_group()
    src.add_argument('--frame',type=Path)
    src.add_argument('--session',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'depth_diagnostics')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.near and a.camera!='orbbec':p.error('--near applies only to Orbbec')
    if a.session:
        rows=[]
        for capture in sorted(a.session.glob('*/capture.json')):
            path=capture.parent/'rgbd.npz'
            f=load_frame(path)
            rows.append(dict(frame=str(path.resolve()),**statistics(f.depth_m)))
        if not rows:p.error('No complete captures')
        fractions=[r['valid_fraction'] for r in rows]
        report=dict(created=datetime.now().astimezone().isoformat(),frames=rows,
                    valid_fraction_min_median_max=np.percentile(fractions,[0,50,100]).tolist(),
                    scope='native measured depth before application filtering; no physical cause inferred')
        (a.output/'session-health.json').write_text(json.dumps(report,indent=2))
        print('FRAMES',len(rows),'VALID min/median/max',report['valid_fraction_min_median_max'])
        return
    if a.frame:
        frame=load_frame(a.frame)
        report=statistics(frame.depth_m)
        (a.output/'frame-health.json').write_text(json.dumps(report,indent=2))
        cv2.imwrite(str(a.output/'measured-depth.png'),depth_image(frame.depth_m))
        print(json.dumps(report));return
    print('Camera only. Drag ROI on depth window. S saves raw RGB-D + statistics; Q/Esc exits.',flush=True)
    name='Measured depth - drag target ROI; S save; Q exit'
    start=None;roi=None;shape=None
    def mouse(event,x,y,flags,unused):
        nonlocal start,roi
        if shape is None:return
        x=int(np.clip(x,0,shape[1]-1));y=int(np.clip(y,0,shape[0]-1))
        if a.rotate==180:x,y=shape[1]-1-x,shape[0]-1-y
        if event==cv2.EVENT_LBUTTONDOWN:start=(x,y)
        if event==cv2.EVENT_LBUTTONUP and start is not None:
            x0,x1=sorted([start[0],x]);y0,y1=sorted([start[1],y]);start=None
            roi=[x0,y0,x1+1,y1+1] if x1>x0 and y1>y0 else None
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(name,mouse)
    try:
        device_status=dict(camera_backend=a.camera)
        device_status['color_requested']=not a.depth_only
        context=(orbbec(with_color=not a.depth_only,diagnostics=device_status,near=a.near) if a.camera=='orbbec'
                 else real_sense(with_color=not a.depth_only,diagnostics=device_status))
        with context as read:
            (a.output/'device-status.json').write_text(json.dumps(device_status,indent=2))
            print('DEVICE STATUS',a.output/'device-status.json',flush=True)
            for _ in range(15):read()
            ready=False
            while True:
                if ready and cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
                f=read();shape=f.depth_m.shape
                report=dict(full=statistics(f.depth_m),roi=statistics(f.depth_m,roi) if roi else None,
                            measurements_filled=False,robot_port_opened=False,device_status=device_status,display_rotation_deg=a.rotate)
                view=depth_image(f.depth_m)
                if a.rotate==180:view=cv2.rotate(view,cv2.ROTATE_180)
                cv2.putText(view,f'Full measured valid: {100*report["full"]["valid_fraction"]:.1f}%',(8,25),0,.65,(255,255,255),2)
                if roi:
                    box=roi if a.rotate==0 else [shape[1]-roi[2],shape[0]-roi[3],shape[1]-roi[0],shape[0]-roi[1]]
                    cv2.rectangle(view,tuple(box[:2]),tuple(box[2:]),(255,255,255),2)
                    cv2.putText(view,f'ROI valid: {100*report["roi"]["valid_fraction"]:.1f}% | pixels: {report["roi"]["pixels"]}',(8,52),0,.65,(255,255,255),2)
                cv2.imshow(name,view)
                if f.raw_bgr is not None:cv2.imshow('Original RGB - depth diagnostics',cv2.rotate(f.raw_bgr,cv2.ROTATE_180) if a.rotate==180 else f.raw_bgr)
                key=cv2.waitKey(1)&255;ready=True
                if key in (27,ord('q')):break
                if key==ord('s'):
                    folder=a.output/str(time.time_ns());folder.mkdir()
                    save_frame(folder/'rgbd.npz',f)
                    (folder/'health.json').write_text(json.dumps(report,indent=2))
                    cv2.imwrite(str(folder/'depth.png'),view)
                    print('SAVED',folder,report,flush=True)
    finally:cv2.destroyAllWindows()


if __name__=='__main__':main()
