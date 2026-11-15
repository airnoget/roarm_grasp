"""Record raw RGB tag corners, factory camera calibration and stationary wrist angles."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from arm import Arm
from camera import orbbec,save_frame
from wrist_pose import stationary_frame


def detect(image,tag_id):
    params=cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),params)
    corners,ids,_=detector.detectMarkers(image)
    matches=[] if ids is None else [c.reshape(4,2) for c,i in zip(corners,ids.flatten()) if int(i)==tag_id]
    if len(matches)!=1: raise ValueError(f'Need exactly one visible tag ID {tag_id}')
    c=matches[0];h,w=image.shape[:2]
    if c[:,0].min()<4 or c[:,1].min()<4 or c[:,0].max()>w-5 or c[:,1].max()>h-5:
        raise ValueError('Tag is too close to image border')
    if abs(cv2.contourArea(c.astype(np.float32)))<1600:
        raise ValueError('Tag is too small; change the view before recording')
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='COM6');p.add_argument('--tag-id',type=int,default=0)
    p.add_argument('--tag-mm',type=float,default=50.)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not np.isfinite(a.tag_mm) or a.tag_mm<=0: p.error('tag size must be a finite measured length in mm')
    root=a.output or Path('calibration')/f'tag-{time.time_ns()}'
    root.mkdir(parents=True,exist_ok=False)
    setup=dict(schema=1,family='tag36h11',tag_id=a.tag_id,tag_size_mm=a.tag_mm,
               port=a.port,camera='orbbec',camera_pose_frame='raw_rgb',
               target_must_remain_fixed=True,robot_mount='link5',motion_commands=False)
    (root/'session.json').write_text(json.dumps(setup,indent=2),encoding='utf-8')
    print(f'SESSION {root.resolve()}\nF=fit C=check S=save Q=exit. Keep target and robot base fixed.',flush=True)
    print('Opening the serial port may reset the controller; keep the arm movement area clear.',flush=True)
    group='fit';counts={'fit':0,'check':0}
    try:
        with Arm(a.port) as arm,orbbec(with_color=True) as read:
            for _ in range(15):read()
            while True:
                frame=read()
                if frame.raw_bgr is None: raise RuntimeError('Raw RGB required for tag calibration')
                view=frame.raw_bgr.copy()
                try:
                    corners=detect(frame.raw_bgr,a.tag_id)
                    cv2.polylines(view,[corners.astype(np.int32)],True,(0,255,0),2)
                    status=f'ID {a.tag_id} ready'
                except ValueError as e:status=str(e)
                cv2.putText(view,f'{group} {counts} | {status}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
                cv2.namedWindow('Tag wrist capture - raw RGB',cv2.WINDOW_NORMAL)
                cv2.imshow('Tag wrist capture - raw RGB',view)
                key=cv2.waitKey(1)&255
                if key in (27,ord('q')):break
                if key==ord('f'):group='fit'
                if key==ord('c'):group='check'
                if key==ord('s'):
                    try:
                        captured,joints=stationary_frame(arm,read)
                        corners=detect(captured.raw_bgr,a.tag_id)
                        if any(getattr(captured,k) is None for k in ('raw_K','raw_dist','T_raw_from_depth_mm','raw_dist_model')):
                            raise ValueError('Factory RGB camera calibration is missing')
                        dest=root/group/f'{counts[group]:03d}'
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        with dest.with_suffix('.npz').open('xb') as out:save_frame(out,captured,joints)
                        record=dict(setup,group=group,corners_raw_px=corners.tolist(),
                                    joints_rad=joints.tolist(),snapshot=dest.with_suffix('.npz').name)
                        dest.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
                        counts[group]+=1
                        print(f'SAVED {dest} {counts}',flush=True)
                    except (ValueError,RuntimeError,TimeoutError) as e: print(f'NOT SAVED: {e}',flush=True)
    finally:cv2.destroyAllWindows()


if __name__=='__main__':main()
