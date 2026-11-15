"""Manual parallel-jaw grasps in the native depth raster (pixels, not robot poses)."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from dataset_labels import atomic_text
from geometry import sample
from preview_live import target_roi


def make_grasp(points, shape):
    """Two opposing finger contacts, then a point defining half contact-band height."""
    p=np.asarray(points,float)
    h,w=shape
    if p.shape!=(3,2) or not np.isfinite(p).all():raise ValueError('Need three finite points')
    if np.any(p<0) or np.any(p[:,0]>=w) or np.any(p[:,1]>=h):raise ValueError('Points outside depth image')
    a,b,c=p;center=(a+b)/2;axis=b-a;width=float(np.linalg.norm(axis))
    if width<4:raise ValueError('Finger separation must be at least 4 pixels')
    axis/=width;normal=np.array([-axis[1],axis[0]])
    height=float(2*abs(np.dot(c-center,normal)))
    if height<4:raise ValueError('Contact band must be at least 4 pixels high')
    corners=np.array([a-normal*height/2,b-normal*height/2,b+normal*height/2,a+normal*height/2])
    if np.any(corners<0) or np.any(corners[:,0]>=w) or np.any(corners[:,1]>=h):raise ValueError('Rectangle crosses image border')
    angle=float((np.arctan2(-axis[1],axis[0])+np.pi/2)%np.pi-np.pi/2)
    return dict(contacts_xy=[a.tolist(),b.tolist()],center_xy=center.tolist(),
                corners_xy=corners.tolist(),opening_px=width,contact_band_px=height,
                angle_rad_ggcnn=angle,grasp_type='left_right_opposing_fingers')


def depth_check(frame,grasp):
    corners=np.asarray(grasp['corners_xy']);lo=corners.min(axis=0);hi=corners.max(axis=0)
    valid=np.isfinite(frame.depth_m)&(frame.depth_m>0)&(frame.depth_m<10)
    mask=np.zeros(valid.shape,np.uint8);cv2.fillConvexPoly(mask,np.rint(corners).astype(np.int32),1)
    result=dict(rectangle_valid_fraction=float(valid[mask.astype(bool)].mean()),usable_for_current_predictor=False)
    try:
        x,y,size=target_roi([*lo,*hi],valid.shape)
        result['roi_xy_size']=[x,y,size]
        result['roi_hole_fraction']=float(1-valid[y:y+size,x:x+size].mean())
        u,v=np.rint(grasp['center_xy']).astype(int)
        result['center_camera_mm']=(sample(frame,int(u),int(v))*1000).tolist()
        result['usable_for_current_predictor']=result['roi_hole_fraction']<=.25
    except ValueError as exc:result['reason']=str(exc)
    return result


def save_grasps(folder,frame,grasps,*,skipped=False):
    folder=Path(folder)
    if skipped and grasps:raise ValueError('Clear annotations before skipping')
    if not skipped and not grasps:raise ValueError('Draw a grasp, or explicitly skip')
    entries=[]
    for g in grasps:
        # Reconstruct and validate on save; never accept unchecked mutable corners.
        a,b=np.asarray(g['contacts_xy']);axis=(b-a)/np.linalg.norm(b-a)
        third=(a+b)/2+np.array([-axis[1],axis[0]])*g['contact_band_px']/2
        clean=make_grasp([a,b,third],frame.depth_m.shape)
        clean['depth_check']=depth_check(frame,clean);entries.append(clean)
    meta=json.loads((folder/'capture.json').read_text(encoding='utf-8-sig'))
    result=dict(schema=1,source='manual_three_clicks',reviewed_time_ns=time.time_ns(),
                coordinate_space='native_depth_xy',image_size=[frame.depth_m.shape[1],frame.depth_m.shape[0]],
                display_rotation=180,source_frame='rgbd.npz',
                source_sha256=hashlib.sha256((folder/'rgbd.npz').read_bytes()).hexdigest(),
                camera_profile=meta.get('camera_profile'),split_group=meta.get('split_group'),
                grasps=entries,skipped=skipped,unmarked_grasps_are_negatives=False,
                physical_gripper_validated=False,motion_enabled=False)
    atomic_text(folder/'grasp_annotation.json',json.dumps(result,indent=2,allow_nan=False))
    return result
