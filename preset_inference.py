"""Shared native RGB-D geometry and real YOLO/GG-CNN image inference."""
import cv2
import numpy as np
from camera import Frame,pinhole_rays
from rgbd_projection import depth_to_rgb
from geometry import sample
from preview_live import render


def factory_geometry(depth_profile,color_profile):
    di,dd=depth_profile.get_intrinsic(),depth_profile.get_distortion()
    ci,cd=color_profile.get_intrinsic(),color_profile.get_distortion()
    coefficients=lambda d:np.array([d.k1,d.k2,d.p1,d.p2,d.k3,d.k4,d.k5,d.k6],float)
    rays=pinhole_rays((di.height,di.width),di.fx,di.fy,di.cx,di.cy,coefficients(dd))
    ext=depth_profile.get_extrinsic_to(color_profile);transform=np.eye(4)
    transform[:3,:3]=np.asarray(ext.rot).reshape(3,3);transform[:3,3]=np.asarray(ext.transform)
    return rays,np.array([[ci.fx,0,ci.cx],[0,ci.fy,ci.cy],[0,0,1]],float),coefficients(cd),transform,str(cd.model)


def native_frame(depth_m,rgb,geometry):
    rays,k,dist,transform,model=geometry
    if rays.shape!=(*depth_m.shape,3):raise ValueError('Depth profile/ray shape mismatch')
    frame=Frame(depth_m,rays,np.zeros((*depth_m.shape,3),np.uint8),rgb,k,dist,transform,model)
    uv=depth_to_rgb(frame)
    valid=np.isfinite(uv).all(axis=2)
    uv=np.where(valid[...,None],uv,-1).astype(np.float32)
    frame.bgr=cv2.remap(rgb,uv[...,0],uv[...,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    return frame


def infer(frame,detector,predictor,wrist=None,joints=None,hand_eye=None):
    _,result=render(frame,detector,predictor,.25)
    for proposal in result['proposals']:
        try:
            camera_mm=sample(frame,*proposal['pixel'])*1000
            proposal['camera_xyz_mm']=camera_mm.tolist()
            if wrist is not None and joints is not None:
                r,t=wrist.camera_pose(joints,hand_eye)
                proposal['candidate_base_xyz_mm']=(r@camera_mm+t).tolist()
        except ValueError as exc:proposal['localization_error']=str(exc)
    result['scope']='image proposals and unvalidated candidate coordinates; no IK or motion'
    result['motion_enabled']=False
    return result
