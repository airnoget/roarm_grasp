"""Post-filter fixed gripper regions in native RGB without changing raw pixels."""
import json
from pathlib import Path
import cv2
import numpy as np

PROFILE=Path(__file__).resolve().parent/'camera_profiles/gripper-exclusion.json'


class GripperExclusion:
    def __init__(self,path=PROFILE):
        self.spec=json.loads(Path(path).read_text(encoding='utf-8'));self.cached=None

    def mask(self,shape):
        height,width=shape[:2]
        if not self.spec['enabled'] or [width,height]!=self.spec['reference_size_wh']:return None
        if self.cached is None:
            mask=np.zeros((height,width),np.uint8)
            cv2.fillPoly(mask,[np.array(p,np.int32) for p in self.spec['polygons_px']],1)
            self.cached=mask.astype(bool)
        return self.cached

    def filter(self,detections,shape):
        mask=self.mask(shape)
        if mask is None:return detections,[],None
        height,width=mask.shape;kept=[];ignored=[]
        for name,box in detections:
            box=np.asarray(box,float)
            if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError('Invalid detection bounds')
            x0,y0=np.clip(np.floor(box[:2]),0,[width,height]).astype(int)
            x1,y1=np.clip(np.ceil(box[2:]),0,[width,height]).astype(int)
            if x1<=x0 or y1<=y0:continue
            fraction=float(mask[y0:y1,x0:x1].mean())
            cx,cy=np.rint((box[:2]+box[2:])/2).astype(int)
            centre=bool(mask[np.clip(cy,0,height-1),np.clip(cx,0,width-1)])
            if fraction>=self.spec['box_overlap_threshold'] or (centre and fraction>=self.spec['centre_overlap_threshold']):
                ignored.append(dict(name=name,xyxy=box.tolist(),reason='fixed_gripper_region',overlap_fraction=fraction))
            else:kept.append((name,box))
        return kept,ignored,mask


def project_exclusion(mask,uv):
    if mask is None:return np.zeros(uv.shape[:2],bool)
    maps=np.where(np.isfinite(uv),uv,-1).astype(np.float32)
    return cv2.remap(mask.astype(np.uint8),maps[...,0],maps[...,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
