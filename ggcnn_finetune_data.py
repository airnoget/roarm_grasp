"""Verified manual positives, using the same YOLO-derived crop as live inference."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from camera import load_frame
from geometry import sample
from preview_live import target_roi
from rgbd_projection import depth_to_rgb,boxes_to_depth


def load_samples(session,preset_sha):
    samples=[];excluded=[]
    for path in sorted(Path(session).glob('*/grasp_annotation.json')):
        label=json.loads(path.read_text());folder=path.parent
        if label['coordinate_space']!='native_depth_xy':raise ValueError('Unsupported label coordinates')
        if label['source_sha256']!=hashlib.sha256((folder/'rgbd.npz').read_bytes()).hexdigest():
            raise ValueError(f'Frame changed: {folder}')
        if label.get('camera_profile',{}).get('preset_sha256')!=preset_sha:raise ValueError('Preset mismatch')
        if label.get('skipped'):continue
        frame=load_frame(folder/'rgbd.npz');manual=json.loads((folder/'annotation.json').read_text())
        boxes=boxes_to_depth([('target',b) for b in manual['boxes_xyxy']],depth_to_rgb(frame))
        valid=np.isfinite(frame.depth_m)&(frame.depth_m>0)&(frame.depth_m<10)
        for index,(_,box) in enumerate(boxes):
            try:
                x,y,size=target_roi(box,valid.shape)
                good=[]
                for grasp in label['grasps']:
                    u,v=grasp['center_xy']
                    if box[0]<=u<box[2] and box[1]<=v<box[3]:
                        corners=np.array(grasp['corners_xy'])
                        if (corners[:,0]<x).any() or (corners[:,0]>=x+size).any() or (corners[:,1]<y).any() or (corners[:,1]>=y+size).any():continue
                        sample(frame,int(round(u)),int(round(v)))
                        good.append(grasp)
                if not good:raise ValueError('No manually labeled grasp with reliable measured center')
                bad=~valid[y:y+size,x:x+size]
                if bad.mean()>.25:raise ValueError('More than 25% holes in live-style crop')
                crop=frame.depth_m[y:y+size,x:x+size].astype(np.float32).copy();crop[bad]=0
                crop=cv2.inpaint(crop,bad.astype(np.uint8),2,cv2.INPAINT_NS)
                image=cv2.resize(crop,(300,300));image=np.clip(image-image.mean(),-1,1)
                # Partial annotations: supervise only compact positive regions;
                # unmarked positions are not assigned negative ground truth.
                targets=np.zeros((4,300,300),np.float32);mask=np.zeros((300,300),np.float32)
                for g in good:
                    p=np.array(g['corners_xy']);center=np.array(g['center_xy'])
                    axis=np.array(g['contacts_xy'][1])-g['contacts_xy'][0];axis/=np.linalg.norm(axis)
                    # Same compact central-third convention as upstream GG-CNN.
                    p-=np.sign((p-center)@axis)[:,None]*axis*g['opening_px']/3
                    points=np.rint((p-[x,y])*300/size).astype(np.int32)
                    area=np.zeros((300,300),np.uint8);cv2.fillConvexPoly(area,points,1)
                    sel=area.astype(bool);mask[sel]=1
                    targets[:,sel]=np.array([1,np.cos(2*g['angle_rad_ggcnn']),np.sin(2*g['angle_rad_ggcnn']),
                                            min(g['opening_px']*300/size,150)/150],np.float32)[:,None]
                if not mask.any():raise ValueError('Empty positive region')
                samples.append(dict(image=image[None],target=targets,mask=mask[None],
                                    group=label.get('split_group') or folder.name,
                                    source=str(path),box_index=index,roi=[x,y,size]))
            except ValueError as exc:excluded.append(dict(source=str(path),box_index=index,reason=str(exc)))
    return samples,excluded


def split_samples(samples,seed=42):
    groups=sorted({s['group'] for s in samples})
    if len(groups)<5:raise ValueError(f'Only {len(groups)} independent usable poses; need at least 5 for a pilot split')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    val=set(groups[:max(2,round(len(groups)*.2))])
    return [s for s in samples if s['group'] not in val],[s for s in samples if s['group'] in val]
