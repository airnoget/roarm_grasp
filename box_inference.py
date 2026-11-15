"""YOLO-gated box geometry; no learned grasp model, IK or motion commands."""
import numpy as np
from box_geometry import fit_box


def infer_boxes(frame,detector,wrist=None,joints=None,hand_eye=None):
    detections=detector(frame)
    if getattr(detector,'image_source',None)!='native_rgb':
        raise ValueError('Box geometry requires native RGB factory calibration')
    uv=detector.last_depth_to_rgb;valid=np.isfinite(uv).all(axis=2)
    native=detector.last_native_detections;masks=[]
    for name,box in native:
        x0,y0,x1,y1=box
        masks.append(valid&(uv[...,0]>=x0)&(uv[...,0]<x1)&(uv[...,1]>=y0)&(uv[...,1]<y1))
    occupied=np.logical_or.reduce(masks) if masks else np.zeros(valid.shape,bool)
    proposals=[];rejected=[];diagnostics=[]
    h,w=valid.shape
    for index,((name,_),mask) in enumerate(zip(native,masks)):
        if name not in detector.allowed:continue
        measured=mask&np.isfinite(frame.depth_m)&(frame.depth_m>0)
        diagnostic=dict(target=index+1,name=name,associated_pixels=int(mask.sum()),
                        measured_points=int(measured.sum()),
                        measured_fraction=None)
        # Projection exists only for valid returns. Dividing valid returns by
        # projected mask size falsely reports 100% even when depth has holes.
        rows,cols=np.nonzero(mask)
        if len(rows):
            roi=frame.depth_m[rows.min():rows.max()+1,cols.min():cols.max()+1]
            diagnostic['measured_fraction']=float((np.isfinite(roi)&(roi>0)).mean())
        else:diagnostic['measured_fraction']=0.
        diagnostic['fraction_scope']='Approximate mapped depth bounding rectangle; not RGB object area'
        diagnostics.append(diagnostic)
        try:
            try:geometry=fit_box(frame,mask,occupied)
            except ValueError as depth_error:
                from rgb_box_geometry import fit_rgb_top
                try:
                    geometry=fit_rgb_top(frame,mask,occupied,native[index][1],uv)
                    geometry['depth_only_rejection']=str(depth_error)
                except ValueError as rgb_error:
                    raise ValueError(f'{depth_error}; {rgb_error}') from rgb_error
            u,v=geometry['pixel'];xyz=np.array(geometry['camera_surface_xyz_mm']);axis=np.array(geometry['camera_closing_axis'])
            # Differential perspective projection of the fitted 3D closing axis.
            if not (0<u<w-1 and 0<v<h-1):raise ValueError('Fitted centre at image border')
            du=(frame.rays[v,u+1]-frame.rays[v,u-1])/2
            dv=(frame.rays[v+1,u]-frame.rays[v-1,u])/2
            dr=(axis-frame.rays[v,u]*axis[2])/xyz[2]
            image_axis=np.linalg.lstsq(np.stack((du,dv),axis=1),dr,rcond=None)[0]
            norm=float(np.linalg.norm(image_axis))
            if norm<1e-6:raise ValueError('Closing direction projects to zero')
            p=dict(pixel=[u,v],angle_rad=float(np.arctan2(image_axis[1],image_axis[0])),
                   width_px=geometry['observed_opening_mm']*norm,score=1.0,
                   camera_xyz_mm=xyz.tolist(),geometry=geometry,backend='box_geometry',score_is_probability=False)
            if wrist is not None and joints is not None:
                r,t=wrist.camera_pose(joints,hand_eye);p['candidate_base_xyz_mm']=(r@xyz+t).tolist()
            proposals.append(p)
            diagnostic.update(fit=True,reason='RGB boundary + measured plane' if geometry.get('boundary_source') else 'OK',top_points=geometry['top_points'])
        except ValueError as exc:
            diagnostic.update(fit=False,reason=str(exc))
            rejected.append(f'Target {index+1}: {exc}')
    return dict(native_detections=[dict(name=n,xyxy=list(map(float,b))) for n,b in native],
                detections=[dict(name=n,xyxy=list(map(int,b))) for n,b in detections],
                proposals=proposals,rejected=rejected,depth_diagnostics=diagnostics,backend='box_geometry',motion_enabled=False,
                scope='Geometric surface fit under isolated-box/planar-support assumptions; not a robot grasp acceptance')
