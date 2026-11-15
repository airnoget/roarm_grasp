"""Measured point-cloud fit for an isolated box on a locally planar support.

Outputs are camera-frame surface estimates, not collision-checked robot grasps.
No missing depth is filled or silently interpreted as a measured centre.
"""
import cv2
import numpy as np


def plane_fit(points,tolerance=3.,seed=42):
    points=np.asarray(points,float)
    if len(points)<80:raise ValueError('Insufficient support depth (need 80 points)')
    rng=np.random.default_rng(seed)
    cloud=points[rng.choice(len(points),min(len(points),2000),replace=False)]
    best=None
    for _ in range(100):
        a,b,c=cloud[rng.choice(len(cloud),3,replace=False)]
        n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<1e-6:continue
        n/=length;d=-np.dot(n,a);inliers=np.abs(cloud@n+d)<=tolerance
        if best is None or inliers.sum()>best.sum():best=inliers
    if best is None or best.mean()<.7:raise ValueError('Surrounding support is not a reliable single plane')
    selected=cloud[best];center=selected.mean(0)
    _,sv,vh=np.linalg.svd(selected-center,full_matrices=False)
    if sv[1]<5:raise ValueError('Support points nearly collinear')
    n=vh[-1]
    if np.dot(n,center)>0:n=-n  # normal points towards the camera
    d=-float(n@center)
    residual=np.abs(selected@n+d)
    if np.percentile(residual,95)>tolerance:raise ValueError('Support plane residual too large')
    return n,d,float(np.sqrt(np.mean(residual**2)))


def fit_box(frame,target_mask,all_targets=None):
    depth=np.asarray(frame.depth_m);valid=np.isfinite(depth)&(depth>0)&(depth<3)
    mask=np.asarray(target_mask,bool)
    if mask.shape!=depth.shape:raise ValueError('Target mask differs from native depth raster')
    v,u=np.nonzero(mask)
    if not len(u):raise ValueError('No measured points associated with target')
    y0,y1=int(v.min()),int(v.max()+1);x0,x1=int(u.min()),int(u.max()+1)
    pad=max(20,int(max(x1-x0,y1-y0)*.6));ring=np.zeros(depth.shape,bool)
    ring[max(0,y0-pad):min(depth.shape[0],y1+pad),max(0,x0-pad):min(depth.shape[1],x1+pad)]=True
    ring&=~(mask if all_targets is None else all_targets)
    xyz=frame.rays*depth[...,None]*1000
    n,d,rms=plane_fit(xyz[ring&valid])
    height=xyz@n+d
    foreground=mask&valid&(height>8)&(height<200)
    # A one/two-pixel missing-depth seam must not split one measured surface
    # into competing targets. Closing is used ONLY for association; fitted
    # points below remain the original measured foreground, never filled pixels.
    association=cv2.morphologyEx(foreground.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(association,8)
    sizes=np.bincount(labels[foreground],minlength=count)[1:]
    if not len(sizes) or sizes.max()<80:raise ValueError('Insufficient connected object depth above support')
    order=np.argsort(sizes)[::-1]
    component=(labels==int(order[0]+1))&foreground
    # Small holes split one top face into components at high native resolution.
    # Associate by physical gap AND matching height, not a fixed large pixel
    # kernel. Keep every fitted point measured; never synthesize missing depth.
    ray_step=(np.median(np.linalg.norm(np.diff(frame.rays,axis=1),axis=2))+
              np.median(np.linalg.norm(np.diff(frame.rays,axis=0),axis=2)))/2
    mm_per_pixel=max(1e-6,float(ray_step*np.median(depth[component])*1000))
    reference_height=np.percentile(height[component],[50,90])
    remaining=list(order[1:])
    needs_association=any(sizes[index]>.35*sizes[order[0]] for index in remaining)
    for _ in range(3 if needs_association else 0):
        changed=False
        distances=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5)
        for index in remaining.copy():
            if sizes[index]<max(20,.1*sizes[order[0]]):continue
            candidate=(labels==int(index+1))&foreground
            heights_candidate=np.percentile(height[candidate],[50,90])
            if (np.max(np.abs(heights_candidate-reference_height))<=3 and
                    float(distances[candidate].min())*mm_per_pixel<=3):
                component|=candidate;remaining.remove(index);changed=True
        if not changed:break
    if any(sizes[index]>.35*component.sum() for index in remaining):
        raise ValueError('Multiple separated surfaces inside target (gap >3mm or inconsistent height)')
    heights=height[component];top_height=float(np.percentile(heights,90))
    top=component&(np.abs(height-top_height)<=3)
    if top.sum()<60:raise ValueError('Insufficient measured top-face points')
    # Camera X projected into the support plane defines left/right preference.
    ex=np.array([1.,0,0])-n*n[0]
    if np.linalg.norm(ex)<.2:raise ValueError('Viewing direction unsuitable for left/right closing axis')
    ex/=np.linalg.norm(ex);ey=np.cross(n,ex)
    points=xyz[top];coords=np.column_stack((points@ex,points@ey)).astype(np.float32)
    rect=cv2.minAreaRect(coords);corners=cv2.boxPoints(rect)
    a,b=corners[1]-corners[0],corners[2]-corners[1]
    spans=np.array([np.linalg.norm(a),np.linalg.norm(b)])
    if spans.min()<10 or spans.max()>160:raise ValueError('Observed footprint outside 10..160 mm pilot range')
    hull_area=cv2.contourArea(cv2.convexHull(coords));area=float(np.prod(spans))
    if hull_area/area<.7:
        raise ValueError(f'Top-face hull/rectangle coverage {100*hull_area/area:.1f}% < 70%; incomplete boundary, no approach target')
    # Both halves and the centre must have measurements. A corner/edge fragment
    # must not be presented as a complete box just because minAreaRect fits it.
    center2=np.asarray(rect[0]);axis_a=a/spans[0];axis_b=b/spans[1]
    grid=(coords-center2)@np.stack((axis_a,axis_b),axis=1)/spans+.5
    bins=np.clip((grid*5).astype(int),0,4)
    coverage=len(np.unique(bins[:,0]+5*bins[:,1]))/25
    if coverage<.6:raise ValueError('Measured top-face coverage too low')
    choose=0 if abs(axis_a[0])>=abs(axis_b[0]) else 1
    axis2=(axis_a,axis_b)[choose]
    axis3=ex*axis2[0]+ey*axis2[1]
    if axis3@ex<0:axis3=-axis3
    surface=center2[0]*ex+center2[1]*ey+(top_height-d)*n
    # Locate the inferred centre in the native depth raster without sampling
    # its potentially missing depth. This is a fitted point, not a depth return.
    ray=surface/surface[2];dist=np.sum((frame.rays[y0:y1,x0:x1]-ray)**2,axis=2)
    row,col=np.unravel_index(np.argmin(dist),dist.shape);pixel=[int(x0+col),int(y0+row)]
    return dict(camera_surface_xyz_mm=surface.tolist(),camera_closing_axis=axis3.tolist(),
                camera_support_normal=n.tolist(),height_above_support_mm=top_height,
                observed_opening_mm=float(spans[choose]),observed_footprint_mm=spans.tolist(),
                pixel=pixel,center_is_fitted=True,center_has_depth=bool(valid[pixel[1],pixel[0]]),
                top_points=int(top.sum()),top_grid_coverage=coverage,support_rms_mm=rms,
                scope='Isolated box on a local planar support; observed footprint may be partial. No robot motion.',
                motion_enabled=False)
