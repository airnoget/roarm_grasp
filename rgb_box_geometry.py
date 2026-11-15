"""RGB top-face boundary + measured depth plane fallback for isolated boxes.

No depth pixels are filled. RGB corners are rays intersected with a measured
top plane; outputs remain fitted surface estimates, not measured corner returns.
"""
import cv2
import numpy as np
from box_geometry import plane_fit


def fit_rgb_top(frame,mask,occupied,box,uv):
    depth=frame.depth_m;valid=np.isfinite(depth)&(depth>0)&(depth<3)
    v,u=np.nonzero(mask)
    if len(u)<80:raise ValueError('RGB fallback: insufficient measured target depth')
    y0,y1=v.min(),v.max()+1;x0,x1=u.min(),u.max()+1
    pad=max(20,int(max(y1-y0,x1-x0)*.6));ring=np.zeros(depth.shape,bool)
    ring[max(0,y0-pad):min(depth.shape[0],y1+pad),max(0,x0-pad):min(depth.shape[1],x1+pad)]=True
    ring&=~occupied
    xyz=frame.rays*depth[...,None]*1000
    n,d,rms=plane_fit(xyz[ring&valid])
    height=xyz@n+d;foreground=mask&valid&(height>8)&(height<200)
    if foreground.sum()<80:raise ValueError('RGB fallback: no measured elevated object')
    top_height=float(np.percentile(height[foreground],90))
    top=foreground&(np.abs(height-top_height)<=3)
    points=xyz[top];pixels=uv[top]
    if len(points)<60:raise ValueError('RGB fallback: insufficient measured top face')
    image=frame.raw_bgr;ih,iw=image.shape[:2]
    bx0,by0,bx1,by1=np.asarray(box,float)
    xa,ya=max(0,int(bx0)-5),max(0,int(by0)-5)
    xb,yb=min(iw,int(bx1)+6),min(ih,int(by1)+6)
    gray=cv2.cvtColor(image[ya:yb,xa:xb],cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    edges=cv2.Canny(gray,25,75)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # Top-face RGB texture can recover a closed outline when its weak internal
    # edge does not form a Canny contour. Seed appearance ONLY from measured
    # top-plane pixels, rather than hard-coding blue/green thresholds.
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    sample=np.rint(pixels).astype(int)
    sample[:,0]=np.clip(sample[:,0],0,iw-1);sample[:,1]=np.clip(sample[:,1],0,ih-1)
    colors=lab[sample[:,1],sample[:,0]];center_color=np.median(colors,axis=0)
    spread=np.linalg.norm(colors-center_color,axis=1)
    threshold=max(6.,min(25.,float(np.percentile(spread,90))+2))
    color_mask=(np.linalg.norm(lab[ya:yb,xa:xb]-center_color,axis=2)<=threshold).astype(np.uint8)
    color_mask=cv2.morphologyEx(color_mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    color_contours,_=cv2.findContours(color_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=list(contours)+list(color_contours)
    transform=frame.T_raw_from_depth_mm;rotation=transform[:3,:3];origin=-rotation.T@transform[:3,3]
    ex=np.array([1.,0,0])-n*n[0];ex/=np.linalg.norm(ex);ey=np.cross(n,ex)
    options=[]
    for contour in contours:
        perimeter=cv2.arcLength(contour,True)
        quad=cv2.approxPolyDP(contour,.025*perimeter,True)
        if len(quad)!=4 or not cv2.isContourConvex(quad):continue
        quad=quad.reshape(4,2).astype(float)+[xa,ya]
        area=cv2.contourArea(quad.astype(np.float32))
        if not .08*(bx1-bx0)*(by1-by0)<area<.95*(bx1-bx0)*(by1-by0):continue
        inside=np.array([cv2.pointPolygonTest(quad.astype(np.float32),tuple(map(float,p)),True)>=-2 for p in pixels[::max(1,len(pixels)//1500)]])
        if inside.mean()<.85:continue
        distortion=np.zeros(8) if str(frame.raw_dist_model).endswith('.NONE') else frame.raw_dist
        rays=cv2.undistortPoints(quad.reshape(-1,1,2),frame.raw_K,distortion).reshape(-1,2)
        directions=np.c_[rays,np.ones(4)]@rotation
        denominators=directions@n
        if np.any(np.abs(denominators)<1e-5):continue
        lengths=(top_height-d-origin@n)/denominators
        if np.any(lengths<=0):continue
        corners=origin+directions*lengths[:,None]
        coords=np.c_[corners@ex,corners@ey].astype(np.float32)
        rect=cv2.minAreaRect(coords);rc=cv2.boxPoints(rect)
        a,b=rc[1]-rc[0],rc[2]-rc[1];spans=np.array([np.linalg.norm(a),np.linalg.norm(b)])
        if spans.min()<10 or spans.max()>160:continue
        rectangularity=cv2.contourArea(coords)/float(np.prod(spans))
        if rectangularity<.9:continue
        center=np.asarray(rect[0]);axes=np.stack((a/spans[0],b/spans[1]),axis=1)
        grid=(np.c_[points@ex,points@ey]-center)@axes/spans+.5
        in_rect=np.all((grid>=-.05)&(grid<=1.05),axis=1)
        if in_rect.mean()<.85:continue
        bins=np.clip((grid[in_rect]*5).astype(int),0,4)
        coverage=len(np.unique(bins[:,0]+5*bins[:,1]))/25
        if coverage<.6:continue
        cloud_hull=cv2.contourArea(cv2.convexHull(np.c_[points@ex,points@ey].astype(np.float32)))
        # Do not infer a whole face from a tiny patch at one corner.
        measured_extent=cloud_hull/float(np.prod(spans))
        if not .55<=measured_extent<=1.2:continue
        options.append((float(inside.mean())+coverage+rectangularity,center,axes,spans,coverage,measured_extent,quad))
    if not options:raise ValueError('RGB fallback: no depth-supported rectangular top boundary')
    _,center,axes,spans,coverage,extent,quad=max(options,key=lambda item:item[0])
    choose=0 if abs(axes[0,0])>=abs(axes[0,1]) else 1
    axis=ex*axes[0,choose]+ey*axes[1,choose]
    if axis@ex<0:axis=-axis
    surface=center[0]*ex+center[1]*ey+(top_height-d)*n
    ray=surface/surface[2];dist=np.sum((frame.rays[y0:y1,x0:x1]-ray)**2,axis=2)
    row,col=np.unravel_index(np.argmin(dist),dist.shape);pixel=[int(x0+col),int(y0+row)]
    return dict(camera_surface_xyz_mm=surface.tolist(),camera_closing_axis=axis.tolist(),
                camera_support_normal=n.tolist(),height_above_support_mm=top_height,
                observed_opening_mm=float(spans[choose]),observed_footprint_mm=spans.tolist(),pixel=pixel,
                center_is_fitted=True,center_has_depth=bool(valid[pixel[1],pixel[0]]),
                top_points=int(top.sum()),top_grid_coverage=coverage,support_rms_mm=rms,
                boundary_source='RGB quadrilateral intersected with measured depth top plane',
                rgb_top_quad=quad.tolist(),measured_extent_fraction=extent,motion_enabled=False)
