"""Convert SDK factory calibration to the project's native RGB-D convention."""
import numpy as np


def factory_metadata(depth_profile,color_profile):
    intr=color_profile.as_video_stream_profile().get_intrinsics()
    model=str(intr.model).split('.')[-1].upper()
    # Inverse and modified Brown models must not be passed to OpenCV as
    # ordinary forward Brown-Conrady coefficients.
    if model not in ('NONE','BROWN_CONRADY'):
        raise ValueError(f'Unsupported RealSense RGB projection distortion: {intr.model}')
    k=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],float)
    dist=np.zeros(5) if model=='NONE' else np.asarray(intr.coeffs,float)
    ext=depth_profile.get_extrinsics_to(color_profile)
    transform=np.eye(4)
    # rs2_extrinsics.rotation is column-major; translation is in metres.
    transform[:3,:3]=np.asarray(ext.rotation,float).reshape(3,3,order='F')
    transform[:3,3]=np.asarray(ext.translation,float)*1000
    r=transform[:3,:3]
    if (not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or dist.shape!=(5,)
            or not np.isfinite(dist).all() or not np.isfinite(transform).all()
            or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1)):
        raise ValueError('Invalid RealSense factory calibration')
    return k,dist,transform,'RealSense.'+model
