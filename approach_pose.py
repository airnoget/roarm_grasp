"""Convert a fitted surface to explicit TCP and wrist pre-approach poses.

This pure geometry stage has no robot connection and does not establish IK,
collision clearance or physical TCP calibration.
"""
import numpy as np


def rigid(transform):
    t=np.asarray(transform,float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('Expected finite homogeneous transform')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1):
        raise ValueError('Expected rigid rotation')
    return t


def preapproach(geometry,base_from_camera,link5_from_tcp,base_from_tcp_rotation,clearance_mm=80):
    camera=rigid(base_from_camera);offset=rigid(link5_from_tcp)
    rotation=np.asarray(base_from_tcp_rotation,float)
    desired=np.eye(4);desired[:3,:3]=rotation;rigid(desired)
    if not np.isfinite(clearance_mm) or not 30<=clearance_mm<=200:
        raise ValueError('Pilot pre-approach clearance must be 30..200 mm')
    point=np.asarray(geometry['camera_surface_xyz_mm'],float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Invalid fitted surface')
    surface=camera[:3,:3]@point+camera[:3,3]
    # Positive base Z, not image Y or optical Z. This is a hover reference,
    # not a contact grasp or a descent command.
    desired[:3,3]=surface+[0,0,clearance_mm]
    wrist=desired@np.linalg.inv(offset)
    return dict(surface_base_mm=surface.tolist(),tcp_preapproach_base=desired.tolist(),
                link5_preapproach_base=wrist.tolist(),clearance_base_z_mm=float(clearance_mm),
                ik_checked=False,collision_checked=False,tcp_physically_validated=False,
                motion_enabled=False,scope='Geometry-only pre-approach pose; not an executable trajectory')
