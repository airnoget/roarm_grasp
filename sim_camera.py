"""Deterministic depth observations from the MuJoCo scene, in optical coordinates."""
import mujoco
import cv2
import numpy as np

from camera import Frame, pinhole_rays
from sim_model import WRIST_CAMERA_NAME


def overhead_frame(model, data, eye=(0.28, 0, 0.62)):
    rotation = np.diag([1., -1., -1.])
    rays = pinhole_rays((160, 160), 235, 235, 79.5, 79.5)
    return raycast(model, data, eye, rotation, rays, [1, 0, 0, 0, 0, 0])


def rendered_wrist_rgb(model, data, rays):
    """Render illumination and remap the RGB view onto supplied optical rays."""
    if not np.isfinite(rays).all() or np.any(rays[...,2] <= 0):
        raise ValueError('RGB projection requires finite forward optical rays')
    xy = rays[...,:2] / rays[...,2:]
    resolution = 640
    extent = float(np.max(np.abs(xy))) * (1 + 4/resolution)
    if not 0 < extent < 100:
        raise ValueError('Unsupported optical field for RGB rendering')
    cam = model.camera(WRIST_CAMERA_NAME).id
    old_fov = float(model.cam_fovy[cam])
    old_width,old_height = model.vis.global_.offwidth,model.vis.global_.offheight
    try:
        model.cam_fovy[cam] = np.rad2deg(2*np.arctan(extent))
        model.vis.global_.offwidth = max(resolution,old_width)
        model.vis.global_.offheight = max(resolution,old_height)
        options = mujoco.MjvOption()
        with mujoco.Renderer(model,height=resolution,width=resolution) as renderer:
            renderer.update_scene(data,camera=WRIST_CAMERA_NAME,scene_option=options)
            rgb=renderer.render().copy()
        focal=resolution/(2*extent)
        uv=xy*focal+(resolution-1)/2
        return cv2.remap(rgb,uv[...,0].astype(np.float32),uv[...,1].astype(np.float32),
                         cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[...,::-1].copy()
    finally:
        model.cam_fovy[cam]=old_fov
        model.vis.global_.offwidth,model.vis.global_.offheight=old_width,old_height


def wrist_frame(model, data, with_color=False, rays=None, rgb_renderer='flat'):
    cam = model.camera(WRIST_CAMERA_NAME).id
    rotation = data.cam_xmat[cam].reshape(3, 3) @ np.diag([1., -1., -1.])
    if rays is None:
        size = 160
        center = (size - 1) / 2
        focal = size / (2 * np.tan(np.deg2rad(model.cam_fovy[cam]) / 2))
        rays = pinhole_rays((size, size), focal, focal, center, center)
    result = raycast(model, data, data.cam_xpos[cam], rotation, rays, [1, 1, 0, 0, 0, 0],
                     with_color and rgb_renderer=='flat')
    if with_color and rgb_renderer=='mujoco':
        result[0].bgr=rendered_wrist_rgb(model,data,rays)
    elif with_color and rgb_renderer!='flat':
        raise ValueError('Unknown RGB renderer')
    return result


def raycast(model, data, eye, rotation, rays, groups, with_color=False):
    world_rays = (rays @ rotation.T).astype(np.float64)
    depth = np.zeros(rays.shape[:2], np.float32)
    bgr = np.zeros((*depth.shape, 3), np.uint8) if with_color else None
    if with_color:
        colors = np.rint(np.clip(model.geom_rgba[:, 2::-1], 0, 1) * 255).astype(np.uint8)
        pixels = bgr.reshape(-1, 3)
    group = np.array(groups, dtype=np.uint8)
    geom_id = np.empty(1, np.int32)
    eye = np.asarray(eye, dtype=np.float64)
    for i, ray in enumerate(world_rays.reshape(-1, 3)):
        distance = mujoco.mj_ray(model, data, eye, ray, group, True, -1, geom_id)
        if distance > 0:
            depth.flat[i] = distance
            if with_color:
                pixels[i] = colors[geom_id[0]]
    return Frame(depth, rays, bgr), rotation, eye * 1000

