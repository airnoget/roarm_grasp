"""Associate native RGB detections with measured depth pixels; never fill missing depth."""
import cv2
import numpy as np


def supports_native(frame):
    return (getattr(frame, 'raw_bgr', None) is not None and
            all(getattr(frame, k, None) is not None for k in
                ('raw_K', 'raw_dist', 'T_raw_from_depth_mm', 'raw_dist_model')) and
            ('BROWN_CONRADY' in str(frame.raw_dist_model) or str(frame.raw_dist_model).endswith('.NONE')))


def depth_to_rgb(frame):
    if not supports_native(frame):
        raise ValueError('Factory depth-to-native-RGB calibration unavailable')
    depth = np.asarray(frame.depth_m)
    rays = np.asarray(frame.rays)
    if rays.shape != (*depth.shape, 3):
        raise ValueError('Ray map shape differs from depth raster')
    k = np.asarray(frame.raw_K, float)
    t = np.asarray(frame.T_raw_from_depth_mm, float)
    dist = np.asarray(frame.raw_dist, float)
    if (k.shape != (3, 3) or t.shape != (4, 4) or
            not all(np.isfinite(v).all() for v in (k, t, dist))):
        raise ValueError('Invalid factory projection calibration')
    if str(frame.raw_dist_model).endswith('.NONE'):
        dist = np.zeros(8)
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(rays).all(axis=2)
    indices = np.flatnonzero(valid)
    points = (rays.reshape(-1, 3)[indices]*depth.ravel()[indices, None]*1000) @ t[:3, :3].T+t[:3, 3]
    front = points[:, 2] > 0
    indices, points = indices[front], points[front]
    uv = np.full((*depth.shape, 2), np.nan, np.float32)
    if len(points):
        projected = cv2.projectPoints(points.astype(float), np.zeros(3), np.zeros(3), k, dist)[0].reshape(-1, 2)
        h, w = frame.raw_bgr.shape[:2]
        inside = (np.isfinite(projected).all(axis=1) & (projected[:, 0] >= 0) &
                  (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h))
        # RGB/depth cameras have a baseline: discard farther samples occluded in RGB.
        bins = np.floor(projected[inside]).astype(int)
        linear = bins[:, 1]*w+bins[:, 0]
        nearest = np.full(h*w, np.inf)
        np.minimum.at(nearest, linear, points[inside, 2])
        visible = points[inside, 2] <= nearest[linear]+1.
        accepted = np.flatnonzero(inside)[visible]
        uv.reshape(-1, 2)[indices[accepted]] = projected[accepted]
    return uv


def boxes_to_depth(detections, uv, min_pixels=5):
    mapped = []
    valid = np.isfinite(uv).all(axis=2)
    for name, box in detections:
        x0, y0, x1, y1 = box
        v, u = np.nonzero(valid & (uv[..., 0] >= x0) & (uv[..., 0] < x1) &
                          (uv[..., 1] >= y0) & (uv[..., 1] < y1))
        if len(u) >= min_pixels:
            mapped.append((name, (int(u.min()), int(v.min()), int(u.max()+1), int(v.max()+1))))
    return mapped


def rgb_click_to_depth(uv, point, tolerance_px=3.):
    distance = np.sum((uv-np.asarray(point))**2, axis=2)
    if not np.isfinite(distance).any():
        raise ValueError('No measured depth projects into RGB')
    index = int(np.nanargmin(distance))
    if distance.ravel()[index] > tolerance_px**2:
        raise ValueError('No valid depth at clicked RGB point')
    v, u = np.unravel_index(index, distance.shape)
    return int(u), int(v)


def letterbox(image, shape):
    h, w = shape
    ih, iw = image.shape[:2]
    scale = min(w/iw, h/ih)
    rw, rh = max(1, round(iw*scale)), max(1, round(ih*scale))
    ox, oy = (w-rw)//2, (h-rh)//2
    canvas = np.full((h, w, 3), 24, np.uint8)
    canvas[oy:oy+rh, ox:ox+rw] = cv2.resize(image, (rw, rh))
    return canvas, dict(scale=scale, offset=[ox, oy], raw_shape=[ih, iw], space='native_rgb')


def to_panel(points, layout):
    return np.asarray(points)*layout['scale']+layout['offset']
