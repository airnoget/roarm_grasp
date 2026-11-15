from contextlib import contextmanager
from dataclasses import dataclass, replace
import time

import numpy as np


@dataclass
class Frame:
    depth_m: np.ndarray
    rays: np.ndarray  # H x W x 3, optical axes right/down/forward
    bgr: np.ndarray | None = None  # registered to depth when present
    raw_bgr: np.ndarray | None = None  # original color raster, separate from depth-aligned BGR
    raw_K: np.ndarray | None = None
    raw_dist: np.ndarray | None = None
    T_raw_from_depth_mm: np.ndarray | None = None
    raw_dist_model: str | None = None


def median_depth_three(frames):
    """Fuse three stationary observations, requiring two valid depth returns."""
    first, middle, last = frames
    if not (first.depth_m.shape == middle.depth_m.shape == last.depth_m.shape
            and np.array_equal(first.rays, middle.rays) and np.array_equal(first.rays, last.rays)):
        raise ValueError("Depth fusion requires matching rasters and calibrated rays")
    # A fused depth raster still uses the middle observation's RGB and factory
    # calibration. Mixing camera profiles/extrinsics would corrupt projection.
    for name in ('raw_K','raw_dist','T_raw_from_depth_mm','raw_dist_model'):
        reference = getattr(middle,name)
        for frame in (first,last):
            value = getattr(frame,name)
            if ((reference is None) != (value is None) or
                    (reference is not None and not np.array_equal(reference,value))):
                raise ValueError(f'Depth fusion requires matching camera calibration: {name}')
    stack = np.stack([frame.depth_m for frame in frames])
    valid = np.isfinite(stack) & (stack > 0)
    ordered = np.sort(np.where(valid, stack, np.inf), axis=0)
    depth = np.zeros(first.depth_m.shape, np.float32)
    count = valid.sum(axis=0)
    two = count == 2
    depth[two] = (ordered[0, two] + ordered[1, two]) * .5
    depth[count == 3] = ordered[1, count == 3]
    return replace(middle, depth_m=depth)


def pinhole_rays(shape, fx, fy, cx, cy, distortion=None):
    v, u = np.indices(shape, dtype=np.float32)
    if distortion is None:
        xy = np.stack(((u - cx) / fx, (v - cy) / fy), axis=-1)
    else:
        import cv2
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        xy = cv2.undistortPoints(
            np.stack((u, v), axis=-1).reshape(-1, 1, 2), matrix, distortion
        ).reshape(*shape, 2)
    return np.concatenate((xy, np.ones((*shape, 1), dtype=np.float32)), axis=-1)


@contextmanager
def real_sense(with_color=False, diagnostics=None):
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    config = rs.config()
    # Pin an explicitly supported synchronized profile. Leaving the resolution
    # implicit can make librealsense fail to resolve depth+RGB requests on the
    # restored D435 firmware (notably the Windows 2.59.1 runtime).
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if with_color:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.depth) if with_color else None
    profile = pipe.start(config)
    try:
        intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        metadata=(None,None,None,None)
        if with_color:
            from realsense_geometry import factory_metadata
            metadata=factory_metadata(profile.get_stream(rs.stream.depth),profile.get_stream(rs.stream.color))
        if diagnostics is not None:
            diagnostics.update(depth_scale_m=float(scale),depth_profile=dict(width=intr.width,height=intr.height,fps=30),
                               native_geometry=True,depth_to_rgb_factory_calibration=bool(with_color))
            device=profile.get_device()
            for field in ('name','serial_number','firmware_version','usb_type_descriptor'):
                try:diagnostics[field]=device.get_info(getattr(rs.camera_info,field))
                except RuntimeError as exc:diagnostics[field]=dict(error=str(exc))
        # Use SDK deprojection for the active lens model.
        rays = np.array([
            rs.rs2_deproject_pixel_to_point(intr, [u, v], 1.0)
            for v in range(intr.height) for u in range(intr.width)
        ], dtype=np.float32).reshape(intr.height, intr.width, 3)

        def read():
            frames = pipe.wait_for_frames(3000)
            original = frames.get_color_frame() if with_color else None
            raw_bgr = np.asanyarray(original.get_data()).copy() if original else None
            frames = align.process(frames) if align else frames
            depth = frames.get_depth_frame()
            if not depth:
                raise RuntimeError("RealSense depth frame missing")
            meters = np.asarray(depth.get_data(), dtype=np.float32) * scale
            if meters.shape != rays.shape[:2]:
                raise RuntimeError("RealSense depth raster differs from the calibrated stream")
            bgr = None
            if with_color:
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError("RealSense registered color frame missing")
                bgr = np.asarray(color.get_data()).copy()
                if bgr.dtype != np.uint8 or bgr.shape != (*meters.shape, 3):
                    raise RuntimeError("RealSense did not produce BGR in the native depth raster")
            return Frame(meters, rays, bgr, raw_bgr,*metadata)

        yield read
    finally:
        pipe.stop()


@contextmanager
def orbbec(with_color=False, diagnostics=None, near=False):
    import pyorbbecsdk as ob

    pipe = ob.Pipeline()
    profiles = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    profile = (profiles.get_video_stream_profile(1280,800,ob.OBFormat.Y16,30) if near
               else profiles.get_default_video_stream_profile())
    if profile.get_format() != ob.OBFormat.Y16:
        raise ValueError("Orbbec default depth profile must be Y16")
    intr = profile.get_intrinsic()
    dist = profile.get_distortion()
    # Brown-Conrady/rational coefficients used by supported Orbbec depth profiles.
    coeffs = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3,
                       dist.k4, dist.k5, dist.k6])
    rays = pinhole_rays((intr.height, intr.width), intr.fx, intr.fy,
                       intr.cx, intr.cy, coeffs)
    config = ob.Config()
    config.enable_stream(profile)
    align = None
    raw_K = raw_dist = raw_extrinsic = raw_dist_model = None
    if with_color:
        colors = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = colors.get_video_stream_profile(0, 0, ob.OBFormat.RGB, 0)
        ci,cd = color_profile.get_intrinsic(),color_profile.get_distortion()
        raw_K = np.array([[ci.fx,0,ci.cx],[0,ci.fy,ci.cy],[0,0,1.]],float)
        raw_dist = np.array([cd.k1,cd.k2,cd.p1,cd.p2,cd.k3,cd.k4,cd.k5,cd.k6],float)
        raw_dist_model = str(cd.model)
        extrinsic = profile.get_extrinsic_to(color_profile)
        raw_extrinsic = np.eye(4)
        raw_extrinsic[:3,:3] = np.asarray(extrinsic.rot).reshape(3,3)
        raw_extrinsic[:3,3] = np.asarray(extrinsic.transform)
        config.enable_stream(color_profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipe.enable_frame_sync()
        align = ob.AlignFilter(align_to_stream=ob.OBStreamType.DEPTH_STREAM)
    restore=None
    if near:
        from orbbec_near import configure
        restore=configure(pipe.get_device(),ob)
    try:
        pipe.start(config)
    except Exception:
        if restore:restore()
        raise
    try:
        if diagnostics is not None:
            from orbbec_status import snapshot
            diagnostics.update(snapshot(pipe.get_device(),ob))
            diagnostics['settings_modified']=bool(near)
            diagnostics['near_mode_requested']=bool(near)
            diagnostics['active_depth_profile']=dict(width=intr.width,height=intr.height,
                                                     format=str(profile.get_format()),fps=profile.get_fps())
        def read():
            for _ in range(5):
                frames = pipe.wait_for_frames(1000)
                depth = frames.get_depth_frame() if frames is not None else None
                if not depth:
                    continue
                raw = np.frombuffer(depth.get_data(), dtype=np.uint16)
                if raw.size != intr.height * intr.width:
                    raise RuntimeError("Orbbec depth raster differs from the calibrated stream")
                meters = raw.reshape(intr.height, intr.width).astype(np.float32)
                meters *= depth.get_depth_scale() / 1000
                bgr = None
                raw_bgr = None
                if align is not None:
                    original = frames.get_color_frame()
                    if original and original.get_format() == ob.OBFormat.RGB:
                        raw_bgr = np.frombuffer(original.get_data(), dtype=np.uint8).reshape(
                            original.get_height(), original.get_width(), 3)[..., ::-1].copy()
                    registered = align.process(frames)
                    color = registered.get_color_frame() if registered else None
                    if not color:
                        continue
                    if (color.get_format() != ob.OBFormat.RGB
                            or (color.get_height(), color.get_width()) != meters.shape):
                        raise RuntimeError("Orbbec did not produce RGB in the native depth raster")
                    rgb = np.frombuffer(color.get_data(), dtype=np.uint8).reshape(*meters.shape, 3)
                    bgr = rgb[..., ::-1].copy()
                return Frame(meters, rays, bgr, raw_bgr, raw_K, raw_dist, raw_extrinsic, raw_dist_model)
            raise RuntimeError("Orbbec depth or registered color frame missing")

        yield read
    finally:
        try:pipe.stop()
        finally:
            if restore:restore()


def save_frame(path, frame, joints=None):
    fields = dict(depth_m=frame.depth_m, rays=frame.rays, capture_time_ns=time.time_ns())
    if frame.bgr is not None:
        fields["bgr"] = frame.bgr
    if frame.raw_bgr is not None:
        fields["raw_bgr"] = frame.raw_bgr
    for name in ('raw_K','raw_dist','T_raw_from_depth_mm','raw_dist_model'):
        if getattr(frame,name) is not None:
            fields[name] = getattr(frame,name)
    if joints is not None:
        fields["joints_rad"] = joints
    np.savez_compressed(path, **fields)


def load_frame(path):
    with np.load(path, allow_pickle=False) as data:
        depth, rays = data["depth_m"], data["rays"]
        bgr = data["bgr"] if "bgr" in data else None
        raw_bgr = data["raw_bgr"] if "raw_bgr" in data else None
        metadata = [data[name] if name in data else None for name in
                    ('raw_K','raw_dist','T_raw_from_depth_mm','raw_dist_model')]
    if depth.ndim != 2 or rays.shape != (*depth.shape, 3) or not np.isfinite(rays).all():
        raise ValueError("Invalid depth frame or ray map")
    if bgr is not None and (bgr.dtype != np.uint8 or bgr.shape != (*depth.shape, 3)):
        raise ValueError("Invalid registered BGR raster")
    if raw_bgr is not None and (raw_bgr.dtype != np.uint8 or raw_bgr.ndim != 3 or raw_bgr.shape[2] != 3):
        raise ValueError("Invalid original color raster")
    if metadata[-1] is not None: metadata[-1] = str(metadata[-1].item())
    return Frame(depth, rays, bgr, raw_bgr, *metadata)



