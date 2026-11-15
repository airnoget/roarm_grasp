"""SDK-only depth stream: no RGB, alignment, ray maps, models, or robot."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
for name in reversed(('.deps','.orbbec-python-deps')):sys.path.insert(0,str(ROOT/name))
import argparse
import json
import time
import cv2
import numpy as np
import pyorbbecsdk as ob


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width',type=int,default=0);p.add_argument('--height',type=int,default=0)
    p.add_argument('--preset',type=Path,help='Viewer exported JSON; SDK loads device settings, not Viewer rendering')
    a=p.parse_args()
    if bool(a.width)!=bool(a.height):p.error('Specify both width and height')
    pipeline=ob.Pipeline()
    backup=None;preset=None
    if a.preset:
        preset=json.loads(a.preset.read_text(encoding='utf-8-sig'))
        info=pipeline.get_device().get_device_info()
        compatible=preset['device_info']['compatible_devices']
        if not any(int(v['pid'],0)==info.get_pid() and int(v['vid'],0)==info.get_vid() for v in compatible):
            raise ValueError('Preset does not match connected camera VID/PID')
        if preset['parameters']['sensor_depth']['disparity_to_depth_mode']!='hardware':
            raise ValueError('This raw diagnostic requires hardware depth conversion')
        profile_cfg=preset['application_config']['sensors']['Depth']['profile']
        if profile_cfg['format']!='Y16' or profile_cfg['fps']!=30:raise ValueError('Expected Y16 30fps preset')
        a.width,a.height=profile_cfg['width'],profile_cfg['height']
        backup=ROOT/'depth_diagnostics'/f'before-preset-{time.time_ns()}.json'
        backup.parent.mkdir(parents=True,exist_ok=True)
        pipeline.get_device().export_settings_as_preset_json_file(str(backup))
        try:
            device=pipeline.get_device()
            device.load_preset_from_json_file(str(a.preset.resolve()))
            search=preset['parameters']['sensor_depth']['disparity_search']
            mode={64:0,128:1,256:2}[search['range_mode']]
            for prop,value in [(ob.OBPropertyID.OB_PROP_DISP_SEARCH_RANGE_MODE_INT,mode),
                               (ob.OBPropertyID.OB_PROP_DISP_SEARCH_OFFSET_INT,search['offset'])]:
                if device.get_int_property(prop)!=value:raise ValueError('Preset disparity readback mismatch')
        except Exception:
            pipeline.get_device().load_preset_from_json_file(str(backup));raise
        print('SDK LOADED PRESET',a.preset,'RESTORE FILE',backup,flush=True)
    profiles=pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    profile=(profiles.get_video_stream_profile(a.width,a.height,ob.OBFormat.Y16,30)
             if a.width else profiles.get_default_video_stream_profile())
    if profile.get_format()!=ob.OBFormat.Y16:raise ValueError('Expected Y16')
    config=ob.Config();config.enable_stream(profile)
    try:pipeline.start(config)
    except Exception:
        if backup:pipeline.get_device().load_preset_from_json_file(str(backup))
        raise
    name='SDK depth ONLY - S save; Q exit';created=False
    try:
        print('DEPTH ONLY',profile,flush=True)
        from orbbec_status import snapshot
        status=snapshot(pipeline.get_device(),ob)
        while True:
            if created and cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
            frames=pipeline.wait_for_frames(1000)
            if frames is None:continue
            frame=frames.get_depth_frame()
            if frame is None:continue
            raw=np.frombuffer(frame.get_data(),np.uint16).copy().reshape(frame.get_height(),frame.get_width())
            scale=float(frame.get_depth_scale());depth=raw.astype(np.float32)*scale
            valid=raw>0
            stats=dict(width=frame.get_width(),height=frame.get_height(),scale_mm=scale,
                       nonzero_fraction=float(valid.mean()),raw_min=int(raw.min()),raw_max=int(raw.max()),
                       depth_mm_percentiles=(np.percentile(depth[valid],[5,50,95]).tolist() if valid.any() else None),
                       rgb_enabled=False,alignment_enabled=False,device_status=status)
            if a.preset:
                import hashlib
                stats['preset']=dict(path=str(a.preset.resolve()),sha256=hashlib.sha256(a.preset.read_bytes()).hexdigest(),
                                     viewer_rendering_applied=False,host_post_filters_applied=False)
            image=np.zeros(raw.shape,np.uint8)
            if valid.any():
                lo,hi=np.percentile(depth[valid],[2,98]);hi=max(hi,lo+1)
                image[valid]=np.clip((depth[valid]-lo)/(hi-lo)*255,0,255).astype(np.uint8)
            view=cv2.applyColorMap(image,cv2.COLORMAP_TURBO);view[~valid]=0
            cv2.putText(view,f'SDK native nonzero: {100*valid.mean():.1f}%',(8,25),0,.7,(255,255,255),2)
            if not created:cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE);created=True
            cv2.imshow(name,view);key=cv2.waitKey(1)&255
            if key in (27,ord('q')):break
            if key==ord('s'):
                folder=ROOT/'depth_diagnostics'/f'raw-{time.time_ns()}';folder.mkdir(parents=True)
                np.savez_compressed(folder/'sensor-raw.npz',raw_u16=raw,scale_mm=scale)
                (folder/'status.json').write_text(json.dumps(stats,indent=2));cv2.imwrite(str(folder/'depth.png'),view)
                print('SAVED',folder,stats['nonzero_fraction'],flush=True)
    finally:
        try:pipeline.stop()
        finally:
            if backup:pipeline.get_device().load_preset_from_json_file(str(backup))
            cv2.destroyAllWindows()


if __name__=='__main__':main()
