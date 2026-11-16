from active_handeye import active_report, fingerprint
"""Viewer device preset + trained YOLO. Left native RGB, right measured depth."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
for name in reversed(('.deps','.orbbec-python-deps','.torchdeps','.semdeps')):sys.path.insert(0,str(ROOT/name))
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import pyorbbecsdk as ob
from trained_model import latest_detector
from depth_health import depth_image
from rgbd_projection import letterbox,to_panel


def load_configuration(path):
    spec=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    depth=spec['parameters']['sensor_depth']
    if depth['disparity_to_depth_mode']!='hardware':raise ValueError('Expected hardware depth conversion')
    active=[f['filter_name'] for f in depth.get('post_processing_filter',[]) if f.get('enable') and f['filter_name']!='DisparityTransform']
    if active:raise ValueError(f'Preset requests unimplemented host filters: {active}')
    # Native geometry is retained; rotate only the displayed images and boxes.
    for sensor in ('sensor_depth','sensor_color'):
        spec['parameters'][sensor]['image_orientation']=dict(mirror=False,flip=False,rotation=0)
    return spec


def display_point(point,shape,rotation):
    x,y=point;h,w=shape
    return (w-1-x,h-1-y) if rotation==180 else (x,y)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',type=Path,default=ROOT/'camera_profiles/gemini335l-viewer-near.json')
    p.add_argument('--rotate',type=int,choices=[0,180],default=180)
    p.add_argument('--port',help='Optional COM6 five-joint feedback for candidate base XYZ; may reset controller')
    p.add_argument('--collect',action='store_true',help='Five-object 200-image far/near collection; no models or arm')
    p.add_argument('--world',action='store_true',help='Use pretrained YOLO-World instead of the trained target detector')
    p.add_argument('--geometry',action='store_true',help='Use box point-cloud geometry instead of GG-CNN; preview only')
    p.add_argument('--step-capture',action='store_true',help='S waits for a motion-ready observation, then saves and exits; never moves')
    p.add_argument('--approach',action='store_true',help='Same-session hover: P plan, G move, X hold; camera stays open')
    p.add_argument('--hover-mm',type=float,default=80.,help='G: TCP height above detected TOP face, mm')
    p.add_argument('--prompt',default='colored cube',help='YOLO-World target description')
    a=p.parse_args();spec=load_configuration(a.preset)
    if not np.isfinite(a.hover_mm) or a.hover_mm<0:p.error('--hover-mm must be finite and nonnegative')
    if a.collect and a.port:p.error('Collection does not open the arm port')
    if a.collect and a.world:p.error('Collection does not use a detector')
    if a.collect and a.geometry:p.error('Collection does not run geometry')
    if a.step_capture and (not a.geometry or not a.port):p.error('--step-capture requires --geometry and --port')
    if a.approach and (not a.geometry or not a.port or a.step_capture):p.error('--approach requires --geometry --port and cannot use --step-capture')
    from target_semantics import TargetDetector
    from ggcnn import Predictor
    from preset_inference import factory_geometry,native_frame,infer
    from geometry import sample
    from camera import save_frame
    detector=None if a.collect else (TargetDetector(str(ROOT/'yolov8s-worldv2.pt'),[a.prompt],'cpu',prompts=[a.prompt],exclude_gripper=True)
                                    if a.world else TargetDetector(str(latest_detector(ROOT)),['artificial_object'],'cpu',exclude_gripper=True))
    predictor=None if a.collect or a.geometry else Predictor(ROOT/'.vendor/ggcnn',ROOT/'.vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell','cpu')
    if a.geometry:
        from box_inference import infer_boxes
        from geometry_tracking import GeometryTracker
        tracker=GeometryTracker()
        infer=infer_boxes
    collection=None
    from exploration import TargetRegistry
    targets=TargetRegistry();selected_id=None
    if a.collect:
        from distance_capture_plan import DistancePlan,banner
        plan=DistancePlan();base=ROOT/'datasets/competition/raw';base.mkdir(parents=True,exist_ok=True)
        pointer=base/'distance-active.json'
        profile_id=hashlib.sha256(a.preset.read_bytes()).hexdigest()
        policy=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))
        if profile_id!=policy['sha256']:raise ValueError('Collection preset differs from the confirmed training policy')
        if pointer.is_file():
            previous=json.loads(pointer.read_text());candidate=Path(previous['session'])
            if candidate.parent.resolve()!=base.resolve():raise ValueError('Invalid collection pointer')
            if (candidate/'progress.json').is_file():
                progress=json.loads((candidate/'progress.json').read_text())
                if progress['total_saved']<200:
                    if previous['preset_sha256']!=profile_id:raise ValueError('Preset changed during collection; finish or archive the prior session explicitly')
                    collection=candidate;plan.restore(progress)
                    actual={key:0 for key in plan.counts}
                    for manifest in collection.glob('*/capture.json'):
                        entry=json.loads(manifest.read_text(encoding='utf-8-sig'))
                        if entry.get('camera_profile',{}).get('preset_sha256')!=profile_id:
                            raise ValueError('Capture preset mismatch in resumed session')
                        actual[entry['collection_stage']]+=1
                    plan.counts=actual
        if collection is None:
            collection=base/f'session-{time.time_ns()}';collection.mkdir()
            (collection/'session.json').write_text(json.dumps(dict(schema=1,collection_plan='five-types-far-near-200',preset_sha256=profile_id,display_rotation=a.rotate,annotation_status='unlabelled'),indent=2))
            (collection/'progress.json').write_text(json.dumps(plan.export(),indent=2))
            pointer.write_text(json.dumps(dict(session=str(collection),preset_sha256=profile_id),indent=2))
        (collection/'camera-preset.json').write_bytes(a.preset.read_bytes())
        print('COLLECTION',collection,'F far / N near, 1-5 object, M mixed, B background, S save, Q quit',flush=True)
    arm=wrist=hand_eye=None;controller=None;last_state=None
    if a.port:
        from wrist_pose import Wrist
        report=json.loads((active_report()).read_text())
        hand_eye=np.asarray(report['wrist_from_depth_mm']);hand_eye=(hand_eye[:3,:3],hand_eye[:3,3])
        wrist=Wrist(ROOT/'.vendor/roarm_ws',report['joint_mapping']['joint_signs'],report['joint_mapping']['joint_offsets_rad'])
        print('COM feedback requested; controller may reset/home. Candidate extrinsics NOT motion-validated.',flush=True)
        print('HAND-EYE',active_report(),'SHA256',fingerprint(),flush=True)
    folder=ROOT/'field_recordings'/f'preset-{time.time_ns()}';folder.mkdir(parents=True)
    applied=folder/'applied-device-preset.json';applied.write_text(json.dumps(spec,indent=2))
    pipe=ob.Pipeline();device=pipe.get_device();info=device.get_device_info()
    if not any(int(v['pid'],0)==info.get_pid() and int(v['vid'],0)==info.get_vid() for v in spec['device_info']['compatible_devices']):
        raise ValueError('Preset is incompatible with connected camera')
    backup=folder/'before-device-preset.json';device.export_settings_as_preset_json_file(str(backup))
    started=False;name=('5 OBJECT DATASET | F far N near | 1-5 M B | S save Q quit' if a.collect
                        else ('YOLO-World: '+a.prompt+' + GG-CNN | LEFT RGB - RIGHT DEPTH | S save Q quit' if a.world
                              else 'YOLO + GG-CNN | LEFT RGB - RIGHT DEPTH | S save Q quit'))
    if a.geometry:name='YOLO + BOX GEOMETRY | LEFT RGB - RIGHT DEPTH | S save Q quit'
    if a.approach:name='STEP MODE | N unload K return B full unload+return H home U retract | P/G/D C/O W record | A auto X hold'
    try:
        device.load_preset_from_json_file(str(applied))
        search=spec['parameters']['sensor_depth']['disparity_search']
        for prop,value in [(ob.OBPropertyID.OB_PROP_DISP_SEARCH_RANGE_MODE_INT,{64:0,128:1,256:2}[search['range_mode']]),
                           (ob.OBPropertyID.OB_PROP_DISP_SEARCH_OFFSET_INT,search['offset'])]:
            if device.get_int_property(prop)!=value:raise ValueError('Preset disparity readback mismatch')
        for prop,value,kind in [(ob.OBPropertyID.OB_PROP_DEPTH_ROTATE_INT,0,'int'),
                                (ob.OBPropertyID.OB_PROP_COLOR_ROTATE_INT,0,'int'),
                                (ob.OBPropertyID.OB_PROP_DEPTH_MIRROR_BOOL,False,'bool'),
                                (ob.OBPropertyID.OB_PROP_DEPTH_FLIP_BOOL,False,'bool'),
                                (ob.OBPropertyID.OB_PROP_COLOR_MIRROR_BOOL,False,'bool'),
                                (ob.OBPropertyID.OB_PROP_COLOR_FLIP_BOOL,False,'bool')]:
            if getattr(device,f'get_{kind}_property')(prop)!=value:raise ValueError('Native image orientation readback mismatch')
        d=spec['application_config']['sensors']['Depth']['profile']
        c=spec['application_config']['sensors']['Color']['profile']
        dp=pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(d['width'],d['height'],ob.OBFormat.Y16,d['fps'])
        cp=pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(c['width'],c['height'],ob.OBFormat.MJPG,c['fps'])
        geometry=factory_geometry(dp,cp)
        cfg=ob.Config();cfg.enable_stream(dp);cfg.enable_stream(cp)
        cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipe.enable_frame_sync();pipe.start(cfg);started=True
        fields=('b','s','e','t','r','v')
        if a.port:
            from arm import Arm
            arm=Arm(a.port);arm.feedback(timeout=12,required_fields=fields)
            if a.approach:
                from live_approach import LiveApproach,TCP_OFFSET
                controller=LiveApproach(wrist,arm,a.hover_mm)
        print('PRESET LOADED',a.preset,'SHA256',hashlib.sha256(a.preset.read_bytes()).hexdigest(),flush=True)
        print('Overrides: enable RGB MJPG; native orientation=0; display rotation=',a.rotate,flush=True)
        print('Device settings loaded; Viewer rendering not copied.',flush=True)
        print(f'P plan; G hover {a.hover_mm:g}mm above TOP; D approach cube centre; C close to recorded angle; O configured open; B taught placement + return; X hold. TCP link5 XYZ {TCP_OFFSET.tolist()}mm approximate; collision geometry NOT checked.' if a.approach else 'No motion commands.',flush=True)
        cv2.namedWindow(name,cv2.WINDOW_NORMAL);cv2.resizeWindow(name,1280,450)
        ready=False;pending_save=None
        while True:
            if ready and cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
            before=arm.feedback(required_fields=fields) if arm else None
            frames=pipe.wait_for_frames(1000)
            if frames is None:continue
            depth,color=frames.get_depth_frame(),frames.get_color_frame()
            if depth is None or color is None:continue
            raw=np.frombuffer(depth.get_data(),np.uint16).copy().reshape(depth.get_height(),depth.get_width())
            mm=raw.astype(np.float32)*depth.get_depth_scale()
            rgb=cv2.imdecode(np.frombuffer(color.get_data(),np.uint8),cv2.IMREAD_COLOR)
            if rgb is None:raise RuntimeError('Cannot decode native MJPG color frame')
            joints=None;pose_error=None
            if arm:
                after=arm.feedback(required_fields=fields)
                last_state=after
                q0,q1=[np.array([s[k] for k in fields[:5]],float) for s in (before,after)]
                delta=(q1-q0+np.pi)%(2*np.pi)-np.pi
                if min(before['v'],after['v'])<700:pose_error='Servo power unavailable'
                elif np.max(np.abs(delta))>.003:pose_error='Arm moving; base XYZ hidden'
                else:joints=q0+delta/2
            frame=native_frame(mm/1000,rgb,geometry)
            if a.collect:
                result=dict(native_detections=[],proposals=[],rejected=[],motion_enabled=False)
            elif a.geometry:
                result=infer(frame,detector,wrist,joints,hand_eye)
                tracker.update(result['proposals'],time.monotonic(),invalid=pose_error is not None)
            else:
                result=infer(frame,detector,predictor,wrist,joints,hand_eye)
            result.update(pose_error=pose_error,robot_joints_rad=None if joints is None else joints.tolist(),
                          handeye_sha256=fingerprint() if arm else None,
                          detector_mode='yolo_world' if a.world else 'trained',prompt=a.prompt if a.world else None,
                          preset_sha256=hashlib.sha256(a.preset.read_bytes()).hexdigest(),display_rotation=a.rotate)
            if a.geometry and joints is not None:
                from observation_pose import compare
                result['observation_pose']=compare(joints)
                observations=[];located=[]
                for proposal in result['proposals']:
                    if proposal.get('candidate_base_xyz_mm') is None:continue
                    index=proposal.get('native_detection_index')
                    if index is None:continue
                    observations.append(dict(name=result['native_detections'][index]['name'],
                                             position_m=[v/1000 for v in proposal['candidate_base_xyz_mm']]))
                    located.append(proposal)
                now=time.monotonic()
                identities=targets.update(observations,now)
                for proposal,identity in zip(located,identities):
                    proposal['target_id']=identity
                    result['native_detections'][proposal['native_detection_index']]['target_id']=identity
                if not controller or not (controller.active or controller.auto_phase):
                    if selected_id not in identities:targets.selected=None
                    selected_id=targets.choose((0,0,0),now)
                result['selected_target_id']=selected_id
                for proposal in located:
                    identity=proposal.get('target_id')
                    if identity is not None:
                        detection=result['native_detections'][proposal['native_detection_index']]
                        detection['distance_m']=float(np.linalg.norm(np.asarray(proposal['candidate_base_xyz_mm'])[:2]))/1000
            if a.geometry and a.port:
                from relative_target import publish_relative_target
                publish_relative_target(ROOT/'exploration_output/relative-target.json',result)
            view=depth_image(mm/1000)
            rgb_display=cv2.rotate(rgb,cv2.ROTATE_180) if a.rotate==180 else rgb
            depth_display=cv2.rotate(view,cv2.ROTATE_180) if a.rotate==180 else view
            left,layout=letterbox(rgb_display,(450,640));right,depth_layout=letterbox(depth_display,(450,640))
            detections=[]
            for detection in result['native_detections']:
                box=np.asarray(detection['xyxy'])
                detections.append(box.tolist())
                points=[display_point(box[:2],rgb.shape[:2],a.rotate),display_point(box[2:],rgb.shape[:2],a.rotate)]
                points=to_panel(np.asarray(points),layout).astype(int)
                lo,hi=np.minimum(*points),np.maximum(*points)
                cv2.rectangle(left,tuple(lo),tuple(hi),(0,255,255),2)
                identity=detection.get('target_id')
                label=detection['name']+(f' #{identity}'+(' SELECTED' if identity==selected_id else '') if identity is not None else ' depth?')
                if detection.get('distance_m') is not None:label+=f' {detection["distance_m"]:.2f}m'
                cv2.putText(left,label,(int(lo[0]),max(20,int(lo[1])-4)),0,.55,(0,255,255),1)
            # These boxes are projected through the factory RGB/depth geometry,
            # not copied between rasters with different resolutions/viewpoints.
            for detection in result.get('detections',[]):
                box=np.asarray(detection['xyxy'])
                points=[display_point(box[:2],raw.shape,a.rotate),display_point(box[2:],raw.shape,a.rotate)]
                points=to_panel(np.asarray(points),depth_layout).astype(int)
                lo,hi=np.minimum(*points),np.maximum(*points)
                cv2.rectangle(right,tuple(lo),tuple(hi),(0,255,255),2)
                cv2.putText(right,'YOLO target',(int(lo[0]),max(20,int(lo[1])-4)),0,.45,(0,255,255),1)
            for proposal in result['proposals']:
                from perception import Candidate,draw_grasp
                u,v=proposal['pixel'];angle=proposal['angle_rad'];width=proposal['width_px']
                x,y=display_point((u,v),raw.shape,a.rotate)
                point=to_panel(np.array([x,y]),depth_layout)
                candidate=Candidate(float(point[0]),float(point[1]),angle+(np.pi if a.rotate==180 else 0),
                                    width*depth_layout['scale'],proposal['score'])
                draw_grasp(right,candidate)
                if a.geometry and proposal.get('tracking'):
                    tr=proposal['tracking'];g=proposal['geometry']
                    status='CONSISTENT' if tr['camera_relative_stable'] else f'CHECK {tr["frames"]}/5'
                    method='RGB+D' if g.get('boundary_source') else 'DEPTH'
                    text=f'T{tr["id"]} {method} W{g["observed_opening_mm"]:.0f} H{g["height_above_support_mm"]:.0f}mm {status}'
                    cv2.putText(right,text,(max(0,min(365,int(point[0])-70)),max(125,int(point[1])-18)),0,.38,(0,255,255),1)
            if result['proposals']:
                proposal=result['proposals'][0]
                xyz=proposal.get('candidate_base_xyz_mm') or proposal.get('camera_xyz_mm')
                label='Candidate BASE' if proposal.get('candidate_base_xyz_mm') is not None else 'CAMERA'
                if xyz:cv2.putText(right,f'{label} XYZ mm: '+', '.join(f'{v:.1f}' for v in xyz),(8,78),0,.5,(0,255,255),1)
            valid=float((raw>0).mean())
            cv2.putText(left,'RGB DATA COLLECTION - no detector' if a.collect else f'RGB | targets: {len(detections)} | gripper ignored: {len(getattr(detector,"last_ignored_gripper",[]))}',(8,25),0,.6,(0,255,0),2)
            cv2.putText(right,f'Measured depth valid: {valid*100:.1f}%',(8,25),0,.6,(255,255,255),2)
            kind='Geometry fits' if a.geometry else 'GG-CNN proposals'
            cv2.putText(right,'Depth recorded; labels still required' if a.collect else f'{kind}: {len(result["proposals"])}',(8,50),0,.55,(0,255,0),2)
            if a.geometry:
                cv2.putText(right,'FITTED surface | '+('LIVE HOVER MODE' if a.approach else 'camera-relative consistency | NO MOTION'),(8,440),0,.4,(0,180,255),1)
                if result.get('observation_pose'):
                    reference=result['observation_pose']
                    cv2.putText(left,f'View ref {reference["reference_name"]}: max joint difference {reference["max_difference_deg"]:.1f} deg',
                                (8,440),0,.42,(0,255,255),1)
            if a.step_capture:
                motion_ready=(joints is not None and pose_error is None and len(result['native_detections'])==1
                              and len(result['proposals'])==1 and result['proposals'][0].get('tracking',{}).get('camera_relative_stable',False))
                caption=('READY: S save + exit' if motion_ready else 'WAIT: S request stable capture')
                if pending_save is not None:caption='Waiting for 5 consistent frames; Q cancels'
                cv2.putText(left,caption,(8,50),0,.5,(0,255,255),2)
            if pose_error:cv2.putText(right,pose_error,(8,102),0,.45,(0,0,255),1)
            elif result['rejected']:cv2.putText(right,result['rejected'][0],(8,102),0,.42,(0,0,255),1)
            canvas=np.hstack((left,right))
            if controller:
                position=controller.observe(last_state)
                from wrist_ik import downward_tilt_deg
                footer=np.zeros((230,canvas.shape[1],3),np.uint8)
                rows=[controller.status,'TCP BASE XYZ mm: '+', '.join(f'{v:.1f}' for v in position),
                      f'Tool-axis tilt from DOWN: {downward_tilt_deg(wrist,[last_state[k] for k in fields[:5]]):.1f} deg | TCP link5 {TCP_OFFSET.tolist()}mm; approximate']
                if controller.plan and controller.plan.get('phase')=='unload':
                    label='RETURN' if controller.plan.get('route_return') else 'UNLOAD'
                    rows.append(f'{label} point {controller.plan["route_step"]}/{len(controller.unload_steps["points"])}: '+controller.plan['route_point'])
                    rows.append('Target joint deg: '+', '.join(f'{v:.1f}' for v in np.degrees(controller.plan['goal_joints_rad']))+(' | gripper open' if controller.plan.get('route_return') else ' | gripper held until final release'))
                elif controller.plan and controller.plan.get('phase') in ('retract','home'):
                    rows.append('RETRACT target TCP BASE XYZ mm: '+', '.join(f'{v:.1f}' for v in controller.plan['hover_base_mm']))
                    rows.append('HOME base 0.44 deg | gripper held' if controller.plan['phase']=='home' else 'RETRACT joint deg: '+', '.join(f'{v:.1f}' for v in np.degrees(controller.plan['goal_joints_rad']))+' | gripper held')
                elif controller.plan:
                    rows+=['LOCKED object BASE XYZ mm: '+', '.join(f'{v:.1f}' for v in controller.plan['surface_base_mm']),
                           'HOVER TCP BASE XYZ mm: '+', '.join(f'{v:.1f}' for v in controller.plan['hover_base_mm'])]
                    rows[-1]+=f' | tilt {controller.plan["downward_tilt_deg"]:.1f}deg'
                    alignment=controller.plan.get('closing_axis_error_deg')
                    if alignment is not None:rows[-1]+=f' | jaw-axis error {alignment:.1f}deg'
                    rows.append(f'Reference-to-target distance: {np.linalg.norm(position-np.array(controller.plan["hover_base_mm"])):.1f} mm')
                    if controller.grasp_clearance is not None:
                        centre=np.array(controller.plan['surface_base_mm'])+[0,0,controller.grasp_clearance]
                        rows.append(f'TCP minus D endpoint Z: {position[2]-centre[2]:+.1f}mm | D extra DOWN {controller.extra_down_mm:g}mm')
                elif result['proposals']:
                    prop=result['proposals'][0]
                    for label,field in [('Object CAMERA','camera_xyz_mm'),('Object BASE','candidate_base_xyz_mm')]:
                        if prop.get(field) is not None:rows.append(label+' XYZ mm: '+', '.join(f'{v:.1f}' for v in prop[field]))
                rows.append(controller.gripper_status+(' | measured '+f'{np.degrees(last_state["g"]):.1f}deg' if 'g' in last_state else ' | NO feedback'))
                if controller.display_left_base is not None:
                    correction=float(controller.manual_offset_base@controller.display_left_base)
                    rows.append(f'TEMP view-left bias {correction:+.0f}mm | J/L adjust 5mm | CLEAR BEFORE TCP CALIBRATION')
                if controller.auto_phase:rows.append('AUTO phase: '+controller.auto_phase+' | X cancels')
                if controller.notice:rows.append('LAST KEY: '+controller.notice)
                for diag in result.get('depth_diagnostics',[])[:max(0,8-len(rows))]:
                    rows.append(f'T{diag["target"]} depth {diag["measured_fraction"]*100:.0f}% ({diag["measured_points"]}pts): {diag["reason"]}')
                if len(rows)>8:footer=np.zeros((len(rows)*28+10,canvas.shape[1],3),np.uint8)
                for line,text in enumerate(rows):cv2.putText(footer,text,(10,25+line*28),0,.55,(0,255,255),1)
                canvas=np.vstack((canvas,footer))
            if a.collect:canvas=np.vstack((banner(plan,canvas.shape[1]),canvas))
            cv2.imshow(name,canvas);key=cv2.waitKey(1)&255;ready=True
            if ord('A')<=key<=ord('Z'):key+=ord('a')-ord('A')
            if key==9 and (not controller or not (controller.active or controller.auto_phase)):
                choices=sorted(p['target_id'] for p in result['proposals'] if p.get('target_id') is not None)
                if choices:
                    selected_id=choices[(choices.index(selected_id)+1)%len(choices)] if selected_id in choices else choices[0]
                    targets.selected=selected_id
                    result['selected_target_id']=selected_id
                    print('SELECTED TARGET',selected_id,flush=True)
            if key in (27,ord('q')):
                if controller:controller.hold(last_state)
                break
            if controller:
                try:
                    if controller.active and controller.plan.get('phase') in ('retract','home') and key not in (255,ord('x'),ord('w'),ord('h'),ord('b'),ord('u')):
                        raise ValueError('Retracting with gripper held; X holds before other manual controls')
                    if controller.rear.phase and key not in (255,ord('x'),ord('h'),ord('b'),ord('u')):
                        raise ValueError('Base rear transfer in progress; X cancels before manual controls')
                    if key in map(ord,'abhknupgdxcorjlw[]'):
                        controller.notice=chr(key).upper()+' received'
                        print('KEY',chr(key).upper(),flush=True)
                    if key==ord('a'):controller.start_auto(result,joints,last_state)
                    elif key==ord('p'):controller.prepare(result,joints)
                    elif key==ord('g'):controller.execute(last_state)
                    elif key==ord('d'):controller.descend(last_state)
                    elif key==ord('x'):controller.hold(last_state)
                    elif key==ord('c'):controller.set_gripper(last_state,True)
                    elif key==ord('o'):controller.set_gripper(last_state,False)
                    elif key==ord(']'):controller.jog_gripper(last_state,True)
                    elif key==ord('['):controller.jog_gripper(last_state,False)
                    elif key==ord('b'):controller.transfer_rear(last_state)
                    elif key==ord('n'):controller.next_unload_point(last_state)
                    elif key==ord('k'):controller.next_return_point(last_state)
                    elif key==ord('u'):controller.retract(last_state)
                    elif key==ord('h'):controller.return_home(last_state)
                    elif key==ord('w'):controller.save_waypoint(last_state)
                    elif key==ord('r'):controller.remember_closed(last_state)
                    elif key==ord('j'):controller.nudge_lateral(last_state,True)
                    elif key==ord('l'):controller.nudge_lateral(last_state,False)
                except ValueError as exc:
                    controller.notice=str(exc);controller.status=str(exc);print('APPROACH:',exc,flush=True)
            if a.step_capture:
                if key==ord('s'):pending_save=time.monotonic()
                if pending_save is not None:
                    if time.monotonic()-pending_save>30:
                        pending_save=None;print('Capture request timed out: no stable target. No motion was requested.',flush=True)
                    elif motion_ready:key=ord('s')
                    else:key=255
            if a.collect and plan.select(key):
                (collection/'progress.json').write_text(json.dumps(plan.export(),indent=2))
                continue
            if a.collect and key==ord('s'):
                if plan.counts[plan.active[1]]>=plan.active[3]:
                    print(plan.hint(),flush=True);continue
                from data_capture import save_capture
                scene=f'{collection.name}-layout-{sum(plan.counts.values()):04d}'
                destination=save_capture(collection,frame,scene,5,plan.active[1],dict(
                    preset_sha256=profile_id,preset_file='camera-preset.json',distance=plan.distance,
                    native_orientation=0,display_rotation=a.rotate,depth_profile=d,color_profile=c,
                    color_enabled_override=True))
                plan.saved();(collection/'progress.json').write_text(json.dumps(plan.export(),indent=2))
                print('SAVED',destination,plan.export(),flush=True);continue
            if key==ord('s'):
                sample=folder/str(time.time_ns());sample.mkdir()
                cv2.imwrite(str(sample/'rgb.png'),rgb);cv2.imwrite(str(sample/'preview.png'),canvas)
                np.savez_compressed(sample/'depth.npz',raw_u16=raw,scale_mm=depth.get_depth_scale())
                save_frame(sample/'rgbd.npz',frame,joints)
                (sample/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
                print('SAVED',sample,flush=True)
                if a.geometry:
                    if not result['proposals']:
                        print('NOT READY FOR APPROACH:', '; '.join(result['rejected']) or 'No target detected / no usable fit. No motion target saved.',flush=True)
                    elif joints is None:
                        print('NOT READY FOR APPROACH: missing stationary joint feedback.',flush=True)
                    else:
                        print('Geometry saved with measured joints. Approach also requires one target, five consistent frames, freshness and live start-pose checks.',flush=True)
                if a.step_capture:
                    print('READY SNAPSHOT SAVED. Closing camera and COM6. Next: python hover_step.py --execute',flush=True)
                    break
    finally:
        try:
            if controller and controller.active and last_state is not None:controller.hold(last_state)
            if started:pipe.stop()
        finally:
            try:device.load_preset_from_json_file(str(backup))
            finally:
                if arm:arm.__exit__(None,None,None)
                cv2.destroyAllWindows()


if __name__=='__main__':main()
