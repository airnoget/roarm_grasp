"""Offline predictions on frames without manual grasp labels; no hardware."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
for n in reversed(('.deps','.torchdeps','.semdeps')):sys.path.insert(0,str(ROOT/n))
import argparse
import json
import time
import cv2
import numpy as np
from camera import load_frame
from trained_model import latest_detector
from target_semantics import TargetDetector
from ggcnn import Predictor
from preset_inference import infer
from rgbd_projection import letterbox,to_panel
from perception import Candidate,draw_grasp


def panel(frame,result):
    rgb=frame.raw_bgr
    depth=frame.depth_m;valid=np.isfinite(depth)&(depth>0)&(depth<10)
    values=depth[valid];low,high=np.percentile(values,[5,95]) if len(values) else (0,1)
    image=cv2.applyColorMap(np.uint8(np.clip((np.where(valid,depth,low)-low)/max(high-low,.001),0,1)*255),cv2.COLORMAP_TURBO)
    image[~valid]=0
    left,ll=letterbox(cv2.rotate(rgb,cv2.ROTATE_180),(450,640))
    right,dl=letterbox(cv2.rotate(image,cv2.ROTATE_180),(450,640))
    for target,shape,layout,entries in [(left,rgb.shape[:2],ll,result['native_detections']),
                                         (right,depth.shape,dl,result['detections'])]:
        h,w=shape
        for entry in entries:
            pts=to_panel(np.array([w-1,h-1])-np.array(entry['xyxy']).reshape(2,2),layout).astype(int)
            a,b=np.minimum(*pts),np.maximum(*pts)
            cv2.rectangle(target,tuple(a),tuple(b),(0,255,255),2)
    h,w=depth.shape
    for g in result['proposals']:
        point=to_panel(np.array([w-1,h-1])-g['pixel'],dl)
        draw_grasp(right,Candidate(float(point[0]),float(point[1]),g['angle_rad']+np.pi,
                                  g['width_px']*dl['scale'],g['score']))
    cv2.putText(left,'Trained YOLO | no manual grasps on this image',(8,23),0,.5,(0,255,255),1)
    title='Box geometry fits' if result.get('backend')=='box_geometry' else 'Cornell GG-CNN proposals'
    cv2.putText(right,f'{title}: {len(result["proposals"])}',(8,23),0,.5,(0,255,255),1)
    if result['rejected']:cv2.putText(right,result['rejected'][0],(8,45),0,.4,(0,0,255),1)
    return np.hstack((left,right))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count',type=int,default=12);p.add_argument('--view',action='store_true')
    p.add_argument('--finetuned',action='store_true',help='Use the latest local pilot weights explicitly')
    p.add_argument('--geometry',action='store_true',help='Use geometric box fitting, no GG-CNN')
    a=p.parse_args()
    if a.geometry and a.finetuned:p.error('Select geometry or fine-tuned GG-CNN, not both')
    if a.count<1:p.error('--count must be positive')
    session=Path(json.loads((ROOT/'datasets/competition/raw/distance-active.json').read_text())['session'])
    files=[]
    for meta in sorted(session.glob('*/capture.json')):
        if (meta.parent/'grasp_annotation.json').exists():continue
        if json.loads(meta.read_text()).get('contains_target'):files.append(meta.parent)
    if not files:raise ValueError('No positive frames without grasp annotations')
    ids=np.linspace(0,len(files)-1,min(a.count,len(files)),dtype=int)
    detector=TargetDetector(str(latest_detector(ROOT)),['artificial_object'],'cpu')
    weights=ROOT/'.vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
    if a.finetuned:
        import hashlib
        record=json.loads((ROOT/'runs/ggcnn/latest-finetune.json').read_text())
        weights=Path(record['best_weights'])
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=record['best_sha256']:raise ValueError('Pilot checkpoint changed')
    predictor=None if a.geometry else Predictor(ROOT/'.vendor/ggcnn',weights,'cpu')
    run_inference=infer
    if a.geometry:
        from box_inference import infer_boxes
        run_inference=infer_boxes
    out=ROOT/'candidate_preview'/f'unlabelled-{time.time_ns()}';out.mkdir(parents=True)
    records=[]
    for i in ids:
        folder=files[i];frame=load_frame(folder/'rgbd.npz')
        result=run_inference(frame,detector) if a.geometry else run_inference(frame,detector,predictor)
        result.update(source=str(folder),ggcnn_weights=None if a.geometry else str(weights),
                      grasp_annotation_present=False,evaluation_scope='qualitative preview; no grasp ground truth or success rate')
        picture=panel(frame,result);destination=out/f'{folder.name}.jpg'
        if a.finetuned:
            cv2.rectangle(picture,(640,0),(1279,35),(0,0,0),-1)
            cv2.putText(picture,f'Local fine-tune PILOT | proposals: {len(result["proposals"])}',(648,23),0,.5,(0,255,255),1)
        if not cv2.imwrite(str(destination),picture):raise OSError(destination)
        (out/f'{folder.name}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        records.append(dict(source=str(folder),preview=str(destination),targets=len(result['native_detections']),proposals=len(result['proposals'])))
        print(folder.name,'targets',records[-1]['targets'],'grasps',records[-1]['proposals'],flush=True)
        if a.view:
            cv2.imshow('Offline RGB | DEPTH - Cornell pretrained | any key next Q quit',picture)
            if cv2.waitKey(0)&255 in (27,ord('q')):break
    (out/'report.json').write_text(json.dumps(records,indent=2))
    print('OUTPUT',out,flush=True)
    if a.view:cv2.destroyAllWindows()

if __name__=='__main__':main()
