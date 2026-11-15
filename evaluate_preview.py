"""Replay held-out RGB-D through trained YOLO and GG-CNN, without hardware."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for name in reversed(('.deps','.torchdeps','.semdeps')):
    sys.path.insert(0,str(ROOT/name))
import json
import time
import numpy as np
from camera import load_frame
from ggcnn import Predictor
from preview_live import render
from target_semantics import TargetDetector
from trained_model import latest_detector


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(overlap/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-overlap))


def main():
    pointer=json.loads((ROOT/'datasets/competition/latest-training.json').read_text())
    dataset=Path(pointer['dataset']).parent
    manifest=json.loads((dataset/'manifest.json').read_text())
    detector=TargetDetector(str(latest_detector(ROOT)),['artificial_object'],'cpu')
    predictor=Predictor(ROOT/'.vendor/ggcnn',ROOT/'.vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell','cpu')
    rows=[];started=time.monotonic()
    for record in manifest['splits']['val']:
        folder=Path(record['source_image']).parent
        annotation=json.loads((folder/'annotation.json').read_text())
        ground_truth=annotation['boxes_xyxy']
        frame=load_frame(folder/'rgbd.npz')
        _,result=render(frame,detector,predictor,.25)
        predicted=[row['xyxy'] for row in result['native_detections'] if row['name']=='artificial_object']
        # Greedy one-to-one matching, confidence threshold is detector default .35.
        pairs=sorted([(iou(p,g),i,j) for i,p in enumerate(predicted) for j,g in enumerate(ground_truth)],reverse=True)
        used_p=set();used_g=set()
        for score,i,j in pairs:
            if score>=.5 and i not in used_p and j not in used_g: used_p.add(i);used_g.add(j)
        rows.append(dict(frame=str(folder/'rgbd.npz'),expected_boxes=len(ground_truth),
                         true_positive_boxes=len(used_p),false_positive_boxes=len(predicted)-len(used_p),
                         missed_boxes=len(ground_truth)-len(used_g),**result))
        print(len(rows),'/',len(manifest['splits']['val']),result['status'],flush=True)
    tp=sum(r['true_positive_boxes'] for r in rows);fp=sum(r['false_positive_boxes'] for r in rows);fn=sum(r['missed_boxes'] for r in rows)
    report=dict(scope='offline image grasp proposals only; no IK, collision, grasp success or deployment validation',
                detector_weights=str(latest_detector(ROOT)),frames=len(rows),true_positive_boxes=tp,
                false_positive_boxes=fp,missed_boxes=fn,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                target_frames=sum(r['expected_boxes']>0 for r in rows),
                target_frames_with_proposals=sum(r['expected_boxes']>0 and bool(r['proposals']) for r in rows),
                negative_frames_with_proposals=sum(r['expected_boxes']==0 and bool(r['proposals']) for r in rows),
                seconds=time.monotonic()-started,trials=rows,motion_enabled=False)
    output=ROOT/'candidate_preview/heldout-pipeline.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('REPORT',output,flush=True)


if __name__=='__main__':main()
