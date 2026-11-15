"""Replay saved detector boxes against geometry; no camera/robot or training."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
from camera import load_frame
from rgbd_projection import depth_to_rgb,boxes_to_depth
from box_inference import infer_boxes

def audit(session):
    rows=[]
    for p in sorted(session.glob('*/result.json')):
        old=json.loads(p.read_text());frame=load_frame(p.parent/'rgbd.npz')
        class Detector:
            image_source='native_rgb';allowed={'artificial_object'}
            last_native_detections=[(d['name'],d['xyxy']) for d in old['native_detections']]
            last_depth_to_rgb=depth_to_rgb(frame)
            def __call__(self,frame):return boxes_to_depth(self.last_native_detections,self.last_depth_to_rgb)
        new=infer_boxes(frame,Detector())
        row=dict(capture=str(p.parent),before=len(old['proposals']),after=len(new['proposals']),
                 rejected=new['rejected'],proposals=new['proposals'])
        rows.append(row);print(p.parent.name,row['before'],'->',row['after'],row['rejected'],flush=True)
    out=ROOT/'depth_diagnostics'/f'{session.name}-replay.json'
    out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('TOTAL',sum(r['before'] for r in rows),'->',sum(r['after'] for r in rows),'REPORT',out)

if __name__=='__main__':audit(Path(sys.argv[1]))
