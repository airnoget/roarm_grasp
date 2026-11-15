"""Download/verify the colored-cube model on a saved RGB image; no camera or arm."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
for name in reversed(('.deps','.torchdeps','.semdeps')):
    sys.path.insert(0,str(ROOT/name))
import json
import cv2
from ultralytics import YOLOWorld

def main():
    model=YOLOWorld(str(ROOT/'yolov8s-worldv2.pt'))
    model.set_classes(['colored cube'])
    files=sorted((ROOT/'datasets/competition/raw').glob('session-*/*/original_rgb.png'))
    if not files:raise RuntimeError('No saved RGB image available for verification')
    path=files[-1]
    result=model.predict(cv2.imread(str(path)),device='cpu',conf=.35,verbose=False)[0]
    folder=ROOT/'candidate_preview/world';folder.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(folder/'offline.jpg'),result.plot())
    report=dict(image=str(path),prompt='colored cube',detections=result.summary(),motion_enabled=False)
    (folder/'offline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('WORLD CHECK PASSED',path,'detections',len(result.boxes),flush=True)

if __name__=='__main__':main()
