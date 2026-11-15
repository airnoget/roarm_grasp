"""Manual object boxes on recorded RGB; never opens a camera or robot."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.deps'))
import argparse
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rgbd_projection import letterbox, to_panel
from dataset_labels import save_labels


class LabelWindow:
    def __init__(self, session):
        self.folders = sorted(p.parent for p in Path(session).glob('*/capture.json'))
        if not self.folders:
            raise ValueError('No completed captures in this session')
        self.reviewed = {f for f in self.folders if
                         json.loads((f/'capture.json').read_text()).get('annotation_status')=='reviewed'}
        self.index = next((i for i,f in enumerate(self.folders) if f not in self.reviewed), 0)
        self.boxes = []
        self.start = self.drag = None
        self.message = '三种物块都标同一类；一张图里的所有目标都要框。'
        self.dirty = False
        self.font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19) if Path('C:/Windows/Fonts/msyh.ttc').is_file() else ImageFont.load_default()
        self.load()

    def load(self):
        self.folder = self.folders[self.index]
        self.meta = json.loads((self.folder/'capture.json').read_text(encoding='utf-8-sig'))
        self.image = cv2.imdecode(np.fromfile(self.folder/self.meta['annotation_image'],dtype=np.uint8),cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f'Cannot read {self.folder}')
        self.background, self.layout = letterbox(self.image, (640, 1280))
        self.boxes = []
        if (self.folder/'annotation.json').exists():
            annotation = json.loads((self.folder/'annotation.json').read_text())
            if annotation['image'] != self.meta['annotation_image'] or annotation['image_size'] != [self.image.shape[1],self.image.shape[0]]:
                raise ValueError('Existing label/image mismatch')
            self.boxes = annotation['boxes_xyxy']
        self.start = self.drag = None
        self.dirty = False

    def point(self, x, y, clamp=False):
        p = (np.array([x, y-115])-self.layout['offset'])/self.layout['scale']
        h,w=self.image.shape[:2]
        if not clamp and not (0 <= p[0] <= w and 0 <= p[1] <= h): return None
        return np.clip(p,[0,0],[w,h])

    def mouse(self, event, x, y, flags, unused):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start=self.point(x,y);self.drag=self.start
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.drag=self.point(x,y,True)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            end=self.point(x,y,True)
            low=np.minimum(self.start,end);high=np.maximum(self.start,end)
            if np.min(high-low)>=2:
                self.boxes.append([*low.tolist(),*high.tolist()]);self.dirty=True
            self.start=self.drag=None

    def save(self, empty=False):
        if self.start is not None:
            self.message='先松开鼠标完成当前矩形，再保存。';return
        if empty and self.boxes:
            self.message='已有框，不能标为空图；确实没有目标时先按R清空。';return
        try:
            save_labels(self.folder,self.boxes,[self.image.shape[1],self.image.shape[0]],empty)
        except ValueError as e:
            self.message='没有框：请框目标，或按B确认无目标。';return
        self.reviewed.add(self.folder)
        self.message=f'已保存第{self.index+1}张。'
        if self.index+1<len(self.folders):
            self.index+=1;self.load()
        else:
            self.dirty=False;self.meta['annotation_status']='reviewed'
            remaining=len(self.folders)-len(self.reviewed)
            self.message=f'本图已保存，还有{remaining}张未完成。' if remaining else '全部已保存；可按Q退出。'

    def draw(self):
        view=self.background.copy()
        boxes=list(self.boxes)
        if self.start is not None and self.drag is not None:
            boxes.append([*np.minimum(self.start,self.drag),*np.maximum(self.start,self.drag)])
        for i,b in enumerate(boxes):
            a,z=to_panel(np.array(b).reshape(2,2),self.layout).astype(int)
            cv2.rectangle(view,tuple(a),tuple(z),(0,255,0),2)
            cv2.putText(view,str(i+1),tuple(a),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)
        top=Image.new('RGB',(1280,115),(25,30,39));d=ImageDraw.Draw(top)
        stage=self.meta.get('collection_stage','unknown')
        d.text((12,8),f'第{self.index+1}/{len(self.folders)}张 | 已完成{len(self.reviewed)} | 批次 {stage} | 框数 {len(self.boxes)}'+(' | 未保存' if self.dirty else ''),font=self.font,fill='white')
        d.text((12,35),'鼠标拖框，可画多个；S保存并下一张；B确认无目标并下一张；退格撤销；R清空。',font=self.font,fill=(145,220,170))
        d.text((12,62),'A上一张 / D下一张（不保存）；Q退出。标注的是原始RGB，不要只框棋盘或按颜色漏标。',font=self.font,fill=(190,210,240))
        d.text((12,88),self.message,font=self.font,fill=(240,210,120))
        return np.vstack([np.asarray(top)[...,::-1],view])

    def run(self):
        name='Object box labels - recorded RGB only'
        cv2.namedWindow(name,cv2.WINDOW_NORMAL);cv2.setMouseCallback(name,self.mouse)
        ready=False
        try:
            while True:
                if ready:
                    try:
                        if cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
                    except cv2.error:break
                cv2.imshow(name,self.draw());key=cv2.waitKey(20)&255;ready=True
                if key in (27,ord('q')):break
                if key==ord('s'):self.save()
                if key==ord('b'):self.save(True)
                if key in (8,127) and self.boxes:self.boxes.pop();self.dirty=True
                if key==ord('r'):self.boxes=[];self.dirty=True
                if key in (ord('a'),ord('d')):
                    self.index=int(np.clip(self.index+(-1 if key==ord('a') else 1),0,len(self.folders)-1));self.load()
        finally:cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',nargs='?',type=Path);a=p.parse_args()
    if a.session is None:
        candidates=[p for p in (ROOT/'datasets/competition/raw').glob('session-*') if any(p.glob('*/capture.json'))]
        if not candidates:p.error('No recorded dataset. Run python debug.py data first')
        a.session=max(candidates,key=lambda p:p.name)
    print('LABEL SESSION',a.session.resolve(),flush=True)
    LabelWindow(a.session).run()


if __name__=='__main__':main()
