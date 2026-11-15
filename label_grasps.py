"""Offline three-click left/right grasp labeling; no camera or serial connection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import argparse
from collections import defaultdict,deque
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from camera import load_frame
from dataset_labels import atomic_text
from grasp_labels import make_grasp,save_grasps,depth_check
from rgbd_projection import letterbox,to_panel


def select_frames(session,limit):
    groups=defaultdict(deque)
    for f in sorted(Path(session).glob('*/capture.json')):
        m=json.loads(f.read_text(encoding='utf-8-sig'))
        if m.get('annotation_status')=='reviewed' and m.get('contains_target'):
            groups[m['collection_stage']].append(f.parent.name)
    chosen=[]
    while groups and len(chosen)<limit:
        for key in list(groups):
            chosen.append(groups[key].popleft())
            if not groups[key]:del groups[key]
            if len(chosen)==limit:break
    return chosen


class GraspWindow:
    name='LEFT-RIGHT GRASP LABELS | 3 clicks | S save K skip Q quit'
    def __init__(self,session,limit=50):
        self.session=Path(session);queue=self.session/'grasp-label-queue.json'
        if queue.exists():ids=json.loads(queue.read_text())['captures']
        else:
            ids=select_frames(session,limit)
            if not ids:raise ValueError('No reviewed positive RGB-D captures')
            atomic_text(queue,json.dumps(dict(captures=ids,limit=limit),indent=2))
        if any(Path(i).name!=i for i in ids):raise ValueError('Invalid capture path')
        self.folders=[self.session/i for i in ids]
        self.reviewed={f for f in self.folders if (f/'grasp_annotation.json').exists()}
        self.index=next((i for i,f in enumerate(self.folders) if f not in self.reviewed),0)
        self.font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
        self.message='先标一对左右夹持面；方向随物块旋转，不强制图像水平。'
        self.load()

    def load(self):
        self.folder=self.folders[self.index];self.frame=load_frame(self.folder/'rgbd.npz')
        if self.frame.bgr is None:raise ValueError('No depth-registered RGB')
        self.grasps=[];saved=self.folder/'grasp_annotation.json'
        if saved.exists():self.grasps=json.loads(saved.read_text())['grasps']
        self.pending=[];self.dirty=False
        rgb=cv2.rotate(self.frame.bgr,cv2.ROTATE_180)
        depth=self.frame.depth_m;valid=np.isfinite(depth)&(depth>0)&(depth<10)
        values=depth[valid];lo,hi=np.percentile(values,[5,95]) if len(values) else (0,1)
        v=np.uint8(np.clip((np.where(valid,depth,lo)-lo)/max(hi-lo,.001),0,1)*255)
        view=cv2.applyColorMap(v,cv2.COLORMAP_TURBO);view[~valid]=0
        self.left,self.layout=letterbox(rgb,(500,640))
        self.right,_=letterbox(cv2.rotate(view,cv2.ROTATE_180),(500,640))

    def point(self,x,y):
        if x>=640:return None
        p=(np.array([x,y-150])-self.layout['offset'])/self.layout['scale']
        h,w=self.frame.depth_m.shape
        if not (0<=p[0]<w and 0<=p[1]<h):return None
        return np.array([w-1,h-1])-p

    def panel(self,points):
        h,w=self.frame.depth_m.shape
        return np.rint(to_panel(np.array([w-1,h-1])-np.asarray(points),self.layout)).astype(int)

    def mouse(self,event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        p=self.point(x,y)
        if p is None:return
        self.pending.append(p);self.dirty=True
        if len(self.pending)==3:
            try:
                g=make_grasp(self.pending,self.frame.depth_m.shape);self.grasps.append(g)
                quality=depth_check(self.frame,g)
                self.message=f'已加矩形；深度可用于当前预测器：{quality["usable_for_current_predictor"]}（不代表实机可抓）'
            except ValueError as e:self.message=str(e)
            self.pending=[]

    def draw(self):
        left=self.left.copy();right=self.right.copy()
        for g in self.grasps:
            points=self.panel(g['corners_xy']);contacts=self.panel(g['contacts_xy'])
            for img in (left,right):
                cv2.polylines(img,[points],True,(0,255,0),2)
                # Opposing finger pads, at the two ends of the closing axis.
                cv2.line(img,tuple(points[0]),tuple(points[3]),(0,0,255),4)
                cv2.line(img,tuple(points[1]),tuple(points[2]),(0,0,255),4)
                cv2.line(img,tuple(contacts[0]),tuple(contacts[1]),(255,255,255),1)
        for p in self.panel(self.pending) if self.pending else []:cv2.circle(left,tuple(p),5,(0,255,255),-1)
        if len(self.pending)==2:
            pts=self.panel(self.pending);cv2.line(left,tuple(pts[0]),tuple(pts[1]),(0,255,255),2)
        header=Image.new('RGB',(1280,150),(25,30,39));d=ImageDraw.Draw(header)
        lines=[f'左右夹标注 {self.index+1}/{len(self.folders)} | 已处理{len(self.reviewed)} | 本图{len(self.grasps)}个'+(' | 未保存' if self.dirty else ''),
               '在左图依次点击：①左夹指接触点 ②对面右夹指接触点 ③偏离连线，确定接触带宽度。',
               '红线=两根夹指；白线=闭合方向；绿框=抓取矩形。只标左右夹，跟随物块方向。',
               'S保存下一张；K跳过差深度/不可判断图；退格撤销；R清空；A/D翻页（先保存）；Q退出。',
               '左：深度坐标对齐RGB（空洞处黑色） | 右：实测深度；均旋转180°。不改原始数据。',self.message]
        for i,line in enumerate(lines):d.text((8,3+24*i),line,font=self.font,fill=(235,235,235))
        return np.vstack((np.asarray(header)[...,::-1],np.hstack((left,right))))

    def save(self,skip=False):
        if self.pending:self.message='请完成3次点击，或用退格撤销当前点。';return
        try:save_grasps(self.folder,self.frame,self.grasps,skipped=skip)
        except ValueError as e:self.message=str(e);return
        self.reviewed.add(self.folder);self.dirty=False
        if self.index+1<len(self.folders):self.index+=1;self.load()
        self.message='已保存。全部处理完成，可按Q退出。' if len(self.reviewed)==len(self.folders) else '已保存；继续下一张。'

    def run(self):
        cv2.namedWindow(self.name,cv2.WINDOW_NORMAL);cv2.setMouseCallback(self.name,self.mouse)
        ready=False
        try:
            while True:
                if ready and cv2.getWindowProperty(self.name,cv2.WND_PROP_VISIBLE)<1:break
                cv2.imshow(self.name,self.draw());key=cv2.waitKey(30)&255;ready=True
                if key in (27,ord('q')):break
                if key==ord('s'):self.save()
                if key==ord('k'):self.save(True)
                if key in (8,127):
                    if self.pending:self.pending.pop()
                    elif self.grasps:self.grasps.pop()
                    self.dirty=True
                if key==ord('r'):self.pending=[];self.grasps=[];self.dirty=True
                if key in (ord('a'),ord('d')):
                    if self.dirty:self.message='当前修改未保存，请先S保存，或R清空后K跳过。';continue
                    self.index=int(np.clip(self.index+(-1 if key==ord('a') else 1),0,len(self.folders)-1));self.load()
        finally:cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path);p.add_argument('--limit',type=int,default=50)
    a=p.parse_args()
    if a.limit<1:p.error('--limit must be positive')
    if a.session is None:
        a.session=Path(json.loads((ROOT/'datasets/competition/raw/distance-active.json').read_text())['session'])
    print('GRASP LABEL SESSION',a.session,flush=True)
    GraspWindow(a.session,a.limit).run()

if __name__=='__main__':main()
