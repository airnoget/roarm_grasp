"""Offline depth-fit audit using human RGB boxes, independent of YOLO weights."""
from pathlib import Path
import sys,json,collections
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import numpy as np
from camera import load_frame
from rgbd_projection import depth_to_rgb
from box_geometry import fit_box


def main():
    groups=collections.defaultdict(list)
    for path in sorted((ROOT/'datasets/competition/raw/session-1794492392969744500').glob('*/capture.json')):
        metadata=json.loads(path.read_text())
        if (path.parent/'annotation.json').is_file():groups[metadata['collection_stage']].append(path.parent)
    rows=[]
    for stage,folders in groups.items():
        for folder in folders[::max(1,len(folders)//2)][:2]:
            frame=load_frame(folder/'rgbd.npz');uv=depth_to_rgb(frame)
            boxes=json.loads((folder/'annotation.json').read_text())['boxes_xyxy']
            valid=np.isfinite(uv).all(axis=2)
            masks=[valid&(uv[...,0]>=b[0])&(uv[...,0]<b[2])&(uv[...,1]>=b[1])&(uv[...,1]<b[3]) for b in boxes]
            occupied=np.logical_or.reduce(masks) if masks else np.zeros(valid.shape,bool)
            for index,mask in enumerate(masks):
                row=dict(stage=stage,folder=str(folder),box=boxes[index],associated_returns=int(mask.sum()))
                try:row.update(fit=True,geometry=fit_box(frame,mask,occupied))
                except ValueError as exc:row.update(fit=False,reason=str(exc))
                rows.append(row)
        subset=[r for r in rows if r['stage']==stage]
        print(stage,'OK',sum(r['fit'] for r in subset),'/',len(subset),collections.Counter(r.get('reason','OK') for r in subset),flush=True)
    out=ROOT/'depth_diagnostics/box-audit.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows,indent=2),encoding='utf-8');print('REPORT',out)

if __name__=='__main__':main()
