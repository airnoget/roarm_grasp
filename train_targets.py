"""Fine-tune the target detector only; never opens a camera or robot port."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))
for name in reversed(('.traindeps', '.cudadeps', '.deps', '.torchdeps', '.semdeps')):
    sys.path.insert(0, str(ROOT/name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--weights', type=Path, default=Path('D:/Lab/挑战杯/LunarRover_HUST-master/weights/best.pt'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='auto', help='auto selects CUDA device 0 when available; otherwise CPU')
    parser.add_argument('--check', action='store_true', help='Validate inputs and load model, without training')
    args = parser.parse_args()
    if args.data is None:
        pointer = ROOT/'datasets/competition/latest-export.json'
        if not pointer.is_file(): parser.error('Run python debug.py prepare first')
        args.data = Path(json.loads(pointer.read_text())['dataset'])
    if not args.data.is_file() or not args.weights.is_file():
        parser.error('Dataset YAML or initial weights missing')
    manifest_path=args.data.parent/'manifest.json'
    policy_path=ROOT/'camera_profiles/training-policy.json'
    if policy_path.is_file():
        policy=json.loads(policy_path.read_text(encoding='utf-8'))
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
        if manifest.get('camera_preset_sha256')!=policy['sha256']:
            parser.error('Dataset is not verified under the current preset. Collect with python debug.py data5, label, then prepare. Existing weights and legacy data are retained.')
    if args.epochs < 1 or args.batch < 1: parser.error('epochs and batch must be positive')
    import torch
    from ultralytics import YOLO
    import yaml
    config = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    if config.get('names') != {0: 'artificial_object'}:
        parser.error('Expected one target class: artificial_object')
    if args.device == 'auto':
        args.device = '0' if torch.cuda.is_available() else 'cpu'
    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('This Python environment has no CUDA; use --device cpu')
    model = YOLO(str(args.weights), task='detect')
    name = f'targets-{time.time_ns()}'
    options = dict(data=str(args.data.resolve()), epochs=args.epochs, imgsz=640,
                   batch=args.batch, device=args.device, workers=0, seed=42, deterministic=True,
                   project=str(ROOT/'runs/targets'), name=name, exist_ok=False,
                   cache=False, amp=False, cls_remap=False, patience=25, close_mosaic=min(10,args.epochs),
                   hsv_h=.015, hsv_s=.4, hsv_v=.3, fliplr=.5,
                   plots=True, save=True)
    print('DATASET', args.data.resolve(), flush=True)
    print('INITIAL WEIGHTS', args.weights.resolve(), flush=True)
    print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available(), flush=True)
    if args.device != 'cpu':
        print('GPU', torch.cuda.get_device_name(int(args.device)), flush=True)
    print('TRAIN OPTIONS', json.dumps(options), flush=True)
    if args.check:
        import numpy as np
        import cv2
        import scipy
        import matplotlib.pyplot as plt
        from ultralytics.data.utils import check_det_dataset
        checked = check_det_dataset(str(args.data.resolve()), autodownload=False)
        for split in ('train','val'):
            images = list(Path(checked[split]).glob('*.png'))
            if not images: raise RuntimeError(f'No {split} images')
            sample = cv2.imread(str(images[0]))
            if sample is None: raise RuntimeError(f'Failed to decode {images[0]}')
        output = ROOT/'.test-artifacts/training-preflight.png'
        output.parent.mkdir(exist_ok=True)
        fig, ax = plt.subplots(); ax.plot([0,1],[0,1]); fig.savefig(output); plt.close(fig)
        x = torch.randn(8,8,device='cuda:0' if args.device != 'cpu' else 'cpu',requires_grad=True)
        (x@x).square().mean().backward()
        if args.device != 'cpu': torch.cuda.synchronize()
        print('NUMPY', np.__version__, 'OPENCV', cv2.__version__, 'SCIPY', scipy.__version__)
        print('CHECK PASSED: image decode, dataset paths, plotting, tensor backward; no training started')
        return
    model.train(**options)
    best = Path(model.trainer.save_dir)/'weights/best.pt'
    if not best.is_file(): raise RuntimeError('Training returned without best.pt')
    from dataset_labels import atomic_text
    atomic_text(ROOT/'datasets/competition/latest-training.json', json.dumps(dict(
        best_weights=str(best.resolve()), dataset=str(args.data.resolve()), class_names={0:'artificial_object'},
        scope='YOLO target detector only; GG-CNN unchanged; no motion validation'), indent=2))
    print('BEST WEIGHTS', best.resolve(), flush=True)


if __name__ == '__main__': main()
