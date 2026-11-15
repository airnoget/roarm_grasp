"""Start the local Gemini 335L preview: python start_preview.py."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
for name in reversed(('.deps', '.orbbec-python-deps', '.torchdeps', '.semdeps')):
    sys.path.insert(0, str(root / name))
os.chdir(root)
extra = sys.argv[1:]
weights = 'D:/Lab/挑战杯/LunarRover_HUST-master/weights/best.pt'
if '--trained' in extra:
    from trained_model import latest_detector
    weights = str(latest_detector(root))
    extra.remove('--trained')
    print('TRAINED DETECTOR',weights,flush=True)
source = [] if '--frame' in extra or '--camera' in extra else ['--camera', 'orbbec']
sys.argv = [str(root / 'preview_live.py'), *source,
            '--target-weights', weights,
            '--upstream', str(root / '.vendor/ggcnn'),
            '--trusted-checkpoint', str(root / '.vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'),
            *extra]
runpy.run_path(str(root / 'preview_live.py'), run_name='__main__')
