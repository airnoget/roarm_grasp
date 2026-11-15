"""Short camera-only YOLO-World + Cornell GG-CNN preview command."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([sys.executable, str(root/'preset_preview.py'),
                                     '--world', *sys.argv[1:]], cwd=root))
