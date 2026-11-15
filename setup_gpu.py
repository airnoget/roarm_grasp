"""Install project-local CUDA PyTorch, then verify GPU computation. Does not train."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def install():
    # CUDA 12.6 wheels support this laptop's driver 566.24; do not use cu128 here.
    from gpu_download import official_wheel, download
    if sys.platform != 'win32' or sys.maxsize <= 2**32:
        raise RuntimeError('This setup entry is for 64-bit Windows')
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    wheels = []
    for package, version in [('torch','2.7.1'),('torchvision','0.22.1')]:
        url, filename, digest = official_wheel(package, version, tag)
        wheels.append(str(download(url, ROOT/'.gpu-downloads'/filename, digest)))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
                    '--timeout', '180', '--retries', '10',
                    '--target', str(ROOT/'.cudadeps'), *wheels,
                    '--index-url', 'https://download.pytorch.org/whl/cu126'], check=True)
    subprocess.run([sys.executable,'-m','pip','install','--upgrade','--target',str(ROOT/'.traindeps'),
                    '-r',str(ROOT/'requirements-training.txt'),'--timeout','180','--retries','10'],check=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(str(ROOT/p) for p in ('.traindeps','.cudadeps','.deps','.torchdeps','.semdeps'))
    code = '''import torch, torchvision
print('TORCH', torch.__version__, 'CUDA', torch.version.cuda, flush=True)
assert torch.cuda.is_available(), 'CUDA unavailable after installation'
x = torch.randn(64,64,device='cuda'); y=x@x
torch.cuda.synchronize()
boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device='cuda')
torchvision.ops.nms(boxes,torch.tensor([.9,.8],device='cuda'),.5)
torch.cuda.synchronize()
print('GPU CHECK PASSED:',torch.cuda.get_device_name(0),flush=True)
'''
    subprocess.run([sys.executable,'-c',code], cwd=ROOT, env=env, check=True)
    from process_output import run_with_output
    run_with_output([sys.executable,str(ROOT/'train_targets.py'),'--check','--device','0'], cwd=ROOT, env=env, check=True)
    print('READY: python debug.py train', flush=True)


def main():
    import msvcrt
    lock = (ROOT/'.gpu-setup.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise SystemExit('Another GPU setup is running. Stop it before restarting.')
    try:
        install()
    except KeyboardInterrupt:
        print('\nStopped. Partial downloads retained; rerun python setup_gpu.py to resume.', flush=True)
        raise SystemExit(130)
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__=='__main__':main()
