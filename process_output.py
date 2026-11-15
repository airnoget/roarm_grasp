"""Forward child output through a pipe, avoiding Windows console handle wrappers."""
import os
import subprocess
import sys


def run_with_output(argv, *, cwd, env=None, check=False):
    child_env = dict(os.environ if env is None else env)
    child_env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    with subprocess.Popen(argv, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                          errors='replace', bufsize=1) as process:
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
        except KeyboardInterrupt:
            # Do not leave a training process holding the GPU after its launcher exits.
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if check and code:
        raise subprocess.CalledProcessError(code, argv)
    return subprocess.CompletedProcess(argv, code)
