"""Local commissioning console. approach permits motion after P plan and G execute."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HELP = '''
v / view     Gemini 335L YOLO + GG-CNN preview (Q/Esc returns here, S saves a frame)
p / ports    List serial ports, without opening them
c / capture  Save one aligned RGB-D snapshot; no arm feedback or motion
b / board    Check printed tag36h11 IDs and chessboard 7x5 corners, no arm connection
tag          Collect 50 mm ID 0 tag views with stationary COM6 wrist feedback
chess        Collect printed 7x5 chessboard, 20 mm squares, COM6 wrist feedback
tcp          Read-only single grasp-point TCP: 8 S fit poses + 3 V check poses
tcpsolve     Solve latest cube TCP session offline; does not apply it
tcpapply     Apply latest validated cube TCP and clear temporary left/down bias
solve        Solve the latest complete chessboard session offline (no robot motion)
preview14    Bench wood_block YOLO/GG-CNN + XYZ using 14.93 mm candidate; no motion commands
data         Fast native RGB + depth dataset capture; no models and no arm connection
data5        Fixed preset, 5 objects, 200 layouts (140 far / 60 near), resumable
label        Manually draw target boxes on the latest recorded dataset (no hardware)
grasp        Label left/right grasps on 50 existing RGB-D images (no hardware)
prepare      Validate reviewed boxes and export grouped train/validation data
train        Train target YOLO on the prepared dataset, CUDA preferred (no robot connection)
trained      Preview latest trained target YOLO + GG-CNN (camera only)
geom         Preview latest YOLO + box depth geometry with the frozen preset (no motion)
locate       Same geometry preview with COM6 feedback; S saves target + joints (opening serial may reset arm)
stepview     S waits for 5 consistent frames, saves one observation and exits; no motion
approach     Live coordinates + downward pose: P plan, G hover, D descend, X hold
hover        Plan latest saved geometry observation, link5 reference; offline, no motion
xyz          Print latest single-target camera/base coordinates and measured joints
trained14    Same trained detector with candidate wrist XYZ (opens COM6 feedback)
depth        Raw measured depth health, draw ROI and S save (camera only)
rawdepth     Independent SDK depth-only test, no RGB/alignment/models
s / sim      Run wrist-camera GG-CNN simulation with original vendor gripper
sy / semantic Run the same simulation with real YOLO target confirmation too
r / replay   Replay the first available trace from the last s run
h / help     Show commands
q / quit     Exit

Calibration: current wrist solver estimates depth-camera -> link5.
Joint feedback + robot FK then maps link5 -> fixed robot base frame.
link5 is a wrist link; it is distinct from the gripper contact TCP.
'''


def run(command):
    env = os.environ.copy()
    paths = [str(ROOT/name) for name in ('.deps','.simdeps','.orbbec-python-deps','.torchdeps','.semdeps')]
    if env.get('PYTHONPATH'): paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    if command in ('v','view'):
        argv = [str(ROOT/'start_preview.py')]
    elif command == 'rawdepth':
        argv = [str(ROOT/'raw_depth.py')]
    elif command == 'depth':
        argv = [str(ROOT/'depth_health.py')]
    elif command == 'trained':
        argv = [str(ROOT/'preset_preview.py'),
                '--preset',str(ROOT/'camera_profiles/gemini335l-viewer-near.json'),
                '--rotate','180']
    elif command == 'geom':
        argv = [str(ROOT/'preset_preview.py'),'--geometry','--rotate','180']
    elif command == 'locate':
        argv = [str(ROOT/'preset_preview.py'),'--geometry','--port','COM6','--rotate','180']
    elif command == 'stepview':
        argv = [str(ROOT/'preset_preview.py'),'--geometry','--port','COM6','--rotate','180','--step-capture']
    elif command == 'approach':
        argv = [str(ROOT/'preset_preview.py'),'--geometry','--port','COM6','--rotate','180','--approach']
    elif command == 'hover':
        argv = [str(ROOT/'plan_hover.py'),'--link5']
    elif command == 'xyz':
        argv = [str(ROOT/'check_localization.py')]
    elif command == 'trained14':
        argv = [str(ROOT/'preview_candidate.py'),'--trained']
    elif command in ('p','ports'):
        argv = ['-m','serial.tools.list_ports','-v']
    elif command in ('b','board'):
        argv = [str(ROOT/'board_preview.py')]
    elif command == 'prepare':
        argv = [str(ROOT/'prepare_targets.py')]
    elif command == 'train':
        argv = [str(ROOT/'train_targets.py')]
    elif command == 'label':
        argv = [str(ROOT/'label_dataset.py')]
    elif command == 'grasp':
        argv = [str(ROOT/'label_grasps.py')]
    elif command == 'data5':
        argv = [str(ROOT/'preset_preview.py'),'--collect']
    elif command == 'data':
        argv = [str(ROOT/'data_capture.py')]
    elif command == 'preview14':
        argv = [str(ROOT/'preview_candidate.py')]
    elif command == 'tag':
        argv = [str(ROOT/'tag_capture.py')]
    elif command == 'chess':
        argv = [str(ROOT/'chess_capture.py')]
    elif command in ('tcp','tcpsolve','tcpapply'):
        argv=[str(ROOT/'tcp_calibrate.py'),{'tcp':'collect','tcpsolve':'solve','tcpapply':'apply'}[command]]
    elif command == 'solve':
        sessions = [p for p in (ROOT/'calibration').glob('chess-*')
                    if (p/'session.json').is_file() and len(list((p/'fit').glob('*.json')))>=10
                    and len(list((p/'check').glob('*.json')))>=5]
        if not sessions:
            print('No complete chessboard session. Run chess; collect >=10 fit and >=5 check views.')
            return 2
        latest=max(sessions,key=lambda p:p.name)
        print(f'Solving {latest}',flush=True)
        argv=[str(ROOT/'solve_chess_handeye.py'),str(latest)]
    elif command in ('c','capture'):
        output = ROOT/'field_recordings'/f'snapshot-{time.time_ns()}.npz'
        argv = [str(ROOT/'capture.py'),'orbbec',str(output),'--color','--no-preview']
        print(f'Snapshot destination: {output}',flush=True)
    elif command in ('s','sim','sy','semantic'):
        output = ROOT/'sim_output'/f'debug-{time.time_ns()}'/'run.json'
        output.parent.mkdir(parents=True,exist_ok=False)
        print(f'Simulation report: {output}',flush=True)
        print('Model: original complete gripper; current physical servo removal is not modeled.',flush=True)
        argv = [str(ROOT/'simulate.py'),'--camera','wrist','--gripper','vendor-split','--trace',
                '--upstream',str(ROOT/'.vendor/ggcnn'),
                '--trusted-checkpoint',str(ROOT/'.vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'),
                '--report',str(output)]
        if command in ('sy','semantic'):
            argv += ['--target-weights','D:/Lab/挑战杯/LunarRover_HUST-master/weights/best.pt']
            print('YOLO is enabled. Simplified rendered RGB may fail target confirmation.',flush=True)
    elif command in ('r','replay'):
        pointer = ROOT/'sim_output/latest-debug-run.json'
        if not pointer.exists():
            print('No recorded s run yet. Run s first.')
            return 2
        traces = json.loads(pointer.read_text(encoding='utf-8'))['traces']
        trace = next((Path(p) for p in traces if Path(p).is_file()),None)
        if trace is None:
            print('No trace exists from the last run. Inspect its report before rerunning.')
            return 2
        print(f'Replay: {trace}',flush=True)
        argv = [str(ROOT/'replay.py'),str(trace),'--depth-view','--inference-view']
    else:
        print(HELP if command in ('h','help') else 'Unknown command; type h for help.')
        return
    if command == 'train':
        from process_output import run_with_output
        result = run_with_output([sys.executable,*argv],cwd=ROOT,env=env)
    else:
        result = subprocess.run([sys.executable,*argv],cwd=ROOT,env=env)
    if command in ('s','sim','sy','semantic'):
        traces = sorted(p for p in output.parent.glob('run_*.npz') if p.stem[4:].isdigit())
        pointer = 'latest-semantic-debug-run.json' if command in ('sy','semantic') else 'latest-debug-run.json'
        (ROOT/'sim_output'/pointer).write_text(
            json.dumps(dict(report=str(output),exit_code=result.returncode,
                            traces=[str(p) for p in traces]),indent=2),encoding='utf-8')
        print('Exit code 1 means at least one trial failed; inspect the report.',flush=True)
        if command in ('s','sim'):
            print('Run r to inspect saved traces.',flush=True)
    if result.returncode:
        print(f'Command ended with code {result.returncode}; see the error above.')
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=['v','view','p','ports','c','capture','b','board','tag','chess','tcp','tcpsolve','tcpapply','solve','preview14','data','data5','label','grasp','prepare','train','trained','geom','locate','stepview','approach','hover','xyz','trained14','depth','rawdepth','s','sim','sy','semantic','r','replay','h','help'])
    args=parser.parse_args()
    if args.command:
        raise SystemExit(run(args.command) or 0)
    print(HELP)
    while True:
        try: command=input('debug> ').strip().lower()
        except (EOFError,KeyboardInterrupt):
            print()
            break
        if command in ('q','quit','exit'): break
        if command:
            try: run(command)
            except KeyboardInterrupt: print('\nReturned to debug console.')


if __name__=='__main__': main()
