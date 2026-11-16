"""Offline first-version exploration demo or fixed-frame JSON replay; no hardware."""
import argparse
import json
from pathlib import Path
from exploration import TargetRegistry,CoverageGrid,PreviewTransport,ExplorationCoordinator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,help='JSON list: stamp, origin_m, observations(name,position_m)')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'exploration_output/latest.json')
    args=parser.parse_args()
    frames=json.loads(args.replay.read_text(encoding='utf-8')) if args.replay else [
        dict(stamp=1,origin_m=[0,0,0],observations=[dict(name='cube',position_m=[1.,.2,0]),dict(name='cube',position_m=[.4,.1,0])]),
        dict(stamp=2,origin_m=[.2,0,0],observations=[dict(name='cube',position_m=[.401,.1,0]),dict(name='cube',position_m=[1.,.2,0])]),
        dict(stamp=3,origin_m=[.3,0,0],observations=[])]
    registry=TargetRegistry();transport=PreviewTransport();coordinator=ExplorationCoordinator(registry,transport)
    events=[]
    for frame in frames:
        ids=registry.update(frame['observations'],frame['stamp'])
        command=coordinator.tick(frame['origin_m'],frame['stamp'],frame.get('stopped',False),frame.get('grasp_ready',False))
        events.append(dict(ids=ids,phase=coordinator.phase,**command))
    grid=CoverageGrid([[0,0,0,1,-1],[0,1,0,1,-1],[0,0,0,0,-1]],.2)
    grid.observe((0,0),0,1.57,.6)
    report=dict(hardware_commands_sent=False,events=events,tracks=list(registry.tracks.values()),
                coverage=dict(seen=sorted(grid.seen),next=grid.next_scan((0,0)),scope='floor only; not proof all targets found'))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for e in events:print(e)
    print('REPORT',args.output)


if __name__=='__main__':main()
