"""Hardware-independent exploration primitives. Coordinates are metres in a fixed frame."""
from dataclasses import dataclass
import math


def distance(a,b):
    return math.dist(a,b)


class TargetRegistry:
    def __init__(self, gate_m=.08, stale_s=3.):
        self.gate=gate_m;self.stale=stale_s;self.tracks={};self.next_id=1;self.selected=None

    def update(self, observations, stamp):
        """Only unambiguous one-to-one matches inherit identities; ambiguity stays unresolved."""
        candidates=[]
        for obs in observations:
            p=obs['position_m']
            if len(p)!=3 or not all(math.isfinite(v) for v in p):raise ValueError('Invalid target position')
            candidates.append([k for k,t in self.tracks.items() if t['state']!='collected'
                               and t['name']==obs['name'] and distance(p,t['position_m'])<self.gate])
        usage={k:sum(k in c for c in candidates) for k in self.tracks}
        assigned=[]
        for obs,c in zip(observations,candidates):
            if len(c)>1 or (c and usage[c[0]]>1):
                assigned.append(None);continue
            if c:identity=c[0]
            else:identity=self.next_id;self.next_id+=1
            self.tracks[identity]=dict(obs,id=identity,last_seen=stamp,state='visible')
            assigned.append(identity)
        for k,t in self.tracks.items():
            if t['state']!='collected' and t['last_seen']!=stamp:t['state']='occluded'
        return assigned

    def choose(self, origin, stamp):
        if self.selected is not None:return self.selected
        options=[t for t in self.tracks.values() if t['state']=='visible' and stamp-t['last_seen']<=self.stale]
        if options:self.selected=min(options,key=lambda t:distance(origin[:2],t['position_m'][:2]))['id']
        return self.selected

    def collected(self):
        if self.selected is not None:self.tracks[self.selected]['state']='collected'
        self.selected=None


class CoverageGrid:
    """0 free, 1 occupied, -1 unknown. Traversability and visual search coverage differ."""
    def __init__(self, cells, resolution_m, origin=(0.,0.), clearance_m=0.):
        if resolution_m<=0 or not cells or not cells[0] or any(len(r)!=len(cells[0]) for r in cells):raise ValueError('Invalid map')
        if any(v not in (-1,0,1) for row in cells for v in row):raise ValueError('Invalid occupancy')
        self.cells=cells;self.resolution=resolution_m;self.origin=origin;self.seen=set()
        self.h=len(cells);self.w=len(cells[0]);self.clearance=math.ceil(clearance_m/resolution_m)

    def free(self,c):
        x,y=c;r=self.clearance
        return all(0<=i<self.w and 0<=j<self.h and self.cells[j][i]==0
                   for i in range(x-r,x+r+1) for j in range(y-r,y+r+1))

    def reachable(self,start):
        if not self.free(start):return set()
        found={start};todo=[start]
        for x,y in todo:
            for c in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if c not in found and self.free(c):found.add(c);todo.append(c)
        return found

    def observe(self, cell, yaw, fov, range_m, usable=True):
        """Caller confirms valid camera pose/image. Ray casting prevents seeing through walls."""
        if not usable:return
        for y in range(self.h):
            for x in range(self.w):
                dx=x-cell[0];dy=y-cell[1];d=math.hypot(dx,dy)
                if d*self.resolution>range_m:continue
                angle=math.atan2(dy,dx)-yaw
                if d and abs(math.atan2(math.sin(angle),math.cos(angle)))>fov/2:continue
                steps=max(abs(dx),abs(dy))*4
                ray=[(round(cell[0]+dx*i/max(1,steps)),round(cell[1]+dy*i/max(1,steps))) for i in range(int(steps)+1)]
                if all(0<=i<self.w and 0<=j<self.h and self.cells[j][i]==0 for i,j in ray):self.seen.add((x,y))

    def next_scan(self,start):
        reachable=self.reachable(start)
        # Alternating rows yield a sweep order; path itself is reconstructed through free cells.
        remaining=reachable-self.seen
        if not remaining:return dict(goal=None,path=[],status='known_reachable_floor_observed',unknown=sum(v==-1 for r in self.cells for v in r))
        goal=min(remaining,key=lambda c:(c[1],c[0] if c[1]%2==0 else -c[0]))
        parents={start:None};todo=[start]
        for x,y in todo:
            if (x,y)==goal:break
            for c in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if c in reachable and c not in parents:parents[c]=(x,y);todo.append(c)
        path=[];c=goal
        while c is not None:path.append(c);c=parents[c]
        return dict(goal=goal,path=path[::-1],status='searching',unknown=sum(v==-1 for r in self.cells for v in r))


@dataclass
class ChassisIntent:
    mode:str
    target_id:int|None=None
    goal_m:tuple|None=None
    reason:str=''


class PreviewTransport:
    """Adapter contract; deliberately no serial device ownership."""
    def __init__(self):self.commands=[]
    def send(self,intent):self.commands.append(intent);return dict(sent_to_hardware=False,intent=vars(intent))


class ScanEnvelope:
    def __init__(self,config):self.config=config
    def edge(self,start,end):
        poses=self.config.get('poses',{})
        if start not in poses or end not in poses:raise ValueError('Exploration poses not taught')
        if [start,end] not in self.config.get('edges',[]):raise ValueError('Exploration edge not commissioned')
        return [poses[start],poses[end]]


class ExplorationCoordinator:
    def __init__(self,registry,transport):self.registry=registry;self.transport=transport;self.phase='scan'
    def tick(self,origin,stamp,stopped=False,grasp_ready=False):
        identity=self.registry.choose(origin,stamp)
        if identity is None:self.phase='scan';return self.transport.send(ChassisIntent('scan',reason='No confirmed 3D target'))
        t=self.registry.tracks[identity]
        if t['state']!='visible' or stamp-t['last_seen']>self.registry.stale:
            self.phase='reobserve';return self.transport.send(ChassisIntent('stop',identity,reason='Selected target not currently confirmed'))
        if stopped and grasp_ready:
            self.phase='grasp';return self.transport.send(ChassisIntent('stop',identity,reason='Relocalize and lock grasp before extending arm'))
        self.phase='dock';return self.transport.send(ChassisIntent('request_docking_plan',identity,tuple(t['position_m']),reason='Target position, not an executable chassis goal'))
