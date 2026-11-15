"""Base-only half-turn followed by gripper release; no arm IK or lowering."""
import numpy as np
from motion_speeds import SPEEDS


def half_turn_target(current_rad):
    if not np.isfinite(current_rad) or abs(current_rad)>np.pi+.02:raise ValueError('Invalid base angle')
    # Original ID11 mapping: round(2047 - angle * 4096 / (2*pi)).
    lower=-np.pi;upper=2047*2*np.pi/4096
    candidates=[float(np.clip(current_rad+sign*np.pi,lower,upper)) for sign in (-1,1)]
    return min(candidates,key=lambda value:abs(abs(value-current_rad)-np.pi))


class RearTransfer:
    def __init__(self,arm,open_deg=110):
        self.arm=arm;self.open_deg=float(open_deg);self.phase=None
        self.target=None;self.samples=0;self.events=[];self.status=''

    def start(self,state):
        if self.phase:raise ValueError('Base transfer already running; X stops')
        if state['v']<700:raise ValueError('Servo power unavailable')
        self.target=half_turn_target(state['b']);self.samples=0
        self.turn_speed=SPEEDS['base_fast']
        command=dict(T=101,joint=1,rad=self.target,spd=self.turn_speed,acc=5)
        self.arm.send(command);self.events=[dict(kind='base_command',command=command)]
        self.phase='turning';self.status=f'BASE only: {np.degrees(state["b"]):.1f} -> {np.degrees(self.target):.1f}deg; no lowering'

    def update(self,state):
        if not self.phase:return
        if state['v']<700:self.status='TRANSFER power unavailable; X holds';return
        if self.phase=='turning':
            error=abs(float(state['b'])-self.target)
            if np.degrees(error)<=SPEEDS['base_slow_angle_deg'] and self.turn_speed!=SPEEDS['base_near']:
                self.turn_speed=SPEEDS['base_near']
                command=dict(T=101,joint=1,rad=self.target,spd=self.turn_speed,acc=5)
                self.arm.send(command);self.events.append(dict(kind='base_deceleration',command=command))
            self.samples=self.samples+1 if error<np.radians(2) else 0
            self.status=f'BASE turning: remaining {np.degrees(error):.1f}deg | other arm joints unchanged | X hold'
            if self.samples>=3:
                command=dict(T=101,joint=6,rad=float(np.radians(self.open_deg)),spd=SPEEDS['gripper_release'],acc=5)
                self.arm.send(command);self.events.append(dict(kind='release_command',command=command))
                self.phase='releasing';self.samples=0
        elif self.phase=='releasing':
            angle=state.get('g',np.nan)
            self.samples=self.samples+1 if np.isfinite(angle) and abs(np.degrees(angle)-self.open_deg)<2 else 0
            self.status='REAR releasing gripper; fixed arm posture/height | X hold'
            if self.samples>=3:self.phase=None;self.status='REAR transfer finished; release not visually verified'

    def stop(self,state):
        if self.phase:
            for joint,key in ((1,'b'),(6,'g')):
                if key in state:self.arm.send(dict(T=101,joint=joint,rad=float(state[key]),spd=80,acc=5))
            self.events.append(dict(kind='user_stop'))
        self.phase=None;self.status='REAR transfer stopped'
