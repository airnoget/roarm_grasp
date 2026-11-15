"""Five objects, 200 layouts, 70% far / 30% near; photos are not annotations."""
from capture_plan import CapturePlan


class DistancePlan(CapturePlan):
    def __init__(self):
        self.distance='far'
        self.stages=[]
        for distance in ('far','near'):
            label='远景' if distance=='far' else '近景'
            self.stages += [(str(i),f'{distance}_single_{i}',f'{label}第{i}种',25 if distance=='far' else 10) for i in range(1,6)]
            self.stages += [('m',f'{distance}_mixed',f'{label}混放',10 if distance=='far' else 5),
                            ('b',f'{distance}_background',f'{label}背景',5)]
        self.counts={s[1]:0 for s in self.stages};self.active=self.stages[0]

    def select(self,key):
        char=chr(key).lower()
        if char in ('f','n'):
            self.distance='far' if char=='f' else 'near'
            self.active=next(s for s in self.stages if s[1].startswith(self.distance+'_'))
            return True
        for stage in self.stages:
            if stage[0]==char and stage[1].startswith(self.distance+'_'):
                self.active=stage;return True
        return False

    def export(self):
        return dict(super().export(),distance=self.distance,object_types=5,far_goal=140,near_goal=60)

    def restore(self,state):
        if set(state['counts'])!=set(self.counts):raise ValueError('Wrong collection plan')
        for key,n in state['counts'].items():
            if type(n) is not int or n<0:raise ValueError('Invalid capture count')
        self.counts=dict(state['counts']);self.active=next(s for s in self.stages if s[1]==state['active_stage'])
        self.distance=self.active[1].split('_')[0]

    def hint(self):
        if self.counts[self.active[1]]>=self.active[3]:
            pending=[s for s in self.stages if self.counts[s[1]]<s[3]]
            if not pending:return '200张已采齐，按Q退出；下一步画框，尚未训练。'
            s=pending[0];key='F' if s[1].startswith('far_') else 'N'
            return f'当前批次已满；换摆放后按{key}、{s[0].upper()}切到{s[2]}。'
        return 'F远景 / N近景；1～5选物块，M混放，B背景；每次换位置或角度只按S拍1张。'


def banner(plan,width):
    from capture_plan import header
    # The old header lays out all stages; filter only for display, preserve totals.
    return header(plan,width)
