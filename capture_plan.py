"""Small explicit collection plan; counts are not training labels."""
import math
from pathlib import Path
import numpy as np


class CapturePlan:
    def __init__(self, object_types=3):
        if not 1 <= object_types <= 9:
            raise ValueError('Use 1 to 9 target types')
        self.stages = [(str(i), f'single_{i}', f'第{i}种单独', 20) for i in range(1, object_types+1)]
        self.stages += [('m', 'mixed', '多个混放', 30), ('b', 'background', '无目标背景', 10)]
        self.counts = {v[1]: 0 for v in self.stages}
        self.active = self.stages[0]

    def select(self, key):
        for stage in self.stages:
            if stage[0] == chr(key).lower():
                self.active = stage
                return True
        return False

    def saved(self):
        self.counts[self.active[1]] += 1

    def export(self):
        return dict(active_stage=self.active[1], counts=dict(self.counts),
                    goals={v[1]: v[3] for v in self.stages},
                    total_saved=sum(self.counts.values()), total_goal=sum(v[3] for v in self.stages),
                    photos_per_layout=1, annotation_status='unlabelled')

    def hint(self):
        if self.counts[self.active[1]] >= self.active[3]:
            remaining = [s for s in self.stages if self.counts[s[1]] < s[3]]
            if remaining:
                next_stage = remaining[0]
                return f'当前批次已完成！换物块或场景后，按{next_stage[0].upper()}切到“{next_stage[2]}”。'
            return f"首轮{self.export()['total_goal']}张计划已完成，可按Q退出；这些图片仍需人工标注。"
        key = self.active[1]
        if key == 'mixed':
            return '同放2～3种物块，换数量、间距或遮挡；每种摆放只拍1张。'
        if key == 'background':
            return '移走所有目标物块，保留桌面和干扰物；换背景后再拍。'
        return f'只放第{self.active[0]}种：换位置、旋转、翻面或垫高；不要只在棋盘纸上拍。'


def header(plan, width):
    from PIL import Image, ImageDraw, ImageFont
    columns = min(5, len(plan.stages))
    rows = math.ceil(len(plan.stages)/columns)
    height = 42+rows*53+66
    image = Image.new('RGB', (width, height), (25, 30, 39))
    draw = ImageDraw.Draw(image)
    fonts = [Path('C:/Windows/Fonts/msyh.ttc'),
             Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    font_file = next((p for p in fonts if p.is_file()), None)
    size = 20 if width >= 1100 else 15
    font = ImageFont.truetype(str(font_file), size) if font_file else ImageFont.load_default()
    state = plan.export()
    title = f"采集进度 {state['total_saved']}/{state['total_goal']} 张 | 每种摆放1张 | 只移动物块，不用Wi-Fi调姿"
    draw.text((12, 8), title, font=font, fill='white')
    for index, stage in enumerate(plan.stages):
        x = index % columns * width//columns
        y = 39+index//columns*53
        selected = stage[1] == plan.active[1]
        draw.rounded_rectangle((x+6, y, x+width//columns-6, y+45), radius=6,
                               fill=(20, 94, 172) if selected else (48, 58, 72))
        draw.text((x+15, y+8), f'{stage[0].upper()} {stage[2]} {plan.counts[stage[1]]}/{stage[3]}',
                  font=font, fill='white')
    y = 42+rows*53
    controls=('F远景/N近景；1～5/M/B切批次；换摆放停稳后S拍1张，Q退出。' if hasattr(plan,'distance')
              else '换摆放 → 停稳 → S拍1张；按1/2/3/M/B切换批次，Q退出。不要原地连拍。')
    draw.text((12, y), controls,
              font=font, fill=(145, 220, 170))
    draw.text((12, y+29), plan.hint(), font=font, fill=(230, 210, 150))
    return np.asarray(image)[..., ::-1].copy()
