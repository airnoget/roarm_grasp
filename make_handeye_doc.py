from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Mm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parent/'docs'/'handeye_print'
OUT.mkdir(parents=True,exist_ok=True)
d=Document(); sec=d.sections[0]
for style in d.styles:
 for border in list(style.element.iter(qn('w:pBdr'))):
  border.getparent().remove(border)
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Mm(18);sec.left_margin=sec.right_margin=Mm(19)
for name in ('Normal','Title','Heading 1','Heading 2'):
 s=d.styles[name];s.font.name='Microsoft YaHei';s._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
 s.font.color.rgb=__import__('docx').shared.RGBColor(0,0,0)
d.styles['Normal'].font.size=Pt(10.5)
d.styles['Normal'].paragraph_format.space_after=Pt(5)
d.styles['Normal'].paragraph_format.line_spacing=1.12
d.styles['Title'].font.size=Pt(22)
d.styles['Heading 1'].font.size=Pt(14)
d.styles['Heading 2'].font.size=Pt(11.5)
footer=sec.footer.paragraphs[0];footer.alignment=2
footer.add_run('RoArm M3 S  腕部相机标定   ')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)

def p(t='',style=None):return d.add_paragraph(t,style)
def h(t):d.add_heading(t,1)
def table(headers,rows,widths=None):
 t=d.add_table(rows=1,cols=len(headers));t.style='Table Grid'
 for c,txt in zip(t.rows[0].cells,headers):
  c.text=txt
  for r in c.paragraphs[0].runs:r.bold=True
 for row in rows:
  for c,txt in zip(t.add_row().cells,row):c.text=str(txt)
 for row in t.rows:
  trPr=row._tr.get_or_add_trPr();trPr.append(OxmlElement('w:cantSplit'))
  for c in row.cells:
   for par in c.paragraphs:
    par.paragraph_format.space_after=Pt(4);par.paragraph_format.space_before=Pt(4)
    for r in par.runs:r.font.size=Pt(9.5)
 if widths:
  t.autofit=False
  for row in t.rows:
   for c,w in zip(row.cells,widths):c.width=Mm(w)
 return t
def code(t):
 par=p(t);par.paragraph_format.space_after=Pt(4)
 for r in par.runs:r.font.name='Consolas';r.font.size=Pt(8.5)

d.add_heading('腕部相机手眼标定准备与记录',0)
p('RoArm M3 S 与 Gemini 335L    现场打印版    2026年11月11日')
p('目的：取得深度相机相对腕部安装连杆的固定位置与朝向，并用独立观测检查物体在底座坐标系中的定位误差。本文件提供准备清单、现有脚本操作步骤、记录表和可选打印靶板。')
h('先确认安装状态')
p('相机必须与某个腕部连杆刚性连接。工程当前默认安装参考为 link5；该坐标系与夹爪实际夹取中心 TCP 分开。物体先从相机坐标换算到 link5，再由关节角和正运动学换算到底座固定坐标系。底座关节转动已包含在关节链中。')
p('当前已知：末端拆下了一个舵机，新舵机暂不安装。拆除的具体 ID 尚未确认。若只是夹爪 ID 17，且相机支架及前五关节完整，可继续核对腕部定位；若拆除腕部 ID 15 或 16，则不能直接沿用现有五关节模型。改变开机角度、舵机中位、安装支架后，应重新检查关节映射和标定有效性。')
h('出发前准备物品')
table(['物品','用途与要求'],[
('相机 机械臂 电脑 电源','Gemini 335L；机械臂底座固定；电脑能读取彩深帧及串口反馈。'),
('哑光打印纸与平整硬板','将末页靶板贴牢在硬板上，保持平整；避免反光覆膜。也可使用已有精密靶板。'),
('量具与底座参考','直角尺、游标卡尺或高度尺；明确底座固定坐标原点和 XYZ 方向。普通尺粗测不能保证毫米级精度。'),
('固定工具与记录工具','胶带或夹具固定靶板；笔记录点号、实测尺寸、姿态和异常。'),
('可见且有有效深度的靶点','优先选择平坦表面内部、易重复定位的点；避开物体边缘、反光区域和深度空洞。')],[45,130])
p('不需要先训练 YOLO 或 GG-CNN 才能标定。手眼标定记录几何关系；目标训练数据与抓取效果验证另行进行。')

d.add_page_break();h('采集与求解步骤')
for t in [
'1  固定相机、底座和靶板。记录相机安装连杆、底座原点、轴向和舵机 ID。先确认关节反馈的方向、零位及正运动学适用。',
'2  在底座固定坐标系中独立测量靶点 XYZ，单位毫米。不要直接用未验证的机械臂自报坐标当作实测真值。可使用有尺寸依据的治具；若用探针触点，须先验证探针尖端与工具中心的关系。',
'3  建议先采集 8至12 组不同静止姿态，变化腕部位置和朝向，覆盖拟使用的工作区域。靶点保持固定且清楚可见。姿态数量只是采集建议，最终以误差和几何覆盖验收。',
'4  每组停稳后录帧，保存深度、射线和 b s e t r 五个关节角。现有脚本检查采集窗口内关节变化，但不等同于硬件同步。使用同一相机、同一深度流和同一安装状态。',
'5  在保存的深度或彩深叠加画面点击同一物理靶点，记录像素 u v。点上需有有效深度；使用内部平面点，不选深度突变边缘。',
'6  将一部分独立姿态留作检查组，不参与拟合。工程要求至少4组非退化三维对应点进行拟合，以及至少2个独立验证点；实际应增加验证姿态和空间覆盖。',
'7  求解后检查每点误差，再在不同姿态观察同一固定点，核对其底座 XYZ 是否一致。通过静止定位验收后，才能进入低速悬停、工具中心及夹爪调试。']:
 p(t)
h('现有工程命令')
p('先关闭占用相机或串口的其它程序。在 PowerShell 中进入电脑端工程；下面的采集命令只查询状态，不发送关节运动命令。COM6 应改为现场实际端口。')
code('cd D:\\codex\\lab\\Star\\roarm_grasp')
code('$env:PYTHONPATH="$PWD\\.deps;$PWD\\.orbbec-python-deps"')
code('python capture.py orbbec calib/p01.npz --arm-port COM6 --color')
p('每次换新文件名。点击后记录 u v；CSV 表头为 frame,u,v,x_mm,y_mm,z_mm，frame 使用相对 CSV 的录帧路径。也可用 --points-csv 与 --point-base-mm X Y Z 自动追加一个点，XYZ 必须换成实测值。')
code('python calibrate.py calib/pairs.csv link5_from_depth.npy `\n  --check calib/check.csv --wrist-config config.json --max-error-mm 5')
p('运行求解前需准备实际 config.json，核对 vendor、joint_signs、joint_offsets_rad；保留 commissioned=false。5毫米只是初次筛查门限，不代表满足比赛抓取公差。最终按夹爪间隙和测量误差预算确定。')

d.add_page_break();h('现场记录表')
p('操作者____________  日期____________  相机序列号____________')
p('相机安装连杆____________  拆除舵机ID____________  深度流____________')
p('底座原点定义__________________________________________________')
p('X正方向____________  Y正方向____________  Z正方向____________')
p('靶板是否固定________  打印格实测边长________毫米  支架是否松动________')
h('靶点独立测量')
table(['点号','X 毫米','Y 毫米','Z 毫米','测量工具或依据'],[['P1','','','',''],['P2','','','',''],['P3','','','',''],['P4','','','','']],[18,28,28,28,73])
h('姿态与像素记录')
p('分组填“拟合”或“检查”。不同组使用独立录帧；不要将检查组用于拟合。')
table(['序号','分组','录帧文件','点号','u 像素','v 像素'],[[str(i),'拟合','','','',''] for i in range(1,7)],[16,22,68,19,25,25])
d.add_page_break();h('独立验证与验收记录')
p('这些录帧不参与外参拟合。更换机械臂姿态、重复观察固定靶点，检查是否仍得到正确的底座 XYZ。靶点实测坐标沿用前页，并记录使用的点号。')
table(['序号','分组','录帧文件','点号','u 像素','v 像素'],[[str(i),'检查','','','',''] for i in range(7,13)],[16,22,68,19,25,25])
p('异常记录______________________________________________________')
p('拟合最大误差________毫米  独立验证最大误差________毫米')
p('同一固定点跨姿态最大差________毫米  判定______________________')
p('注意：这张表中的打印靶板尺寸是靶板自身坐标，尚未自动成为底座 XYZ；必须独立建立靶板与底座的关系。')
h('现场坐标与安装草图')
p('画出底座原点、XYZ 正方向、相机支架固定位置与靶板位置；记录改动和量具。')
for _ in range(5):p('________________________________________________________________')

d.add_page_break();h('可选打印棋盘格')
p('棋盘格为8列6行方格，每格设计边长20毫米；内部角点为7列5行。请按100%或“实际大小”打印，关闭“适应页面”。打印后用尺核对单格边长和下方100毫米标尺，若有缩放，以实测值为准。')
im=Image.new('RGB',(1600,1200),'white');dr=ImageDraw.Draw(im)
for y in range(6):
 for x in range(8):
  if (x+y)%2==0:dr.rectangle((x*200,y*200,(x+1)*200-1,(y+1)*200-1),fill='black')
im.save(OUT/'checkerboard.png');d.add_picture(str(OUT/'checkerboard.png'),width=Mm(160))
bar=Image.new('RGB',(1200,170),'white');dr=ImageDraw.Draw(bar)
dr.line((100,55,1100,55),fill='black',width=5)
for x in range(100,1101,100):dr.line((x,30,x,80),fill='black',width=4)
bar.save(OUT/'ruler.png');d.add_picture(str(OUT/'ruler.png'),width=Mm(120))
p('上方两端长刻线间距为100毫米，每小格10毫米。')
p('此页可用于打印尺度检查、配准检查及后续自动标定扩展。当前工程没有棋盘格自动检测与标准手眼求解入口，现阶段仍按第2页的深度三维点配对流程操作；不要把“打印并拍摄此板”当作完成标定。')
p('若选取棋盘上的点做深度采样，使用平坦方格内部可重复定位的点，并先检查有效深度；棋盘交角附近的深度混合可能产生误差。纸面纹理、光照和距离会影响深度质量，靶板应贴在平整硬板上。')
p('工程参考：capture.py、calibrate.py、wrist_pose.py、validate_wrist.py。文档状态截至2026年11月11日；未填写现场测量值，未声明实机标定或抓取通过。')
path=OUT/'RoArm_M3S_Gemini335L_手眼标定打印手册.docx';d.save(path)
print(path)
