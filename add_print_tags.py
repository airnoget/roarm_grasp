from pathlib import Path
from PIL import Image
from docx import Document
from docx.shared import Mm,Pt
from docx.oxml.ns import qn

root=Path(__file__).resolve().parent/'docs/handeye_print'
source=root/'RoArm_M3S_Gemini335L_手眼标定打印手册.docx'
doc=Document(source)
doc.add_page_break()
doc.add_heading('AprilTag 打印标签',level=1)
doc.add_paragraph('标签族为 tag36h11，ID为0至3。每个标签黑色外框边长50毫米，外侧四周各保留10毫米白边。单张图总宽70毫米。按100%或实际大小打印，关闭适应页面；打印后实测黑框边长。')
table=doc.add_table(rows=2,cols=2);table.autofit=False
for i,cell in enumerate(c for row in table.rows for c in row.cells):
 cell.width=Mm(85)
 small=Image.open(root/'tags'/f'tag36_11_{i:05d}.png').convert('L')
 assert small.size==(10,10),small.size
 crop=small.crop((1,1,9,9))
 assert all(v==0 for v in list(crop.crop((0,0,8,1)).getdata()))
 assert all(v==0 for v in list(crop.crop((0,7,8,8)).getdata()))
 assert all(v==0 for v in list(crop.crop((0,0,1,8)).getdata()))
 assert all(v==0 for v in list(crop.crop((7,0,8,8)).getdata()))
 img=Image.new('L',(700,700),255)
 img.paste(crop.resize((500,500),Image.Resampling.NEAREST),(100,100))
 path=root/'tags'/f'tag36h11_id{i}_50mm.png';img.save(path)
 p=cell.paragraphs[0];p.alignment=1
 p.paragraph_format.space_after=Pt(0)
 p.add_run().add_picture(str(path),width=Mm(70))
 p=cell.add_paragraph(f'tag36h11   ID {i}   黑框50毫米');p.alignment=1
 p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(6)
 for r in p.runs:r.font.size=Pt(10)
p=doc.add_paragraph('检测程序中的 tag_size 或 markerLength 应填实测黑色外框边长，设计值为0.050米，不含白边。剪裁时保留完整白边，贴在平整哑光硬板上；避免折痕、反光覆膜和遮挡。')
p=doc.add_paragraph('这些标签可供后续 Tag 检测与位姿估计使用。当前工程尚无 Tag 自动手眼求解入口；识别出ID并不等于完成标定。若作为多Tag板使用，另需测量每个标签角点在板上的位置，不能假定Word表格间距。')
p=doc.add_paragraph('原始码图来自 AprilRobotics/apriltag-imgs 官方仓库，采用BSD 2-Clause许可。来源与许可：https://github.com/AprilRobotics/apriltag-imgs')
for r in p.runs:r.font.size=Pt(8)
out=root/'RoArm_M3S_Gemini335L_手眼标定打印手册_含AprilTag.docx'
doc.save(out)
print(out)
