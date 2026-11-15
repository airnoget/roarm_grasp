"""Read STEP assembly with OpenCascade; preserve occurrence transforms and mm units."""
from pathlib import Path
import sys,json,argparse,re
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.caddeps'))
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label,TDF_LabelSequence
from OCP.TopLoc import TopLoc_Location
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer

def name(label):
    attribute=TDataStd_Name()
    return attribute.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(),attribute) else ''

def main():
    p=argparse.ArgumentParser();p.add_argument('step',type=Path);p.add_argument('--output',type=Path,default=ROOT/'cad_inspection/modified-step');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    document=TDocStd_Document(TCollection_ExtendedString('MDTV-XCAF'))
    reader=STEPCAFControl_Reader();reader.SetNameMode(True)
    status=reader.ReadFile(str(a.step.resolve()))
    if int(status)!=1:raise RuntimeError(f'STEP read failed: {status}')
    if not reader.Transfer(document):raise RuntimeError('STEP transfer failed')
    tool=XCAFDoc_DocumentTool.ShapeTool_s(document.Main());roots=TDF_LabelSequence();tool.GetFreeShapes(roots)
    records=[];exports=[]
    def visit(label,parent,path,export_children=False):
        referenced=TDF_Label()
        ref=tool.GetReferredShape_s(label,referenced)
        definition=referenced if ref else label
        title=name(definition) or name(label)
        location=parent.Multiplied(tool.GetLocation_s(label))
        shape=tool.GetShape_s(definition)
        # Definition's own location is already present on shape. For components,
        # add only accumulated occurrence placements to the definition geometry.
        world=shape.Moved(location) if ref else shape.Moved(parent)
        box=Bnd_Box();BRepBndLib.Add_s(world,box)
        bounds=list(box.Get()) if not box.IsVoid() else None
        transform=location.Transformation()
        row=dict(name=title,path=path+[title],assembly=tool.IsAssembly_s(definition),
                 transform_mm=[[transform.Value(i,j) for j in range(1,5)] for i in range(1,4)],
                 box_assembly_mm=bounds)
        if bounds:row['size_assembly_mm']=[bounds[i+3]-bounds[i] for i in range(3)]
        records.append(row)
        selected='夹爪装配' in title
        if selected:
            BRepMesh_IncrementalMesh(world,.35,False,.4,True)
            writer=StlAPI_Writer();writer.ASCIIMode=False
            target=a.output/f'gripper-assembly-{len(exports)}.stl'
            if not writer.Write(world,str(target)):raise RuntimeError('STL export failed')
            exports.append(dict(name=title,file=str(target),coordinates='STEP assembly mm',box_assembly_mm=bounds))
        if row['assembly']:
            children=TDF_LabelSequence();tool.GetComponents_s(definition,children,False)
            for i in range(1,children.Length()+1):visit(children.Value(i),location,path+[title],export_children or selected)
    for i in range(1,roots.Length()+1):visit(roots.Value(i),TopLoc_Location(),[])
    report=dict(source=str(a.step.resolve()),units='mm',roots=roots.Length(),occurrences=len(records),
                components=records,gripper_exports=exports,tcp_verified=False,
                note='STEP assembly frame is not assumed to be robot base_link or link5. STL units are mm.')
    (a.output/'assembly.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    selected=[r for r in records if any(k in r['name'] for k in ['夹爪','上压','垫块','ST3215'])]
    (a.output/'gripper-components.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in selected:print(row['name'],'SIZE mm',row.get('size_assembly_mm'),flush=True)
    print('REPORT',a.output/'assembly.json','OCCURRENCES',len(records),'GRIPPER EXPORTS',len(exports),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
