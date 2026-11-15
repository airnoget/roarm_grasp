"""Read-only SolidWorks assembly inventory, component transforms in metres."""
import argparse,json
from pathlib import Path
import pythoncom
import win32com.client
from win32com.client import VARIANT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('assembly',type=Path)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    pythoncom.CoInitialize()
    try:app=win32com.client.GetActiveObject('SldWorks.Application')
    except pythoncom.com_error:app=win32com.client.Dispatch('SldWorks.Application')
    err=VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
    warn=VARIANT(pythoncom.VT_BYREF|pythoncom.VT_I4,0)
    # Silent + read-only. No saves, rebuilds, exports or hardware connections.
    document=app.OpenDoc6(str(a.assembly.resolve()),2,3,'',err,warn)
    if document is None:raise RuntimeError(f'OpenDoc6 failed errors={err.value} warnings={warn.value}')
    def value(obj,name,*args):
        member=getattr(obj,name)
        return member(*args) if callable(member) else member
    rows=[]
    for component in value(document,'GetComponents',False) or []:
        row={}
        for name in ('Name2','GetPathName','GetSuppression','IsSuppressed'):
            try:row[name]=value(component,name)
            except Exception as exc:row[name+'_error']=str(exc)
        try:row['transform_array']=list(value(value(component,'Transform2'),'ArrayData'))
        except Exception as exc:row['transform_error']=str(exc)
        try:row['box_m']=list(value(component,'GetBox',False,False))
        except Exception as exc:row['box_error']=str(exc)
        rows.append(row)
    report=dict(assembly=str(a.assembly.resolve()),read_only=True,errors=err.value,warnings=warn.value,
                components=rows,units='metres; SolidWorks MathTransform array layout',
                tcp_verified=False,scope='CAD component inventory; no link5/TCP correspondence assumed')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPORT',a.output,'COMPONENTS',len(rows),flush=True)
    for row in rows:print(row.get('Name2'),row.get('GetPathName'),flush=True)

if __name__=='__main__':main()
