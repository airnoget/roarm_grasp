"""Continue Cornell weights on partial manual positives; offline pilot only."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND','Agg')
for n in reversed(('.traindeps','.cudadeps','.deps','.torchdeps','.semdeps')):sys.path.insert(0,str(ROOT/n))
sys.path.insert(0,str(ROOT/'.vendor/ggcnn'))
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from ggcnn_finetune_data import load_samples,split_samples


def objective(pred,target,mask,teacher):
    # Sparse positive annotation is not a complete negative grasp map.
    supervised=sum((((p-target[:,i:i+1])**2)*mask).sum()/mask.sum().clamp_min(1)
                   for i,p in enumerate(pred))
    # Retain the original model away from labeled positives rather than invent negatives.
    preserve=sum((((p-t)**2)*(1-mask)).sum()/(1-mask).sum().clamp_min(1) for p,t in zip(pred,teacher))
    return supervised+.1*preserve,supervised


def evaluate(model,teacher,data,device):
    model.eval();values=[]
    with torch.no_grad():
        for s in data:
            x=torch.from_numpy(s['image'][None]).to(device)
            y=torch.from_numpy(s['target'][None]).to(device);m=torch.from_numpy(s['mask'][None]).to(device)
            pred=model(x);_,loss=objective(pred,y,m,teacher(x));values.append(float(loss))
    return float(np.mean(values))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.epochs<1 or a.lr<=0:p.error('Positive epochs and learning rate required')
    torch.manual_seed(42);np.random.seed(42)
    session=Path(json.loads((ROOT/'datasets/competition/raw/distance-active.json').read_text(encoding='utf-8'))['session'])
    sha=json.loads((ROOT/'camera_profiles/training-policy.json').read_text(encoding='utf-8'))['sha256']
    samples,excluded=load_samples(session,sha);train,val=split_samples(samples)
    out=ROOT/'runs/ggcnn'/f'finetune-{time.time_ns()}';out.mkdir(parents=True)
    summarize=lambda data:[{k:v for k,v in s.items() if k not in ('image','mask','target')} for s in data]
    manifest=dict(train=summarize(train),validation=summarize(val),excluded=excluded,seed=42,preset_sha256=sha,
                  note='Split by original physical scene; partial positives, not complete negative labels')
    (out/'data.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DATA',len(train),'train crops',len(val),'validation crops; excluded',len(excluded),flush=True)
    checkpoint=ROOT/'.vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=torch.load(checkpoint,map_location=device,weights_only=False).to(device)
    teacher=copy.deepcopy(model).eval()
    for v in teacher.parameters():v.requires_grad_(False)
    for v in model.parameters():v.requires_grad_(True)
    baseline=evaluate(model,teacher,val,device);best=baseline
    torch.save(model.cpu(),out/'best.pt');model.to(device)
    print('DEVICE',device,'BASELINE positive-region MSE',baseline,flush=True)
    if a.check:print('CHECK PASSED',out);return
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    history=[];best_epoch=0
    for epoch in range(1,a.epochs+1):
        model.train();losses=[]
        for i in np.random.permutation(len(train)):
            s=train[i];x=torch.from_numpy(s['image'][None]).to(device)
            y=torch.from_numpy(s['target'][None]).to(device);m=torch.from_numpy(s['mask'][None]).to(device)
            with torch.no_grad():original=teacher(x)
            optimizer.zero_grad();loss,_=objective(model(x),y,m,original)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss.detach()))
        score=evaluate(model,teacher,val,device)
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_positive_mse=score))
        if score<best:
            best=score;best_epoch=epoch;torch.save(model.cpu(),out/'best.pt');model.to(device)
        print(f'EPOCH {epoch}/{a.epochs} train={np.mean(losses):.5f} val={score:.5f} best={best:.5f}',flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2))
    torch.save(model.cpu(),out/'last.pt')
    report=dict(initial_checkpoint=str(checkpoint),initial_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                best_weights=str(out/'best.pt'),best_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),
                best_epoch=best_epoch,baseline_validation_positive_mse=baseline,best_validation_positive_mse=best,
                learning_rate=a.lr,epochs=a.epochs,train_crops=len(train),validation_crops=len(val),
                initialization='Cornell pretrained; all layers updated by backpropagation',
                objective='positive-region MSE on Q/cos2theta/sin2theta/width + 0.1 teacher preservation outside positives',
                scope='Small-data pilot. Validation loss is not grasp accuracy. No deployment promotion.',motion_enabled=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    # Do not silently replace the production Cornell checkpoint.
    (ROOT/'runs/ggcnn/latest-finetune.json').write_text(json.dumps(report,indent=2))
    print('RESULT',out,'BEST EPOCH',best_epoch,flush=True)

if __name__=='__main__':main()
