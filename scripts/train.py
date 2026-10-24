"""Research training only. No test-set loader and no human disease prediction."""
from common import *
import argparse, random, time, math
from datetime import datetime
import numpy as np
import torch, torchvision
from PIL import Image, ImageOps
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

class TickImages(Dataset):
    def __init__(self, records, classes, target, augment=False):
        self.records,self.classes,self.target=records,classes,target
        ops=[torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomVerticalFlip(),torchvision.transforms.RandomRotation(20,fill=(240,240,240))] if augment else []
        self.transform=torchvision.transforms.Compose(ops+[torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(resolve_image_path(r['source_path'])) as src: im=ImageOps.exif_transpose(src).convert('RGB')
        # Penn-specific strip masking, matching the historic experiment, not phone preprocessing.
        im=im.crop((0,0,im.width,round(im.height*.96)))
        im=ImageOps.pad(im,(224,224),method=Image.Resampling.LANCZOS,color=(240,240,240))
        return self.transform(im),self.classes.index(r[self.target])

def run(config, smoke=False, output_root=None):
    require_training_files()
    cfg=json.loads(Path(config).read_text()); target=cfg['target'];seed=cfg['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    check=json.loads((WORK/'notes/preflight.json').read_text()); manifest=WORK/'data/penn_train_dev.csv'
    assert digest(SOURCE)==check['source_sha256'], 'Source changed: rerun preflight and inspect changes.'
    assert digest(manifest)==check['manifest_sha256'], 'Manifest changed: rerun preflight.'
    records=rows(manifest);assert all(r['provisional_split'] in ['train','dev'] for r in records)
    for r in records:
        image_path=resolve_image_path(r['source_path'])
        if not image_path.is_file():
            raise FileNotFoundError(f'Missing photo: {image_path}. Set TICKPREDICT_IMAGE_DIR to the original photo folder.')
        assert digest(image_path)==r['sha256'],str(image_path)
    train=[r for r in records if r['provisional_split']=='train'];dev=[r for r in records if r['provisional_split']=='dev']
    classes=sorted({r[target] for r in train});assert {r[target] for r in dev}<=set(classes)
    if smoke:
        train=[next(r for r in train if r[target]==c) for c in classes]*2
        dev=[next(r for r in dev if r[target]==c) for c in classes]*2
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    output_root=Path(output_root) if output_root is not None else WORK/'runs'
    out=output_root/((datetime.now().strftime('%Y%m%d-%H%M%S-%f'))+'_'+target+('_smoke' if smoke else ''))
    weight_hash=digest(WEIGHTS);assert weight_hash.startswith('f37072fd')
    out.mkdir(parents=True)
    model=torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    for p in model.parameters():p.requires_grad=False
    # Adapt the last residual block and the task head; early blocks remain pretrained.
    for p in model.layer4.parameters():p.requires_grad=True
    model.fc=torch.nn.Linear(model.fc.in_features,len(classes));model.to(device)
    optim=torch.optim.AdamW([{'params':model.layer4.parameters(),'lr':cfg['backbone_lr']},{'params':model.fc.parameters(),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
    counts=np.array([sum(r[target]==c for r in train) for c in classes])
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(len(classes)*counts),dtype=torch.float32,device=device))
    trainloader=DataLoader(TickImages(train,classes,target,True),batch_size=cfg['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    devloader=DataLoader(TickImages(dev,classes,target),batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
    provenance={'config':cfg,'classes':classes,'train_n':len(train),'validation_n':len(dev),'device':str(device),'smoke_only':smoke,'weights_sha256':weight_hash,'manifest_sha256':digest(manifest),'source_sha256':digest(SOURCE),'script_sha256':digest(__file__),'torch':torch.__version__,'torchvision':torchvision.__version__,'test_images_used':0,'preprocessing':'Penn only: EXIF transpose, bottom 4% strip removed, letterbox 224; ImageNet normalization','uncertainty':'Softmax scores are uncalibrated; not clinical or phone validation.'}
    write_json(out/'provenance.json',provenance)
    history=[];best=-1;stale=0
    print('Run:',out,'Device:',device,flush=True)
    for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
        start=time.time();model.train()
        # Freeze running statistics: tiny batches should not alter pretrained BN statistics.
        for m in model.modules():
            if isinstance(m,torch.nn.BatchNorm2d):m.eval()
        total=0;seen=0
        for x,y in trainloader:
            x,y=x.to(device),y.to(device);optim.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
            total+=loss.item()*len(y);seen+=len(y)
        model.eval();truth=[];pred=[];probs=[]
        with torch.inference_mode():
            for x,y in devloader:
                prob=model(x.to(device)).softmax(1).cpu().numpy();probs.extend(prob.tolist());pred.extend(prob.argmax(1).tolist());truth.extend(y.tolist())
        score=float(balanced_accuracy_score(truth,pred))
        entry={'epoch':epoch,'training_loss':total/seen,'validation_balanced_accuracy':score,'validation_macro_f1':float(f1_score(truth,pred,labels=list(range(len(classes))),average='macro',zero_division=0)),'seconds':time.time()-start}
        history.append(entry);write_json(out/'history.json',history);print(json.dumps(entry),flush=True)
        if score>best:
            best=score;stale=0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'classes':classes,'target':target,'epoch':epoch,'architecture':'resnet18'},out/'best.pt')
            write_json(out/'validation.json',{'balanced_accuracy':score,'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(len(classes)))).tolist(),'classes':classes,'predictions':[{'tick_id':r['tick_id'],'true':classes[t],'predicted':classes[p],'uncalibrated_probabilities':v} for r,t,p,v in zip(dev,truth,pred,probs)]})
        else:stale+=1
        if stale>=cfg['patience']:break
    write_json(out/'completion.json',{'status':'completed','best_validation_balanced_accuracy':best,'epochs':len(history),'smoke_only':smoke})
    print('Saved:',out,flush=True);return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--output-root',type=Path);a=p.parse_args();run(a.config,a.smoke,a.output_root)
