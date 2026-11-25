"""One representative neural model under matched information and scalar targets."""
from context import *
from statistics_utils import *
import torch,time
from torch import nn
torch.set_num_threads(2)
rows=[];selection=[];preds={};checkpoint=OUT/'matched_mlp_predictions.pkl'
if checkpoint.exists():preds=pd.read_pickle(checkpoint)
tree=pd.read_pickle(RES/'tab_final.pkl')
for h in [1,6,24]:
    for reg in ['past','feed']:
        for anchored in [False,True]:
            key=(h,reg,anchored)
            if key in preds:continue
            x=features(h,reg);mean=x.loc[TR].mean();std=x.loc[TR].std().replace(0,1)
            tensors={k:torch.tensor(((x.loc[orig]-mean)/std).values,dtype=torch.float32) for k,orig in [('tr',TR),('va',SELECT),('te',TE)]}
            y=target(h);base=x['Biogas|L0'];scale=float(y.loc[TR].std());offset=0 if anchored else float(y.loc[TR].mean())
            yt=torch.tensor(((y-base if anchored else y).loc[TR].values-offset)/scale,dtype=torch.float32)
            yv=y.loc[SELECT].values;bt=base.loc[TE].values if anchored else 0;bv=base.loc[SELECT].values if anchored else 0
            options=[]
            for lr in [.0005,.001]:
                pv=[];pt=[];records=[]
                for seed in range(3):
                    torch.manual_seed(SEED+seed);rng=np.random.default_rng(SEED+seed)
                    model=nn.Sequential(nn.Linear(x.shape[1],128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
                    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
                    best=np.inf;state=None;pat=0;start=time.time()
                    for epoch in range(80):
                        model.train()
                        for indices in np.array_split(rng.permutation(len(TR)),int(np.ceil(len(TR)/128))):
                            opt.zero_grad();loss=((model(tensors['tr'][indices]).ravel()-yt[indices])**2).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                        model.eval()
                        with torch.no_grad():v=model(tensors['va']).ravel().numpy()*scale+offset+bv
                        score=np.sqrt(np.mean((v-yv)**2))
                        if score<best-1e-8:best=score;state={k:v.detach().clone() for k,v in model.state_dict().items()};pat=0;best_epoch=epoch+1
                        else:pat+=1
                        if pat>=12:break
                    model.load_state_dict(state);model.eval()
                    with torch.no_grad():
                        pv.append(model(tensors['va']).ravel().numpy()*scale+offset+bv)
                        pt.append(model(tensors['te']).ravel().numpy()*scale+offset+bt)
                    records.append(dict(h=h,regime=reg,anchored=anchored,lr=lr,seed=seed,best_epoch=best_epoch,epochs=epoch+1,selection_RMSE_m3_h=best*CFM,seconds=time.time()-start))
                score=np.sqrt(np.mean((np.mean(pv,axis=0)-yv)**2));options.append((score,np.mean(pt,axis=0),lr,records))
            best=min(options,key=lambda q:q[0]);preds[key]={'prediction':best[1],'lr':best[2],'selection':sum([option[3] for option in options],[])}
            pd.to_pickle(preds,checkpoint);print('matched MLP',key,'LR',best[2],metric(y.loc[TE].values,best[1]),flush=True)
for (h,reg,anchor),result in preds.items():
    y=target(h).loc[TE].values;p=result['prediction'];selection.extend(result['selection'])
    label=f'LGB-Δ [{reg}]' if anchor else ('LightGBM' if reg=='past' else 'LightGBM +feed')
    ref=tree['pred'][h]['test'][label].values
    rows.append(dict(h=h,regime=reg,anchor=anchor,model='Matched MLP',lr=result['lr'],**metric(y,p),**rmse_difference(y,p,ref,block_indices(len(y),72))))
    rows.append(dict(h=h,regime=reg,anchor=anchor,model='Matched LightGBM',lr=.02,**metric(y,ref)))
frame('matched_MLP_metrics.csv',rows);frame('matched_MLP_selection.csv',selection)
print('MATCHED MLP COMPLETE',flush=True)
