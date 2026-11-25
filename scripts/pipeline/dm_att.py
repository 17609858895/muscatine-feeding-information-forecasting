import pandas as pd, numpy as np, pickle, torch
from scipy import stats
from common import *
fp=pickle.load(open('final_preds.pkl','rb')); d=pd.read_pickle('preds_dl_all.pkl')
def dm(y,p1,p2,h):
    e=(y-p1)**2-(y-p2)**2; n=len(e); m=e.mean(); g=[np.mean((e[k:]-m)*(e[:n-k]-m)) for k in range(h)]
    v=(g[0]+2*sum(g[1:]))/n; s=m/np.sqrt(max(v,1e-12)); hl=np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    s*=hl; return s, 2*stats.t.sf(abs(s),n-1)
rows=[]
for h in [1,6,24]:
    T=fp['pred'][h]['test']; y=T.y.values
    for prop in ['PG-HQ ensemble [past]','PG-HQ ensemble [feed]']:
        for b in ['Persistence','XGBoost','CatBoost','LightGBM','AR-Ridge (gas only)']:
            s,p=dm(y,T[prop].values,T[b].values,h); rows.append(dict(h=h,proposed=prop,baseline=b,DM=s,p=p))
        for b in ['PatchTST','N-BEATS','TCN','LSTM']:
            s,p=dm(y,T[prop].values,d['P'][b][:,h-1],h); rows.append(dict(h=h,proposed=prop,baseline=b,DM=s,p=p))
DM=pd.DataFrame(rows); DM.to_csv('dm_tests.csv',index=False); print(DM.round(4).to_string())
# intrinsic attention of PG-HQ (past) vs permutation HR-PGA, by process group
import dl
A=[]
for s in range(3):
    m=dl.PGHQ2(d=32,use_feed=False); m.load_state_dict(torch.load(f'pghq_past_s{s}.pt')); m.eval()
    X,Y,B,C,F=dl.D['te']; ws=[]
    with torch.no_grad():
        for i in range(0,len(X),512): m(X[i:i+512],C[i:i+512],F[i:i+512]); ws.append(m.last_att)
    A.append(torch.cat(ws).mean(0).numpy())  # HMAX x tokens(ng*nb)
A=np.mean(A,0).reshape(24,len(GROUPS),len(BINS))
np.save('pghq_attention.npy',A)
hr=pickle.load(open('hrpga.pkl','rb')); HS=hr['HS']
G=A.sum(2)  # horizon x group
Ag=hr['past']['Ag'][:,:len(GROUPS)]
for i,h in enumerate(HS):
    print(h,'spearman(group attention, group permutation)=',round(stats.spearmanr(G[h-1],Ag[i]).correlation,2), np.round(G[h-1],3))
