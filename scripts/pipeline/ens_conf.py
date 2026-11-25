import pandas as pd, numpy as np, pickle
from scipy.optimize import nnls
from common import *
o=pickle.load(open('tab_final.pkl','rb')); d=pd.read_pickle('preds_dl_all.pkl')
HS=[1,2,3,4,6,9,12,18,24]
rows=[]; ens={}
yv_all=pd.Series(H['Biogas'].values,index=idx)
for h in HS:
    T=o['pred'][h]['test']; V=o['pred'][h]['val']
    for reg,nm in [('past','PG-HQ (proposed, past-only)'),('feed','PG-HQ (proposed, feed-conditioned)')]:
        T[f'PG-HQ [{reg}]']=d['P'][nm][:,h-1]; V[f'PG-HQ [{reg}]']=d['P'][nm+'__val'][:,h-1]
        comps=[f'LGB-Δ [{reg}]',f'Cat-Δ [{reg}]',f'PG-HQ [{reg}]']
        w,_=nnls(V[comps].values,V.y.values); w=w/w.sum()
        pv=V[comps].values@w; pt=T[comps].values@w
        key=f'PG-HQ ensemble [{reg}]'; T[key]=pt; V[key]=pv
        # split conformal (absolute) and volatility-normalised conformal
        vol_v=yv_all.rolling(24).std().reindex(V.index).values+1.0; vol_t=yv_all.rolling(24).std().reindex(T.index).values+1.0
        rv=np.abs(V.y.values-pv); n=len(rv)
        for a in [0.2,0.1,0.05]:
            ql=min(1,np.ceil((n+1)*(1-a))/n)
            q=np.quantile(rv,ql); qn=np.quantile(rv/vol_v,ql)
            cov=np.mean(np.abs(T.y.values-pt)<=q); covn=np.mean(np.abs(T.y.values-pt)<=qn*vol_t)
            rows.append(dict(h=h,regime=reg,nominal=1-a,cov_split=cov,width_split=2*q,cov_norm=covn,width_norm=np.mean(2*qn*vol_t)))
        ens[(h,reg)]=dict(w=dict(zip(comps,w)))
    o['pred'][h]['test']=T; o['pred'][h]['val']=V
C=pd.DataFrame(rows); C.to_csv('conformal.csv',index=False)
pickle.dump(dict(pred=o['pred'],ens=ens),open('final_preds.pkl','wb'))
# metrics table for all models
rows=[]
for h in HS:
    T=o['pred'][h]['test']; y=T.y.values
    for c in T.columns.drop('y'): rows.append(dict(h=h,model=c,**metrics(y,T[c].values)))
# DL baselines (only 1/6/24 plus all h for PG-HQ)
Y=d['Y']
for nm,p in d['P'].items():
    if nm.endswith('__val') or 'PG-HQ' in nm: continue
    for h in HS: rows.append(dict(h=h,model=nm,**metrics(Y[:,h-1],p[:,h-1])))
R=pd.DataFrame(rows); pers=R[R.model=='Persistence'].set_index('h').RMSE; R['Skill']=1-R.RMSE/R.h.map(pers)
R.to_csv('all_metrics.csv',index=False)
print(R[R.h.isin([1,6,24])].pivot(index='model',columns='h',values=['R2','Skill']).sort_values(('R2',24)).round(3).to_string())
print(pd.DataFrame({k:v['w'] for k,v in ens.items()}).round(2).T.to_string())
print(C[C.h.isin([1,6,24])].round(3).to_string())
