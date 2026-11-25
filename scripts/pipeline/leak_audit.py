import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from common import *
import lightgbm as lgb
X0,_=build()
LGB=lambda: lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=2,verbose=-1,random_state=0,n_jobs=1)
rows=[]
rng=np.random.default_rng(0)
allo=np.concatenate([o_tr,o_va,o_te]); PERM=rng.permutation(allo); 
for h in [1,6,24]:
    X=pd.concat([X0,calendar(X0.index,h)],axis=1); yy=pd.Series(target(h).values,index=idx)
    leak=pd.DataFrame({'V-burner_FT3@t+h':H['V-burner_FT3'].shift(-h),'V-Boiler_FT3@t+h':H['V-Boiler_FT3'].shift(-h),'hour@t+h':(idx+pd.Timedelta(hours=h)).hour},index=idx)
    y24=pd.Series(H['Biogas'][::-1].rolling(24).mean()[::-1].shift(-1).values,index=idx)  # mean over (t,t+24]
    tr,te=idx[o_tr],idx[o_te]
    perm=PERM; ntr=int(0.8*len(perm)); rtr,rte=idx[np.sort(perm[:ntr])],idx[np.sort(perm[ntr:])]
    def run(name,XX,target,trn,tst):
        ok=XX.notna().all(1)&target.notna(); trn=trn[ok[trn].values]; tst=tst[ok[tst].values]
        m=LGB().fit(XX.loc[trn],target[trn]); p=m.predict(XX.loc[tst]); r=metrics(target[tst].values,p)
        rows.append(dict(h=h,setting=name,**r, n_test=len(tst))); print(h,name,round(r['R2'],3),flush=True)
    run('A Strict chronological, past-only',X,yy,tr,te)
    run('B Random 80/20 split, past-only',X,yy,rtr,rte)
    run('C Random split + target-time cumulative counters',pd.concat([X,leak],axis=1),yy,rtr,rte)
    run('D Chronological + target-time cumulative counters',pd.concat([X,leak],axis=1),yy,tr,te)
    if h==24:
        run('E Strict chronological, next-24 h mean target',X,y24,tr,te)
        run('F Random split, next-24 h mean target',X,y24,rtr,rte)
pd.DataFrame(rows).to_csv('leakage_audit.csv',index=False)
