import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from common import *
import lightgbm as lgb
LGB=lambda: lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=2,verbose=-1,random_state=0,n_jobs=1)
# conventional engineering: raw lags 0..23 + rolling mean/std 6/24/168 for each variable
cols={}
for v in VARS:
    s=H[v]
    for l in range(24): cols[f'{v}|lag{l}']=s.shift(l)
    for w in [6,24,168]: cols[f'{v}|rm{w}']=s.rolling(w).mean(); cols[f'{v}|rs{w}']=s.rolling(w).std()
XR=pd.DataFrame(cols,index=idx)
X0,_=build()
rows=[]
for h in [1,6,24]:
    yy=pd.Series(target(h).values,index=idx); cal=calendar(idx,h); tr,te=idx[o_tr],idx[o_te]; base=H['Biogas']
    for nm,XX in [('raw lags + rolling stats',pd.concat([XR,cal],axis=1)),('process-group lag-bins',pd.concat([X0,cal],axis=1))]:
        m=LGB().fit(XX.loc[tr],(yy-base)[tr]); p=m.predict(XX.loc[te])+base[te].values
        rows.append(dict(h=h,features=nm,n_features=XX.shape[1],**metrics(yy[te].values,p)))
    print(rows[-2:],flush=True)
pd.DataFrame(rows).to_csv('ablation_features.csv',index=False)
