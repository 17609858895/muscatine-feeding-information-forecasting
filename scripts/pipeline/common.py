import pandas as pd, numpy as np
from feats import *
HMAX=24; L=168
idx=H.index; y=H['Biogas'].values
VARS=[v for vs in GROUPS.values() for v in vs]
Zv=H[VARS].values
valid=~np.isnan(Zv).any(1)
okv=np.array([valid[t-L+1:t+HMAX+1].all() and not np.isnan(y[t:t+HMAX+1]).any() if t>=L and t<len(idx)-HMAX else False for t in range(len(idx))])
T=np.arange(len(idx)); end=idx+pd.Timedelta(hours=HMAX)
o_tr=T[okv&(end<SPLIT_VAL)]; o_va=T[okv&(idx>=SPLIT_VAL)&(end<SPLIT_TEST)]; o_te=T[okv&(idx>=SPLIT_TEST)]
hsw=(H['Q-HSW1_GPM']+H['Q-HSW2_GPM'])
def future_feed(h):
    f=lambda s: s[::-1].rolling(h).mean()[::-1].shift(-1)
    return pd.DataFrame({'Planned HSW|F':f(hsw),'Planned PS|F':f(H['Q-PS_on']),'Planned TWAS|F':f(H['Q-TWAS_GPM'])})
def metrics(yv,p):
    e=yv-p; return dict(RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(np.abs(e)),R2=1-np.sum(e**2)/np.sum((yv-yv.mean())**2))
