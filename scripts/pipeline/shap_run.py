import pandas as pd, numpy as np, pickle, shap
from common import *
o=pickle.load(open('tab_final.pkl','rb')); X0,_=build()
out={}
for h in [1,6,24]:
    m=o['models'][(h,'LGB-Δ [past]')]; Xp=pd.concat([X0,calendar(X0.index,h)],axis=1).loc[idx[o_te]]
    sv=shap.TreeExplainer(m).shap_values(Xp); out[h]=dict(sv=sv,X=Xp)
    imp=pd.Series(np.abs(sv).mean(0),index=Xp.columns).sort_values(ascending=False); print(h,imp.head(10).round(2).to_dict())
pickle.dump(out,open('shap.pkl','wb'))
