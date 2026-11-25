import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from common import *
import lightgbm as lgb
o=pickle.load(open('tab_final.pkl','rb')); X0,_=build()
# ---------- ALE (LGB-Δ past) ----------
def ale1d(m,X,f,nb=20):
    q=np.unique(np.quantile(X[f],np.linspace(0,1,nb+1)))
    if len(q)<3: return None
    b=np.clip(np.searchsorted(q,X[f],side='right')-1,0,len(q)-2); eff=np.zeros(len(q)-1); n=np.zeros(len(q)-1)
    for k in range(len(q)-1):
        r=X[b==k]
        if len(r)==0: continue
        lo=r.copy(); hi=r.copy(); lo[f]=q[k]; hi[f]=q[k+1]
        eff[k]=np.mean(m.predict(hi)-m.predict(lo)); n[k]=len(r)
    acc=np.concatenate([[0],np.cumsum(eff)]); cen=acc-np.sum(((acc[:-1]+acc[1:])/2)*n)/n.sum()
    return q,cen
ALE={}
feats=['Q-HSW2_GPM|L0','Q-HSW1_GPM|L0','Q-HSW1_GPM|L48-71','H-Dig1_FT|L0','D1_TEMPERATURE|L0']
allo=np.sort(np.concatenate([o_tr,o_va,o_te]))
for h in [1,6,24]:
    m=o['models'][(h,'LGB-Δ [past]')]; X=pd.concat([X0,calendar(X0.index,h)],axis=1).loc[idx[o_tr]]
    for f in feats:
        r=ale1d(m,X,f); ALE[(h,f)]=r
        if r: print('ALE',h,f,'range',round(r[1].max()-r[1].min(),2),'cfm')
# ---------- horizon-resolved DML (partially linear, blocked cross-fitting) ----------
LG=lambda: lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,verbose=-1,random_state=0,n_jobs=2)
def fw(s,a,b):  # mean of s over (t+a, t+b]
    return s[::-1].rolling(b-a).mean()[::-1].shift(-(a+1))
def dml(Y,T,X,h,K=5,buf=None):
    ok=Y.notna()&T.notna()&X.notna().all(1); Y,T,X=Y[ok],T[ok],X[ok]; n=len(Y); buf=buf or (h+168)
    ry=np.zeros(n); rt=np.zeros(n); edges=np.linspace(0,n,K+1).astype(int)
    for k in range(K):
        a,b=edges[k],edges[k+1]; tr=np.r_[0:max(0,a-buf),min(n,b+buf):n]
        ry[a:b]=Y.values[a:b]-LG().fit(X.iloc[tr],Y.iloc[tr]).predict(X.iloc[a:b])
        rt[a:b]=T.values[a:b]-LG().fit(X.iloc[tr],T.iloc[tr]).predict(X.iloc[a:b])
    th=np.sum(rt*ry)/np.sum(rt*rt); psi=(ry-th*rt)*rt/np.mean(rt*rt)
    L=h+24; g=[np.mean(psi[l:]*psi[:n-l]) for l in range(L+1)]; v=(g[0]+2*sum((1-l/(L+1))*g[l] for l in range(1,L+1)))/n
    return th,np.sqrt(v),rt,ry,X
rows=[]; het=[]
idx_all=idx[allo]
for h in [1,2,3,4,6,9,12,18,24]:
    Y=pd.Series(target(h).values,index=idx)-H['Biogas']
    T=fw(hsw,0,h); Tfut=fw(hsw,h,2*h)
    X=pd.concat([X0,calendar(X0.index,h),fw(H['Q-PS_on'],0,h).rename('PS_plan'),fw(H['Q-TWAS_GPM'],0,h).rename('TWAS_plan')],axis=1)
    Xs=X.loc[idx_all]; Ys=Y.loc[idx_all]
    th,se,rt,ry,Xo=dml(Ys,T.loc[idx_all],Xs,h)
    rows.append(dict(h=h,test='HSW in (t, t+h] → y(t+h)−y(t)',theta=th,se=se))
    thp,sep,_,_,_=dml(Ys,Tfut.loc[idx_all],pd.concat([Xs,T.loc[idx_all].rename('T_actual')],axis=1),h)
    rows.append(dict(h=h,test='Placebo: HSW in (t+h, t+2h]',theta=thp,se=sep))
    # random-block placebo (shuffle 24-h blocks of T)
    tb=T.loc[idx_all].values.copy(); nb=len(tb)//24; perm=np.random.default_rng(h).permutation(nb); tb2=np.concatenate([tb[i*24:(i+1)*24] for i in perm]+[tb[nb*24:]])
    thr,ser,_,_,_=dml(Ys,pd.Series(tb2,index=idx_all),Xs,h)
    rows.append(dict(h=h,test='Placebo: block-shuffled treatment',theta=thr,se=ser))
    # heterogeneity by gas-storage tertile (residual-on-residual within subgroup)
    st=Xo['H-Dig1_FT|L0']; tq=pd.qcut(st,3,labels=['low','mid','high'])
    for lv in ['low','mid','high']:
        s=(tq==lv).values; het.append(dict(h=h,storage=lv,theta=np.sum(rt[s]*ry[s])/np.sum(rt[s]**2)))
    print(h,[f"{r['test'][:12]} {r['theta']:.2f}±{1.96*r['se']:.2f}" for r in rows[-3:]],flush=True)
D=pd.DataFrame(rows); D['ci_lo']=D.theta-1.96*D.se; D['ci_hi']=D.theta+1.96*D.se; D.to_csv('dml_horizon.csv',index=False)
Hh=pd.DataFrame(het); Hh.to_csv('dml_heterogeneity_storage.csv',index=False); print(Hh.pivot(index='h',columns='storage',values='theta').round(2).to_string())
pickle.dump(ALE,open('ale.pkl','wb'))
print('HSW mean gpm',round(hsw.mean(),2),'sd',round(hsw.std(),2))
