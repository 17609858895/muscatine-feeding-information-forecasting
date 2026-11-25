"""Extra analyses needed by the figure set (run once; outputs under extra/)."""
import pandas as pd, numpy as np, pickle, warnings, time, os, sys; warnings.filterwarnings('ignore')
from common import *
import lightgbm as lgb, xgboost as xgb, catboost as cb
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
os.makedirs('extra',exist_ok=True)
X0,meta=build(); HS=[1,2,3,4,6,9,12,18,24]
LGB=lambda nj=2: lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=2,verbose=-1,random_state=0,n_jobs=nj)
BASE={
 'Ridge':lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))),
 'SVR':lambda: make_pipeline(StandardScaler(),SVR(C=30,epsilon=2,gamma='scale')),
 'KNN':lambda: make_pipeline(StandardScaler(),KNeighborsRegressor(25,weights='distance')),
 'Random forest':lambda: RandomForestRegressor(400,min_samples_leaf=5,max_features=0.33,n_jobs=2,random_state=0),
 'Extra trees':lambda: ExtraTreesRegressor(400,min_samples_leaf=5,max_features=0.5,n_jobs=2,random_state=0),
 'HistGB':lambda: HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.0,random_state=0),
 'LightGBM':LGB,
 'XGBoost':lambda: xgb.XGBRegressor(n_estimators=800,learning_rate=0.02,max_depth=4,min_child_weight=10,subsample=0.8,colsample_bytree=0.6,reg_lambda=2,random_state=0,n_jobs=2),
 'CatBoost':lambda: cb.CatBoostRegressor(iterations=1200,learning_rate=0.03,depth=5,l2_leaf_reg=5,verbose=0,random_seed=0,thread_count=2),
}
step=sys.argv[1:] or ['E1','E2','E3','E4','E5','E6','E7','E8']
if os.environ.get('ONLY'): step=os.environ['ONLY'].split(',')
tr,va,te=idx[o_tr],idx[o_va],idx[o_te]
def Xpast(h): return pd.concat([X0,calendar(X0.index,h)],axis=1)
if 'E1' in step:   # baselines at all horizons
    o=pickle.load(open('tab_final.pkl','rb')); rows=[]
    for h in HS:
        T=o['pred'][h]['test']; X=Xpast(h); yy=pd.Series(target(h).values,index=idx)
        if 'Ridge' not in T.columns:
            T['Seasonal naive (24 h)']=H['Biogas'].shift(24-h)[te].values
            ar=[c for c in X.columns if c.startswith('Biogas|')]+['hod_sin','hod_cos']
            T['AR-Ridge (gas only)']=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))).fit(X.loc[tr,ar],yy[tr]).predict(X.loc[te,ar])
            for nm,f in BASE.items(): T[nm]=f().fit(X.loc[tr],yy[tr]).predict(X.loc[te])
        for c in T.columns.drop('y'): rows.append(dict(h=h,model=c,**metrics(T.y.values,T[c].values)))
        print('E1',h,flush=True)
    R=pd.DataFrame(rows); d=pd.read_pickle('preds_dl_all.pkl')
    for nm,p in d['P'].items():
        if nm.endswith('__val') or 'PG-HQ' in nm: continue
        for h in HS: rows.append(dict(h=h,model=nm,**metrics(d['Y'][:,h-1],p[:,h-1])))
    fp=pickle.load(open('final_preds.pkl','rb'))
    for h in HS:
        T=fp['pred'][h]['test']
        for c in ['PG-HQ [past]','PG-HQ [feed]','PG-HQ ensemble [past]','PG-HQ ensemble [feed]']: rows.append(dict(h=h,model=c,**metrics(T.y.values,T[c].values)))
    R=pd.DataFrame(rows).drop_duplicates(['h','model']); pers=R[R.model=='Persistence'].set_index('h').RMSE; R['Skill']=1-R.RMSE/R.h.map(pers)
    R.to_csv('extra/metrics_all_horizons.csv',index=False)
if 'E2' in step:   # leakage A/B at all horizons + 24 h predictions
    rng=np.random.default_rng(0); allo=np.concatenate([o_tr,o_va,o_te]); perm=rng.permutation(allo); ntr=int(0.8*len(perm))
    rtr,rte=idx[np.sort(perm[:ntr])],idx[np.sort(perm[ntr:])]; rows=[]; P={}
    for h in HS:
        X=Xpast(h); yy=pd.Series(target(h).values,index=idx)
        for nm,a,b in [('chronological',tr,te),('random',rtr,rte)]:
            m=LGB().fit(X.loc[a],yy[a]); p=m.predict(X.loc[b]); rows.append(dict(h=h,split=nm,**metrics(yy[b].values,p)))
            if h==24: P[nm]=pd.DataFrame({'y':yy[b].values,'p':p,'ptr':np.nan},index=b); P[nm+'_train']=pd.DataFrame({'y':yy[a].values,'p':m.predict(X.loc[a])},index=a)
        print('E2',h,flush=True)
    pd.DataFrame(rows).to_csv('extra/leakage_all_horizons.csv',index=False); pickle.dump(P,open('extra/leakage_preds24.pkl','wb'))
if 'E3' in step:   # rolling origin at all horizons
    folds=[('2022-07-01','2022-08-15'),('2022-08-15','2022-10-01'),('2022-10-01','2022-11-15'),('2022-11-15','2023-01-01'),('2023-01-01','2023-02-28')]
    allo=np.sort(np.concatenate([o_tr,o_va,o_te])); ti=idx[allo]; rows=[]
    for h in HS:
        Xp=Xpast(h); Xf=pd.concat([Xp,future_feed(h)],axis=1); yy=pd.Series(target(h).values,index=idx); base=Xp['Biogas|L0']
        for k,(a,b) in enumerate(folds):
            trk=ti[(ti+pd.Timedelta(hours=24))<a]; tek=ti[(ti>=a)&(ti<b)]; yt=yy[tek].values
            rows.append(dict(h=h,fold=k+1,model='Persistence',**metrics(yt,base[tek].values)))
            rows.append(dict(h=h,fold=k+1,model='LightGBM',**metrics(yt,LGB().fit(Xp.loc[trk],yy[trk]).predict(Xp.loc[tek]))))
            for reg,XX in [('past',Xp),('feed',Xf)]:
                rows.append(dict(h=h,fold=k+1,model=f'LGB-Δ [{reg}]',**metrics(yt,LGB().fit(XX.loc[trk],(yy-base)[trk]).predict(XX.loc[tek])+base[tek].values)))
        print('E3',h,flush=True)
    R=pd.DataFrame(rows); P=R[R.model=='Persistence'].set_index(['h','fold']).RMSE; R['Skill']=1-R.RMSE/[P[(a,b)] for a,b in zip(R.h,R.fold)]
    R.to_csv('extra/rolling_all_horizons.csv',index=False)
# ---------------- DML helpers ----------------
def fw(s,a,b): return s[::-1].rolling(b-a).mean()[::-1].shift(-(a+1))
def nuis(kind):
    if kind=='LightGBM': return lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,verbose=-1,random_state=0,n_jobs=2)
    if kind=='Random forest': return RandomForestRegressor(150,min_samples_leaf=10,max_features=0.33,n_jobs=2,random_state=0)
    if kind=='CatBoost': return cb.CatBoostRegressor(iterations=400,learning_rate=0.05,depth=5,verbose=0,random_seed=0,thread_count=2)
    if kind=='Lasso': return make_pipeline(StandardScaler(),LassoCV(cv=3,alphas=20,max_iter=3000))
def dml(Y,T,X,h,K=5,buf=None,kind='LightGBM'):
    ok=Y.notna()&T.notna()&X.notna().all(1); Y,T,X=Y[ok],T[ok],X[ok]; n=len(Y); buf=h+168 if buf is None else buf
    ry=np.zeros(n); rt=np.zeros(n); e=np.linspace(0,n,K+1).astype(int)
    for k in range(K):
        a,b=e[k],e[k+1]; trn=np.r_[0:max(0,a-buf),min(n,b+buf):n]
        ry[a:b]=Y.values[a:b]-nuis(kind).fit(X.iloc[trn],Y.iloc[trn]).predict(X.iloc[a:b])
        rt[a:b]=T.values[a:b]-nuis(kind).fit(X.iloc[trn],T.iloc[trn]).predict(X.iloc[a:b])
    th=np.sum(rt*ry)/np.sum(rt*rt); psi=(ry-th*rt)*rt/np.mean(rt*rt); L=h+24
    g=[np.mean(psi[l:]*psi[:n-l]) for l in range(L+1)]; v=(g[0]+2*sum((1-l/(L+1))*g[l] for l in range(1,L+1)))/n
    return th,np.sqrt(v)
allo=np.sort(np.concatenate([o_tr,o_va,o_te])); ia=idx[allo]
def dml_inputs(h):
    Y=(pd.Series(target(h).values,index=idx)-H['Biogas']).loc[ia]; T=fw(hsw,0,h).loc[ia]
    X=pd.concat([X0,calendar(X0.index,h),fw(H['Q-PS_on'],0,h).rename('PS_plan'),fw(H['Q-TWAS_GPM'],0,h).rename('TWAS_plan')],axis=1).loc[ia]
    return Y,T,X
if 'E4' in step:   # DML robustness + refutations
    rows=[]
    for h in [3,24]:
        Y,T,X=dml_inputs(h)
        for kind in ['LightGBM','Random forest','CatBoost','Lasso']:
            t0=time.time(); th,se=dml(Y,T,X,h,kind=kind); rows.append(dict(h=h,check='Nuisance model',setting=kind,theta=th,se=se)); print('E4',h,kind,round(th,2),round(time.time()-t0),flush=True)
        for buf in [h+24,h+336]:
            th,se=dml(Y,T,X,h,buf=buf); rows.append(dict(h=h,check='Buffer length',setting=f'{buf} h',theta=th,se=se))
        for K in [3,10]:
            th,se=dml(Y,T,X,h,K=K); rows.append(dict(h=h,check='Cross-fitting folds',setting=f'K = {K}',theta=th,se=se))
        rc=pd.Series(np.random.default_rng(1).standard_normal(len(X)),index=X.index,name='rand')
        th,se=dml(Y,T,pd.concat([X,rc],axis=1),h); rows.append(dict(h=h,check='Refutation',setting='Random common cause',theta=th,se=se))
        n=len(Y)//2
        for nm,sl in [('First half',slice(0,n)),('Second half',slice(n,None))]:
            th,se=dml(Y.iloc[sl],T.iloc[sl],X.iloc[sl],h); rows.append(dict(h=h,check='Subsample',setting=nm,theta=th,se=se))
    pd.DataFrame(rows).to_csv('extra/dml_robustness.csv',index=False)
if 'E5' in step:   # 2-D ALE
    o=pickle.load(open('tab_final.pkl','rb')); out={}
    f1,f2='Q-HSW2_GPM|L0','H-Dig1_FT|L0'
    for h in [6,24]:
        m=o['models'][(h,'LGB-Δ [past]')]; X=Xpast(h).loc[tr]
        q1=np.unique(np.quantile(X[f1][X[f1]>0],np.linspace(0,1,11))); q1=np.r_[0,q1] if q1[0]>0 else q1
        q2=np.unique(np.quantile(X[f2],np.linspace(0,1,11)))
        b1=np.clip(np.searchsorted(q1,X[f1],side='right')-1,0,len(q1)-2); b2=np.clip(np.searchsorted(q2,X[f2],side='right')-1,0,len(q2)-2)
        D=np.zeros((len(q1)-1,len(q2)-1)); N=np.zeros_like(D)
        for i in range(len(q1)-1):
            for j in range(len(q2)-1):
                r=X[(b1==i)&(b2==j)]
                if len(r)==0: continue
                def P(a,b): z=r.copy(); z[f1]=a; z[f2]=b; return m.predict(z)
                D[i,j]=np.mean(P(q1[i+1],q2[j+1])-P(q1[i],q2[j+1])-P(q1[i+1],q2[j])+P(q1[i],q2[j])); N[i,j]=len(r)
        A=np.zeros((len(q1),len(q2))); A[1:,1:]=np.cumsum(np.cumsum(D,0),1)
        out[h]=dict(q1=q1,q2=q2,A=A,N=N); print('E5',h,flush=True)
    pickle.dump(out,open('extra/ale2d.pkl','wb'))
if 'E6' in step:   # train-set predictions for pred-obs panels (feed ensemble)
    import torch, dl
    o=pickle.load(open('tab_final.pkl','rb')); fp=pickle.load(open('final_preds.pkl','rb')); out={}
    Ps=[]
    for s in range(3):
        m=dl.PGHQ2(d=32,use_feed=True); m.load_state_dict(torch.load(f'pghq_feed_s{s}.pt')); Ps.append(dl.predict(m,'tr'))
    Ptr=np.mean(Ps,0)
    for h in [1,6,24]:
        Xf=pd.concat([Xpast(h),future_feed(h)],axis=1).loc[tr]; base=Xf['Biogas|L0'].values
        w=fp['ens'][(h,'feed')]['w']
        comp={'LGB-Δ [feed]':o['models'][(h,'LGB-Δ [feed]')].predict(Xf)+base,'Cat-Δ [feed]':o['models'][(h,'Cat-Δ [feed]')].predict(Xf)+base,'PG-HQ [feed]':Ptr[:,h-1]}
        out[h]=pd.DataFrame({'y':pd.Series(target(h).values,index=idx)[tr].values,'p':sum(w[k]*comp[k] for k in w)},index=tr)
    pickle.dump(out,open('extra/train_preds_feed_ens.pkl','wb')); print('E6 done')
if 'E7' in step:   # ablation ladder: raw lags (level)
    cols={}
    for v in VARS:
        s=H[v]
        for l in range(24): cols[f'{v}|lag{l}']=s.shift(l)
        for w in [6,24,168]: cols[f'{v}|rm{w}']=s.rolling(w).mean(); cols[f'{v}|rs{w}']=s.rolling(w).std()
    XR=pd.DataFrame(cols,index=idx); rows=[]
    for h in [1,6,24]:
        yy=pd.Series(target(h).values,index=idx); XX=pd.concat([XR,calendar(idx,h)],axis=1)
        rows.append(dict(h=h,step='Raw lags (level)',**metrics(yy[te].values,LGB().fit(XX.loc[tr],yy[tr]).predict(XX.loc[te]))))
    pd.DataFrame(rows).to_csv('extra/ablation_rawlevel.csv',index=False); print('E7 done')
if 'E8' in step:   # scenario response vs DML + ops metrics + erank null
    o=pickle.load(open('tab_final.pkl','rb')); rows=[]
    for h in HS:
        m=o['models'][(h,'LGB-Δ [feed]')]; X=pd.concat([Xpast(h),future_feed(h)],axis=1).loc[te]
        for f in [0.75,1.25]:
            Z=X.copy(); Z['Planned HSW|F']=Z['Planned HSW|F']*f; dT=(Z['Planned HSW|F']-X['Planned HSW|F']).values
            dp=m.predict(Z)-m.predict(X); s=np.abs(dT)>0.5
            rows.append(dict(h=h,scenario=f'{int(round((f-1)*100)):+d}%',resp_per_gpm=np.median(dp[s]/dT[s]),q25=np.percentile(dp[s]/dT[s],25),q75=np.percentile(dp[s]/dT[s],75)))
    pd.DataFrame(rows).to_csv('extra/scenario_response.csv',index=False)
    hr=pickle.load(open('hrpga.pkl','rb'))
    def erank(A):
        A=np.nan_to_num(np.clip(A,0,None)); A=A/(A.sum(1,keepdims=True)+1e-12); s=np.linalg.svd(A,compute_uv=False); p=s/s.sum(); return float(np.exp(-(p*np.log(p+1e-12)).sum()))
    Ab=np.nan_to_num(hr['past']['Ab']); rng=np.random.default_rng(0); null=[]
    for _ in range(2000):
        B=Ab.copy()
        for j in range(B.shape[1]): B[:,j]=rng.permutation(B[:,j])
        null.append(erank(B))
    pd.DataFrame({'null_erank':null}).assign(observed=erank(Ab)).to_csv('extra/erank_null.csv',index=False)
    print('E8 done; erank',erank(Ab),'null mean',np.mean(null),'p(lower)',np.mean(np.array(null)<=erank(Ab)))
