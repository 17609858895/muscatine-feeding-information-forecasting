from pathlib import Path
"""Analyses behind the supplementary figures (outputs in supp/)."""
import pandas as pd, numpy as np, pickle, os, sys, time, warnings; warnings.filterwarnings('ignore')
from common import *
import lightgbm as lgb, xgboost as xgb, optuna
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
optuna.logging.set_verbosity(optuna.logging.WARNING)
X0,meta=build(); HS=[1,2,3,4,6,9,12,18,24]
tr,va,te=idx[o_tr],idx[o_va],idx[o_te]
def Xpast(h): return pd.concat([X0,calendar(X0.index,h)],axis=1)
def Y(h): return pd.Series(target(h).values,index=idx)
LGBp=dict(n_estimators=800,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=2)
LGB=lambda **k: lgb.LGBMRegressor(**{**LGBp,**k},verbose=-1,random_state=0,n_jobs=2)
steps=os.environ.get('ONLY','A,B,C,D,E,F,G,H,I').split(',')
rmse=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
# A. Optuna (validation RMSE; test never used)
if 'A' in steps:
    out=[]
    def run(name,h,make,delta,n=40):
        X=Xpast(h); y=Y(h); base=X['Biogas|L0']
        def obj(trial):
            m=make(trial); tgt=(y-base) if delta else y
            m.fit(X.loc[tr],tgt[tr]); p=m.predict(X.loc[va])+(base[va].values if delta else 0); return rmse(y[va].values,p)
        st=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=0)); t0=time.time(); st.optimize(obj,n_trials=n,callbacks=[lambda s_,t_: print('  trial',name,h,t_.number,round(t_.value,2),round(time.time()-t0),flush=True)])
        imp=optuna.importance.get_param_importances(st)
        for t in st.trials: out.append(dict(model=name,h=h,trial=t.number,value=t.value,**{f'p_{k}':v for k,v in t.params.items()}))
        pickle.dump(dict(best=st.best_params,imp=imp),open(f'supp/optuna_{name}_{h}.pkl','wb')); print('A',name,h,round(st.best_value,2),round(time.time()-t0),flush=True)
    run('LGB-Δ',6,lambda t: lgb.LGBMRegressor(n_estimators=t.suggest_int('n_estimators',200,1000,step=100),learning_rate=t.suggest_float('learning_rate',0.005,0.1,log=True),num_leaves=t.suggest_int('num_leaves',7,47),min_child_samples=t.suggest_int('min_child_samples',10,120),colsample_bytree=t.suggest_float('colsample_bytree',0.3,1.0),reg_lambda=t.suggest_float('reg_lambda',1e-3,20,log=True),subsample=0.8,subsample_freq=1,verbose=-1,random_state=0,n_jobs=2),True,25)
    run('LGB-Δ',24,lambda t: lgb.LGBMRegressor(n_estimators=t.suggest_int('n_estimators',200,1000,step=100),learning_rate=t.suggest_float('learning_rate',0.005,0.1,log=True),num_leaves=t.suggest_int('num_leaves',7,47),min_child_samples=t.suggest_int('min_child_samples',10,120),colsample_bytree=t.suggest_float('colsample_bytree',0.3,1.0),reg_lambda=t.suggest_float('reg_lambda',1e-3,20,log=True),subsample=0.8,subsample_freq=1,verbose=-1,random_state=0,n_jobs=2),True,25)
    run('XGBoost',24,lambda t: xgb.XGBRegressor(n_estimators=t.suggest_int('n_estimators',200,1000,step=100),learning_rate=t.suggest_float('learning_rate',0.005,0.1,log=True),max_depth=t.suggest_int('max_depth',2,6),min_child_weight=t.suggest_float('min_child_weight',1,50,log=True),colsample_bytree=t.suggest_float('colsample_bytree',0.3,1.0),subsample=0.8,random_state=0,n_jobs=2),False,20)
    run('SVR',6,lambda t: make_pipeline(StandardScaler(),SVR(C=t.suggest_float('C',0.1,300,log=True),epsilon=t.suggest_float('epsilon',0.1,10,log=True),gamma=t.suggest_float('gamma',1e-4,0.05,log=True))),False,20)
    run('KNN',6,lambda t: make_pipeline(StandardScaler(),KNeighborsRegressor(t.suggest_int('n_neighbors',5,150,log=True),weights=t.suggest_categorical('weights',['uniform','distance']),p=t.suggest_categorical('p',[1,2]))),False,30)
    pd.DataFrame(out).to_csv('supp/optuna_trials.csv',index=False)
# B. refit tuned SVR/KNN at all horizons
if 'B' in steps:
    bs=pickle.load(open('supp/optuna_SVR_6.pkl','rb'))['best']; bk=pickle.load(open('supp/optuna_KNN_6.pkl','rb'))['best']; rows=[]
    for h in HS:
        X=Xpast(h); y=Y(h)
        for nm,m in [('SVR (tuned)',make_pipeline(StandardScaler(),SVR(**bs))),('KNN (tuned)',make_pipeline(StandardScaler(),KNeighborsRegressor(**bk)))]:
            m.fit(X.loc[tr],y[tr]); rows.append(dict(h=h,model=nm,**metrics(y[te].values,m.predict(X.loc[te]))))
        print('B',h,flush=True)
    pd.DataFrame(rows).to_csv('supp/tuned_svr_knn.csv',index=False)
# C. learning curves (training length, months before the fixed end of training)
if 'C' in steps:
    rows=[]
    for h in [1,6,24]:
        X=Xpast(h); y=Y(h); base=X['Biogas|L0']
        for months in [1,2,3,4,5,6,7,8]:
            start=pd.Timestamp(tr[-1])-pd.DateOffset(months=months); trk=tr[tr>=start]
            m=LGB().fit(X.loc[trk],(y-base)[trk])
            rows.append(dict(h=h,months=months,n=len(trk),R2_train=metrics(y[trk].values,m.predict(X.loc[trk])+base[trk].values)['R2'],R2_val=metrics(y[va].values,m.predict(X.loc[va])+base[va].values)['R2'],R2_test=metrics(y[te].values,m.predict(X.loc[te])+base[te].values)['R2']))
        print('C',h,flush=True)
    pd.DataFrame(rows).to_csv('supp/learning_curves.csv',index=False)
# D. deep-model training histories (one seed)
if 'D' in steps:
    import torch, dl
    hist={}
    def train_log(model,lr,epochs=80,patience=12):
        torch.manual_seed(0); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        Xtr,Ytr,Btr,Ctr,Ftr=dl.D['tr']; tgt=(Ytr-Btr[:,None])/dl.ysd; H_=[]; best=1e9; bad=0
        for ep in range(epochs):
            model.train(); perm=torch.randperm(len(Xtr)); tl=[]
            for i in range(0,len(perm),128):
                j=perm[i:i+128]; opt.zero_grad(); l=torch.nn.functional.smooth_l1_loss(model(Xtr[j],Ctr[j],Ftr[j]),tgt[j],beta=0.5); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl.append(l.item())
            v=dl.evalp(model,'va'); H_.append((ep,np.mean(tl),v))
            if v<best-1e-4: best=v; bad=0
            else:
                bad+=1
                if bad>=patience: break
        return H_
    for nm,mk,lr in [('PG-HQ (past)',lambda: dl.PGHQ2(d=32,use_feed=False),1e-3),('PG-HQ (feed)',lambda: dl.PGHQ2(d=32,use_feed=True),1e-3),('DLinear',dl.DLinear,5e-4),('TCN',dl.TCN,5e-4),('PatchTST',dl.PatchTST,5e-4)]:
        t0=time.time(); hist[nm]=train_log(mk(),lr); print('D',nm,len(hist[nm]),round(time.time()-t0),flush=True)
    pickle.dump(hist,open('supp/dl_histories.pkl','wb'))
# E. conditional permutation (within Biogas-L0 quintiles) vs marginal, group level, past LGB-Δ
if 'E' in steps:
    o=pickle.load(open('tab_final.pkl','rb')); rng=np.random.default_rng(0); rows=[]
    for h in [1,6,24]:
        m=o['models'][(h,'LGB-Δ [past]')]; X=Xpast(h).loc[te]; base=X['Biogas|L0'].values; yt=Y(h)[te].values
        r0=rmse(yt,m.predict(X)+base); strata=pd.qcut(X['Biogas|L0'],5,labels=False).values
        for g,vs in list(GROUPS.items())+[('Calendar',None)]:
            cols=[c for c in X.columns if c.split('|')[0] in vs] if vs else ['hod_sin','hod_cos','dow_sin','dow_cos']
            for mode in ['marginal','conditional']:
                d=[]
                for r in range(5):
                    Z=X.copy()
                    if mode=='marginal': Z[cols]=Z[cols].values[rng.permutation(len(Z))]
                    else:
                        V=Z[cols].values.copy()
                        for s in range(5):
                            ii=np.where(strata==s)[0]; V[ii]=V[rng.permutation(ii)]
                        Z[cols]=V
                    d.append(rmse(yt,m.predict(Z)+base)-r0)
                rows.append(dict(h=h,group=g,mode=mode,dRMSE=np.mean(d)))
        print('E',h,flush=True)
    pd.DataFrame(rows).to_csv('supp/conditional_permutation.csv',index=False)
# F. lag-bin design sensitivity: equal-width bins
if 'F' in steps:
    cols={}
    for v in VARS:
        s=H[v]
        for k in range(8): cols[f'{v}|E{k}']=s.shift(21*k).rolling(21).mean()
    XE=pd.DataFrame(cols,index=idx); rows=[]
    for h in [1,6,24]:
        y=Y(h); base=H['Biogas']; XX=pd.concat([XE,calendar(idx,h)],axis=1)
        m=LGB().fit(XX.loc[tr],(y-base)[tr]); rows.append(dict(h=h,design='Equal-width bins (8 × 21 h)',**metrics(y[te].values,m.predict(XX.loc[te])+base[te].values)))
    A=pd.read_csv('ablation_features.csv'); A['design']=A.features.map({'raw lags + rolling stats':'Raw lags + rolling statistics','process-group lag-bins':'Log-spaced bins (PG-LB)'})
    pd.concat([pd.DataFrame(rows),A[['h','design','RMSE','MAE','R2']]]).to_csv('supp/lagbin_design.csv',index=False); print('F done')
# G. next-24-h mean target benchmark
if 'G' in steps:
    rows=[]; ym=pd.Series(H['Biogas'][::-1].rolling(24).mean()[::-1].shift(-1).values,index=idx)
    X=Xpast(24); Xf=pd.concat([X,future_feed(24)],axis=1); base=X['Biogas|L0']; r24=H['Biogas'].rolling(24).mean()
    P={'Persistence (last hour)':base[te].values,'Persistence (last 24 h mean)':r24[te].values}
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    import catboost as cb
    for nm,m,XX,d in [('Ridge',make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))),X,False),('Random forest',RandomForestRegressor(300,min_samples_leaf=5,max_features=0.33,n_jobs=2,random_state=0),X,False),
                      ('LightGBM',LGB(),X,False),('XGBoost',xgb.XGBRegressor(n_estimators=800,learning_rate=0.02,max_depth=4,min_child_weight=10,subsample=0.8,colsample_bytree=0.6,random_state=0,n_jobs=2),X,False),
                      ('CatBoost',cb.CatBoostRegressor(iterations=1200,learning_rate=0.03,depth=5,verbose=0,random_seed=0,thread_count=2),X,False),('LGB-Δ [past]',LGB(),X,True),('LGB-Δ [feed]',LGB(),Xf,True)]:
        tgt=(ym-r24) if d else ym; m.fit(XX.loc[tr],tgt[tr]); P[nm]=m.predict(XX.loc[te])+(r24[te].values if d else 0)
    for k,p in P.items(): rows.append(dict(model=k,**metrics(ym[te].values,p)))
    pd.DataFrame(rows).to_csv('supp/next24_mean_benchmark.csv',index=False); print(pd.DataFrame(rows).round(3))
# H. conformal variants: split, normalised, ACI (online), CQR (LightGBM quantiles)
if 'H' in steps:
    fp=pickle.load(open('final_preds.pkl','rb')); rows=[]; a=0.1
    for h in HS:
        V=fp['pred'][h]['val']; T=fp['pred'][h]['test']; key='PG-HQ ensemble [feed]'
        rv=np.abs(V.y-V[key]).values; rt=np.abs(T.y-T[key]).values; n=len(rv); ql=min(1,np.ceil((n+1)*(1-a))/n)
        q=np.quantile(rv,ql); rows.append(dict(h=h,method='Split',coverage=np.mean(rt<=q),width=2*q))
        vol=H['Biogas'].rolling(24).std()+1; vv=vol.reindex(V.index).values; vt=vol.reindex(T.index).values
        qn=np.quantile(rv/vv,ql); rows.append(dict(h=h,method='Normalised',coverage=np.mean(rt<=qn*vt),width=np.mean(2*qn*vt)))
        from aci import aci_halfwidths
        qa=aci_halfwidths(V.y.values-V[key].values,T.y.values-T[key].values,h,alpha=a); cov=rt<=qa; wid=2*qa
        rows.append(dict(h=h,method='ACI',coverage=np.mean(cov),width=np.mean(wid)))
        # CQR
        X=pd.concat([Xpast(h),future_feed(h)],axis=1); y=Y(h); base=X['Biogas|L0']
        lo=lgb.LGBMRegressor(objective='quantile',alpha=0.05,**LGBp,verbose=-1,random_state=0,n_jobs=2).fit(X.loc[tr],(y-base)[tr]); hi=lgb.LGBMRegressor(objective='quantile',alpha=0.95,**LGBp,verbose=-1,random_state=0,n_jobs=2).fit(X.loc[tr],(y-base)[tr])
        lv,hv=lo.predict(X.loc[va])+base[va].values,hi.predict(X.loc[va])+base[va].values; s=np.maximum(lv-y[va].values,y[va].values-hv); qc=np.quantile(s,ql)
        lt,ht=lo.predict(X.loc[te])+base[te].values-qc,hi.predict(X.loc[te])+base[te].values+qc; yt=y[te].values
        rows.append(dict(h=h,method='CQR',coverage=np.mean((yt>=lt)&(yt<=ht)),width=np.mean(ht-lt)))
        print('H',h,flush=True)
    pd.DataFrame(rows).to_csv('supp/conformal_methods.csv',index=False)
# I. LAB merge + error stratification by FOS/TAC
if 'I' in steps:
    L=pd.read_csv(os.environ.get('LABS_CSV',str(Path(__file__).resolve().parents[2]/'data/LABS-raw.csv')),encoding='utf-8-sig')
    L['date']=pd.to_datetime(L[['Year','Month','Day']]); L=L.set_index('date')
    L.to_pickle('supp/labs.pkl')
    fp=pickle.load(open('final_preds.pkl','rb')); rows=[]
    for h in [1,6,24]:
        T=fp['pred'][h]['test'].copy(); T['date']=(T.index+pd.Timedelta(hours=h)).normalize()-pd.Timedelta(days=1)  # lab result of previous day (available)
        ft=L[['Dig1-FOS-TAC','Dig2-FOS-TAC']].mean(1); T['fostac']=T['date'].map(ft)
        T['ae']=np.abs(T.y-T['PG-HQ ensemble [feed]']); T['ae_p']=np.abs(T.y-T['PG-HQ ensemble [past]'])
        T=T.dropna(subset=['fostac']); T['q']=pd.qcut(T.fostac,4,labels=['Q1','Q2','Q3','Q4'])
        for qn,g in T.groupby('q'): rows.append(dict(h=h,quartile=qn,fostac_lo=g.fostac.min(),fostac_hi=g.fostac.max(),mae_feed=g.ae.mean(),mae_past=g.ae_p.mean(),n=len(g)))
    pd.DataFrame(rows).to_csv('supp/error_by_fostac.csv',index=False); print('I done')
# J. CATE by lid-height tertile with moving-block bootstrap CIs
if 'J' in steps:
    LG=lambda: lgb.LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,verbose=-1,random_state=0,n_jobs=2)
    fw=lambda s,a,b: s[::-1].rolling(b-a).mean()[::-1].shift(-(a+1))
    allo=np.sort(np.concatenate([o_tr,o_va,o_te])); ia=idx[allo]; rows=[]; rng=np.random.default_rng(0)
    for h in HS:
        Yv=(Y(h)-H['Biogas']).loc[ia]; T=fw(hsw,0,h).loc[ia]
        X=pd.concat([X0,calendar(X0.index,h),fw(H['Q-PS_on'],0,h).rename('PS_plan'),fw(H['Q-TWAS_GPM'],0,h).rename('TWAS_plan')],axis=1).loc[ia]
        ok=Yv.notna()&T.notna()&X.notna().all(1); Yv,T,X=Yv[ok],T[ok],X[ok]; n=len(Yv); buf=h+168; ry=np.zeros(n); rt=np.zeros(n); e=np.linspace(0,n,6).astype(int)
        for k in range(5):
            a,b=e[k],e[k+1]; trn=np.r_[0:max(0,a-buf),min(n,b+buf):n]
            ry[a:b]=Yv.values[a:b]-LG().fit(X.iloc[trn],Yv.iloc[trn]).predict(X.iloc[a:b]); rt[a:b]=T.values[a:b]-LG().fit(X.iloc[trn],T.iloc[trn]).predict(X.iloc[a:b])
        tq=pd.qcut(X['H-Dig1_FT|L0'],3,labels=['low','mid','high']).values
        for lv in ['low','mid','high']:
            idx_l=np.where(tq==lv)[0]; th=np.sum(rt[idx_l]*ry[idx_l])/np.sum(rt[idx_l]**2)
            B=168; nb=int(np.ceil(n/B)); boots=[]
            for r in range(300):
                st=rng.integers(0,n-B,nb); ii=np.concatenate([np.arange(s,s+B) for s in st])[:n]; m=tq[ii]==lv
                boots.append(np.sum(rt[ii][m]*ry[ii][m])/np.sum(rt[ii][m]**2))
            rows.append(dict(h=h,storage=lv,theta=th,lo=np.percentile(boots,2.5),hi=np.percentile(boots,97.5)))
        print('J',h,flush=True)
    pd.DataFrame(rows).to_csv('supp/cate_storage_bootstrap.csv',index=False)
