import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings('ignore')
from feats import *
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb, xgboost as xgb, catboost as cb
X0,meta=build()
def metr(y,p):
    e=y-p; rmse=np.sqrt(np.mean(e**2)); mae=np.mean(np.abs(e)); r2=1-np.sum(e**2)/np.sum((y-y.mean())**2)
    return rmse,mae,r2
rows=[]; preds={}
for h in HORIZONS:
    X=pd.concat([X0,calendar(X0.index,h)],axis=1); y=target(h)
    ok=X.notna().all(1)&y.notna(); X=X[ok]; y=y[ok]
    tr,va,te=split_idx(X.index)
    trva=tr|va
    yte=y[te].values
    P={}
    P['Persistence']=X.loc[te,'Biogas|L0'].values
    P['Seasonal naive (24 h)']=H['Biogas'].shift(24-h).reindex(X.index)[te].values if h<=24 else None
    P['Moving average (24 h)']=X.loc[te,'Biogas|L12-23'].values*0.5+X.loc[te,'Biogas|L6-11'].values*0.25+X.loc[te,'Biogas|L3-5'].values*0.125+X.loc[te,'Biogas|L1-2'].values*(1/12)+X.loc[te,'Biogas|L0'].values*(1/24)
    ar_cols=[c for c in X.columns if c.startswith('Biogas|')]+['hod_sin','hod_cos']
    m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))).fit(X.loc[trva,ar_cols],y[trva]); P['AR-Ridge (gas only)']=m.predict(X.loc[te,ar_cols])
    models={
     'Ridge':make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))),
     'SVR':make_pipeline(StandardScaler(),SVR(C=30,epsilon=2,gamma='scale')),
     'KNN':make_pipeline(StandardScaler(),KNeighborsRegressor(25,weights='distance')),
     'Random forest':RandomForestRegressor(400,min_samples_leaf=5,max_features=0.33,n_jobs=-1,random_state=0),
     'Extra trees':ExtraTreesRegressor(400,min_samples_leaf=5,max_features=0.5,n_jobs=-1,random_state=0),
     'HistGB':HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.0,random_state=0),
     'LightGBM':lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=2,verbose=-1,random_state=0),
     'XGBoost':xgb.XGBRegressor(n_estimators=800,learning_rate=0.02,max_depth=4,min_child_weight=10,subsample=0.8,colsample_bytree=0.6,reg_lambda=2,random_state=0,n_jobs=2),
     'CatBoost':cb.CatBoostRegressor(iterations=1200,learning_rate=0.03,depth=5,l2_leaf_reg=5,verbose=0,random_seed=0,thread_count=2),
    }
    for nm,m in models.items():
        t0=time.time(); m.fit(X[trva],y[trva]); P[nm]=m.predict(X[te]); 
        ptr=m.predict(X[tr]); r2tr=metr(y[tr].values,ptr)[2]
        print(h,nm,round(time.time()-t0,1),'s train R2',round(r2tr,3),flush=True)
    for nm,p in P.items():
        rmse,mae,r2=metr(yte,p); rows.append(dict(h=h,model=nm,RMSE=rmse,MAE=mae,R2=r2))
    preds[h]=pd.DataFrame(P,index=X.index[te]).assign(y=yte)
R=pd.DataFrame(rows)
pers=R[R.model=='Persistence'].set_index('h').RMSE
R['Skill_vs_persist']=1-R.RMSE/R.h.map(pers)
R.to_csv('bench_tab.csv',index=False)
pd.to_pickle(preds,'preds_tab.pkl')
print(R.pivot(index='model',columns='h',values=['R2','Skill_vs_persist']).round(3).to_string())
