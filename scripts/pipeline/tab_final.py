import pandas as pd, numpy as np, time, warnings, pickle; warnings.filterwarnings('ignore')
from common import *
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb, xgboost as xgb, catboost as cb
X0,meta=build(); meta.to_csv('feature_meta.csv',index=False)
LGB=lambda: lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=2,verbose=-1,random_state=0,n_jobs=1)
CAT=lambda: cb.CatBoostRegressor(iterations=1200,learning_rate=0.03,depth=5,l2_leaf_reg=5,verbose=0,random_seed=0,thread_count=1)
BASE={
 'Ridge':lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))),
 'SVR':lambda: make_pipeline(StandardScaler(),SVR(C=30,epsilon=2,gamma='scale')),
 'KNN':lambda: make_pipeline(StandardScaler(),KNeighborsRegressor(25,weights='distance')),
 'Random forest':lambda: RandomForestRegressor(400,min_samples_leaf=5,max_features=0.33,n_jobs=1,random_state=0),
 'Extra trees':lambda: ExtraTreesRegressor(400,min_samples_leaf=5,max_features=0.5,n_jobs=1,random_state=0),
 'HistGB':lambda: HistGradientBoostingRegressor(max_iter=600,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.0,random_state=0),
 'LightGBM':LGB,
 'XGBoost':lambda: xgb.XGBRegressor(n_estimators=800,learning_rate=0.02,max_depth=4,min_child_weight=10,subsample=0.8,colsample_bytree=0.6,reg_lambda=2,random_state=0,n_jobs=1),
 'CatBoost':CAT,
}
HS_ALL=[1,2,3,4,6,9,12,18,24]
out={'rows':[],'pred':{},'models':{}}
for h in HS_ALL:
    Xp=pd.concat([X0,calendar(X0.index,h)],axis=1)
    Xf=pd.concat([Xp,future_feed(h)],axis=1)
    yy=pd.Series(target(h).values,index=idx)
    tr,va,te=idx[o_tr],idx[o_va],idx[o_te]
    base=Xp['Biogas|L0']
    P={};V={}
    P['Persistence']=base[te].values; V['Persistence']=base[va].values
    if h in (1,6,24):
        P['Seasonal naive (24 h)']=H['Biogas'].shift(24-h)[te].values
        ar=[c for c in Xp.columns if c.startswith('Biogas|')]+['hod_sin','hod_cos']
        m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))).fit(Xp.loc[tr,ar],yy[tr]); P['AR-Ridge (gas only)']=m.predict(Xp.loc[te,ar])
        for nm,f in BASE.items():
            t0=time.time(); m=f().fit(Xp.loc[tr],yy[tr]); P[nm]=m.predict(Xp.loc[te]); 
            out['rows'].append(dict(h=h,model=nm,regime='past-only',train_R2=metrics(yy[tr].values,m.predict(Xp.loc[tr]))['R2'],**{('test_'+k):v for k,v in metrics(yy[te].values,P[nm]).items()}))
            print(h,nm,round(time.time()-t0,1),flush=True)
        for nm in ['LightGBM','CatBoost','Ridge']:   # feed-conditioned versions of strong baselines (same information as proposed)
            m=BASE[nm]().fit(Xf.loc[tr],yy[tr]); P[nm+' +feed']=m.predict(Xf.loc[te])
    # proposed tabular components: persistence-anchored delta on process-group lag-bin features
    for reg,XX in [('past',Xp),('feed',Xf)]:
        for nm,f in [('LGB-Δ',LGB),('Cat-Δ',CAT)]:
            m=f().fit(XX.loc[tr],(yy-base)[tr]); key=f'{nm} [{reg}]'
            P[key]=m.predict(XX.loc[te])+base[te].values; V[key]=m.predict(XX.loc[va])+base[va].values
            out['models'][(h,key)]=m
            out['rows'].append(dict(h=h,model=key,regime=reg,train_R2=metrics(yy[tr].values,m.predict(XX.loc[tr])+base[tr].values)['R2'],**{('test_'+k):v for k,v in metrics(yy[te].values,P[key]).items()}))
        print(h,reg,'Δ done',flush=True)
    out['pred'][h]=dict(test=pd.DataFrame(P,index=te).assign(y=yy[te].values),val=pd.DataFrame(V,index=va).assign(y=yy[va].values))
pickle.dump(out,open('tab_final.pkl','wb'))
print('done')
