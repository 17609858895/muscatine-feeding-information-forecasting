import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from common import *
import lightgbm as lgb
X0,_=build()
LGB=lambda: lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=2,verbose=-1,random_state=0,n_jobs=1)
folds=[('2022-07-01','2022-08-15'),('2022-08-15','2022-10-01'),('2022-10-01','2022-11-15'),('2022-11-15','2023-01-01'),('2023-01-01','2023-02-28')]
allo=np.sort(np.concatenate([o_tr,o_va,o_te]))
rows=[]
for h in [1,6,24]:
    Xp=pd.concat([X0,calendar(X0.index,h)],axis=1); Xf=pd.concat([Xp,future_feed(h)],axis=1); yy=pd.Series(target(h).values,index=idx); base=Xp['Biogas|L0']
    for k,(a,b) in enumerate(folds):
        ti=idx[allo]; tr=ti[(ti+pd.Timedelta(hours=24))<a]; te=ti[(ti>=a)&(ti<b)]
        yt=yy[te].values
        rows.append(dict(h=h,fold=k+1,start=a,model='Persistence',**metrics(yt,base[te].values)))
        m=LGB().fit(Xp.loc[tr],yy[tr]); rows.append(dict(h=h,fold=k+1,start=a,model='LightGBM',**metrics(yt,m.predict(Xp.loc[te]))))
        for reg,XX in [('past',Xp),('feed',Xf)]:
            m=LGB().fit(XX.loc[tr],(yy-base)[tr]); rows.append(dict(h=h,fold=k+1,start=a,model=f'LGB-Δ [{reg}]',**metrics(yt,m.predict(XX.loc[te])+base[te].values)))
        print(h,k,flush=True)
R=pd.DataFrame(rows); R.to_csv('rolling_origin.csv',index=False)
print(R.pivot_table(index=['model'],columns=['h','fold'],values='R2').round(2).to_string())
