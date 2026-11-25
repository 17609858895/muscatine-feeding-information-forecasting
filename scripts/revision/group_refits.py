"""Group-removal refits of historical-input LGB-Δ (27 fixed-configuration refits; Table S4).

Extracted unchanged from the refit section of the original export_models_and_loco.py;
the legacy interface-model export in that file is omitted because the current interface
models are produced by rebuild_interface.py. Outputs: results/group_refits/.
"""
from pathlib import Path
import os, sys
import numpy as np,pandas as pd,lightgbm as lgb
PKG=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get('MUSCATINE_RESULTS',PKG/'results')).resolve()
OUTDIR=DATA/'group_refits';OUTDIR.mkdir(parents=True,exist_ok=True)
os.chdir(DATA);sys.path.insert(0,str(PKG/'scripts'/'pipeline'))
from common import H,build,calendar,future_feed,target,idx,o_tr,o_va,o_te,GROUPS,metrics
import pickle
HORIZONS=[1,2,3,4,6,9,12,18,24]
out=pickle.load(open('tab_final.pkl','rb')); X0,_=build(); base=H.Biogas
tr,va,te=idx[o_tr],idx[o_va],idx[o_te]
rows=[]
for h in HORIZONS:
    Xp=pd.concat([X0,calendar(X0.index,h)],axis=1);yy=target(h); full=out['pred'][h]['test'][f'LGB-Δ [past]'].values;yv=yy[te].values
    for g in ['Gas storage','Co-substrate feed','Calendar']:
        drop=[c for c in Xp if any(c.startswith(v+'|') for v in GROUPS.get(g,[]))] if g!='Calendar' else ['hod_sin','hod_cos','dow_sin','dow_cos']
        XX=Xp.drop(columns=drop);m=lgb.LGBMRegressor(n_estimators=800,learning_rate=.02,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.6,reg_lambda=2,verbose=-1,random_state=0,n_jobs=1)
        m.fit(XX.loc[tr],(yy-base)[tr]);p=m.predict(XX.loc[te])+base[te].values
        rmse=metrics(yv,p)['RMSE'];rmfull=metrics(yv,full)['RMSE']; diff=(rmse-rmfull)*1.69901
        rows.append({'h':h,'removed_group':g,'removed_features':len(drop),'n_test':len(te),'full_RMSE_m3_h':rmfull*1.69901,'retrained_RMSE_m3_h':rmse*1.69901,'delta_RMSE_m3_h':diff,'retrained_R2':metrics(yv,p)['R2']})
        print('LOCO',h,g,round(diff,3),flush=True)
        np.savez_compressed(OUTDIR/f'loco_{h}_{g.replace(" ","_")}.npz',prediction=p*1.69901,origin=te.astype(str).values,y=yv*1.69901)
    pd.DataFrame(rows).to_csv(OUTDIR/'loco_process_groups.csv',index=False)
print('Completed 27 fixed-configuration retraining controls',flush=True)
