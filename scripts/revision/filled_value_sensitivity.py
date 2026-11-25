"""Retain all events; separately evaluate exclusions of filled target values."""
from context import *
from statistics_utils import *
import copy
raw=pd.read_pickle(RES/'hourly.pkl');models=pd.read_pickle(RES/'tab_final.pkl')['models']
rows=[];affected=[];preds={}
for h in HS:
    for name,orig in [('training',TR),('selection',SELECT),('calibration',CAL),('test',TE)]:
        missing=raw.Biogas.reindex(orig+pd.Timedelta(hours=h)).isna().values
        for t in orig[missing]:
            affected.append(dict(h=h,split=name,origin=t,target=t+pd.Timedelta(hours=h),type='filled_target'))
        for t in orig:
            window=raw.loc[t+pd.Timedelta(hours=1):t+pd.Timedelta(hours=h),['Q-HSW1_GPM','Q-HSW2_GPM','Q-PS_on','Q-TWAS_GPM']]
            if window.isna().any().any():
                affected.append(dict(h=h,split=name,origin=t,target=t+pd.Timedelta(hours=h),type='filled_future_feed_window',variables=','.join(window.columns[window.isna().any()])))
    missing_train=raw.Biogas.reindex(TR+pd.Timedelta(hours=h)).isna().values
    missing_test=raw.Biogas.reindex(TE+pd.Timedelta(hours=h)).isna().values
    if not(missing_train.any() or missing_test.any()):continue
    for reg in ['past','feed']:
        x=features(h,reg);y=target(h);base=x['Biogas|L0'];keep_train=TR[~missing_train]
        for model_name in ['LGB-Δ','Cat-Δ']:
            original=models[h,f'{model_name} [{reg}]'];model=copy.deepcopy(original)
            model.fit(x.loc[keep_train],(y-base).loc[keep_train]);p=model.predict(x.loc[TE])+base.loc[TE].values
            original_p=original.predict(x.loc[TE])+base.loc[TE].values
            preds[h,reg,model_name]=p
            for variant,ypred,mask in [('original-all',original_p,np.ones(len(TE),bool)),('original-observed-targets',original_p,~missing_test),('refit-observed-targets',p,~missing_test)]:
                rows.append(dict(h=h,regime=reg,model=model_name,variant=variant,n_train=len(TR) if variant.startswith('original') else len(keep_train),n_test=int(mask.sum()),**metric(y.loc[TE].values[mask],ypred[mask])))
            print('filled sensitivity',h,reg,model_name,flush=True)
frame('filled_input_inventory.csv',affected);frame('filled_target_sensitivity.csv',rows);pd.to_pickle(preds,OUT/'filled_target_refits.pkl')
print('FILLED TARGET SENSITIVITY COMPLETE',flush=True)
