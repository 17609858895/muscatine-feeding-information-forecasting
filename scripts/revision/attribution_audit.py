"""Separate correction-only sensitivity from the consistent complete forecast."""
from context import *
from statistics_utils import *
import lightgbm as lgb

models=pd.read_pickle(RES/'tab_final.pkl')['models'];R=20
groups={**GROUPS,'Calendar':[],'Planned feed':[]}
rows=[];lagrows=[];locorows=[];binrows=[]

def columns(x,g):
    if g=='Calendar':return [c for c in x if c.endswith(('_sin','_cos'))]
    if g=='Planned feed':return [c for c in x if c.endswith('|F')]
    return [c for c in x if c.split('|')[0] in GROUPS[g]]

def evaluate(model,x,y,cols,seed):
    base=x['Biogas|L0'].values;reference=np.sqrt(np.mean((model.predict(x)+base-y)**2))
    gas='Biogas|L0' in cols;values={'correction':[],'complete':[]}
    rng=np.random.default_rng(seed)
    for rep in range(R):
        ix=rng.permutation(len(x));z=x.copy();z[cols]=z[cols].values[ix];delta=model.predict(z)
        values['correction'].append((np.sqrt(np.mean((delta+base-y)**2))-reference)*CFM)
        fullbase=base[ix] if gas else base
        values['complete'].append((np.sqrt(np.mean((delta+fullbase-y)**2))-reference)*CFM)
    return {kind:dict(mean=np.mean(v),permutation_sd=np.std(v,ddof=1),permutation_low=np.quantile(v,.025),permutation_high=np.quantile(v,.975)) for kind,v in values.items()}

for h in HS:
    for reg in ['past','feed']:
        x=features(h,reg).loc[TE];y=target(h).loc[TE].values;m=models[h,f'LGB-Δ [{reg}]']
        windows=[('all',np.arange(len(TE)))]
        if h in [1,6,24]:windows += [(f'third-{k+1}',v) for k,v in enumerate(np.array_split(np.arange(len(TE)),3))]
        for window,ix in windows:
            for group in groups:
                cols=columns(x,group)
                if not cols:continue
                result=evaluate(m,x.iloc[ix],y[ix],cols,SEED+h)
                for mapping,value in result.items():rows.append(dict(h=h,regime=reg,window=window,group=group,mapping=mapping,n=len(ix),repeats=R,**value))
            if h in [1,6,12,24] and reg=='past':
                for lab in BIN_LAB:
                    cols=[f'{v}|L{lab}' for v in GROUPS['Co-substrate feed']]
                    value=evaluate(m,x.iloc[ix],y[ix],cols,SEED+h)['complete']
                    lagrows.append(dict(h=h,window=window,lag=lab,n=len(ix),repeats=R,**value))
        if reg=='past':
            p=m.predict(x)+x['Biogas|L0'].values
            for group in ['Gas storage','Co-substrate feed','Calendar']:
                f=RES/'group_refits'/f'loco_{h}_{group.replace(" ","_")}.npz'
                data=np.load(f);loco=data['prediction']/CFM
                assert np.allclose(data['y']/CFM,y)
                for length in [24,72,168]:
                    value=rmse_difference(y,loco,p,block_indices(len(y),length))
                    locorows.append(dict(h=h,group=group,block_h=length,repeats=2000,**value))
        print('attribution',h,reg,flush=True)
    frame('attribution_consistency.csv',rows);frame('HSW_lag_stability.csv',lagrows);frame('loco_paired_uncertainty.csv',locorows)

for h in [12,24]:
    for start,end in [(40,63),(48,71),(56,79)]:
        x=features(h,'past').copy()
        for v in GROUPS['Co-substrate feed']:
            for lab,a,b in [('24-47',24,start-1),('48-71',start,end),('72-167',end+1,167)]:
                x[f'{v}|L{lab}']=H[v].shift(a).rolling(b-a+1).mean()
        y=target(h);base=x['Biogas|L0'];model=lgb.LGBMRegressor(**models[h,'LGB-Δ [past]'].get_params())
        model.fit(x.loc[TR],(y-base).loc[TR]);p=model.predict(x.loc[TE])+base.loc[TE].values
        cols=[f'{v}|L48-71' for v in GROUPS['Co-substrate feed']]
        value=evaluate(model,x.loc[TE],y.loc[TE].values,cols,SEED+h)['complete']
        binrows.append(dict(h=h,start_h=start,end_h=end,**metric(y.loc[TE].values,p),**value,repeats=R))
        print('bin sensitivity',h,start,end,flush=True)
frame('lag_bin_sensitivity.csv',binrows)
print('ATTRIBUTION AUDIT COMPLETE',flush=True)
