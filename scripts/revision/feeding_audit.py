"""Forecast-origin-available feed proxies and paired schedule perturbations."""
from context import *
import pickle

models=pd.read_pickle(RES/'tab_final.pkl')['models']
feedvars=['Planned HSW|F','Planned PS|F','Planned TWAS|F']
source=pd.DataFrame({'HSW':H['Q-HSW1_GPM']+H['Q-HSW2_GPM'],'PS':H['Q-PS_on'],'TWAS':H['Q-TWAS_GPM']})
proxy_rows=[];perturb_rows=[];predictions={}
NREPS=20

def window(series,h):
    return series.iloc[::-1].rolling(h).mean().iloc[::-1].shift(-1).loc[TE].values

def input_error(pred,truth):
    return dict(window_NRMSE=float(np.sqrt(np.mean((pred-truth)**2))/max(np.mean(truth),1e-12)),window_WAPE=float(np.abs(pred-truth).sum()/max(truth.sum(),1e-12)))

for h in HS:
    x=features(h,'feed').loc[TE];base=x['Biogas|L0'].values;y=target(h).loc[TE].values
    model=models[h,'LGB-Δ [feed]'];actual=x[feedvars].values
    for method,weeks in [('Weekly feed',1),('Four-week feed median',4)]:
        blocks=[]
        for j in range(1,h+1):
            observed=[source.shift(168*k-j).loc[TE].values for k in range(1,weeks+1)]
            blocks.append(np.median(observed,axis=0))
        predicted=np.mean(blocks,axis=0)
        # Every source timestamp precedes the forecast origin by at least 144 h.
        xx=x.copy();xx[feedvars]=predicted;p=model.predict(xx)+base
        predictions[h,method]=p
        for j,variable in enumerate(['HSW','PS','TWAS']):
            proxy_rows.append(dict(h=h,method=method,variable=variable,**metric(y,p),**input_error(predicted[:,j],actual[:,j])))
    if h not in [6,24]:continue
    truth=x['Planned HSW|F'].values
    noises=np.random.default_rng(SEED).normal(size=(NREPS,len(H)))
    for shift in [-6,-3,-2,-1,0,1,2,3,6]:
        shifted=source.HSW.shift(shift) # positive shift = delivery delayed
        for sigma in [0,.1,.2,.4]:
            for rep in range(NREPS if sigma else 1):
                perturbed=shifted*np.maximum(0,1+sigma*noises[rep])
                value=window(perturbed,h);xx=x.copy();xx['Planned HSW|F']=value
                p=model.predict(xx)+base
                perturb_rows.append(dict(h=h,scenario='hourly_noise',shift_h=shift,magnitude=sigma,rep=rep,**metric(y,p),**input_error(value,truth)))
    for bias in [-.4,-.2,-.1,.1,.2,.4]:
        value=truth*(1+bias);xx=x.copy();xx['Planned HSW|F']=value;p=model.predict(xx)+base
        perturb_rows.append(dict(h=h,scenario='sustained_bias',shift_h=0,magnitude=bias,rep=0,**metric(y,p),**input_error(value,truth)))
    # Feeding episodes are contiguous runs above 1 native GPM; this is a
    # reproducible stress scenario, not an observed cancellation process.
    from statistics_utils import runs
    episodes=runs(source.HSW.values>1)
    draws=np.random.default_rng(SEED+17).random((NREPS,len(episodes)))
    for fraction in [.25,.5]:
        for scenario in ['episode_cancellation','episode_delay_6h']:
            for rep in range(NREPS):
                z=source.HSW.values.copy()
                for k,(start,end) in enumerate(episodes):
                    if draws[rep,k]>=fraction:continue
                    segment=source.HSW.values[start:end].copy();z[start:end]-=segment
                    if scenario=='episode_delay_6h' and start+6<len(z):
                        dest_end=min(end+6,len(z));z[start+6:dest_end]+=segment[:dest_end-start-6]
                value=window(pd.Series(z,index=H.index),h);xx=x.copy();xx['Planned HSW|F']=value;p=model.predict(xx)+base
                perturb_rows.append(dict(h=h,scenario=scenario,shift_h=6 if 'delay' in scenario else 0,magnitude=fraction,rep=rep,**metric(y,p),**input_error(value,truth)))
    print('feeding audit',h,flush=True)
frame('available_feed_controls.csv',proxy_rows);frame('feeding_perturbations.csv',perturb_rows)
pd.to_pickle(predictions,OUT/'available_feed_predictions.pkl')
print('FEEDING AUDIT COMPLETE',flush=True)
