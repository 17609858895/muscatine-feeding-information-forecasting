"""Re-score saved fixed LightGBM models on complete common perturbation support.

No training, parameter search, source changes, or canonical-result overwrites.
Retains all original shifts, random draws, thresholds and preprocessing.
"""
from pathlib import Path
import sys, json, hashlib, warnings
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
from context import H,TE,HS,features,target,CFM,SEED,np,pd,RES
from context import OUT as MAJOR_OUT
from statistics_utils import runs
from scipy.stats import spearmanr

OUT=RES/'feeding_complete_support';OUT.mkdir(exist_ok=True)
models_path=RES/'tab_final.pkl'
tab=pd.read_pickle(models_path);models=tab['models']
source=pd.DataFrame({'HSW':H['Q-HSW1_GPM']+H['Q-HSW2_GPM'],'PS':H['Q-PS_on'],'TWAS':H['Q-TWAS_GPM']})
SHIFTS=[-6,-3,-2,-1,0,1,2,3,6]
NREPS=20

def full_window(series,h):
    return series.iloc[::-1].rolling(h).mean().iloc[::-1].shift(-1).loc[TE]

support=np.ones(len(TE),bool);support_rows=[]
for h in [6,24]:
    for shift in SHIFTS:
        values=full_window(source.HSW.shift(shift),h)
        valid=np.isfinite(values.to_numpy());support &= valid
        support_rows.append({'h':h,'shift_h':shift,'missing_count':int((~valid).sum()),'first_missing':str(TE[~valid].min()) if (~valid).any() else None})
COMMON=TE[support]
assert len(COMMON)>0
assert (COMMON.to_series().diff().dropna()==pd.Timedelta(hours=1)).all()
pd.DataFrame({'origin':TE,'common_complete_window':support}).to_csv(OUT/'feeding_common_support_origins.csv',index=False)
pd.DataFrame(support_rows).to_csv(OUT/'feeding_shift_support.csv',index=False)

def metric(y,p):
    assert np.isfinite(y).all() and np.isfinite(p).all()
    e=y-p
    return dict(RMSE=float(np.sqrt(np.mean(e**2))*CFM),MAE=float(np.mean(np.abs(e))*CFM),R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)))

def input_error(pred,truth):
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    return dict(window_NRMSE=float(np.sqrt(np.mean((pred-truth)**2))/truth.mean()),window_WAPE=float(np.abs(pred-truth).sum()/truth.sum()))

rows=[];baselines=[];all_prediction=[];all_window=[]
noises=np.random.default_rng(SEED).normal(size=(NREPS,len(H)))
episodes=runs(source.HSW.values>1)
draws=np.random.default_rng(SEED+17).random((NREPS,len(episodes)))
for h in [6,24]:
    x=features(h,'feed').loc[COMMON];base=x['Biogas|L0'].to_numpy();y=target(h).loc[COMMON].to_numpy()
    model=models[h,'LGB-Δ [feed]'];truth=x['Planned HSW|F'].to_numpy()
    for reg in ['past','feed']:
        key=f'LGB-Δ [{reg}]';old=tab['pred'][h]['test'][key].loc[COMMON].to_numpy()
        p=models[h,key].predict(features(h,reg).loc[COMMON])+base
        maxdiff=float(np.max(np.abs(p-old)))
        assert maxdiff<1e-10,(h,reg,maxdiff)
        baselines.append(dict(h=h,model=key,n=len(COMMON),max_prediction_difference_from_saved=maxdiff,**metric(y,p)))
    def score(value,scenario,shift_h,magnitude,rep):
        xx=x.copy();xx['Planned HSW|F']=value
        p=model.predict(xx)+base
        record=dict(h=h,scenario=scenario,shift_h=shift_h,magnitude=magnitude,rep=rep,n=len(COMMON),**metric(y,p),**input_error(value,truth))
        rows.append(record);all_prediction.append(p);all_window.append(value)
    for shift in SHIFTS:
        shifted=source.HSW.shift(shift)
        for sigma in [0,.1,.2,.4]:
            for rep in range(NREPS if sigma else 1):
                perturbed=shifted*np.maximum(0,1+sigma*noises[rep])
                score(full_window(perturbed,h).loc[COMMON].to_numpy(),'hourly_noise',shift,sigma,rep)
    for bias in [-.4,-.2,-.1,.1,.2,.4]:
        score(truth*(1+bias),'sustained_bias',0,bias,0)
    for fraction in [.25,.5]:
        for scenario in ['episode_cancellation','episode_delay_6h']:
            for rep in range(NREPS):
                z=source.HSW.to_numpy().copy()
                for k,(start,end) in enumerate(episodes):
                    if draws[rep,k]>=fraction:continue
                    segment=source.HSW.iloc[start:end].to_numpy().copy();z[start:end]-=segment
                    if scenario=='episode_delay_6h' and start+6<len(z):
                        dest_end=min(end+6,len(z));z[start+6:dest_end]+=segment[:dest_end-start-6]
                score(full_window(pd.Series(z,index=H.index),h).loc[COMMON].to_numpy(),scenario,6 if 'delay' in scenario else 0,fraction,rep)
    print('Rescored horizon',h,'n',len(COMMON),flush=True)

new=pd.DataFrame(rows);old=pd.read_csv(MAJOR_OUT/'feeding_perturbations.csv')
new.to_csv(OUT/'feeding_perturbations_complete_support.csv',index=False)
pd.DataFrame(baselines).to_csv(OUT/'feeding_baselines_complete_support.csv',index=False)
keys=['h','scenario','shift_h','magnitude','rep']
paired=old.merge(new,on=keys,suffixes=('_before','_after'),validate='one_to_one')
for col in ['RMSE','MAE','R2','window_WAPE','window_NRMSE']:
    paired[col+'_change']=paired[col+'_after']-paired[col+'_before']
paired.to_csv(OUT/'feeding_score_changes.csv',index=False)
np.savez_compressed(OUT/'feeding_complete_support_predictions.npz',origins=COMMON.to_numpy().astype('datetime64[ns]').astype('int64'),predictions_native=np.asarray(all_prediction),window_HSW_native=np.asarray(all_window))

summ=[]
for h in [6,24]:
    d=new[new.h==h]; rho=float(spearmanr(d.window_WAPE,d.RMSE).statistic)
    for scenario,mag in [('hourly_noise',.4),('sustained_bias',-.4),('sustained_bias',.4),('episode_cancellation',.5),('episode_delay_6h',.5)]:
        s=d[(d.scenario==scenario)&(d.magnitude==mag)]
        if scenario=='hourly_noise':s=s[s.shift_h==0]
        summ.append(dict(h=h,scenario=scenario,magnitude=mag,rows=len(s),mean_RMSE=float(s.RMSE.mean()),min_RMSE=float(s.RMSE.min()),max_RMSE=float(s.RMSE.max()),mean_WAPE=float(s.window_WAPE.mean()),mean_NRMSE=float(s.window_NRMSE.mean())))
pd.DataFrame(summ).to_csv(OUT/'feeding_summary_complete_support.csv',index=False)
p24=new[new.h==24];c=p24[p24.scenario=='episode_cancellation']
report={
    'protocol':'Original fixed LightGBM models, deterministic features, random matrices, contiguous-episode definition and perturbation settings retained; only scoring support changed.',
    'n_original':len(TE),'n_common':len(COMMON),'first':str(COMMON.min()),'last':str(COMMON.max()),'excluded':[str(x) for x in TE[~support]],
    'preprocessing':'Original H.interpolate(limit=3) preserved. Common support requires finite shifted HSW windows at both h=6 and 24 for every shift from -6 to +6.',
    'scenario_rows':len(new),'day_ahead_scenario_rows':len(p24),'missing_corrected_values':int(new[['RMSE','MAE','R2','window_NRMSE','window_WAPE']].isna().sum().sum()),
    'rho_all_24h':float(spearmanr(p24.window_WAPE,p24.RMSE).statistic),'rho_cancellation_24h':float(spearmanr(c.window_WAPE,c.RMSE).statistic),'cancellation_draws':len(c),
    'baselines':baselines,'summaries':summ,'max_abs_RMSE_change':float(paired.RMSE_change.abs().max()),
    'model_file_sha256':hashlib.sha256(models_path.read_bytes()).hexdigest(),
    'random_seed':SEED,'random_draw_count':NREPS,
    'episode_selection':'Each contiguous above-threshold run is selected independently with probability 0.25 or 0.50, not an exact fraction of runs.',
}
(OUT/'feeding_correction_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
