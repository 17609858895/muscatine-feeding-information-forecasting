"""Conditional uncertainty from retained predictions and fixed fitted responses.

No training, tuning, or calibration is repeated. All local differences are
computed using the original model and original training quantile boundaries.
"""
from pathlib import Path
import sys, json, hashlib, warnings
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
import context as ctx
import numpy as np
import pandas as pd
from statistics_utils import block_indices
OUT=ctx.RES/'figure_intervals';OUT.mkdir(exist_ok=True)
REPEATS=2000;BLOCK=72;SEED=20261018
records=[];checks=[];results={}
fp=pd.read_pickle(ctx.OUT/'independent_final_preds.pkl')
bench=pd.read_pickle(ctx.OUT/'benchmark_predictions.pkl')
metrics=pd.read_csv(ctx.OUT/'all_benchmark_metrics.csv')
indices=block_indices(len(ctx.TE),BLOCK,REPEATS,SEED)
for h in ctx.HS:
    d=fp['pred'][h]['test'];assert d.index.equals(ctx.TE)
    y=d.y.to_numpy();pers=bench[h]['Persistence']
    pr=np.sqrt(np.mean((pers-y)**2));prb=np.sqrt(np.mean((pers-y)[indices]**2,axis=1))
    for name in ['LGB-Δ [past]','LGB-Δ [feed]','Cat-Δ [past]','Cat-Δ [feed]',
                 'PG-HQ ensemble [past]','PG-HQ ensemble [feed]','Weekly feed','Four-week feed median']:
        p=np.asarray(bench[h][name]);assert len(p)==len(y) and np.isfinite(p).all()
        rmse=np.sqrt(np.mean((p-y)**2));rb=np.sqrt(np.mean((p-y)[indices]**2,axis=1))
        row=metrics[(metrics.h==h)&(metrics.model==name)].iloc[0]
        assert abs(rmse*ctx.CFM-row.RMSE)<1e-9
        assert abs((1-rmse/pr)-row.Skill)<1e-12
        for quantity,point,b in [('RMSE',rmse*ctx.CFM,rb*ctx.CFM),('Skill',1-rmse/pr,1-rb/prb)]:
            lo,hi=np.quantile(b,[.025,.975])
            records.append(dict(h=h,model=name,quantity=quantity,point=point,lower=lo,upper=hi,n=len(y)))
pd.DataFrame(records).to_csv(OUT/'forecast_pointwise_intervals.csv',index=False)
print('Forecast scores reproduced; intervals saved.',flush=True)

# Bootstrap row multiplicities preserve time ordering inside each circular block.
tr_indices=block_indices(len(ctx.TR),BLOCK,REPEATS,SEED)
weights=np.stack([np.bincount(i,minlength=len(ctx.TR)) for i in tr_indices]).astype(float)
del tr_indices
tree=pd.read_pickle(ctx.RES/'tab_final.pkl')
old_ale=pd.read_pickle(ctx.RES/'ale.pkl')
old_2d=pd.read_pickle(ctx.RES/'extra/ale2d.pkl')
features=['Q-HSW2_GPM|L0','Q-HSW1_GPM|L48-71','H-Dig1_FT|L0']
def predict(model,x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.predict(x,num_threads=2)
for h in [1,6,24]:
    x=ctx.features(h,'past').loc[ctx.TR]
    model=tree['models'][(h,'LGB-Δ [past]')]
    for f in features:
        q,original=old_ale[h,f];q=np.asarray(q);nb=len(q)-1
        bins=np.clip(np.searchsorted(q,x[f],side='right')-1,0,nb-1)
        low=x.copy();high=x.copy();low[f]=q[bins];high[f]=q[bins+1]
        delta=predict(model,high)-predict(model,low)
        indicator=np.eye(nb)[bins]
        counts=indicator.sum(0);sums=delta@indicator
        eff=np.divide(sums,counts,out=np.zeros(nb),where=counts>0)
        acc=np.r_[0,np.cumsum(eff)]
        point=acc-np.sum((acc[:-1]+acc[1:])*.5*counts)/counts.sum()
        discrepancy=float(np.max(np.abs(point-original)))
        assert discrepancy<1e-10,(h,f,discrepancy)
        bn=weights@indicator;bs=weights@(indicator*delta[:,None])
        be=np.divide(bs,bn,out=np.full_like(bs,np.nan),where=bn>0)
        ba=np.column_stack([np.zeros(REPEATS),np.cumsum(be,axis=1)])
        bc=ba-np.sum((ba[:,:-1]+ba[:,1:])*.5*bn,axis=1)[:,None]/bn.sum(1)[:,None]
        valid=np.isfinite(bc).all(1)
        # Do not silently fill empty bootstrap bins or condition on a selected
        # subset of replicates. Missing support is explicitly marked instead.
        assert valid.all(),(h,f,int(valid.sum()))
        lower,upper=np.quantile(bc,[.025,.975],axis=0)
        results[h,f]=dict(q=q,point=point,lower=lower,upper=upper,counts=counts)
        checks.append(dict(kind='first_order',h=h,feature=f,max_point_difference=discrepancy,
                           bootstrap_complete=int(valid.sum()),bins=nb))
    print('First-order ALE',h,'reproduced.',flush=True)
    if h==1:continue
    old=old_2d[h];q1=np.asarray(old['q1']);q2=np.asarray(old['q2'])
    n1,n2=len(q1)-1,len(q2)-1
    b1=np.clip(np.searchsorted(q1,x[features[0]],side='right')-1,0,n1-1)
    b2=np.clip(np.searchsorted(q2,x[features[2]],side='right')-1,0,n2-1)
    def corners(upper1,upper2):
        z=x.copy();z[features[0]]=q1[b1+upper1];z[features[2]]=q2[b2+upper2]
        return predict(model,z)
    delta=corners(1,1)-corners(0,1)-corners(1,0)+corners(0,0)
    indicator=np.eye(n1*n2)[b1*n2+b2]
    counts=indicator.sum(0);sums=delta@indicator
    local=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0).reshape(n1,n2)
    point=np.cumsum(np.cumsum(local,axis=0),axis=1)
    discrepancy=float(np.max(np.abs(point-old['A'][1:,1:])))
    assert discrepancy<1e-10,(h,discrepancy)
    assert np.array_equal(counts.reshape(n1,n2),old['N'])
    bn=weights@indicator;bs=weights@(indicator*delta[:,None])
    be=np.divide(bs,bn,out=np.full_like(bs,np.nan),where=bn>0).reshape(REPEATS,n1,n2)
    ba=np.cumsum(np.cumsum(be,axis=1),axis=2)
    # Each cumulative contrast requires every preceding cell in its rectangle.
    # Show an interval only if that entire support exists in ALL replicates.
    supported=np.isfinite(ba).all(0)
    lower=np.full((n1,n2),np.nan);upper=lower.copy()
    lower[supported],upper[supported]=np.quantile(ba[:,supported],[.025,.975],axis=0)
    results[h,'mixed']=dict(q1=q1,q2=q2,point=point,lower=lower,upper=upper,
                            supported=supported,counts=counts.reshape(n1,n2))
    checks.append(dict(kind='accumulated_mixed_differences',h=h,max_point_difference=discrepancy,
                       supported_cells=int(supported.sum()),total_cells=n1*n2,
                       intervals_crossing_zero=int(((lower<=0)&(upper>=0)&supported).sum())))
    print('Mixed response',h,'supported intervals',int(supported.sum()),'of',n1*n2,flush=True)
pd.to_pickle(results,OUT/'response_pointwise_intervals.pkl')
np.savez_compressed(OUT/'bootstrap_indices.npz',test=indices,
                   training=block_indices(len(ctx.TR),BLOCK,REPEATS,SEED))
sources=[ctx.OUT/'independent_final_preds.pkl',ctx.OUT/'benchmark_predictions.pkl',
         ctx.RES/'tab_final.pkl',ctx.RES/'ale.pkl',ctx.RES/'extra/ale2d.pkl']
audit=dict(repeats=REPEATS,block_hours=BLOCK,seed=SEED,training_n=len(ctx.TR),test_n=len(ctx.TE),
           conditional='Fixed models and fixed feature grids; no refitting or recalibration.',
           heatmap_estimator='Uncentered cumulative mixed finite differences, preserving the original grid and values. Not fully centered second-order ALE.',
           heatmap_support='Intervals only where every cumulative cell is occupied in every bootstrap replicate; no imputation of missing bin effects.',
           checks=checks,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'interval_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(checks,indent=2),flush=True)
