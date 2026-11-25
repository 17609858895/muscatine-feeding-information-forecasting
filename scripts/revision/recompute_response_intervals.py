"""Resample fixed-model response estimates within uninterrupted training segments."""
from pathlib import Path
import sys,json,warnings,hashlib
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
import context as ctx
import numpy as np
import pandas as pd
PRIOR=ctx.RES/'figure_intervals'
OUT=ctx.RES/'figure_intervals_segmented';OUT.mkdir(exist_ok=True)
REPEATS=2000;BLOCK=72;SEED=20261018
old=pd.read_pickle(PRIOR/'response_pointwise_intervals.pkl')
rng=np.random.default_rng(SEED)
breaks=np.flatnonzero(np.diff(ctx.TR.values)!=np.timedelta64(1,'h'))+1
segments=np.split(np.arange(len(ctx.TR)),breaks)
samples=[]
for segment in segments:
    n=len(segment)
    assert n>=BLOCK
    assert np.all(np.diff(ctx.TR[segment].values)==np.timedelta64(1,'h'))
    starts=rng.integers(n,size=(REPEATS,int(np.ceil(n/BLOCK))))
    local=((starts[:,:,None]+np.arange(BLOCK))%n).reshape(REPEATS,-1)[:,:n]
    samples.append(segment[local])
indices=np.concatenate(samples,axis=1)
assert indices.shape==(REPEATS,len(ctx.TR))
weights=np.stack([np.bincount(i,minlength=len(ctx.TR)) for i in indices]).astype(float)
models=pd.read_pickle(ctx.RES/'tab_final.pkl')['models']
results={};checks=[]
features=['Q-HSW2_GPM|L0','Q-HSW1_GPM|L48-71','H-Dig1_FT|L0']
def predict(model,x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.predict(x,num_threads=2)
for h in [1,6,24]:
    x=ctx.features(h,'past').loc[ctx.TR];model=models[(h,'LGB-Δ [past]')]
    for f in features:
        d=old[h,f];q=d['q'];nb=len(q)-1
        bins=np.clip(np.searchsorted(q,x[f],side='right')-1,0,nb-1)
        low=x.copy();high=x.copy();low[f]=q[bins];high[f]=q[bins+1]
        delta=predict(model,high)-predict(model,low)
        indicator=np.eye(nb)[bins];counts=indicator.sum(0)
        acc=np.r_[0,np.cumsum(delta@indicator/counts)]
        point=acc-np.sum((acc[:-1]+acc[1:])*.5*counts)/counts.sum()
        assert np.allclose(point,d['point'],atol=1e-12)
        bn=weights@indicator;bs=weights@(indicator*delta[:,None])
        assert (bn>0).all()
        ba=np.column_stack([np.zeros(REPEATS),np.cumsum(bs/bn,axis=1)])
        ba-=np.sum((ba[:,:-1]+ba[:,1:])*.5*bn,axis=1)[:,None]/bn.sum(1)[:,None]
        lower,upper=np.quantile(ba,[.025,.975],axis=0)
        results[h,f]=dict(q=q,point=point,lower=lower,upper=upper,counts=counts)
        change=max(np.max(abs(lower-d['lower'])),np.max(abs(upper-d['upper'])))*ctx.CFM
        checks.append(dict(h=h,feature=f,complete_replicates=REPEATS,max_endpoint_change_m3h=float(change)))
    print('First-order response intervals',h,flush=True)
    if h==1:continue
    d=old[h,'mixed'];q1,q2=d['q1'],d['q2'];n1,n2=len(q1)-1,len(q2)-1
    b1=np.clip(np.searchsorted(q1,x[features[0]],side='right')-1,0,n1-1)
    b2=np.clip(np.searchsorted(q2,x[features[2]],side='right')-1,0,n2-1)
    def corner(i,j):
        z=x.copy();z[features[0]]=q1[b1+i];z[features[2]]=q2[b2+j]
        return predict(model,z)
    delta=corner(1,1)-corner(0,1)-corner(1,0)+corner(0,0)
    indicator=np.eye(n1*n2)[b1*n2+b2];counts=indicator.sum(0)
    local=np.divide(delta@indicator,counts,out=np.zeros(n1*n2),where=counts>0).reshape(n1,n2)
    point=np.cumsum(np.cumsum(local,axis=0),axis=1)
    assert np.allclose(point,d['point'],atol=1e-12)
    bn=weights@indicator;bs=weights@(indicator*delta[:,None])
    local_b=np.divide(bs,bn,out=np.full_like(bs,np.nan),where=bn>0).reshape(REPEATS,n1,n2)
    accum=np.cumsum(np.cumsum(local_b,axis=1),axis=2)
    supported=np.isfinite(accum).all(0)
    lower=np.full((n1,n2),np.nan);upper=lower.copy()
    lower[supported],upper[supported]=np.quantile(accum[:,supported],[.025,.975],axis=0)
    results[h,'mixed']=dict(q1=q1,q2=q2,point=point,lower=lower,upper=upper,supported=supported,counts=counts.reshape(n1,n2))
    checks.append(dict(h=h,feature='mixed',supported=int(supported.sum()),span_zero=int(((lower<=0)&(upper>=0)&supported).sum())))
    print('Mixed response',h,checks[-1],flush=True)
pd.to_pickle(results,OUT/'response_pointwise_intervals.pkl')
np.savez_compressed(OUT/'bootstrap_indices.npz',training=indices,test=np.load(PRIOR/'bootstrap_indices.npz')['test'])
audit=dict(repeats=REPEATS,block_hours=BLOCK,seed=SEED,training_n=len(ctx.TR),
           segments=[dict(n=len(s),start=str(ctx.TR[s[0]]),end=str(ctx.TR[s[-1]])) for s in segments],
           checks=checks,point_estimates_unchanged=True,
           conditional='Fixed models, quantile grids, and sample count in each uninterrupted training segment.')
(OUT/'interval_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
# Verify full-domain vs actually displayed response ranges.
raw=[ctx.H['Q-HSW2_GPM'],ctx.H['Q-HSW1_GPM'].shift(48).rolling(24).mean(),ctx.H['H-Dig1_FT']]
display=[]
for f,r in zip(features,raw):
    xmin,xmax=np.quantile(r.dropna(),[.01,.99])
    for h in [1,6,24]:
        d=results[h,f];keep=(d['q']>=xmin-1e-9)&(d['q']<=xmax+1e-9)
        display.append(dict(feature=f,h=h,plotted_range_m3h=float(np.ptp(d['point'][keep])*ctx.CFM)))
(OUT/'display_ranges.json').write_text(json.dumps(display,indent=2),encoding='utf8')
print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
