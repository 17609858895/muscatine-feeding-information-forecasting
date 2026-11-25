"""Paired circular block resampling; fitted predictions are held fixed."""
from context import np, pd, SEED, CFM
from scipy.stats import norm

def block_indices(n, length=72, repeats=2000, seed=SEED):
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(repeats,int(np.ceil(n/length))))
    return ((starts[:,:,None]+np.arange(length))%n).reshape(repeats,-1)[:,:n]

def ci(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return tuple(np.quantile(x,[.025,.975])) if len(x) else (np.nan,np.nan)

def rmse_difference(y, candidate, reference, indices):
    a=(np.asarray(candidate)-y)**2;b=(np.asarray(reference)-y)**2
    point=(np.sqrt(a.mean())-np.sqrt(b.mean()))*CFM
    samples=(np.sqrt(a[indices].mean(1))-np.sqrt(b[indices].mean(1)))*CFM
    lo,hi=ci(samples)
    return dict(delta_RMSE_m3_h=point,CI_low=lo,CI_high=hi)

def hac_test(d,lag):
    d=np.asarray(d);n=len(d);z=d-d.mean()
    variance=np.dot(z,z)/n
    for k in range(1,min(lag,n-1)+1):
        variance+=2*(1-k/(lag+1))*np.dot(z[k:],z[:-k])/n
    se=np.sqrt(max(variance,0)/n)
    statistic=d.mean()/se if se else np.nan
    return statistic,2*norm.sf(abs(statistic))

def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p)
    adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty(len(p));out[order]=adjusted
    return out

def runs(mask):
    mask=np.asarray(mask,dtype=bool)
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
    return list(zip(starts,ends))

def acf(x,maxlag=168):
    z=np.asarray(x)-np.mean(x);den=np.dot(z,z)
    return np.array([np.dot(z[k:],z[:len(z)-k])/den for k in range(maxlag+1)])
