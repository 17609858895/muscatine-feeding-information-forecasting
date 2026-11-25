"""Adaptive conformal inference with delayed feedback (Gibbs & Candes 2021), for h-step forecasts:
the residual of origin i is only revealed h hours later."""
import numpy as np
def aci_halfwidths(val_resid, test_resid, h, alpha=0.1, gamma=0.005):
    hist=list(np.abs(val_resid)); at=alpha; q=np.zeros(len(test_resid)); r=np.abs(test_resid)
    for i in range(len(r)):
        q[i]=np.quantile(hist,min(1.0,max(0.0,1-at)))
        j=i-h
        if j>=0:
            err=1.0 if r[j]>q[j] else 0.0
            at=at+gamma*(alpha-err); hist.append(r[j])
    return q
