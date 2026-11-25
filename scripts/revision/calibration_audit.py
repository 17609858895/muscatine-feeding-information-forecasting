"""Independent-calibration evaluation and recorded adaptive-state sensitivities.

Primary rule retains the original finite endpoint and after-issuance update.
Infinite endpoints and feedback at issuance are explicitly separate sensitivities.
No exchangeability or delayed-feedback coverage theorem is asserted.
"""
from context import *
from statistics_utils import *
from bisect import bisect_left, insort
from collections import deque
import pickle

ALPHAS=[.01,.02,.05,.10,.15,.20,.30,.40,.50]
STRATEGIES={'Persistence':'Persistence','Past-only ensemble':'PG-HQ ensemble [past]','Feed-conditioned ensemble':'PG-HQ ensemble [feed]'}

def adaptive(initial,residual,h,alpha,gamma=.005,history=None,boundary='finite',timing='after'):
    queue=deque(np.asarray(initial)[-history:] if history else initial)
    ordered=sorted(queue);a=alpha;n=len(residual)
    qs=np.empty(n);levels=np.empty(n)
    def update(j,a):
        if j<0:return a
        a+=gamma*(alpha-float(residual[j]>qs[j]))
        if history and len(queue)>=history:
            ordered.pop(bisect_left(ordered,queue.popleft()))
        queue.append(residual[j]);insort(ordered,residual[j])
        return a
    for i in range(n):
        if timing=='before':a=update(i-h,a)
        levels[i]=a;p=1-a
        if boundary=='infinite' and p>=1:q=np.inf
        elif boundary=='infinite' and p<=0:q=-np.inf
        else:
            rank=np.clip(p,0,1)*(len(ordered)-1);j=int(np.floor(rank));fraction=rank-j
            q=ordered[j] if j==len(ordered)-1 else ordered[j]*(1-fraction)+ordered[j+1]*fraction
        qs[i]=q
        if timing=='after':a=update(i-h,a)
    return qs,levels

def summary(y,p,q,a):
    c=np.maximum(0,p-q);d=np.minimum(c,y);deficit=np.maximum(c-y,0);mask=c>y
    events=runs(mask);outside=(a<=0)|(a>=1)
    return dict(firm_fraction=d.sum()/y.sum(),shortfall_rate=mask.mean(),
        shortfall_volume_share=deficit.sum()/max(c.sum(),1e-10),commit_m3h=c.mean()*CFM,
        delivered_m3h=d.mean()*CFM,mean_flow_m3h=y.mean()*CFM,n=len(y),
        events=len(events),longest_event_h=max([b-a for a,b in events],default=0),
        deficit_total_m3=deficit.sum()*CFM,max_hourly_deficit_m3_h=deficit.max()*CFM,
        boundary_hours=int(outside.sum()),longest_boundary_run_h=max([b-a for a,b in runs(outside)],default=0),
        infinite_quantile_hours=int((~np.isfinite(q)).sum()),a_min=a.min(),a_max=a.max())

def main():
    fp=pd.read_pickle(OUT/'independent_final_preds.pkl')
    results=[];paths={};intervals=[];interval_paths={};sens=[];traces=[]
    for h in HS:
        v=fp['pred'][h]['val'];t=fp['pred'][h]['test'];v['Persistence']=H.Biogas.loc[v.index].values
        y=t.y.values
        for strategy,col in STRATEGIES.items():
            p=t[col].values;initial=(v[col]-v.y).loc[CAL].values;error=p-y
            for alpha in ALPHAS:
                q,a=adaptive(initial,error,h,alpha)
                paths[h,strategy,alpha]={'q':q,'a':a,'c':np.maximum(0,p-q),'y':y,'p':p}
                results.append(dict(h=h,strategy=strategy,alpha=alpha,method='ACI',n_calibration=len(initial),**summary(y,p,q,a)))
                rank=int(np.ceil((len(initial)+1)*(1-alpha)))
                qs=np.full(len(y),np.sort(initial)[rank-1] if rank<=len(initial) else np.inf)
                results.append(dict(h=h,strategy=strategy,alpha=alpha,method='Split',n_calibration=len(initial),**summary(y,p,qs,np.full(len(y),alpha))))
            qi,ai=adaptive(abs(initial),abs(error),h,.1)
            interval_paths[h,strategy]={'q':qi,'a':ai,'y':y,'p':p}
            intervals.append(dict(h=h,model=col,method='Independent finite-endpoint ACI',coverage=np.mean(abs(error)<=qi),mean_width=np.mean(qi)*2*CFM,n_calibration=len(initial)))
            if h in [1,6,24]:
                for alpha in [.01,.02,.05]:
                    settings=[(g,w,b,'after','independent') for g in [.001,.005,.01] for w in [168,360,720,None] for b in ['finite','infinite']]
                    settings.extend([(.005,None,'finite','before','independent'),(.005,None,'finite','after','reused-696')])
                    for gamma,history,boundary,timing,source in settings:
                        init=initial if source=='independent' else (v[col]-v.y).values
                        q,a=adaptive(init,error,h,alpha,gamma,history,boundary,timing)
                        sens.append(dict(h=h,strategy=strategy,alpha=alpha,gamma=gamma,history=history or 'expanding',boundary=boundary,timing=timing,calibration=source,**summary(y,p,q,a)))
                        if gamma==.005 and history is None and timing=='after' and source=='independent':
                            for i,date in enumerate(TE):
                                traces.append(dict(origin=date,h=h,strategy=strategy,alpha=alpha,boundary=boundary,a=a[i],q_m3_h=q[i]*CFM,commitment_m3_h=max(0,p[i]-q[i])*CFM))
        print('calibration horizon',h,flush=True)
    frame('firm_supply_independent.csv',results);frame('intervals_independent.csv',intervals)
    frame('calibration_sensitivity.csv',sens);frame('adaptive_state_traces.csv',traces)
    pd.to_pickle(paths,OUT/'calibration_paths.pkl')
    pd.to_pickle(interval_paths,OUT/'interval_paths.pkl')
    paired=[];events=[];influence=[];matched=[]
    for h in [1,6,24]:
        for length in [24,72,168]:
            ix=block_indices(len(TE),length)
            for alpha in [.01,.02,.05,.1]:
                past=paths[h,'Past-only ensemble',alpha];feed=paths[h,'Feed-conditioned ensemble',alpha]
                y=past['y'];d=np.minimum(feed['c'],y)-np.minimum(past['c'],y)
                samples=d[ix].sum(1)/y[ix].sum(1);lo,hi=ci(samples)
                paired.append(dict(h=h,alpha=alpha,block_h=length,delta_fulfilled_fraction=d.sum()/y.sum(),CI_low=lo,CI_high=hi,repeats=len(ix)))
            # Curves interpolated only within their common observed-risk range.
            def curve(strategy,indices=None):
                xx=[];yy=[]
                for alpha in ALPHAS:
                    v=paths[h,strategy,alpha];y=v['y'];c=v['c']
                    if indices is not None:y=y[indices];c=c[indices]
                    xx.append(np.mean(c>y));yy.append(np.minimum(c,y).sum()/y.sum())
                u,inv=np.unique(xx,return_inverse=True)
                return u,np.bincount(inv,weights=yy)/np.bincount(inv)
            for risk in [.02,.05,.10]:
                x1,y1=curve('Past-only ensemble');x2,y2=curve('Feed-conditioned ensemble')
                if not(max(x1.min(),x2.min())<=risk<=min(x1.max(),x2.max())):continue
                estimate=np.interp(risk,x2,y2)-np.interp(risk,x1,y1);boot=[]
                for ind in ix:
                    u,v=curve('Past-only ensemble',ind);w,z=curve('Feed-conditioned ensemble',ind)
                    if max(u.min(),w.min())<=risk<=min(u.max(),w.max()):boot.append(np.interp(risk,w,z)-np.interp(risk,u,v))
                lo,hi=ci(boot);matched.append(dict(h=h,risk=risk,block_h=length,delta_fulfilled_fraction=estimate,CI_low=lo,CI_high=hi,valid_bootstrap=len(boot),repeats=len(ix)))
        for alpha in [.01,.02,.05]:
            past=paths[h,'Past-only ensemble',alpha];feed=paths[h,'Feed-conditioned ensemble',alpha];y=past['y']
            for strategy in STRATEGIES:
                v=paths[h,strategy,alpha];deficit=np.maximum(v['c']-y,0)
                for start,end in runs(deficit>0):
                    events.append(dict(h=h,strategy=strategy,alpha=alpha,origin_start=TE[start],origin_end=TE[end-1],duration_h=end-start,deficit_m3=deficit[start:end].sum()*CFM,peak_deficit_m3_h=deficit[start:end].max()*CFM))
            union=(past['c']>y)|(feed['c']>y);d=np.minimum(feed['c'],y)-np.minimum(past['c'],y)
            for start,end in runs(union):
                keep=np.ones(len(y),bool);keep[start:end]=False
                influence.append(dict(h=h,alpha=alpha,event_start=TE[start],event_end=TE[end-1],event_h=end-start,all_event_delta=d.sum()/y.sum(),score_without_event_delta=d[keep].sum()/y[keep].sum(),note='Scores only; original calibration paths retained'))
    frame('commitment_paired_uncertainty.csv',paired);frame('commitment_events.csv',events)
    frame('commitment_event_influence.csv',influence);frame('commitment_matched_risk.csv',matched)
    print('CALIBRATION AUDIT COMPLETE',flush=True)

if __name__=='__main__':main()
