import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from common import *
o=pickle.load(open('tab_final.pkl','rb'))
X0,meta=build()
HS=[1,2,3,4,6,9,12,18,24]
groups=list(GROUPS)+['Calendar','Planned feed']
blocks=[(g,b) for g in GROUPS for b in BIN_LAB]+[('Calendar','—'),('Planned feed','(t, t+h]')]
def block_cols(g,b,cols):
    if g=='Calendar': return [c for c in cols if c.endswith(('_sin','_cos'))]
    if g=='Planned feed': return [c for c in cols if c.endswith('|F')]
    vs=GROUPS[g]; return [f'{v}|L{b}' for v in vs]
rng=np.random.default_rng(0)
res={}
for reg in ['past','feed']:
    Ab=np.zeros((len(HS),len(blocks))); Ag=np.zeros((len(HS),len(groups)))
    for i,h in enumerate(HS):
        m=o['models'][(h,f'LGB-Δ [{reg}]')]
        Xp=pd.concat([X0,calendar(X0.index,h)],axis=1)
        XX=pd.concat([Xp,future_feed(h)],axis=1) if reg=='feed' else Xp
        te=idx[o_te]; Xt=XX.loc[te].copy(); base=Xt['Biogas|L0'].values; yt=pd.Series(target(h).values,index=idx)[te].values
        r0=np.sqrt(np.mean((m.predict(Xt)+base-yt)**2))
        def perm_imp(cols,rep=5):
            out=[]
            for r in range(rep):
                Xq=Xt.copy(); p=rng.permutation(len(Xq)); Xq[cols]=Xq[cols].values[p]
                # persistence anchor is not permuted (anchor = observed current value)
                out.append(np.sqrt(np.mean((m.predict(Xq)+base-yt)**2))-r0)
            return np.mean(out)
        for j,(g,b) in enumerate(blocks):
            cols=[c for c in block_cols(g,b,Xt.columns) if c in Xt.columns]
            Ab[i,j]=perm_imp(cols) if cols else np.nan
        for j,g in enumerate(groups):
            cols=[c for gg,bb in blocks if gg==g for c in block_cols(gg,bb,Xt.columns) if c in Xt.columns]
            Ag[i,j]=perm_imp(cols) if cols else np.nan
        print(reg,h,'done',flush=True)
    res[reg]=dict(Ab=Ab,Ag=Ag)
# faithfulness deletion test (past regime, h=1,6,24): mean-replacement of top-k blocks
def deletion(reg,h,rank,kmax=12):
    m=o['models'][(h,f'LGB-Δ [{reg}]')]
    Xp=pd.concat([X0,calendar(X0.index,h)],axis=1)
    XX=pd.concat([Xp,future_feed(h)],axis=1) if reg=='feed' else Xp
    tr,te=idx[o_tr],idx[o_te]; Xt=XX.loc[te].copy(); base=Xt['Biogas|L0'].values; yt=pd.Series(target(h).values,index=idx)[te].values
    mu=XX.loc[tr].mean(); curve=[]
    for k in range(0,kmax+1):
        Xq=Xt.copy()
        for j in rank[:k]:
            g,b=blocks[j]; cols=[c for c in block_cols(g,b,Xq.columns) if c in Xq.columns]
            Xq[cols]=mu[cols].values
        curve.append(np.sqrt(np.mean((m.predict(Xq)+base-yt)**2)))
    return np.array(curve)
fa={}
for reg in ['past']:
    Ab=res[reg]['Ab']; valid=~np.isnan(Ab).all(0); glob=np.nanmean(Ab,0)
    for h in [1,6,24]:
        i=HS.index(h); a=np.where(np.isnan(Ab[i]),-1e9,Ab[i])
        spec=np.argsort(-a); gl=np.argsort(-np.where(np.isnan(glob),-1e9,glob))
        cand=[j for j in range(len(blocks)) if valid[j]]
        rnd=np.mean([deletion(reg,h,list(rng.permutation(cand))) for _ in range(10)],0)
        fa[h]=dict(specific=deletion(reg,h,list(spec)),glob=deletion(reg,h,list(gl)),random=rnd)
        print('faith',h,np.round(fa[h]['specific'][[0,3,6,12]],2),np.round(fa[h]['glob'][[0,3,6,12]],2),np.round(fa[h]['random'][[0,3,6,12]],2),flush=True)
def erank(A):
    A=np.nan_to_num(np.clip(A,0,None)); A=A/ (A.sum(1,keepdims=True)+1e-12); s=np.linalg.svd(A,compute_uv=False); p=s/s.sum(); return float(np.exp(-(p*np.log(p+1e-12)).sum()))
res['erank']={reg:erank(res[reg]['Ab']) for reg in ['past','feed']}
res['faith']=fa; res['blocks']=blocks; res['groups']=groups; res['HS']=HS
pickle.dump(res,open('hrpga.pkl','wb'))
print('erank',res['erank'])
for reg in ['past','feed']:
    print(reg); print(pd.DataFrame(res[reg]['Ag'],index=HS,columns=groups).round(2).to_string())
