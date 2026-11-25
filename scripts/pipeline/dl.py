import pandas as pd, numpy as np, torch, torch.nn as nn, time, sys, warnings; warnings.filterwarnings('ignore')
from feats import *
torch.set_num_threads(2)
VARS=[v for vs in GROUPS.values() for v in vs]
L=168; HMAX=24; EMB='2022-11-14'  # embargo: train origins must have target end < SPLIT_VAL
Hs=H[VARS].copy()
tr_mask=Hs.index<SPLIT_VAL
mu=Hs[tr_mask].mean(); sd=Hs[tr_mask].std()+1e-6
Z=((Hs-mu)/sd).values.astype(np.float32)
y=H['Biogas'].values.astype(np.float32); ysd=float(sd['Biogas'])
idx=H.index
hsw=(H['Q-HSW1_GPM']+H['Q-HSW2_GPM']).values
def fut_feed(t):  # mean planned HSW, PS-on, TWAS over (t,t+h] for h=1..24, standardized
    out=[]
    for arr in [hsw,H['Q-PS_on'].values,H['Q-TWAS_GPM'].values]:
        seg=arr[t+1:t+HMAX+1]; c=np.cumsum(seg)/np.arange(1,HMAX+1); out.append(c)
    o=np.stack(out,1); return o
FF=None
def make(origins):
    Xs,Ys,Base,Cal,Fut=[],[],[],[],[]
    for t in origins:
        Xs.append(Z[t-L+1:t+1]); Ys.append(y[t+1:t+HMAX+1]); Base.append(y[t])
        tt=idx[t]+pd.to_timedelta(np.arange(1,HMAX+1),'h')
        Cal.append(np.stack([np.sin(2*np.pi*tt.hour/24),np.cos(2*np.pi*tt.hour/24),np.sin(2*np.pi*tt.dayofweek/7),np.cos(2*np.pi*tt.dayofweek/7)],1))
        Fut.append(fut_feed(t))
    return [torch.tensor(np.array(a),dtype=torch.float32) for a in (Xs,Ys,Base,Cal,Fut)]
valid=~np.isnan(Z).any(1)
okv=np.array([valid[t-L+1:t+HMAX+1].all() and not np.isnan(y[t:t+HMAX+1]).any() for t in range(len(idx)-HMAX)]+[False]*HMAX)
okv[:L]=False
T=np.arange(len(idx))
end=idx+pd.Timedelta(hours=HMAX)
o_tr=T[okv&(end<SPLIT_VAL)]; o_va=T[okv&(idx>=SPLIT_VAL)&(end<SPLIT_TEST)]; o_te=T[okv&(idx>=SPLIT_TEST)]
FFS=np.array([np.nanmean(hsw[:len(o_tr)]),1,1])
D={k:make(o) for k,o in [('tr',o_tr),('va',o_va),('te',o_te)]}
# normalise future feed with train stats
fm=D['tr'][4].mean((0,1)); fs=D['tr'][4].std((0,1))+1e-6
for k in D: D[k][4]=(D[k][4]-fm)/fs
NV=len(VARS); gi=[[VARS.index(v) for v in vs] for vs in GROUPS.values()]
# ---------------- models -----------------
class LSTMm(nn.Module):
    def __init__(s,cell=nn.LSTM): super().__init__(); s.r=cell(NV,64,2,batch_first=True,dropout=0.1); s.o=nn.Linear(64,HMAX)
    def forward(s,x,c,f): o,_=s.r(x); return s.o(o[:,-1])
class GRUm(LSTMm):
    def __init__(s): super().__init__(nn.GRU)
class TCN(nn.Module):
    def __init__(s,ch=48):
        super().__init__(); layers=[]; cin=NV
        for d in [1,2,4,8,16,32]:
            layers+= [nn.Conv1d(cin,ch,3,dilation=d,padding=2*d),nn.GELU(),nn.Dropout(0.1)]; cin=ch
        s.c=nn.ModuleList([l for l in layers]); s.o=nn.Linear(ch,HMAX)
    def forward(s,x,c,f):
        h=x.transpose(1,2)
        for l in s.c:
            h=l(h)
            if isinstance(l,nn.Conv1d): h=h[...,:x.shape[1]]
        return s.o(h[...,-1])
class TransF(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.i=nn.Linear(NV,d); s.p=nn.Parameter(torch.randn(L,d)*0.02)
        s.e=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,128,0.1,batch_first=True),2); s.o=nn.Linear(d,HMAX)
    def forward(s,x,c,f): return s.o(s.e(s.i(x)+s.p)[:,-1])
class PatchTST(nn.Module):  # channel-independent patches, shared transformer, flatten head on all channels
    def __init__(s,P=24,S=12,d=48):
        super().__init__(); s.P,s.S=P,S; n=(L-P)//S+1; s.n=n
        s.i=nn.Linear(P,d); s.p=nn.Parameter(torch.randn(n,d)*0.02)
        s.e=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,96,0.1,batch_first=True),2)
        s.o=nn.Linear(NV*n*d//4,HMAX); s.red=nn.Linear(d,d//4)
    def forward(s,x,c,f):
        B=x.shape[0]; z=x.transpose(1,2).unfold(2,s.P,s.S)  # B,NV,n,P
        z=s.i(z).reshape(B*NV,s.n,-1)+s.p; z=s.red(s.e(z)).reshape(B,-1); return s.o(z)
class DLinear(nn.Module):
    def __init__(s,k=25):
        super().__init__(); s.k=k; s.t=nn.Linear(L*NV,HMAX); s.r=nn.Linear(L*NV,HMAX)
    def forward(s,x,c,f):
        xt=x.transpose(1,2); pad=nn.functional.pad(xt,(s.k-1,0),mode='replicate'); tr=nn.functional.avg_pool1d(pad,s.k,1)
        return s.t(tr.flatten(1))+s.r((xt-tr).flatten(1))
class NBEATS(nn.Module):
    def __init__(s,w=256,nb=3):
        super().__init__(); s.b=nn.ModuleList([nn.Sequential(nn.Linear(L*NV if i==0 else L*NV,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,L*NV+HMAX)) for i in range(nb)])
    def forward(s,x,c,f):
        r=x.flatten(1); out=0
        for b in s.b:
            z=b(r); r=r-z[:,:L*NV]; out=out+z[:,L*NV:]
        return out
class iTrans(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.i=nn.Linear(L,d); s.e=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,128,0.1,batch_first=True),2); s.o=nn.Linear(d,HMAX)
    def forward(s,x,c,f): z=s.e(s.i(x.transpose(1,2))); return s.o(z[:,VARS.index('Biogas')])
class MLPm(nn.Module):
    def __init__(s): super().__init__(); s.m=nn.Sequential(nn.Linear(L*NV,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,HMAX))
    def forward(s,x,c,f): return s.m(x.flatten(1))
# ---- proposed: process-group lag-bin tokens + horizon queries (+ optional feed scenario)
BIN_EDGES=BINS
class PGHQ(nn.Module):
    def __init__(s,d=64,use_feed=False,nl=2):
        super().__init__(); s.use_feed=use_feed
        s.enc=nn.ModuleList([nn.Linear(len(g),d) for g in gi])
        s.gemb=nn.Parameter(torch.randn(len(gi),1,d)*0.02); s.lemb=nn.Parameter(torch.randn(1,len(BINS),d)*0.02)
        s.tok=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,128,0.1,batch_first=True),1)
        s.hemb=nn.Embedding(HMAX,d); s.cal=nn.Linear(4,d); s.fq=nn.Linear(3,d)
        s.att=nn.ModuleList([nn.MultiheadAttention(d,4,dropout=0.1,batch_first=True) for _ in range(nl)])
        s.ff=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,d)) for _ in range(nl)])
        s.ln=nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        s.o=nn.Linear(d,1); s.last_att=None
    def tokens(s,x):
        # x: B,L,NV  -> lag-bin means (lag a..b counted back from last step)
        toks=[]
        for (a,b) in BINS:
            seg=x[:,L-1-b:L-a]  # lags a..b
            toks.append(seg.mean(1))
        tb=torch.stack(toks,1)  # B,nb,NV
        G=[s.enc[k](tb[:,:,g]) for k,g in enumerate(gi)]  # each B,nb,d
        T=torch.stack(G,1)+s.gemb.unsqueeze(0)+s.lemb.unsqueeze(1)  # B,ng,nb,d
        B=x.shape[0]; T=T.reshape(B,-1,T.shape[-1]); return s.tok(T)
    def forward(s,x,c,f):
        T=s.tokens(x); B=x.shape[0]
        q=s.hemb.weight.unsqueeze(0).expand(B,-1,-1)+s.cal(c)
        if s.use_feed: q=q+s.fq(f)
        atts=[]
        for a,ff,ln in zip(s.att,s.ff,s.ln):
            o,w=a(ln(q),T,T,need_weights=True,average_attn_weights=True); q=q+o; q=q+ff(q); atts.append(w)
        s.last_att=atts[-1].detach()
        return s.o(q).squeeze(-1)
class PGHQ2(PGHQ):
    def __init__(s,**kw):
        super().__init__(**kw); s.lin=nn.Linear(len(BINS)*NV,HMAX)
    def forward(s,x,c,f):
        toks=torch.stack([x[:,L-1-b:L-a].mean(1) for (a,b) in BINS],1).flatten(1)
        return super().forward(x,c,f)+s.lin(toks)
def train(model,epochs=80,lr=5e-4,bs=128,seed=0,patience=12):
    torch.manual_seed(seed); np.random.seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    Xtr,Ytr,Btr,Ctr,Ftr=D['tr']; tgt=(Ytr-Btr[:,None])/ysd
    best=1e9; bst=None; bad=0
    for ep in range(epochs):
        model.train(); perm=torch.randperm(len(Xtr))
        for i in range(0,len(perm),bs):
            j=perm[i:i+bs]; opt.zero_grad(); l=nn.functional.smooth_l1_loss(model(Xtr[j],Ctr[j],Ftr[j]),tgt[j],beta=0.5); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        vl=evalp(model,'va',raw=True)
        if vl<best-1e-4: best=vl; bst={k:v.clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(bst); return model,ep
def predict(model,k):
    model.eval(); X,Y,B,C,F=D[k]; out=[]
    with torch.no_grad():
        for i in range(0,len(X),512): out.append(model(X[i:i+512],C[i:i+512],F[i:i+512]))
    return (torch.cat(out)*ysd+B[:,None]).numpy()
def evalp(model,k,raw=False):
    p=predict(model,k); Y=D[k][1].numpy(); return np.sqrt(np.mean((p-Y)**2))
if __name__=='__main__':
    MODELS={'MLP':MLPm,'LSTM':LSTMm,'GRU':GRUm,'TCN':TCN,'Transformer':TransF,'PatchTST':PatchTST,'DLinear':DLinear,'N-BEATS':NBEATS,'iTransformer':iTrans,
            'PG-HQ (proposed, past-only)':lambda:PGHQ2(d=32,use_feed=False),'PG-HQ (proposed, feed-conditioned)':lambda:PGHQ2(d=32,use_feed=True)}
    LR={'PG-HQ (proposed, past-only)':1e-3,'PG-HQ (proposed, feed-conditioned)':1e-3}
    sel=sys.argv[1:] or list(MODELS)
    Y=D['te'][1].numpy(); res=[]; P={}
    for nm in sel:
        Ps=[]; Pv=[]
        for seed in [0,1,2]:
            t0=time.time(); m,ep=train(MODELS[nm](),seed=seed,lr=LR.get(nm,5e-4)); p=predict(m,'te'); Ps.append(p); Pv.append(predict(m,'va'))
            for h in [1,6,24]:
                e=Y[:,h-1]-p[:,h-1]; r2=1-np.sum(e**2)/np.sum((Y[:,h-1]-Y[:,h-1].mean())**2)
                res.append(dict(model=nm,h=h,seed=seed,RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(np.abs(e)),R2=r2))
            print(nm,'ep',ep,round(time.time()-t0),'s',[round(r['R2'],3) for r in res[-3:]],flush=True)
            if 'PG-HQ' in nm: torch.save(m.state_dict(),f"pghq_{'feed' if 'feed' in nm else 'past'}_s{seed}.pt")
        P[nm]=np.mean(Ps,0); P[nm+'__val']=np.mean(Pv,0)
    pd.DataFrame(res).to_csv('bench_dl_'+('_'.join(s.split()[0] for s in sel) if sys.argv[1:] else 'all')+'.csv',index=False)
    pd.to_pickle({'P':P,'Y':Y,'Yv':D['va'][1].numpy(),'origins':idx[o_te],'origins_val':idx[o_va]},'preds_dl_'+('_'.join(s.split()[0] for s in sel) if sys.argv[1:] else 'all')+'.pkl')
    print('n train/val/test origins',len(o_tr),len(o_va),len(o_te))
