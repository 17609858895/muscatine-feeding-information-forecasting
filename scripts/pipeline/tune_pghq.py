from dl import *
import itertools
Y=D['te'][1].numpy()
class PGHQ2(PGHQ):
    def __init__(s,**kw):
        super().__init__(**kw); s.lin=nn.Linear(len(BINS)*NV,HMAX)
    def forward(s,x,c,f):
        toks=torch.stack([x[:,L-1-b:L-a].mean(1) for (a,b) in BINS],1).flatten(1)
        return super().forward(x,c,f)+s.lin(toks)
for (lr,pat,cls,d) in [(3e-4,15,PGHQ,64),(3e-4,15,PGHQ2,64),(1e-3,15,PGHQ2,32)]:
    for feed in [False,True]:
        t0=time.time(); m,ep=train(cls(d=d,use_feed=feed),epochs=80,lr=lr,patience=pat); p=predict(m,'te')
        r=[1-np.sum((Y[:,h-1]-p[:,h-1])**2)/np.sum((Y[:,h-1]-Y[:,h-1].mean())**2) for h in [1,6,24]]
        print(cls.__name__,lr,d,'feed',feed,'ep',ep,round(time.time()-t0),'s',np.round(r,3),flush=True)
