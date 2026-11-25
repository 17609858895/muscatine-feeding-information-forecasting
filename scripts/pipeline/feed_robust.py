import pandas as pd, numpy as np, pickle
from common import *
o=pickle.load(open('tab_final.pkl','rb')); X0,_=build(); rng=np.random.default_rng(1)
rows=[]
for h in [6,24]:
    m=o['models'][(h,'LGB-Δ [feed]')]; te=idx[o_te]; yt=pd.Series(target(h).values,index=idx)[te].values
    Xp=pd.concat([X0,calendar(X0.index,h)],axis=1)
    for shift in [0,1,2,3,6]:
        for noise in [0,0.1,0.2,0.4]:
            r=[]
            for rep in range(5 if noise>0 else 1):
                s=hsw.shift(shift*rng.choice([-1,1])) if shift else hsw
                s=s*(1+noise*rng.standard_normal(len(s))).clip(0)
                ff=future_feed(h); ff['Planned HSW|F']=s[::-1].rolling(h).mean()[::-1].shift(-1)
                X=pd.concat([Xp,ff],axis=1).loc[te]; p=m.predict(X)+X['Biogas|L0'].values; r.append(metrics(yt,p)['R2'])
            rows.append(dict(h=h,timing_shift_h=shift,mult_noise=noise,R2=np.mean(r)))
R=pd.DataFrame(rows); R.to_csv('feed_schedule_robustness.csv',index=False); print(R.pivot_table(index=['h','timing_shift_h'],columns='mult_noise',values='R2').round(3).to_string())
