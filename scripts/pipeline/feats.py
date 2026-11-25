import pandas as pd, numpy as np
H=pd.read_pickle('hourly.pkl')
H=H.interpolate(limit=3)
TARGET='Biogas'
GROUPS={
 'Gas history':['Biogas'],
 'Gas storage':['H-Dig1_FT','H-Dig2_FT'],
 'Co-substrate feed':['Q-HSW1_GPM','Q-HSW2_GPM','H-HSW_ft','V-HSW-DIG2-yest_gal'],
 'Sludge feed':['Q-PS_on','Q-TWAS_GPM'],
 'Digester thermal':['D1_TEMPERATURE','D2_TEMPERATURE'],
 'Plant hydraulics':['Q-influent_MGD','Q-Aeration_SCFM','Q-RAS_MGD','Q-WAS1_MGD','Q-WAS2_MGD'],
}
BINS=[(0,0),(1,2),(3,5),(6,11),(12,23),(24,47),(48,71),(72,167)]
BIN_LAB=['0','1-2','3-5','6-11','12-23','24-47','48-71','72-167']
HORIZONS=[1,6,24]
SPLIT_VAL='2022-11-15'; SPLIT_TEST='2022-12-15'
def build(H=H):
    cols={}; meta=[]
    for g,vs in GROUPS.items():
        for v in vs:
            s=H[v]
            for (a,b),lab in zip(BINS,BIN_LAB):
                # mean of s over lags a..b  (s_{t-a} ... s_{t-b})
                f=s.shift(a).rolling(b-a+1).mean()
                name=f'{v}|L{lab}'; cols[name]=f; meta.append((name,g,v,lab))
    X=pd.DataFrame(cols,index=H.index)
    return X, pd.DataFrame(meta,columns=['feature','group','var','lagbin'])
def calendar(idx,h):
    t=idx+pd.Timedelta(hours=h)
    return pd.DataFrame({'hod_sin':np.sin(2*np.pi*t.hour/24),'hod_cos':np.cos(2*np.pi*t.hour/24),
                         'dow_sin':np.sin(2*np.pi*t.dayofweek/7),'dow_cos':np.cos(2*np.pi*t.dayofweek/7)},index=idx)
def target(h): return H[TARGET].shift(-h)
def split_idx(idx):
    tr=idx<SPLIT_VAL; va=(idx>=SPLIT_VAL)&(idx<SPLIT_TEST); te=idx>=SPLIT_TEST
    return tr,va,te
