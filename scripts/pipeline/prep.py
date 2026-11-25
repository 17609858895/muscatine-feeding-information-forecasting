from pathlib import Path
import pandas as pd, numpy as np
d=pd.read_csv(__import__('os').environ.get('SCADA_CSV',str(Path(__file__).resolve().parents[2]/'data/SCADA-raw.csv')),parse_dates=['Timestamp']).sort_values('Timestamp').set_index('Timestamp')
d['Q-PS_on']=(d['Q-PS_MGD']>1).astype(float)
d['Q-TWAS_on']=(d['Q-TWAS_GPM']>5).astype(float)
h=d.resample('1h').mean()
cnt=d['Biogas'].resample('1h').count()
h['n_min']=cnt
# daily-reset counters: take hourly max of cumulative to see
print(h.shape, (cnt<45).sum(), 'hours with <45 min')
h=h[h.index>='2022-03-18']
h.loc[cnt<30]=np.nan
print(h.isna().any(axis=1).sum(),'nan hours')
h.to_pickle('hourly.pkl')
s=h['Biogas']
print(s.describe())
for lag in [1,2,3,6,12,24,48,168]: print(lag, round(s.autocorr(lag),3))
dH=h[['H-Dig1_FT','H-Dig2_FT']].diff()
print('corr Biogas vs dH1,dH2', s.corr(dH['H-Dig1_FT']), s.corr(dH['H-Dig2_FT']))
print('corr dBiogas vs dH', s.diff().corr(dH['H-Dig1_FT']), s.diff().corr(dH['H-Dig2_FT']))
print(h.corr()['Biogas'].sort_values().round(3).to_string())
print((d['Biogas']<1).mean(), 'fraction minutes near zero biogas')
print(h['Biogas'].groupby(h.index.hour).mean().round(1).tolist())
print(h['Biogas'].groupby(h.index.month).mean().round(1).to_string())
