from pathlib import Path
"""Descriptive split of metered biogas between the waste-gas burner and the boiler
over the modelled SCADA period (hourly aggregation as in prep.py)."""
import os, numpy as np, pandas as pd
src = os.environ.get('SCADA_CSV', str(Path(__file__).resolve().parents[2]/'data/SCADA-raw.csv'))
d = pd.read_csv(src, parse_dates=['Timestamp'], usecols=['Timestamp','Biogas','Biogas_burner','Biogas_boiler']).sort_values('Timestamp').set_index('Timestamp')
h = d.resample('1h').mean(); cnt = d['Biogas'].resample('1h').count()
h = h[h.index >= '2022-03-18']; cnt = cnt[cnt.index >= '2022-03-18']
h.loc[cnt < 30] = np.nan
h = h.dropna()
CFM2M3H = 0.0283168466 * 60
out = []
def rep(name, x):
    tot = x['Biogas_burner'].sum() + x['Biogas_boiler'].sum()
    out.append(f"{name}: hours={len(x)}, burner share={x['Biogas_burner'].sum()/tot:.3f}, boiler share={x['Biogas_boiler'].sum()/tot:.3f}, "
               f"mean total={x['Biogas'].mean()*CFM2M3H:.1f} m3/h, mean burner={x['Biogas_burner'].mean()*CFM2M3H:.1f} m3/h, mean boiler={x['Biogas_boiler'].mean()*CFM2M3H:.1f} m3/h, "
               f"boiler p95={x['Biogas_boiler'].quantile(.95)*CFM2M3H:.1f} m3/h, hours with burner>1 cfm={ (x['Biogas_burner']>1).mean():.3f}, sum check max abs diff={ (x['Biogas']-x['Biogas_burner']-x['Biogas_boiler']).abs().max():.3f} cfm")
rep('all', h)
rep('train 2022-03-25..2022-11-13', h.loc['2022-03-25':'2022-11-13'])
rep('test 2022-12-15..2023-02-27', h.loc['2022-12-15':'2023-02-27'])
for m, g in h.groupby(h.index.to_period('M')):
    tot = g['Biogas_burner'].sum() + g['Biogas_boiler'].sum()
    out.append(f"{m}: burner share={g['Biogas_burner'].sum()/tot:.3f}")
txt = '\n'.join(out); print(txt)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','results','gas_use_split.txt'),'w').write(txt)
