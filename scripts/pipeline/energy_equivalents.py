"""Indicative fuel-energy equivalents of metered biogas flows and forecast metrics.

Methane content was not measured at the Muscatine WRRF. Flows (m3 h-1, uncorrected to
standard conditions) are converted with an assumed methane volume fraction and the lower
heating value (LHV) of methane, 35.8 MJ m-3 (0 degC, 101.325 kPa). 55% and 65% methane
are reported as sensitivity bounds. Output: results/energy_equivalents.csv
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = os.environ.get('SCADA_CSV', str(ROOT / 'data' / 'SCADA-raw.csv'))
LHV_CH4_MJ_M3 = 35.8
CH4_FRACTIONS = {'low': 0.55, 'central': 0.60, 'high': 0.65}
CFM_TO_M3H = 0.0283168466 * 60

def kw_per_m3h(f):
    return f * LHV_CH4_MJ_M3 / 3.6  # MJ m-3 -> kWh m-3, so m3 h-1 -> kW

# Hourly aggregation as in prep.py / gas_use_split.py
d = pd.read_csv(SRC, parse_dates=['Timestamp'], usecols=['Timestamp', 'Biogas', 'Biogas_burner', 'Biogas_boiler'])
d = d.sort_values('Timestamp').set_index('Timestamp')
h = d.resample('1h').mean(); cnt = d['Biogas'].resample('1h').count()
h = h[h.index >= '2022-03-18']; cnt = cnt[cnt.index >= '2022-03-18']
h.loc[cnt < 30] = np.nan
h = h.dropna() * CFM_TO_M3H

rows = []
def add(label, m3h, source):
    r = {'quantity': label, 'flow_m3_h': m3h, 'source': source}
    for k, f in CH4_FRACTIONS.items():
        r[f'kW_{k}'] = m3h * kw_per_m3h(f)
    rows.append(r)

add('Mean metered biogas, all complete-routing hours', h['Biogas'].mean(), 'SCADA, %d h' % len(h))
add('Mean burner (flare) flow, all complete-routing hours', h['Biogas_burner'].mean(), 'SCADA')
add('Mean boiler flow, all complete-routing hours', h['Biogas_boiler'].mean(), 'SCADA')
add('Mean metered biogas, test-period window', h.loc['2022-12-15':'2023-02-27', 'Biogas'].mean(), 'SCADA')
# Forecast metrics reported in the manuscript (Table 2; Sections 3.3 and 3.8)
for lab, v in [('Day-ahead RMSE, past-only ensemble', 57.48), ('Day-ahead RMSE, feed-conditioned ensemble', 45.52),
               ('Day-ahead RMSE reduction, ensemble', 11.96), ('7-day mean flow, minimum', 90.0), ('7-day mean flow, maximum', 290.0),
               ('Mean outcome, commitment test origins', 209.3),
               ('Mean fulfilled commitment, persistence, alpha=0.05', 81.8),
               ('Mean fulfilled commitment, past-only ensemble, alpha=0.05', 101.0),
               ('Mean fulfilled commitment, feed-conditioned ensemble, alpha=0.05', 108.9)]:
    add(lab, v, 'manuscript result')

out = pd.DataFrame(rows)
out['burner_share_of_integrated_flow'] = np.nan
out.loc[0, 'burner_share_of_integrated_flow'] = h['Biogas_burner'].sum() / (h['Biogas_burner'].sum() + h['Biogas_boiler'].sum())
out.loc[0, 'flared_volume_m3'] = h['Biogas_burner'].sum()
out.loc[0, 'flared_energy_MWh_central'] = h['Biogas_burner'].sum() * kw_per_m3h(0.60) / 1000
dst = ROOT / 'results' / 'energy_equivalents.csv'
out.to_csv(dst, index=False, float_format='%.4f')
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)
print(out.round(3).to_string())
