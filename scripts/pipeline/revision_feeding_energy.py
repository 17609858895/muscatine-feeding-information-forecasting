"""Revision analyses (Bioresource Technology, first revision, 2026-11-26).

Answers Reviewer 1 requests for basic feeding data, HSW amounts and yield potential,
practice-oriented response summaries, and a short economic estimate of flared gas.
All quantities are descriptive and use the public SCADA and laboratory records.

Outputs (results/revision_20261126/):
  feeding_by_digester.csv   daily feeding statistics by digester, period and day type
  lab_loading.csv           laboratory volumes, HSW COD/VS loads and HRT
  hsw_daily_association.csv daily regression of metered biogas on HSW volume (HAC SE)
  hsw_lag_summary.csv       detrended HSW-to-biogas cross-correlation summary
  energy_value.csv          indicative energy and electricity value of flared gas
  summary.txt               numbers quoted in the revised manuscript and response
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data'
OUT = ROOT / 'results' / 'revision_20261126'
OUT.mkdir(parents=True, exist_ok=True)

CFM = 0.0283168466 * 60          # ft3 min-1 -> m3 h-1
GPM = 0.003785411784 * 60        # US gal min-1 -> m3 h-1
GAL = 0.003785411784             # US gal -> m3
DIGESTER_VOLUME = 2 * 1840.0     # m3, two digesters (Section 2.1)
START, END = '2022-03-18', '2023-02-28'
TRAIN = ('2022-03-25', '2022-11-13')
TEST = ('2022-12-15', '2023-02-27')
lines = []
def say(s=''):
    print(s); lines.append(s)

# ---------------------------------------------------------------- SCADA, hourly as in prep.py
cols = ['Timestamp', 'Q-HSW1_GPM', 'Q-HSW2_GPM', 'Q-TWAS_GPM', 'Q-PS_MGD', 'Biogas', 'Biogas_burner',
        'Biogas_boiler', 'H-Dig1_FT', 'H-Dig2_FT', 'D1_TEMPERATURE', 'D2_TEMPERATURE']
d = pd.read_csv(os.environ.get('SCADA_CSV', RAW / 'SCADA-raw.csv'), usecols=cols, parse_dates=['Timestamp'])
d = d.sort_values('Timestamp').set_index('Timestamp')
d['PS_on'] = (d['Q-PS_MGD'] > 1).astype(float)
h = d.resample('1h').mean(); cnt = d['Biogas'].resample('1h').count()
h = h[h.index >= START]; cnt = cnt[cnt.index >= START]
h.loc[cnt < 30] = np.nan
h['HSW1'] = h['Q-HSW1_GPM'] * GPM; h['HSW2'] = h['Q-HSW2_GPM'] * GPM
h['HSW'] = h['HSW1'] + h['HSW2']; h['TWAS'] = h['Q-TWAS_GPM'] * GPM
h['gas'] = h['Biogas'] * CFM; h['burner'] = h['Biogas_burner'] * CFM; h['boiler'] = h['Biogas_boiler'] * CFM

# daily volumes from complete days (>= 23 valid hours), scaled to 24 h
valid = h['gas'].notna().resample('D').sum()
day = h[['HSW1', 'HSW2', 'HSW', 'TWAS', 'gas', 'burner', 'boiler', 'PS_on', 'H-Dig1_FT', 'H-Dig2_FT']].resample('D').mean()
day = day[valid >= 23]
for c in ['HSW1', 'HSW2', 'HSW', 'TWAS', 'gas', 'burner', 'boiler']:
    day[c] = day[c] * 24.0                     # m3 d-1
day['weekend'] = day.index.dayofweek >= 5
say(f'SCADA complete days: {len(day)} ({day.index.min().date()} to {day.index.max().date()})')

def period(df, a, b):
    return df.loc[a:b]
rows = []
for pname, (a, b) in {'record': (START, END), 'training': TRAIN, 'test': TEST}.items():
    sub = period(day, a, b)
    for dtype, s2 in [('all days', sub), ('working days', sub[~sub.weekend]), ('weekends', sub[sub.weekend])]:
        r = {'period': pname, 'day_type': dtype, 'n_days': len(s2)}
        for c in ['HSW1', 'HSW2', 'HSW', 'TWAS', 'gas']:
            r[f'{c}_m3_d_mean'] = s2[c].mean(); r[f'{c}_m3_d_sd'] = s2[c].std()
        r['HSW1_feed_day_frac'] = (s2['HSW1'] > 0.5).mean(); r['HSW2_feed_day_frac'] = (s2['HSW2'] > 0.5).mean()
        r['PS_on_frac_mean'] = s2['PS_on'].mean()
        rows.append(r)
fd = pd.DataFrame(rows); fd.to_csv(OUT / 'feeding_by_digester.csv', index=False, float_format='%.4f')
say('\nDaily feeding by digester (m3 d-1, mean; SCADA):')
say(fd[['period', 'day_type', 'n_days', 'HSW1_m3_d_mean', 'HSW2_m3_d_mean', 'HSW_m3_d_mean', 'TWAS_m3_d_mean',
        'gas_m3_d_mean', 'HSW1_feed_day_frac', 'HSW2_feed_day_frac', 'PS_on_frac_mean']].round(3).to_string(index=False))
# hourly feeding intensity when HSW is fed
fh = h['HSW'][h['HSW'] > 0.23]
say(f'\nHSW hours with flow > 0.23 m3/h: {len(fh)} of {h["HSW"].notna().sum()} ({len(fh)/h["HSW"].notna().sum():.3f}); '
    f'median flow while feeding {fh.median():.2f} m3/h, 90th pct {fh.quantile(.9):.2f}')
for nm in ['HSW1', 'HSW2']:
    x = h[nm]
    say(f'{nm}: hours fed (>0.23) fraction {(x > 0.23).mean():.3f}; working-day mean {x[x.index.dayofweek<5].mean():.3f} m3/h, weekend {x[x.index.dayofweek>=5].mean():.3f} m3/h')
say(f'Digester temperatures (degF): D1 {h["D1_TEMPERATURE"].mean():.1f}, D2 {h["D2_TEMPERATURE"].mean():.1f} '
    f'-> {(h["D1_TEMPERATURE"].mean()-32)/1.8:.1f} and {(h["D2_TEMPERATURE"].mean()-32)/1.8:.1f} degC')

# ---------------------------------------------------------------- laboratory daily records
lab = pd.read_csv(os.environ.get('LABS_CSV', RAW / 'LABS-raw.csv'), encoding='utf-8-sig')
lab['date'] = pd.to_datetime(dict(year=lab.Year, month=lab.Month, day=lab.Day))
lab = lab.set_index('date').sort_index()
L = pd.DataFrame(index=lab.index)
L['V_TWAS'] = lab['V-TWAS_gal'] * GAL; L['V_PS'] = lab['V-PS_gal'] * GAL; L['V_HSW'] = lab['V-HSW_gal'] * GAL
L['V_FOG'] = lab['FOG_gal'] * GAL
L['HSW_COD_kg_m3'] = lab['HSW-COD_mgL'] / 1000.0
L['HSW_VS_pct'] = lab['HSW-VS_percent']
L['SRT_d'] = lab['SRT']
L['lab_gas'] = lab['Biogas'] * CFM * 24.0      # m3 d-1 (daily mean flow in ft3/min)
Lr = L.loc[START:END].copy()
# weekday-only COD/VS analyses: carry the last measurement forward for up to 7 days for loading
Lr['COD_ff'] = Lr['HSW_COD_kg_m3'].replace(0, np.nan).ffill(limit=7)
Lr['VS_ff'] = Lr['HSW_VS_pct'].replace(0, np.nan).ffill(limit=7)
Lr['V_feed'] = Lr[['V_TWAS', 'V_PS', 'V_HSW']].sum(axis=1, min_count=3)
Lr['HSW_vol_share'] = Lr['V_HSW'] / Lr['V_feed']
Lr['HRT_d'] = DIGESTER_VOLUME / Lr['V_feed']
Lr['HSW_COD_load_kg_d'] = Lr['V_HSW'] * Lr['COD_ff']
Lr['HSW_VS_load_kg_d'] = Lr['V_HSW'] * Lr['VS_ff'] / 100 * 1000   # density 1 t m-3 assumed
Lr.to_csv(OUT / 'lab_loading.csv', float_format='%.4f')
cod_meas = Lr['HSW_COD_kg_m3'].replace(0, np.nan).dropna()
vs_meas = Lr['HSW_VS_pct'].replace(0, np.nan).dropna()
say(f'\nLaboratory days in record: {len(Lr)}; HSW COD measurements {len(cod_meas)}, VS measurements {len(vs_meas)}')
for c, u in [('V_HSW', 'm3/d'), ('V_PS', 'm3/d'), ('V_TWAS', 'm3/d'), ('V_FOG', 'm3/d'), ('V_feed', 'm3/d'), ('HSW_vol_share', '-'),
             ('HRT_d', 'd'), ('SRT_d', 'd'), ('HSW_COD_load_kg_d', 'kg COD/d'), ('HSW_VS_load_kg_d', 'kg VS/d'), ('lab_gas', 'm3/d')]:
    x = Lr[c].replace([np.inf, -np.inf], np.nan).dropna()
    say(f'  {c:18s} mean {x.mean():10.3f}  median {x.median():10.3f}  IQR {x.quantile(.25):9.3f}-{x.quantile(.75):9.3f} {u}  n={len(x)}')
say(f'  HSW COD (kg/m3): mean {cod_meas.mean():.1f}, median {cod_meas.median():.1f}, IQR {cod_meas.quantile(.25):.1f}-{cod_meas.quantile(.75):.1f}, CV {cod_meas.std()/cod_meas.mean():.2f}')
say(f'  HSW VS (%): mean {vs_meas.mean():.2f}, median {vs_meas.median():.2f}, IQR {vs_meas.quantile(.25):.2f}-{vs_meas.quantile(.75):.2f}, CV {vs_meas.std()/vs_meas.mean():.2f}')
for pname, (a, b) in {'training': TRAIN, 'test': TEST}.items():
    s = Lr.loc[a:b]
    say(f'  {pname}: V_HSW {s.V_HSW.mean():.2f} m3/d, HSW share {s.HSW_vol_share.mean():.3f}, HRT {s.HRT_d.mean():.1f} d, '
        f'HSW COD load {s.HSW_COD_load_kg_d.mean():.0f} kg/d')
# working-day vs weekend lab HSW volumes
wk = Lr.index.dayofweek >= 5
say(f'  Lab HSW volume: working days {Lr.V_HSW[~wk].mean():.2f} m3/d, weekends {Lr.V_HSW[wk].mean():.2f} m3/d')

# agreement between laboratory and SCADA HSW volumes
j = Lr[['V_HSW']].join(day[['HSW']], how='inner').dropna()
say(f'  Lab vs SCADA daily HSW volume: n={len(j)}, r={j.V_HSW.corr(j.HSW):.3f}, lab/SCADA mean ratio {j.V_HSW.mean()/j.HSW.mean():.3f}')

# theoretical methane potential of HSW COD (stoichiometric 0.35 m3 CH4 per kg COD at 0 degC, 1 atm)
B0 = 0.35
cod_load = Lr['HSW_COD_load_kg_d'].dropna()
ch4_pot = cod_load.mean() * B0
gas_day_mean = day['gas'].mean()
say(f'\nHSW COD load mean {cod_load.mean():.0f} kg/d -> stoichiometric CH4 potential {ch4_pot:.0f} m3/d (STP)')
for f in (0.55, 0.60, 0.65):
    say(f'  biogas equivalent at {int(f*100)}% CH4: {ch4_pot/f:.0f} m3/d = {ch4_pot/f/gas_day_mean*100:.1f}% of mean metered biogas ({gas_day_mean:.0f} m3/d)')
say(f'  per m3 HSW: {B0*cod_meas.mean():.1f} m3 CH4 = {B0*cod_meas.mean()/0.60:.1f} m3 biogas at 60% CH4 (mean COD)')

# ---------------------------------------------------------------- daily association: metered biogas vs HSW volume
D = day[['gas', 'HSW', 'TWAS', 'PS_on', 'H-Dig1_FT', 'H-Dig2_FT']].join(Lr[['V_PS', 'V_TWAS']], how='left')
D = D.asfreq('D')
D['HSW_l1'] = D['HSW'].shift(1); D['HSW_l2'] = D['HSW'].shift(2); D['HSW_l3'] = D['HSW'].shift(3)
D['dCover'] = (D['H-Dig1_FT'] + D['H-Dig2_FT']).diff()
specs = {
    'HSW d0+d1': ['HSW', 'HSW_l1'],
    'HSW d0+d1+d2': ['HSW', 'HSW_l1', 'HSW_l2'],
    'HSW d0..d3': ['HSW', 'HSW_l1', 'HSW_l2', 'HSW_l3'],
    'HSW d0+d1 + sludge volumes': ['HSW', 'HSW_l1', 'V_PS', 'V_TWAS'],
    'HSW d0+d1 + sludge + cover change': ['HSW', 'HSW_l1', 'V_PS', 'V_TWAS', 'dCover'],
}
arows = []
for name, xs in specs.items():
    z = D[['gas'] + xs].dropna()
    X = sm.add_constant(z[xs]); m = sm.OLS(z['gas'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    hs = [c for c in xs if c.startswith('HSW')]
    w = np.zeros(len(m.params)); w[[list(m.params.index).index(c) for c in hs]] = 1
    t = m.t_test(w); est = float(np.squeeze(t.effect)); se = float(np.squeeze(t.sd)); ci = np.squeeze(t.conf_int())
    share = est * z['HSW'].mean() / z['gas'].mean()
    arows.append({'spec': name, 'n_days': len(z), 'sum_HSW_coef_m3_per_m3': est, 'se_HAC': se, 'ci_low': ci[0], 'ci_high': ci[1],
                  'mean_HSW_m3_d': z['HSW'].mean(), 'mean_gas_m3_d': z['gas'].mean(), 'implied_share': share, 'r2': m.rsquared})
A = pd.DataFrame(arows); A.to_csv(OUT / 'hsw_daily_association.csv', index=False, float_format='%.4f')
say('\nDaily association of metered biogas with HSW volume (sum of lag coefficients, HAC 7-day):')
say(A.round(3).to_string(index=False))

# long laboratory record sensitivity (2020-2023, laboratory biogas and HSW volume)
LL = L[['lab_gas', 'V_HSW', 'V_PS', 'V_TWAS']].asfreq('D')
LL['V_HSW_l1'] = LL['V_HSW'].shift(1)
z = LL.dropna(); xs = ['V_HSW', 'V_HSW_l1', 'V_PS', 'V_TWAS']
m = sm.OLS(z['lab_gas'], sm.add_constant(z[xs])).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
t = m.t_test([0, 1, 1, 0, 0]); ci = np.squeeze(t.conf_int())
say(f'Laboratory record {z.index.min().date()}..{z.index.max().date()} (n={len(z)}): HSW d0+d1 coefficient '
    f'{float(np.squeeze(t.effect)):.1f} m3 biogas per m3 HSW (95% CI {ci[0]:.1f} to {ci[1]:.1f}); implied share '
    f'{float(np.squeeze(t.effect))*z.V_HSW.mean()/z.lab_gas.mean()*100:.1f}%')

# ---------------------------------------------------------------- detrended HSW-to-biogas lag summary (as Fig. 2d)
def dt(s):
    s = s.interpolate(); return s - s.rolling(168, center=True, min_periods=84).mean()
yb = dt(h['gas']); xb = dt(h['HSW'])
lags = np.arange(0, 97)
r = np.array([xb.corr(yb.shift(-k)) for k in lags])
kmax = int(lags[np.nanargmax(r)])
half = int(lags[(lags > kmax) & (r < r.max() / 2)][0])
zero = int(lags[(lags > kmax) & (r < 0)][0])
kmin = int(lags[np.nanargmin(r)])
pd.DataFrame({'lag_h': lags, 'r': r}).to_csv(OUT / 'hsw_lag_summary.csv', index=False, float_format='%.4f')
say(f'\nDetrended HSW->biogas cross-correlation: peak {r.max():.2f} at {kmax} h; half of peak by {half} h; '
    f'zero crossing at {zero} h; minimum {r.min():.2f} at {kmin} h')
for nm in ['HSW1', 'HSW2']:
    xb2 = dt(h[nm]); rr = np.array([xb2.corr(yb.shift(-k)) for k in lags])
    say(f'  {nm}: peak {np.nanmax(rr):.2f} at {int(lags[np.nanargmax(rr)])} h')

# ---------------------------------------------------------------- indicative energy and electricity value of flared gas
LHV = 35.8          # MJ m-3 methane (0 degC, 101.325 kPa); metered flows are uncorrected
hh = h[['gas', 'burner', 'boiler']].dropna()
burn = hh['burner'].mean(); share = hh['burner'].sum() / (hh['burner'].sum() + hh['boiler'].sum())
erows = []
for f in (0.55, 0.60, 0.65):
    fuel_kw = burn * f * LHV / 3.6
    for eta in (0.30, 0.35):
        e_mwh = fuel_kw * eta * 8760 / 1000
        for price_name, price in [('Iowa industrial 2025', 0.0715), ('Iowa commercial 2025', 0.1084)]:
            erows.append({'ch4_fraction': f, 'electrical_efficiency': eta, 'price_basis': price_name, 'price_usd_kwh': price,
                          'burner_flow_m3_h': burn, 'flared_m3_yr': burn * 8760, 'fuel_kW_LHV': fuel_kw,
                          'fuel_MWh_yr': fuel_kw * 8760 / 1000, 'electric_kW': fuel_kw * eta, 'electric_MWh_yr': e_mwh,
                          'value_kUSD_yr': e_mwh * 1000 * price / 1000})
E = pd.DataFrame(erows); E.to_csv(OUT / 'energy_value.csv', index=False, float_format='%.4f')
say(f'\nFlared gas: burner share {share:.3f}; mean burner flow {burn:.1f} m3/h; {burn*8760/1e6:.2f} million m3/yr (annualized)')
c = E[(E.ch4_fraction == 0.60)]
say(f'  fuel energy at 60% CH4: {c.fuel_kW_LHV.iloc[0]:.0f} kW = {c.fuel_MWh_yr.iloc[0]:.0f} MWh/yr (range 55-65%: '
    f'{E.fuel_MWh_yr.min():.0f}-{E.fuel_MWh_yr.max():.0f})')
say(f'  electricity at 30-35% efficiency: {E.electric_kW.min():.0f}-{E.electric_kW.max():.0f} kW, {E.electric_MWh_yr.min():.0f}-{E.electric_MWh_yr.max():.0f} MWh/yr; '
    f'central (60%, 35%) {c[c.electrical_efficiency==0.35].electric_MWh_yr.iloc[0]:.0f} MWh/yr')
say(f'  gross electricity value: {E.value_kUSD_yr.min():.0f}-{E.value_kUSD_yr.max():.0f} kUSD/yr; central 60%/35%: '
    f"{c[(c.electrical_efficiency==0.35)].value_kUSD_yr.min():.0f}-{c[(c.electrical_efficiency==0.35)].value_kUSD_yr.max():.0f} kUSD/yr")
# energy equivalent of forecast quantities (60% CH4, 35% electrical)
for lab_, v in [('day-ahead ensemble RMSE reduction', 11.96), ('6 h matched-risk commitment gain (4.34 pp of mean outcome 209.3)', 0.0434 * 209.3)]:
    say(f'  {lab_}: {v:.1f} m3/h -> {v*0.6*LHV/3.6:.0f} kW fuel, {v*0.6*LHV/3.6*0.35:.0f} kW electric')

# ---------------------------------------------------------------- SI tables S23-S25 (values as printed)
groups = {'Full record': (START, END, None), 'Training': (*TRAIN, None), 'Test': (*TEST, None),
          'Working days': (START, END, False), 'Weekends': (START, END, True)}
def sel(df, a, b, wkend):
    s = df.loc[a:b]
    return s if wkend is None else s[(s.index.dayofweek >= 5) == wkend]
t23 = {}
for g, (a, b, w) in groups.items():
    s, l = sel(day, a, b, w), sel(Lr, a, b, w)
    t23[g] = {
        'Complete SCADA days': f'{len(s)}',
        'HSW to digester 1 (m³ d⁻¹)': f'{s.HSW1.mean():.1f}',
        'HSW to digester 2 (m³ d⁻¹)': f'{s.HSW2.mean():.1f}',
        'Combined HSW (m³ d⁻¹)': f'{s.HSW.mean():.1f}',
        'Days with HSW to digester 1 (%)': f'{(s.HSW1 > 0.5).mean()*100:.0f}',
        'Days with HSW to digester 2 (%)': f'{(s.HSW2 > 0.5).mean()*100:.0f}',
        'TWAS (m³ d⁻¹)': f'{s.TWAS.mean():.1f}',
        'PS pump on-time (%)': f'{s.PS_on.mean()*100:.1f}',
        'Metered biogas (m³ d⁻¹)': f'{s.gas.mean():,.0f}',
        'Laboratory days': f'{len(l)}',
        'PS volume (m³ d⁻¹)': f'{l.V_PS.mean():.1f}',
        'TWAS volume (m³ d⁻¹)': f'{l.V_TWAS.mean():.1f}',
        'HSW volume (m³ d⁻¹)': f'{l.V_HSW.mean():.1f}',
        'HSW share of fed volume (%)': f'{l.HSW_vol_share.mean()*100:.1f}',
        'Hydraulic retention time (d)': f'{l.HRT_d.replace([np.inf,-np.inf],np.nan).mean():.1f}',
        'HSW COD, median [IQR] (kg m⁻³)': (lambda c: f'{c.median():.0f} [{c.quantile(.25):.0f}–{c.quantile(.75):.0f}]')(l.HSW_COD_kg_m3.replace(0, np.nan).dropna()),
        'HSW VS, median [IQR] (%)': (lambda c: f'{c.median():.1f} [{c.quantile(.25):.1f}–{c.quantile(.75):.1f}]')(l.HSW_VS_pct.replace(0, np.nan).dropna()),
        'HSW COD load (kg d⁻¹)': f'{l.HSW_COD_load_kg_d.mean():,.0f}',
    }
T23 = pd.DataFrame(t23); T23.index.name = 'Quantity'; T23.to_csv(OUT / 'table_S23.csv', encoding='utf-8-sig')
say('\nTable S23'); say(T23.to_string())
t24 = [{'Specification': r.spec, 'Days': r.n_days, 'Summed HSW coefficient (m³ m⁻³)': f'{r.sum_HSW_coef_m3_per_m3:.1f} [{r.ci_low:.1f}, {r.ci_high:.1f}]',
        'Implied share (%)': f'{r.implied_share*100:.0f}', 'R²': f'{r.r2:.2f}'} for r in A.itertuples()]
t24.append({'Specification': 'Laboratory record 2020–2023, HSW d0+d1 + sludge volumes', 'Days': len(z),
            'Summed HSW coefficient (m³ m⁻³)': f'{float(np.squeeze(t.effect)):.1f} [{ci[0]:.1f}, {ci[1]:.1f}]',
            'Implied share (%)': f'{float(np.squeeze(t.effect))*z.V_HSW.mean()/z.lab_gas.mean()*100:.0f}', 'R²': f'{m.rsquared:.2f}'})
pd.DataFrame(t24).to_csv(OUT / 'table_S24.csv', index=False, encoding='utf-8-sig')
say('\nTable S24'); say(pd.DataFrame(t24).to_string(index=False))
t25 = E.pivot_table(index=['ch4_fraction', 'electrical_efficiency'], columns='price_basis', values='value_kUSD_yr').reset_index()
t25 = t25.merge(E.drop_duplicates(['ch4_fraction', 'electrical_efficiency'])[['ch4_fraction', 'electrical_efficiency', 'fuel_MWh_yr', 'electric_kW', 'electric_MWh_yr']])
t25.to_csv(OUT / 'table_S25.csv', index=False, float_format='%.1f', encoding='utf-8-sig')
say('\nTable S25'); say(t25.round(1).to_string(index=False))

(OUT / 'summary.txt').write_text('\n'.join(lines), encoding='utf-8')
