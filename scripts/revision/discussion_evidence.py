"""Supporting calculations for the expanded discussion (Sections 3.3, 3.7, 3.8, 4.1).

Inputs:
  results/hourly.pkl
  results/revision_analysis/independent_final_preds.pkl
  results/revision_analysis/feeding_perturbations.csv
Usage: python discussion_evidence.py <results folder> <output folder>
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
A = root / 'revision_analysis'
res = {}
K = 0.3048 ** 3 * 60  # native ft3 min-1 to m3 h-1 (pred pickle and hourly gas columns are native)

hourly = pd.read_pickle(root / 'hourly.pkl')
preds = pd.read_pickle(A / 'independent_final_preds.pkl')['pred']

# ---------- A. error structure of the two ensembles at 24 h
t = preds[24]['test'] * K
y = t['y']
rows = []
for name in ['Persistence', 'PG-HQ ensemble [past]', 'PG-HQ ensemble [feed]']:
    e = t[name] - y  # positive = overprediction
    rows.append(dict(model=name, n=len(e), RMSE=float(np.sqrt((e**2).mean())), MAE=float(e.abs().mean()),
                     median_abs=float(e.abs().median()), q90_abs=float(e.abs().quantile(.90)),
                     over_q90=float(e.quantile(.90)), over_q95=float(e.quantile(.95)), over_q99=float(e.quantile(.99)),
                     under_q05=float(e.quantile(.05))))
err = pd.DataFrame(rows).set_index('model')
err.to_csv(out / 'error_structure_24h.csv')
res['error_structure_24h'] = err.round(2).to_dict(orient='index')
p, f = err.loc['PG-HQ ensemble [past]'], err.loc['PG-HQ ensemble [feed]']
res['relative_reduction_feed_vs_past_24h'] = {k: round(float(1 - f[k] / p[k]), 3) for k in
                                               ['RMSE', 'MAE', 'median_abs', 'q90_abs', 'over_q90', 'over_q95', 'over_q99']}

# share of squared-error reduction coming from origins with abrupt declines
ep, ef = t['PG-HQ ensemble [past]'] - y, t['PG-HQ ensemble [feed]'] - y
gas = hourly['Biogas'] * K
tgt_time = t.index + pd.Timedelta(hours=24)
drop = (gas.reindex(tgt_time).values - gas.reindex(t.index).values)  # target minus current flow
drop = pd.Series(drop, index=t.index)
q10 = drop.quantile(.10)
dec = drop <= q10
res['abrupt_decline_definition'] = f'y(t+24) - y(t) <= 10th percentile ({q10:.1f} m3/h)'
res['decline_subset'] = {
    'n': int(dec.sum()),
    'RMSE_past': float(np.sqrt((ep[dec]**2).mean())), 'RMSE_feed': float(np.sqrt((ef[dec]**2).mean())),
    'RMSE_past_other': float(np.sqrt((ep[~dec]**2).mean())), 'RMSE_feed_other': float(np.sqrt((ef[~dec]**2).mean())),
    'mean_overprediction_past': float(ep[dec].mean()), 'mean_overprediction_feed': float(ef[dec].mean()),
}

# ---------- B. the 18 h feed-conditioned shortfall event at alpha = 0.01 (origins 2023-01-10 23:00 to 2023-01-11 16:00)
o = pd.date_range('2023-01-10 23:00', '2023-01-11 16:00', freq='h')
hsw = hourly['Q-HSW1_GPM'] + hourly['Q-HSW2_GPM']
ev = pd.DataFrame({'y': y.reindex(o), 'past': t['PG-HQ ensemble [past]'].reindex(o), 'feed': t['PG-HQ ensemble [feed]'].reindex(o)})
ev['target_time'] = o + pd.Timedelta(hours=24)
ev['current_flow'] = gas.reindex(o).values
# mean recorded HSW over the 24 h future window supplied to the feed-conditioned model (labels t+1..t+24)
ev['future_HSW_mean_GPM'] = [hsw.loc[s + pd.Timedelta(hours=1): s + pd.Timedelta(hours=24)].mean() for s in o]
ev.to_csv(out / 'event_20230111.csv')
test_hsw_mean = float(np.mean([hsw.loc[s + pd.Timedelta(hours=1): s + pd.Timedelta(hours=24)].mean() for s in t.index]))
w = hourly.loc['2023-01-11 18:00':'2023-01-12 01:00', ['Biogas', 'Biogas_burner', 'Biogas_boiler', 'H-Dig1_FT', 'H-Dig2_FT', 'Q-HSW1_GPM', 'Q-HSW2_GPM']].copy()
w[['Biogas', 'Biogas_burner', 'Biogas_boiler']] *= K
w[['H-Dig1_FT', 'H-Dig2_FT']] *= 0.3048
w.to_csv(out / 'event_20230111_routing_m3h_m.csv')
res['event_20230111'] = {
    'mean_y': float(ev['y'].mean()), 'mean_feed_forecast': float(ev['feed'].mean()), 'mean_past_forecast': float(ev['past'].mean()),
    'mean_current_flow': float(ev['current_flow'].mean()),
    'future_HSW_mean_GPM_event': float(ev['future_HSW_mean_GPM'].mean()), 'future_HSW_mean_GPM_test': test_hsw_mean,
}

# ---------- C. feeding perturbations: gas RMSE versus window error in the supplied feeding mean (24 h)
fp = pd.read_csv(A / 'feeding_perturbations.csv')
fp24 = fp[fp.h == 24]
ok = fp24.dropna(subset=['window_WAPE'])  # WAPE undefined for advanced (negative-shift) noise runs at 24 h
rho, pval = stats.spearmanr(ok.window_WAPE, ok.RMSE)
res['perturbation_spearman_24h'] = {'rho': float(rho), 'p': float(pval), 'n': int(len(ok))}
canc = fp24[(fp24.scenario == 'episode_cancellation')]
summ = canc.groupby('magnitude').agg(RMSE_mean=('RMSE', 'mean'), RMSE_min=('RMSE', 'min'), RMSE_max=('RMSE', 'max'),
                                     WAPE_mean=('window_WAPE', 'mean'), WAPE_min=('window_WAPE', 'min'), WAPE_max=('window_WAPE', 'max'))
res['cancellation_24h'] = summ.round(3).to_dict(orient='index')
rc, pc = stats.spearmanr(canc.window_WAPE, canc.RMSE)
res['cancellation_spearman_24h'] = {'rho': float(rc), 'p': float(pc), 'n': int(len(canc))}
# episode inventory in the test period
tt = hourly.loc['2022-12-15':'2023-02-28', ['Q-HSW1_GPM', 'Q-HSW2_GPM']].sum(axis=1)
on = tt > 1
runs = (on != on.shift()).cumsum()[on]
lens = runs.value_counts()
vol = tt[on].groupby(runs).sum()
res['test_HSW_episodes'] = {'n_episodes': int(lens.size), 'median_duration_h': float(lens.median()),
                            'max_duration_h': int(lens.max()),
                            'volume_share_largest_5': float(vol.sort_values(ascending=False).head(5).sum() / vol.sum())}
by = fp24.groupby(['scenario', 'magnitude']).agg(RMSE=('RMSE', 'mean'), WAPE=('window_WAPE', 'mean')).reset_index()
by.to_csv(out / 'perturbation_summary_24h.csv', index=False)

# ---------- D. effective sample size heuristic for the training origins
acf1 = 0.92
n_train = 5421
res['effective_n_AR1'] = {'n': n_train, 'rho1': acf1, 'n_eff': n_train * (1 - acf1) / (1 + acf1)}

(out / 'discussion_evidence.json').write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=2, default=float))
