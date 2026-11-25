"""Forecast value expressed as firm biogas supply.

For each forecast origin t and horizon h, a hypothetical gas-use commitment c is set
from the one-sided lower prediction bound of a forecast: c = max(0, yhat - q), where q
is the adaptive (1 - alpha) quantile of signed errors (yhat - y). As in the paper's ACI
intervals, q starts from validation errors and is updated only after each outcome is
observed, i.e. with a delay of h hours (Gibbs & Candes 2021, step 0.005).

Delivered firm gas = min(c, y). Commitments above the outcome are shortfalls that would
require backup fuel or curtailment; gas above c remains available for storage, the boiler
or the flare. Storage buffering is ignored, so the estimate is conservative.

Inputs : results/final_preds.pkl, results/hourly.pkl (same files used for Table 2/Fig. 5)
Outputs: results/firm_supply.csv, results/firm_supply_summary.txt
Units  : native CFM converted with 1 CFM = 1.69901 m3 h-1.
"""
import os, pickle, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('RESULTS_DIR', os.path.join(HERE, '..', '..', 'results'))
CFM = 1.69901
HS = [1, 2, 3, 4, 6, 9, 12, 18, 24]
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
GAMMA = 0.005
STRATS = {'Persistence': 'Persistence',
          'Past-only ensemble': 'PG-HQ ensemble [past]',
          'Feed-conditioned ensemble': 'PG-HQ ensemble [feed]'}


def aci_lower(val_err, test_err, yhat, h, alpha, gamma=GAMMA):
    """One-sided adaptive conformal lower bound with delayed feedback.
    err = yhat - y (positive = over-prediction)."""
    hist = list(val_err); at = alpha
    q = np.zeros(len(test_err))
    for i in range(len(test_err)):
        q[i] = np.quantile(hist, min(1.0, max(0.0, 1 - at)))
        j = i - h
        if j >= 0:
            miss = 1.0 if test_err[j] > q[j] else 0.0
            at = at + gamma * (alpha - miss)
            hist.append(test_err[j])
    return np.maximum(0.0, yhat - q)


def split_lower(val_err, yhat, alpha):
    n = len(val_err); lev = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.maximum(0.0, yhat - np.quantile(val_err, lev))


def main():
    fp = pickle.load(open(os.path.join(RES, 'final_preds.pkl'), 'rb'))
    H = pd.read_pickle(os.path.join(RES, 'hourly.pkl')).interpolate(limit=3)
    rows = []
    for h in HS:
        T = fp['pred'][h]['test']; V = fp['pred'][h]['val'].copy()
        V['Persistence'] = H['Biogas'].reindex(V.index).values
        yt = T['y'].values
        for name, col in STRATS.items():
            ev = (V[col] - V['y']).values; ev = ev[np.isfinite(ev)]
            yhat = T[col].values; et = yhat - yt
            for a in ALPHAS:
                for method in ('ACI', 'Split'):
                    c = aci_lower(ev, et, yhat, h, a) if method == 'ACI' else split_lower(ev, yhat, a)
                    d = np.minimum(c, yt)
                    rows.append(dict(h=h, strategy=name, alpha=a, method=method,
                                     firm_fraction=d.sum() / yt.sum(),
                                     shortfall_rate=np.mean(c > yt),
                                     shortfall_volume_share=np.maximum(c - yt, 0).sum() / max(c.sum(), 1e-9),
                                     commit_m3h=c.mean() * CFM, delivered_m3h=d.mean() * CFM,
                                     mean_flow_m3h=yt.mean() * CFM, n=len(yt)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, 'firm_supply.csv'), index=False)

    # flare context over the same target hours (test origins + 24 h)
    T24 = fp['pred'][24]['test']
    tgt = T24.index + pd.Timedelta(hours=24)
    burner = H['Biogas_burner'].reindex(tgt).mean() * CFM
    boiler = H['Biogas_boiler'].reindex(tgt).mean() * CFM
    out = []
    A = R[(R.method == 'ACI') & (R.alpha == 0.05)].pivot(index='h', columns='strategy')
    out.append('ACI, alpha = 0.05 (95% one-sided reliability)')
    out.append(A[['firm_fraction', 'shortfall_rate', 'delivered_m3h']].round(3).to_string())
    for h in (6, 24):
        g = R[(R.method == 'ACI') & (R.alpha == 0.05) & (R.h == h)].set_index('strategy')
        dp = g.loc['Feed-conditioned ensemble', 'delivered_m3h'] - g.loc['Past-only ensemble', 'delivered_m3h']
        dpe = g.loc['Feed-conditioned ensemble', 'delivered_m3h'] - g.loc['Persistence', 'delivered_m3h']
        out.append(f'h={h}: feed - past = {dp:.1f} m3/h ({dp*24:.0f} m3/d); feed - persistence = {dpe:.1f} m3/h; '
                   f'relative to past {dp/g.loc["Past-only ensemble","delivered_m3h"]*100:.1f}%')
    out.append(f'mean burner flow over 24 h target hours = {burner:.1f} m3/h; boiler = {boiler:.1f} m3/h')
    for a in (0.02, 0.05, 0.10):
        g = R[(R.method == 'ACI') & (R.alpha == a) & (R.h == 24)].set_index('strategy')
        out.append(f'alpha={a}: 24 h firm fraction ' + ', '.join(f'{k}={v:.3f}' for k, v in g.firm_fraction.items())
                   + ' | shortfall ' + ', '.join(f'{k}={v:.3f}' for k, v in g.shortfall_rate.items()))
    S = R[(R.method == 'Split') & (R.alpha == 0.05)].pivot(index='h', columns='strategy')
    out.append('Split conformal, alpha = 0.05')
    out.append(S[['firm_fraction', 'shortfall_rate']].round(3).to_string())
    txt = '\n'.join(out); print(txt)
    open(os.path.join(RES, 'firm_supply_summary.txt'), 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
