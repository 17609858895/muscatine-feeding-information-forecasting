"""After tune_all.py and dl_tune.py: merge tuned baselines into the benchmark tables used by Fig. 4 and S-figures.
Outputs: extra/metrics_all_horizons_tuned.csv, dm_tests_tuned.csv, extra/rolling_all_horizons_tuned.csv,
tune/proposed_tuned_vs_prespecified.csv"""
import pandas as pd, numpy as np, pickle, os, sys, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from common import *
step = sys.argv[1:] or ['merge', 'dm', 'rolling']
TUNED = ['Random forest', 'Extra trees', 'HistGB', 'LightGBM', 'XGBoost', 'CatBoost', 'SVR', 'KNN', 'LightGBM +feed', 'CatBoost +feed']
DEEP = ['MLP', 'LSTM', 'GRU', 'TCN', 'Transformer', 'PatchTST', 'DLinear', 'N-BEATS', 'iTransformer', 'PG-HQ [past]', 'PG-HQ [feed]']
PROPD = ['LGB-Δ [past]', 'Cat-Δ [past]', 'LGB-Δ [feed]', 'Cat-Δ [feed]']
fp = pd.read_pickle('final_preds.pkl'); HS = [1, 2, 3, 4, 6, 9, 12, 18, 24]
if 'merge' in step:
    M = pd.read_csv('extra/metrics_all_horizons.csv'); t = pd.read_csv('tune/tuned_all_horizons.csv')
    keep = M[~M.model.isin(TUNED)]
    tb = t[t.model.isin(TUNED)][['h', 'model', 'RMSE', 'MAE', 'R2']]
    tp = t[t.model.isin(PROPD)][['h', 'model', 'RMSE', 'MAE', 'R2']].copy(); tp['model'] = tp.model + ' (tuned)'
    parts = [keep, tb, tp]
    if os.path.exists('tune/preds_dl5.pkl'):
        P5 = pd.read_pickle('tune/preds_dl5.pkl'); rr = []
        for nm, v in P5.items():
            for h in HS:
                yv = fp['pred'][h]['test'].y.values; rr.append(dict(h=h, model=nm, **metrics(yv, v['te'][:, h - 1])))
        parts[0] = keep[~keep.model.isin(list(P5))]; parts.append(pd.DataFrame(rr))
    N = pd.concat(parts, ignore_index=True)
    pers = N[N.model == 'Persistence'].set_index('h').RMSE; N['Skill'] = 1 - N.RMSE / N.h.map(pers)
    N.to_csv('extra/metrics_all_horizons_tuned.csv', index=False)
    a = pd.read_csv('tune/tuned_anchor.csv'); o = M.set_index(['model', 'h'])
    cmp = []
    for _, r in a[a.model.isin(PROPD)].iterrows():
        cmp.append(dict(model=r.model, h=r.h, val_RMSE_prespec=r.val_RMSE_default, val_RMSE_tuned=r.val_RMSE_tuned,
                        test_R2_prespec=o.loc[(r.model, r.h), 'R2'], test_R2_tuned=r.test_R2))
    C = pd.DataFrame(cmp); C['dR2_test'] = C.test_R2_tuned - C.test_R2_prespec; C.to_csv('tune/proposed_tuned_vs_prespecified.csv', index=False); print(C.round(3).to_string())
    print(N[N.h.isin([1, 6, 24])].pivot_table(index='model', columns='h', values='R2').round(3).sort_values(24).to_string())
def dm(y, p1, p2, h):
    e = (y - p1) ** 2 - (y - p2) ** 2; n = len(e); mu = e.mean(); g = [np.mean((e[k:] - mu) * (e[:n - k] - mu)) for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n; s = mu / np.sqrt(max(v, 1e-12)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n); return s, 2 * stats.t.sf(abs(s), n - 1)
if 'dm' in step:
    TP = pickle.load(open('tune/preds_anchor.pkl', 'rb'))['P']; P5 = pd.read_pickle('tune/preds_dl5.pkl') if os.path.exists('tune/preds_dl5.pkl') else {}
    rows = []
    for h in [1, 6, 24]:
        T = fp['pred'][h]['test']; y = T.y.values
        B = {b: T[b].values for b in ['Persistence', 'AR-Ridge (gas only)', 'Ridge']}
        for nm in TUNED: B[nm] = TP[(nm, h)]['te']
        for nm in DEEP:
            if nm in P5 and 'PG-HQ' not in nm: B[nm] = P5[nm]['te'][:, h - 1]
            elif nm in T and 'PG-HQ' not in nm: B[nm] = T[nm].values
        for prop in ['PG-HQ ensemble [past]', 'PG-HQ ensemble [feed]']:
            for b, pb in B.items():
                s, p = dm(y, T[prop].values, pb, h); rows.append(dict(h=h, proposed=prop, baseline=b, DM=s, p=p))
    pd.DataFrame(rows).to_csv('dm_tests_tuned.csv', index=False); print('dm done')
if 'rolling' in step:
    import lightgbm as lgb
    X0, _ = build(); near = lambda h: 1 if h <= 3 else (6 if h <= 9 else 24)
    folds = [('2022-07-01', '2022-08-15'), ('2022-08-15', '2022-10-01'), ('2022-10-01', '2022-11-15'), ('2022-11-15', '2023-01-01'), ('2023-01-01', '2023-02-28')]
    allo = np.sort(np.concatenate([o_tr, o_va, o_te])); ti = idx[allo]; rows = []
    for h in HS:
        bp = pickle.load(open(f'tune/best_LightGBM_{near(h)}.pkl', 'rb'))['best']
        Xp = pd.concat([X0, calendar(X0.index, h)], axis=1); yy = pd.Series(target(h).values, index=idx)
        for k, (a, b) in enumerate(folds):
            trk = ti[(ti + pd.Timedelta(hours=24)) < a]; tek = ti[(ti >= a) & (ti < b)]
            m = lgb.LGBMRegressor(**bp, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0, n_jobs=2).fit(Xp.loc[trk], yy[trk])
            rows.append(dict(h=h, fold=k + 1, model='LightGBM', **metrics(yy[tek].values, m.predict(Xp.loc[tek]))))
        print('rolling', h, flush=True)
    R = pd.read_csv('extra/rolling_all_horizons.csv'); R = pd.concat([R[R.model != 'LightGBM'], pd.DataFrame(rows)], ignore_index=True)
    Pm = R[R.model == 'Persistence'].set_index(['h', 'fold']).RMSE; R['Skill'] = 1 - R.RMSE / [Pm[(a, b)] for a, b in zip(R.h, R.fold)]
    R.to_csv('extra/rolling_all_horizons_tuned.csv', index=False); print('rolling done')
