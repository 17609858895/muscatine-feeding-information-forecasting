"""Supplementary Tables S1-S7 -> tables/TableS*.csv and tables/Supplementary_Tables_S1-S7.xlsx.
Flows are reported in SI units: biogas CFM x 1.69901 = m3 h-1; DML effects (CFM per GPM) x 7.48 = m3 h-1 per m3 h-1."""
import pandas as pd, numpy as np, os, pickle
from scipy import stats
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
os.makedirs('tables', exist_ok=True)
CFM = 1.69901; TH = 1.69901 / 0.227125
T = {}; TITLE = {}; NOTE = {}

# ---------------- S1 variables ----------------
G = {'Biogas': 'Gas history', 'H-Dig1_FT': 'Gas storage', 'H-Dig2_FT': 'Gas storage', 'Q-HSW1_GPM': 'Co-substrate feed', 'Q-HSW2_GPM': 'Co-substrate feed',
     'H-HSW_ft': 'Co-substrate feed', 'V-HSW-DIG2-yest_gal': 'Co-substrate feed', 'Q-PS_on': 'Sludge feed', 'Q-TWAS_GPM': 'Sludge feed',
     'D1_TEMPERATURE': 'Digester thermal', 'D2_TEMPERATURE': 'Digester thermal', 'Q-influent_MGD': 'Plant hydraulics', 'Q-Aeration_SCFM': 'Plant hydraulics',
     'Q-RAS_MGD': 'Plant hydraulics', 'Q-WAS1_MGD': 'Plant hydraulics', 'Q-WAS2_MGD': 'Plant hydraulics'}
D = [  # tag, description, original unit, SI unit, factor
    ('Biogas', 'Total biogas flow (burner + boiler); forecast target and gas-history input', 'CFM', 'm3 h-1', 1.69901),
    ('H-Dig1_FT', 'Floating-cover (gas storage) height, digester 1', 'ft', 'm', 0.3048),
    ('H-Dig2_FT', 'Floating-cover (gas storage) height, digester 2', 'ft', 'm', 0.3048),
    ('Q-HSW1_GPM', 'High-strength waste feed flow, line 1', 'GPM', 'm3 h-1', 0.227125),
    ('Q-HSW2_GPM', 'High-strength waste feed flow, line 2', 'GPM', 'm3 h-1', 0.227125),
    ('H-HSW_ft', 'High-strength waste receiving-tank level', 'ft', 'm', 0.3048),
    ('V-HSW-DIG2-yest_gal', 'High-strength waste volume fed to digester 2 on the previous day', 'gal', 'm3', 0.00378541),
    ('Q-PS_on', 'Primary-sludge pump on-time fraction', '-', '-', 1.0),
    ('Q-TWAS_GPM', 'Thickened waste activated sludge feed flow', 'GPM', 'm3 h-1', 0.227125),
    ('D1_TEMPERATURE', 'Digester 1 temperature', 'degF', 'degC', np.nan),
    ('D2_TEMPERATURE', 'Digester 2 temperature', 'degF', 'degC', np.nan),
    ('Q-influent_MGD', 'Plant influent flow', 'MGD', 'm3 d-1', 3785.41),
    ('Q-Aeration_SCFM', 'Aeration air flow', 'SCFM', 'm3 h-1', 1.69901),
    ('Q-RAS_MGD', 'Total return activated sludge flow', 'MGD', 'm3 d-1', 3785.41),
    ('Q-WAS1_MGD', 'Waste activated sludge flow, pump 1', 'MGD', 'm3 d-1', 3785.41),
    ('Q-WAS2_MGD', 'Waste activated sludge flow, pump 2', 'MGD', 'm3 d-1', 3785.41),
]
EXC = [
    ('V-burner_FT3', 'Cumulative burner gas counter', 'ft3', 'Derived from the target; counter saturates at 1e5 ft3 (about 20% of minutes)'),
    ('V-Boiler_FT3', 'Cumulative boiler gas counter', 'ft3', 'Derived from the target'),
    ('Biogas_burner', 'Biogas flow to burner', 'CFM', 'Component of the target'),
    ('Biogas_boiler', 'Biogas flow to boiler', 'CFM', 'Component of the target'),
    ('Q-PS_MGD', 'Primary-sludge flow signal (1-min pump pulses)', 'MGD', 'On/off pulse signal whose hourly mean is not a flow; replaced by the hourly pump on-time fraction (Q-PS_on)'),
    ('Q-RAS1/2/3_MGD, RAS-1/2/3-GATE_MGD', 'Individual RAS pump and gate flows', 'MGD', 'Redundant with total RAS flow'),
    ('Q-TWAS_on', 'TWAS pump on-time fraction', '-', 'Redundant with TWAS feed flow'),
]
H = pd.read_pickle('hourly.pkl')
r = []
for tag, desc, u, su, f in D:
    s = H[tag]; s2 = (s - 32) * 5 / 9 if u == 'degF' else s * f
    r.append(dict(Variable=tag, Description=desc, **{'Process group': G[tag]}, **{'Original unit': u, 'SI unit': su},
                  **{'Median (SI)': s2.median(), 'IQR low (SI)': s2.quantile(.25), 'IQR high (SI)': s2.quantile(.75)},
                  **{'Lag bins (h)': '0, 1-2, 3-5, 6-11, 12-23, 24-47, 48-71, 72-167', 'Role': 'Model input (8 lag-bin means)' + ('; target' if tag == 'Biogas' else '')}))
for tag, desc, u, why in EXC:
    r.append(dict(Variable=tag, Description=desc, **{'Process group': '-', 'Original unit': u, 'SI unit': '-', 'Lag bins (h)': '-', 'Role': 'Excluded: ' + why}))
r.append(dict(Variable='hod_sin, hod_cos, dow_sin, dow_cos', Description='Hour-of-day and day-of-week of the target time t+h', **{'Process group': 'Calendar', 'Original unit': '-', 'SI unit': '-', 'Lag bins (h)': '-', 'Role': 'Model input'}))
r.append(dict(Variable='Planned HSW, PS, TWAS', Description='Mean scheduled feed over (t, t+h]', **{'Process group': 'Feed schedule', 'Original unit': 'GPM / - / GPM', 'SI unit': 'm3 h-1 / - / m3 h-1', 'Lag bins (h)': 'future window', 'Role': 'Input in the feed-conditioned regime only'}))
T['S1'] = pd.DataFrame(r)
TITLE['S1'] = 'Table S1. Process variables, units, process groups, lag windows and model roles.'
NOTE['S1'] = 'Median and IQR computed on the 8,349 hourly records in SI units. Past-only inputs: 16 variables x 8 lag bins + 4 calendar features = 132; the feed-conditioned regime adds 3 planned-feed features.'

# ---------------- S2 hyper-parameters ----------------
SPACE = {
    'LightGBM-type (LightGBM, LightGBM +feed, LGB-Δ)': 'n_estimators 200-1200 (step 100); learning_rate 0.005-0.1 (log); num_leaves 7-63; min_child_samples 10-150; colsample_bytree 0.3-1.0; reg_lambda 1e-3-20 (log); subsample 0.8 fixed',
    'CatBoost-type (CatBoost, CatBoost +feed, Cat-Δ)': 'iterations 300-1200 (step 100); learning_rate 0.01-0.15 (log); depth 3-7; l2_leaf_reg 0.5-30 (log)',
    'XGBoost': 'n_estimators 200-1200; learning_rate 0.005-0.1 (log); max_depth 2-8; min_child_weight 1-50 (log); colsample_bytree 0.3-1.0; reg_lambda 1e-3-20 (log)',
    'Random forest / Extra trees': '300 trees; min_samples_leaf 1-40 (log); max_features 0.1-1.0; max_depth {8, 16, None}',
    'HistGB': 'max_iter 200-1200; learning_rate 0.005-0.1 (log); max_leaf_nodes 7-63; min_samples_leaf 10-150; l2_regularization 1e-3-20 (log)',
    'SVR (RBF, standardised inputs)': 'C 0.1-300 (log); epsilon 0.1-10 (log); gamma 1e-4-0.05 (log)',
    'KNN (standardised inputs)': 'n_neighbors 5-150 (log); weights {uniform, distance}; p {1, 2}',
    'Ridge, AR-Ridge': 'alpha selected by efficient leave-one-out over 13 values in 1e-3-1e3 (log)',
    'Deep models (MLP, LSTM, GRU, TCN, Transformer, PatchTST, DLinear, N-BEATS, iTransformer)': 'learning rate {2e-4, 5e-4, 1e-3} at seed 0; AdamW, weight decay 1e-4, batch 128, <= 80 epochs, early stopping patience 12 on validation RMSE; five seeds at the selected rate',
    'PG-HQ': 'd_model, heads, dropout and learning rate selected in a separate validation search (d_model 32, learning rate 1e-3); five seeds',
}
if os.path.exists('tune/tuned_anchor.csv'):
    rows = []
    for k, v in SPACE.items(): rows.append({'Model family': k, 'Horizon (h)': 'all', 'Search space': v})
    for f in sorted(os.listdir('tune')):
        if f.startswith('best_'):
            B = pickle.load(open('tune/' + f, 'rb')); nm, h = f[5:-4].rsplit('_', 1); nm = nm.replace('_', ' ')
            bp = '; '.join(f'{k} = {v:.4g}' if isinstance(v, float) else f'{k} = {v}' for k, v in B['best'].items())
            rows.append({'Model family': nm, 'Horizon (h)': int(h), 'Search space': 'Best: ' + bp,
                         'Trials': len(B['trials']), 'Validation RMSE, default (m3 h-1)': B['default_val'] * CFM, 'Validation RMSE, tuned (m3 h-1)': B['best_val'] * CFM})
    if os.path.exists('tune/dl_lr_search.csv'):
        L = pd.read_csv('tune/dl_lr_search.csv')
        for nm, g in L.groupby('model', sort=False):
            b = g.loc[g.val_RMSE.idxmin()]
            rows.append({'Model family': nm, 'Horizon (h)': '1-24 (joint)', 'Search space': 'Best: learning rate = %.0e; ' % b.lr + ', '.join(f'{lr:.0e}: {v * CFM:.2f}' for lr, v in zip(g.lr, g.val_RMSE)),
                         'Trials': len(g), 'Validation RMSE, tuned (m3 h-1)': b.val_RMSE * CFM})
    T['S2'] = pd.DataFrame(rows)
    TITLE['S2'] = 'Table S2. Hyper-parameter search spaces and selected values.'
    NOTE['S2'] = 'Optuna TPE sampler (seed 0); objective = validation-period RMSE (2022-11-15 to 2022-12-14); the pre-specified configuration was evaluated as trial 0. Parameters selected at 1, 6 and 24 h are reused at 2-3, 4-9 and 12-18 h. Deep-model validation RMSE is averaged over the 24 horizons.'

# ---------------- S3 all-horizon metrics ----------------
m = pd.read_csv('extra/metrics_all_horizons.csv')
TUNED = ['Random forest', 'Extra trees', 'HistGB', 'LightGBM', 'XGBoost', 'CatBoost', 'SVR', 'KNN', 'LightGBM +feed', 'CatBoost +feed']
DEEP = ['MLP', 'LSTM', 'GRU', 'TCN', 'Transformer', 'PatchTST', 'DLinear', 'N-BEATS', 'iTransformer', 'PG-HQ [past]', 'PG-HQ [feed]']
keep = m[~m.model.isin(TUNED + DEEP)].copy(); keep['Configuration'] = np.where(keep.model.isin(['LGB-Δ [past]', 'Cat-Δ [past]', 'LGB-Δ [feed]', 'Cat-Δ [feed]', 'PG-HQ ensemble [past]', 'PG-HQ ensemble [feed]']), 'pre-specified', '-')
parts = [keep[['h', 'model', 'RMSE', 'MAE', 'R2', 'Configuration']]]
if os.path.exists('tune/tuned_all_horizons.csv'):
    t = pd.read_csv('tune/tuned_all_horizons.csv'); t['Configuration'] = 'Optuna (params from ' + t.params_from_h.astype(str) + ' h)'
    t.loc[t.model.str.contains('Δ'), 'model'] = t.loc[t.model.str.contains('Δ'), 'model'] + ' (tuned)'
    parts.append(t[['h', 'model', 'RMSE', 'MAE', 'R2', 'Configuration']])
if os.path.exists('tune/preds_dl5.pkl'):
    P5 = pd.read_pickle('tune/preds_dl5.pkl'); fp0 = pd.read_pickle('final_preds.pkl')
    rr = []
    for nm, v in P5.items():
        for h in [1, 2, 3, 4, 6, 9, 12, 18, 24]:
            yv = fp0['pred'][h]['test'].y.values; e = yv - v['te'][:, h - 1]
            rr.append(dict(h=h, model=nm, RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)), R2=1 - np.sum(e ** 2) / np.sum((yv - yv.mean()) ** 2), Configuration='5-seed mean, selected learning rate'))
    parts.append(pd.DataFrame(rr))
else:
    parts.append(m[m.model.isin(DEEP)].assign(Configuration='3-seed mean')[['h', 'model', 'RMSE', 'MAE', 'R2', 'Configuration']])
S3 = pd.concat(parts, ignore_index=True)
pers = S3[S3.model == 'Persistence'].set_index('h').RMSE
S3['Skill'] = 1 - S3.RMSE / S3.h.map(pers)
S3['RMSE'] *= CFM; S3['MAE'] *= CFM
S3['Period'] = 'Test (2022-12-15 to 2023-02-27)'
ro = pd.read_csv('extra/rolling_all_horizons_tuned.csv' if os.path.exists('extra/rolling_all_horizons_tuned.csv') else 'extra/rolling_all_horizons.csv'); ro['RMSE'] *= CFM; ro['MAE'] *= CFM
ro = ro.groupby(['h', 'model']).agg(RMSE=('RMSE', 'mean'), MAE=('MAE', 'mean'), R2=('R2', 'mean'), Skill=('Skill', 'mean'), R2_sd=('R2', 'std')).reset_index()
ro['Period'] = 'Rolling origin, mean of 5 folds'; ro['Configuration'] = np.where(ro.model == 'LightGBM', 'Optuna (params from nearest anchor horizon)', np.where(ro.model == 'Persistence', '-', 'pre-specified'))
S3 = pd.concat([S3, ro], ignore_index=True)[['Period', 'h', 'model', 'Configuration', 'RMSE', 'MAE', 'R2', 'Skill', 'R2_sd']]
S3 = S3.rename(columns={'h': 'Horizon (h)', 'model': 'Model', 'RMSE': 'RMSE (m3 h-1)', 'MAE': 'MAE (m3 h-1)', 'R2': 'R2', 'R2_sd': 'R2 SD across folds'}).sort_values(['Period', 'Horizon (h)', 'RMSE (m3 h-1)'], ascending=[False, True, True])
T['S3'] = S3
TITLE['S3'] = 'Table S3. Forecast accuracy at 1-24 h on the test period and in five rolling-origin folds.'
NOTE['S3'] = 'Skill = 1 - RMSE/RMSE(persistence) at the same horizon. sMAPE is not reported because hourly biogas flow approaches zero during outages (minimum 2.0 m3 h-1), which makes percentage errors unstable.'

# ---------------- S4 DM matrix ----------------
def dm(y, p1, p2, h):
    e = (y - p1) ** 2 - (y - p2) ** 2; n = len(e); mu = e.mean(); g = [np.mean((e[k:] - mu) * (e[:n - k] - mu)) for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n; s = mu / np.sqrt(max(v, 1e-12)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return s, 2 * stats.t.sf(abs(s), n - 1)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o): run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
    return adj
fp = pd.read_pickle('final_preds.pkl'); rows = []
TP = pickle.load(open('tune/preds_anchor.pkl', 'rb'))['P'] if os.path.exists('tune/preds_anchor.pkl') else {}
P5 = pd.read_pickle('tune/preds_dl5.pkl') if os.path.exists('tune/preds_dl5.pkl') else {}
for h in [1, 6, 24]:
    Tt = fp['pred'][h]['test']; y = Tt.y.values; B = {}
    for b in ['Persistence', 'Seasonal naive (24 h)', 'AR-Ridge (gas only)', 'Ridge', 'Ridge +feed']:
        if b in Tt: B[b] = Tt[b].values
    for nm in TUNED:
        if (nm, h) in TP: B[nm + ' (tuned)'] = TP[(nm, h)]['te']
        elif nm in Tt: B[nm] = Tt[nm].values
    for nm in DEEP:
        if nm in P5: B[nm + (' (5 seeds)' if 'PG-HQ' not in nm else ' (single network, 5 seeds)')] = P5[nm]['te'][:, h - 1]
        elif nm in Tt: B[nm] = Tt[nm].values
    for prop in ['PG-HQ ensemble [past]', 'PG-HQ ensemble [feed]']:
        blk = []
        for b, pb in B.items():
            if len(pb) != len(y): continue
            s, p = dm(y, Tt[prop].values, pb, h); blk.append(dict(h=h, Proposed=prop, Baseline=b, DM=s, p=p,
                                                               **{'RMSE proposed (m3 h-1)': np.sqrt(np.mean((y - Tt[prop].values) ** 2)) * CFM, 'RMSE baseline (m3 h-1)': np.sqrt(np.mean((y - pb) ** 2)) * CFM}))
        ph = holm([r['p'] for r in blk])
        for r, a in zip(blk, ph): r['p (Holm)'] = a; rows.append(r)
S4 = pd.DataFrame(rows).rename(columns={'h': 'Horizon (h)', 'DM': 'DM statistic (HLN)'})
T['S4'] = S4
TITLE['S4'] = 'Table S4. Diebold-Mariano tests of the proposed ensembles against every baseline.'
NOTE['S4'] = 'Squared-error loss; Harvey-Leybourne-Newbold small-sample correction with h-1 autocovariance lags; two-sided t(n-1) reference, n = 1,797 test origins. Negative statistics favour the proposed ensemble. Holm adjustment within each horizon x ensemble family.'

# ---------------- S5 leakage ----------------
la = pd.read_csv('leakage_audit.csv'); la['RMSE'] *= CFM; la['MAE'] *= CFM
la = la.rename(columns={'h': 'Horizon (h)', 'setting': 'Scenario', 'RMSE': 'RMSE (m3 h-1)', 'MAE': 'MAE (m3 h-1)', 'n_test': 'n test'})
lb = pd.read_csv('extra/leakage_all_horizons.csv'); lb['RMSE'] *= CFM; lb['MAE'] *= CFM
lb = lb.pivot(index='h', columns='split', values='R2').reset_index(); lb['Scenario'] = 'A vs B across horizons'
lb = lb.rename(columns={'h': 'Horizon (h)', 'chronological': 'R2 chronological (A)', 'random': 'R2 random split (B)'}); lb['R2 inflation (B - A)'] = lb['R2 random split (B)'] - lb['R2 chronological (A)']
T['S5'] = pd.concat([la.sort_values(['Horizon (h)', 'Scenario']), lb], ignore_index=True)
TITLE['S5'] = 'Table S5. Leakage audit: scenarios A-F and random-split inflation across 1-24 h.'
NOTE['S5'] = 'A: strict chronological split, past-only inputs. B: random 80/20 split of origins. C/D: random or chronological split with target-time cumulative gas counters added. E: chronological split with the target defined as the mean flow over the next 24 h; F: the same target with a random split. Model: LightGBM with the pre-specified configuration.'

# ---------------- S6 cost ----------------
if os.path.exists('tune/tuned_anchor.csv'):
    ta = pd.read_csv('tune/tuned_anchor.csv')
    c1 = ta.groupby('model').agg(**{'Tuning time, 3 horizons (s)': ('tune_sec', 'sum'), 'Fit + predict, one horizon (s)': ('refit_sec', 'mean')}).reset_index().rename(columns={'model': 'Model'})
    c1['Trainable parameters'] = 'n/a (tree/kernel model)'; c1['Type'] = 'Tabular, one model per horizon'
    parts = [c1]
    if os.path.exists('tune/dl_cost.csv'):
        c2 = pd.read_csv('tune/dl_cost.csv').rename(columns={'model': 'Model', 'n_params': 'Trainable parameters', 'train_sec_per_seed': 'Training time per seed (s)', 'infer_ms_per_origin': 'Inference (ms per origin, 24 horizons)'})
        c2['Type'] = 'Deep, joint 24-horizon output'; parts.append(c2.drop(columns=['lr']))
    T['S6'] = pd.concat(parts, ignore_index=True)
    TITLE['S6'] = 'Table S6. Computational cost.'
    NOTE['S6'] = 'Cloud CPU, 2 cores, no GPU; PyTorch 2 threads. Tabular fit + predict covers one training fit and prediction of the validation and test periods.'

# ---------------- S7 DML ----------------
dh = pd.read_csv('dml_horizon.csv')
for c in ['theta', 'se', 'ci_lo', 'ci_hi']: dh[c] *= TH
dh = dh.rename(columns={'h': 'Horizon (h)', 'test': 'Estimand', 'theta': 'theta (m3 h-1 per m3 h-1)', 'se': 'Newey-West SE', 'ci_lo': '95% CI low', 'ci_hi': '95% CI high'})
dh['Passes (CI excludes 0 for main; includes 0 for placebo)'] = [((lo > 0) if 'Placebo' not in e else (lo <= 0 <= hi)) for e, lo, hi in zip(dh['Estimand'], dh['95% CI low'], dh['95% CI high'])]
rb = pd.read_csv('extra/dml_robustness.csv'); rb['theta'] *= TH; rb['se'] *= TH
rb = rb.rename(columns={'h': 'Horizon (h)', 'check': 'Robustness check', 'setting': 'Setting', 'theta': 'theta (m3 h-1 per m3 h-1)', 'se': 'Newey-West SE'})
cb_ = pd.read_csv('supp/cate_storage_bootstrap.csv')
for c in ['theta', 'lo', 'hi']: cb_[c] *= TH
cb_ = cb_.rename(columns={'h': 'Horizon (h)', 'storage': 'Gas-storage tertile', 'theta': 'theta (m3 h-1 per m3 h-1)', 'lo': '95% CI low (block bootstrap)', 'hi': '95% CI high (block bootstrap)'})
spec = pd.DataFrame([
    ('Model', 'Partially linear: Y = theta T + g(X) + e; T = m(X) + v'),
    ('Outcome Y', 'Biogas change y(t+h) - y(t)'),
    ('Treatment T', 'Mean HSW feed (lines 1 + 2) over (t, t+h]'),
    ('Controls X', 'All 132 past-only lag-bin and calendar features at origin t, plus planned PS on-time and TWAS flow over (t, t+h]'),
    ('Nuisance learners', 'LightGBM (400 trees, learning rate 0.03, 15 leaves); Random forest, CatBoost and Lasso as robustness checks'),
    ('Cross-fitting', 'K = 5 contiguous blocks; training folds exclude a buffer of h + 168 h on each side of the held-out block'),
    ('Standard error', 'Newey-West HAC on the orthogonal score, Bartlett kernel, h + 24 lags'),
    ('Placebo 1', 'Future feed over (t+h, t+2h], with the actual (t, t+h] feed added to X'),
    ('Placebo 2', 'Treatment shuffled in 24 h blocks'),
    ('Heterogeneity', 'Residual-on-residual slope within tertiles of digester-1 cover height at t; moving-block bootstrap CI'),
    ('Sample', 'All valid forecast origins (training, validation and test)'),
], columns=['Item', 'Specification'])
T['S7'] = [spec, dh, rb, cb_]
TITLE['S7'] = 'Table S7. Horizon-resolved double machine learning: specification, estimates, placebo tests and robustness.'
NOTE['S7'] = 'Effects converted from CFM per GPM to m3 h-1 biogas per m3 h-1 HSW (factor 7.48).'

# ---- renumber in the order of first citation in the manuscript ----
REN={'S1':'S1','S2':'S2','S7':'S3','S5':'S4','S3':'S5','S4':'S6','S6':'S7'}
T={REN[k]:v for k,v in T.items()}; TITLE={REN[k]:v.replace('Table '+k+'.','Table '+REN[k]+'.') for k,v in TITLE.items()}; NOTE={REN[k]:v for k,v in NOTE.items()}
import glob
for f in glob.glob('tables/TableS*.csv'): os.remove(f)
# ---------------- write ----------------
for k, v in T.items():
    if isinstance(v, list):
        for i, d in enumerate(v): d.to_csv(f'tables/Table{k}{"abcd"[i]}.csv', index=False)
    else: v.to_csv(f'tables/Table{k}.csv', index=False)
wb = Workbook(); wb.remove(wb.active)
F = Font(name='Arial', size=10); FB = Font(name='Arial', size=10, bold=True); FT = Font(name='Arial', size=11, bold=True); FN = Font(name='Arial', size=9, italic=True)
fill = PatternFill('solid', fgColor='E8ECEF'); thin = Side(style='thin', color='7F8C8D'); hb = Border(top=thin, bottom=thin)
def put(ws, df, r0):
    for j, c in enumerate(df.columns, 1):
        x = ws.cell(r0, j, c); x.font = FB; x.fill = fill; x.border = hb; x.alignment = Alignment(wrap_text=True, vertical='center')
    for i, row in enumerate(df.itertuples(index=False), r0 + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (float, np.floating)):
                if np.isnan(v): v = None
                else: v = float(v)
            elif isinstance(v, (np.integer,)): v = int(v)
            elif isinstance(v, (np.bool_, bool)): v = 'yes' if v else 'no'
            x = ws.cell(i, j, v); x.font = F; x.alignment = Alignment(wrap_text=isinstance(v, str) and len(v) > 40, vertical='top')
            if isinstance(v, float):
                c = df.columns[j - 1]
                x.number_format = '0.00E+00' if c.startswith('p') else ('0.000' if ('R2' in c or 'Skill' in c or 'theta' in c or 'SE' in c or 'CI' in c or 'DM' in c) else '0.00')
    return r0 + len(df) + 2
for k in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
    if k not in T: continue
    ws = wb.create_sheet('Table ' + k); ws.cell(1, 1, TITLE[k]).font = FT
    r0 = 3; blocks = T[k] if isinstance(T[k], list) else [T[k]]
    names = ['(a) Specification', '(b) Estimates and placebo tests', '(c) Robustness checks', '(d) Heterogeneity by gas-storage tertile'] if k == 'S3' else [None]
    for d, nm in zip(blocks, names):
        if nm: ws.cell(r0, 1, nm).font = FB; r0 += 1
        r0 = put(ws, d, r0)
    ws.cell(r0, 1, 'Note. ' + NOTE[k]).font = FN
    ws.freeze_panes = 'A4' if k != 'S3' else None
    for j in range(1, ws.max_column + 1):
        L = max((len(str(ws.cell(i, j).value)) for i in range(3, ws.max_row + 1) if ws.cell(i, j).value is not None), default=8)
        ws.column_dimensions[get_column_letter(j)].width = min(max(10, L * 0.95), 60)
wb.save('tables/Supplementary_Tables_S1-S7.xlsx'); print({k: (len(v) if not isinstance(v, list) else [len(x) for x in v]) for k, v in T.items()})
