"""Unified Optuna (TPE) tuning of every tabular model on the validation period.
The pre-specified configuration is enqueued as trial 0, so a tuned model is never worse than
the default on validation. The test period is never used for selection.
Outputs: tune/best_<model>_<h>.pkl, tune/trials.csv, tune/tuned_anchor.csv, tune/tuned_all_horizons.csv,
tune/preds_anchor.pkl"""
import pandas as pd, numpy as np, pickle, os, time, warnings; warnings.filterwarnings('ignore')
from common import *
import lightgbm as lgb, xgboost as xgb, catboost as cb, optuna
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs('tune', exist_ok=True)
X0, meta = build()
tr, va, te = idx[o_tr], idx[o_va], idx[o_te]
def Xpast(h): return pd.concat([X0, calendar(X0.index, h)], axis=1)
def Xfeed(h): return pd.concat([Xpast(h), future_feed(h)], axis=1)
def Y(h): return pd.Series(target(h).values, index=idx)
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

# ---- search spaces (sampler) and builders (params -> estimator) ----
def s_lgb(t): return dict(n_estimators=t.suggest_int('n_estimators', 200, 1200, step=100), learning_rate=t.suggest_float('learning_rate', 0.005, 0.1, log=True),
                          num_leaves=t.suggest_int('num_leaves', 7, 63), min_child_samples=t.suggest_int('min_child_samples', 10, 150),
                          colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0), reg_lambda=t.suggest_float('reg_lambda', 1e-3, 20, log=True))
def b_lgb(p): return lgb.LGBMRegressor(**p, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0, n_jobs=2)
d_lgb = dict(n_estimators=800, learning_rate=0.02, num_leaves=15, min_child_samples=40, colsample_bytree=0.6, reg_lambda=2.0)

def s_cat(t): return dict(iterations=t.suggest_int('iterations', 300, 1200, step=100), learning_rate=t.suggest_float('learning_rate', 0.01, 0.15, log=True),
                          depth=t.suggest_int('depth', 3, 7), l2_leaf_reg=t.suggest_float('l2_leaf_reg', 0.5, 30, log=True))
def b_cat(p): return cb.CatBoostRegressor(**p, verbose=0, random_seed=0, thread_count=2)
d_cat = dict(iterations=1200, learning_rate=0.03, depth=5, l2_leaf_reg=5.0)

def s_xgb(t): return dict(n_estimators=t.suggest_int('n_estimators', 200, 1200, step=100), learning_rate=t.suggest_float('learning_rate', 0.005, 0.1, log=True),
                          max_depth=t.suggest_int('max_depth', 2, 8), min_child_weight=t.suggest_float('min_child_weight', 1, 50, log=True),
                          colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0), reg_lambda=t.suggest_float('reg_lambda', 1e-3, 20, log=True))
def b_xgb(p): return xgb.XGBRegressor(**p, subsample=0.8, random_state=0, n_jobs=2)
d_xgb = dict(n_estimators=800, learning_rate=0.02, max_depth=4, min_child_weight=10.0, colsample_bytree=0.6, reg_lambda=2.0)

def s_rf(t): return dict(min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 40, log=True), max_features=t.suggest_float('max_features', 0.1, 1.0),
                         max_depth=t.suggest_categorical('max_depth', [8, 16, None]))
def b_rf(p): return RandomForestRegressor(300, **p, n_jobs=2, random_state=0)
def b_et(p): return ExtraTreesRegressor(300, **p, n_jobs=2, random_state=0)
d_rf = dict(min_samples_leaf=5, max_features=0.33, max_depth=None)
d_et = dict(min_samples_leaf=5, max_features=0.5, max_depth=None)

def s_hgb(t): return dict(max_iter=t.suggest_int('max_iter', 200, 1200, step=100), learning_rate=t.suggest_float('learning_rate', 0.005, 0.1, log=True),
                          max_leaf_nodes=t.suggest_int('max_leaf_nodes', 7, 63), min_samples_leaf=t.suggest_int('min_samples_leaf', 10, 150),
                          l2_regularization=t.suggest_float('l2_regularization', 1e-3, 20, log=True))
def b_hgb(p): return HistGradientBoostingRegressor(**p, random_state=0)
d_hgb = dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)

def s_svr(t): return dict(C=t.suggest_float('C', 0.1, 300, log=True), epsilon=t.suggest_float('epsilon', 0.1, 10, log=True), gamma=t.suggest_float('gamma', 1e-4, 0.05, log=True))
def b_svr(p): return make_pipeline(StandardScaler(), SVR(**p))
d_svr = dict(C=30.0, epsilon=2.0, gamma=1 / 136)

def s_knn(t): return dict(n_neighbors=t.suggest_int('n_neighbors', 5, 150, log=True), weights=t.suggest_categorical('weights', ['uniform', 'distance']), p=t.suggest_categorical('p', [1, 2]))
def b_knn(p): return make_pipeline(StandardScaler(), KNeighborsRegressor(**p))
d_knn = dict(n_neighbors=25, weights='distance', p=2)

# name: (sampler, builder, default, delta-target?, regime, n_trials)
SPEC = {
    'Random forest': (s_rf, b_rf, d_rf, False, 'past', 15),
    'Extra trees': (s_rf, b_et, d_et, False, 'past', 15),
    'HistGB': (s_hgb, b_hgb, d_hgb, False, 'past', 25),
    'LightGBM': (s_lgb, b_lgb, d_lgb, False, 'past', 30),
    'XGBoost': (s_xgb, b_xgb, d_xgb, False, 'past', 25),
    'CatBoost': (s_cat, b_cat, d_cat, False, 'past', 15),
    'SVR': (s_svr, b_svr, d_svr, False, 'past', 20),
    'KNN': (s_knn, b_knn, d_knn, False, 'past', 25),
    'LightGBM +feed': (s_lgb, b_lgb, d_lgb, False, 'feed', 30),
    'CatBoost +feed': (s_cat, b_cat, d_cat, False, 'feed', 15),
    'LGB-Δ [past]': (s_lgb, b_lgb, d_lgb, True, 'past', 30),
    'Cat-Δ [past]': (s_cat, b_cat, d_cat, True, 'past', 15),
    'LGB-Δ [feed]': (s_lgb, b_lgb, d_lgb, True, 'feed', 30),
    'Cat-Δ [feed]': (s_cat, b_cat, d_cat, True, 'feed', 15),
}
ANCH = [1, 6, 24]
def fitpred(name, h, p, sets):
    s, b, d, delta, reg, n = SPEC[name]
    X = Xfeed(h) if reg == 'feed' else Xpast(h); y = Y(h); base = X['Biogas|L0']
    m = b(p).fit(X.loc[tr], ((y - base) if delta else y)[tr])
    return {k: m.predict(X.loc[ix]) + (base[ix].values if delta else 0) for k, ix in sets.items()}, y

if __name__ == '__main__':
    rows = []
    for name in SPEC:
        s, b, d, delta, reg, n = SPEC[name]
        for h in ANCH:
            f = f'tune/best_{name}_{h}.pkl'.replace(' ', '_')
            if os.path.exists(f): continue
            X = Xfeed(h) if reg == 'feed' else Xpast(h); y = Y(h); base = X['Biogas|L0']; tgt = (y - base) if delta else y
            def obj(t):
                m = b(s(t)).fit(X.loc[tr], tgt[tr]); return rmse(y[va].values, m.predict(X.loc[va]) + (base[va].values if delta else 0))
            st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0)); st.enqueue_trial(d)
            t0 = time.time(); st.optimize(obj, n_trials=n)
            tl = [dict(model=name, h=h, trial=t.number, value=t.value, **{f'p_{k}': v for k, v in t.params.items()}) for t in st.trials]
            try: imp = optuna.importance.get_param_importances(st)
            except Exception: imp = {}
            pickle.dump(dict(best=st.best_params, best_val=st.best_value, default_val=st.trials[0].value, trials=tl, imp=imp, sec=time.time() - t0), open(f, 'wb'))
            print(name, h, 'default', round(st.trials[0].value, 2), 'best', round(st.best_value, 2), round(time.time() - t0), 's', flush=True)
    # refit at anchors: val + test predictions, fit/predict timing
    P = {}; R = []; T = []
    for name in SPEC:
        for h in ANCH:
            B = pickle.load(open(f'tune/best_{name}_{h}.pkl'.replace(' ', '_'), 'rb')); T += B['trials']
            t0 = time.time(); pr, y = fitpred(name, h, B['best'], {'va': va, 'te': te}); sec = time.time() - t0
            P[(name, h)] = pr
            R.append(dict(h=h, model=name, regime=SPEC[name][4], val_RMSE_default=B['default_val'], val_RMSE_tuned=B['best_val'], tune_sec=B['sec'], refit_sec=sec,
                          **{'test_' + k: v for k, v in metrics(y[te].values, pr['te']).items()}))
            print('refit', name, h, round(R[-1]['test_R2'], 3), flush=True)
    pd.DataFrame(T).to_csv('tune/trials.csv', index=False); pd.DataFrame(R).to_csv('tune/tuned_anchor.csv', index=False)
    pickle.dump(dict(P=P, va=va, te=te), open('tune/preds_anchor.pkl', 'wb'))
    # all horizons: parameters from the nearest anchor (1 -> 1-3 h, 6 -> 4-9 h, 24 -> 12-24 h)
    near = lambda h: 1 if h <= 3 else (6 if h <= 9 else 24)
    A = []
    for h in [1, 2, 3, 4, 6, 9, 12, 18, 24]:
        for name in SPEC:
            B = pickle.load(open(f'tune/best_{name}_{near(h)}.pkl'.replace(' ', '_'), 'rb'))
            pr, y = fitpred(name, h, B['best'], {'te': te})
            A.append(dict(h=h, model=name, params_from_h=near(h), **metrics(y[te].values, pr['te'])))
        print('all-h', h, flush=True); pd.DataFrame(A).to_csv('tune/tuned_all_horizons.csv', index=False)
    print('done')
