"""Deep models: learning-rate selection on the validation period (seed 0), then five seeds at the
selected rate. PG-HQ keeps the learning rate selected in tune_pghq.py; seeds 0-2 are reloaded and
seeds 3-4 are trained. Outputs: tune/dl_lr_search.csv, tune/dl_seeds.csv, tune/dl_cost.csv, tune/preds_dl5.pkl"""
import sys, time, os, numpy as np, pandas as pd, torch
torch.set_num_threads(2)
import dl
from dl import *
os.makedirs('tune', exist_ok=True)
MODELS = {'MLP': MLPm, 'LSTM': LSTMm, 'GRU': GRUm, 'TCN': TCN, 'Transformer': TransF, 'PatchTST': PatchTST, 'DLinear': DLinear,
          'N-BEATS': NBEATS, 'iTransformer': iTrans,
          'PG-HQ [past]': lambda: PGHQ2(d=32, use_feed=False), 'PG-HQ [feed]': lambda: PGHQ2(d=32, use_feed=True)}
LRS = [2e-4, 5e-4, 1e-3]
Yte = dl.D['te'][1].numpy(); Yva = dl.D['va'][1].numpy()
def hmet(p, Y, h):
    e = Y[:, h - 1] - p[:, h - 1]; return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                                               R2=float(1 - np.sum(e ** 2) / np.sum((Y[:, h - 1] - Y[:, h - 1].mean()) ** 2)))
lr_rows, seed_rows, cost_rows = [], [], []
P = pd.read_pickle('tune/preds_dl5.pkl') if os.path.exists('tune/preds_dl5.pkl') else {}
for f, L in [('tune/dl_lr_search.csv', lr_rows), ('tune/dl_seeds.csv', seed_rows), ('tune/dl_cost.csv', cost_rows)]:
    if os.path.exists(f): L += pd.read_csv(f).to_dict('records')
sel = sys.argv[1:] or list(MODELS)
for nm in sel:
    if nm in P: continue
    runs = {}
    if nm.startswith('PG-HQ'):
        lr = 1e-3; tag = 'feed' if 'feed' in nm else 'past'
        for s in [0, 1, 2]:
            m = MODELS[nm](); m.load_state_dict(torch.load(f'pghq_{tag}_s{s}.pt')); runs[s] = (m, np.nan, np.nan)
    else:
        for lr_ in LRS:
            t0 = time.time(); m, ep = train(MODELS[nm](), seed=0, lr=lr_); v = evalp(m, 'va')
            lr_rows.append(dict(model=nm, lr=lr_, val_RMSE=v, epochs=ep + 1, sec=time.time() - t0)); runs[('lr', lr_)] = (m, ep, time.time() - t0)
            print(nm, 'lr', lr_, round(v, 2), ep, round(time.time() - t0), flush=True)
        lr = min([r for r in lr_rows if r['model'] == nm], key=lambda r: r['val_RMSE'])['lr']; runs[0] = runs[('lr', lr)]
    for s in range(5):
        if s not in runs:
            t0 = time.time(); m, ep = train(MODELS[nm](), seed=s, lr=lr); runs[s] = (m, ep, time.time() - t0)
            if nm.startswith('PG-HQ'): torch.save(m.state_dict(), f"pghq_{'feed' if 'feed' in nm else 'past'}_s{s}.pt")
            print(nm, 'seed', s, ep, round(time.time() - t0), flush=True)
    Ps, Pv = [], []
    for s in range(5):
        m, ep, sec = runs[s]; p = predict(m, 'te'); Ps.append(p); Pv.append(predict(m, 'va'))
        for h in [1, 6, 24]: seed_rows.append(dict(model=nm, lr=lr, seed=s, h=h, epochs=(ep + 1) if ep == ep else np.nan, train_sec=sec, **hmet(p, Yte, h)))
    m = runs[0][0]; npar = sum(p.numel() for p in m.parameters())
    t0 = time.time(); predict(m, 'te'); inf = (time.time() - t0) / len(Yte) * 1e3
    tsec = np.nanmean([runs[s][2] for s in range(5)])
    cost_rows.append(dict(model=nm, lr=lr, n_params=npar, train_sec_per_seed=tsec, infer_ms_per_origin=inf))
    P[nm] = dict(te=np.mean(Ps, 0), va=np.mean(Pv, 0))
    pd.DataFrame(lr_rows).to_csv('tune/dl_lr_search.csv', index=False); pd.DataFrame(seed_rows).to_csv('tune/dl_seeds.csv', index=False)
    pd.DataFrame(cost_rows).to_csv('tune/dl_cost.csv', index=False); pd.to_pickle(P, 'tune/preds_dl5.pkl')
    print('done', nm, flush=True)
