"""Shared, read-only access to the original research results."""
from pathlib import Path
import os, sys, warnings
HERE=Path(__file__).resolve().parent
# Package layout: <package>/scripts/revision/context.py
PKG=HERE.parents[1]
ROOT=PKG
WORKSPACE=PKG
# Working directory of scripts/pipeline (hourly.pkl, tab_final.pkl, final_preds.pkl, ...).
RES=Path(os.environ.get('MUSCATINE_RESULTS',PKG/'results')).resolve()
# Outputs of the revision analyses (formerly reports/major_revision_20261013/analysis).
OUT=Path(os.environ.get('MUSCATINE_REVISION_OUT',RES/'revision_analysis')).resolve()
OUT.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path.insert(0,str(PKG/'scripts'/'pipeline'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.chdir(RES)
warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
import numpy as np, pandas as pd
from common import H,idx,o_tr,o_va,o_te,build,calendar,future_feed,target,GROUPS,BINS,BIN_LAB
HS=[1,2,3,4,6,9,12,18,24]
CFM=1.69901
SEED=20261013
TR,VA,TE=idx[o_tr],idx[o_va],idx[o_te]
SELECT=VA[VA+pd.Timedelta(hours=24)<pd.Timestamp('2022-11-29')]
CAL=VA[VA>=pd.Timestamp('2022-11-29')]
X0,META=build()
def features(h,reg='past'):
    x=pd.concat([X0,calendar(idx,h)],axis=1)
    return pd.concat([x,future_feed(h)],axis=1) if reg=='feed' else x
def metric(y,p):
    e=np.asarray(y)-np.asarray(p)
    return dict(RMSE=float(np.sqrt(np.mean(e**2))*CFM),MAE=float(np.mean(np.abs(e))*CFM),R2=float(1-np.sum(e**2)/np.sum((y-np.mean(y))**2)))
def frame(path,rows):
    d=pd.DataFrame(rows);d.to_csv(OUT/path,index=False);return d
