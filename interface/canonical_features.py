"""The shared training/interface feature function, with declared numeric precision.

Inputs are rounded to six decimal places in native units. Every lag-bin mean
uses the same float64 window operation and is rounded to six decimal places.
Calendar values are also rounded. No state is carried between forecast origins.
"""
import numpy as np
import pandas as pd
BINS=[(0,0),(1,2),(3,5),(6,11),(12,23),(24,47),(48,71),(72,167)]
LAB=['0','1-2','3-5','6-11','12-23','24-47','48-71','72-167']
PRECISION=6
VARIABLES=['Biogas','H-Dig1_FT','H-Dig2_FT','Q-HSW1_GPM','Q-HSW2_GPM','H-HSW_ft','V-HSW-DIG2-yest_gal','Q-PS_on','Q-TWAS_GPM','D1_TEMPERATURE','D2_TEMPERATURE','Q-influent_MGD','Q-Aeration_SCFM','Q-RAS_MGD','Q-WAS1_MGD','Q-WAS2_MGD']

def canonical_history_features(history):
    z=np.round(history[VARIABLES].iloc[-168:].to_numpy(dtype=np.float64,copy=True),PRECISION)
    values={}
    for j,var in enumerate(VARIABLES):
        for (a,b),label in zip(BINS,LAB):
            values[f'{var}|L{label}']=float(np.round(np.mean(z[-b-1:len(z)-a,j]),PRECISION))
    return values

def feature_row(history,h,plan=None,historical=None):
    values=dict(historical if historical is not None else canonical_history_features(history))
    t=history.index[-1]+pd.Timedelta(hours=h)
    values.update(hod_sin=float(np.round(np.sin(2*np.pi*t.hour/24),PRECISION)),hod_cos=float(np.round(np.cos(2*np.pi*t.hour/24),PRECISION)),dow_sin=float(np.round(np.sin(2*np.pi*t.dayofweek/7),PRECISION)),dow_cos=float(np.round(np.cos(2*np.pi*t.dayofweek/7),PRECISION)))
    if plan is not None:
        # Convert each hourly SI value first, then use the same native precision.
        values.update({'Planned HSW|F':float(np.round(np.mean(np.round(plan.hsw_m3_h.iloc[:h].to_numpy()/.227125,PRECISION)),PRECISION)),
                       'Planned PS|F':float(np.round(np.mean(np.round(plan.ps_on_fraction.iloc[:h].to_numpy(),PRECISION)),PRECISION)),
                       'Planned TWAS|F':float(np.round(np.mean(np.round(plan.twas_m3_h.iloc[:h].to_numpy()/.227125,PRECISION)),PRECISION))})
    return {k.replace(' ','_'):v for k,v in values.items()}

def current_anchor(history):return float(np.round(history.Biogas.iloc[-1],PRECISION))
