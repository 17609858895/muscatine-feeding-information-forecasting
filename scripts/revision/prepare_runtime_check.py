from context import *
import json
CHECK=RES/'interface_runtime_check';CHECK.mkdir(exist_ok=True)
sys.path.insert(0,str(RES/'interface_rebuild'))
import predictor as gui
data=pd.DataFrame({c:H[v]*factor+off for c,v,factor,off in gui.SCHEMA},index=H.index);data.index.name='timestamp';data.to_csv(CHECK/'runtime_history.csv',float_format='%.17g')
plan=pd.DataFrame({'hsw_m3_h':(H['Q-HSW1_GPM']+H['Q-HSW2_GPM'])*.227125,'ps_on_fraction':H['Q-PS_on'],'twas_m3_h':H['Q-TWAS_GPM']*.227125},index=H.index);plan.index.name='timestamp';plan.to_csv(CHECK/'runtime_plan.csv',float_format='%.17g')
(CHECK/'runtime_origins.json').write_text(json.dumps([str(t)for t in TE]),encoding='utf8')
pred=pd.read_pickle(OUT/'gui_canonical_predictions.pkl');np.savez(CHECK/'runtime_reference.npz',**{f'{h}_{r}':v*CFM for (h,r),v in pred.items()})
