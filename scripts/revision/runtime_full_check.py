from pathlib import Path
import sys,json,numpy as np,pandas as pd,lightgbm
import os
RES=Path(os.environ.get('MUSCATINE_RESULTS',Path(__file__).resolve().parents[2]/'results'));HERE=RES/'interface_runtime_check'
sys.path.insert(0,str(RES/'interface_rebuild'))
import predictor as gui
hist=pd.read_csv(HERE/'runtime_history.csv',index_col=0,parse_dates=True,float_precision='round_trip');plan=pd.read_csv(HERE/'runtime_plan.csv',index_col=0,parse_dates=True,float_precision='round_trip');refs=np.load(HERE/'runtime_reference.npz')
origins=pd.to_datetime(json.loads((HERE/'runtime_origins.json').read_text(encoding='utf8')));diff={key:[]for key in refs}
for i,t in enumerate(origins):
    h=hist.loc[t-pd.Timedelta(hours=167):t].to_csv(float_format='%.17g');p=plan.loc[t+pd.Timedelta(hours=1):t+pd.Timedelta(hours=24)].to_csv(float_format='%.17g')
    got=gui.predict(h,p)
    for row in got['rows']:
        for reg in ['past','feed']:
            key=f'{row["horizon_h"]}_{reg}';diff[key].append(abs(row[reg+'_m3_h']-refs[key][i]))
    if i%500==0:print('Installed GUI runtime checked',i,flush=True)
maxima={k:float(np.max(v))for k,v in diff.items()};assert max(maxima.values())<1e-10,maxima
(HERE/'installed_runtime_audit.json').write_text(json.dumps({'n_origins':len(origins),'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lightgbm.__version__,'maximum_difference_m3_h':maxima},indent=2),encoding='utf8')
print('Installed runtime: all 1797 origins match.',flush=True)
