from context import *
import shutil,importlib.util,json
ready=RES/'interface_rebuild'
shutil.copytree(PKG/'interface/examples',ready/'examples',dirs_exist_ok=True)
source=(ready/'predictor.py').read_text(encoding='utf8')
source=source.replace("'model':'Persistence-anchored LightGBM (canonical six-decimal features)'", "'model':'Persistence-anchored LightGBM (LGB-Δ)'")
if "'issued_at':" not in source:source=source.replace("return {'origin':str(origin),", "return {'origin':str(origin),'issued_at':str(origin+pd.Timedelta(hours=1)),")
(ready/'predictor.py').write_text(source,encoding='utf8')
html=(ready/'index.html').read_text(encoding='utf8')
html=html.replace('Intervals are calibrated on validation residuals','Intervals use independent pre-test calibration residuals')
html=html.replace("'Forecast origin · '+d.origin", "'Last complete hour · '+d.origin+' | Issued · '+d.issued_at")
html=html.replace('Forecast horizon (hours from origin)','Forecast horizon (successive complete hours)')
(ready/'index.html').write_text(html,encoding='utf8')
sys.path.insert(0,str(ready))
spec=importlib.util.spec_from_file_location('canonical_gui',ready/'predictor.py');gui=importlib.util.module_from_spec(spec);spec.loader.exec_module(gui)
si=pd.DataFrame({col:H[v]*factor+offset for col,v,factor,offset in gui.SCHEMA},index=H.index);si.index.name='timestamp'
plan=pd.DataFrame({'hsw_m3_h':(H['Q-HSW1_GPM']+H['Q-HSW2_GPM'])*.227125,'ps_on_fraction':H['Q-PS_on'],'twas_m3_h':H['Q-TWAS_GPM']*.227125},index=H.index);plan.index.name='timestamp'
reference=pd.read_pickle(OUT/'gui_canonical_predictions.pkl');differences={(h,r):[] for h in HS for r in ['past','feed']}
for i,t in enumerate(TE):
    hs=si.loc[t-pd.Timedelta(hours=167):t].to_csv(float_format='%.17g')
    ps=plan.loc[t+pd.Timedelta(hours=1):t+pd.Timedelta(hours=24)].to_csv(float_format='%.17g')
    result=gui.predict(hs,ps)
    for row in result['rows']:
        h=row['horizon_h']
        for reg in ['past','feed']:differences[h,reg].append(abs(row[reg+'_m3_h']-reference[h,reg][i]*CFM))
    if i%300==0:print('complete CSV inference',i,flush=True)
rows=[]
for (h,reg),d in differences.items():
    d=np.asarray(d);assert d.max()<1e-10,(h,reg,d.max())
    rows.append(dict(h=h,regime=reg,n=len(d),max_difference_m3_h=d.max(),median=np.median(d),p95=np.quantile(d,.95),p99=np.quantile(d,.99),fraction_above_1e_8=np.mean(d>1e-8)))
frame('gui_full_predict_validation.csv',rows)
example=gui.predict((ready/'examples/history_168h.csv').read_text(encoding='utf8'),(ready/'examples/recorded_feed_oracle_24h.csv').read_text(encoding='utf8'))
(OUT/'interface_example.json').write_text(json.dumps(example,indent=2),encoding='utf8')
print('ALL 1797 CSV PREDICTIONS VERIFIED',flush=True)
