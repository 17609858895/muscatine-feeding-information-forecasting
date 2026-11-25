"""Refit an explicitly versioned interface implementation using canonical features."""
from context import *
from canonical_features import feature_row,canonical_history_features,current_anchor
import importlib.util,json,shutil
original_gui=HERE/'reference_interface_source'
spec=importlib.util.spec_from_file_location('oldgui',original_gui/'predictor.py');gui=importlib.util.module_from_spec(spec);spec.loader.exec_module(gui)
destination=RES/'interface_rebuild';destination.mkdir(exist_ok=True)
for p in original_gui.iterdir():
    if p.is_file():shutil.copy2(p,destination/p.name)
(destination/'models').mkdir(exist_ok=True)
shutil.copy2(HERE/'canonical_features.py',destination/'canonical_features.py')
for name in ['history_bounds.json']:shutil.copy2(original_gui/'models'/name,destination/'models'/name)
si=pd.DataFrame({col:H[v]*factor+offset for col,v,factor,offset in gui.SCHEMA},index=H.index);si.index.name='timestamp'
plan=pd.DataFrame({'hsw_m3_h':(H['Q-HSW1_GPM']+H['Q-HSW2_GPM'])*.227125,'ps_on_fraction':H['Q-PS_on'],'twas_m3_h':H['Q-TWAS_GPM']*.227125},index=H.index);plan.index.name='timestamp'
origins=TR.union(VA).union(TE);hist_cache={};csv_cache={};base=[];csvbase=[]
for i,t in enumerate(origins):
    hist=H.loc[t-pd.Timedelta(hours=167):t];hist_cache[t]=canonical_history_features(hist);base.append(current_anchor(hist))
    if t in TE:
        parsed=gui.read_history(si.loc[hist.index].to_csv(float_format='%.17g'));csv_cache[t]=canonical_history_features(parsed);csvbase.append(current_anchor(parsed))
    if i%1500==0:print('canonical histories',i,flush=True)
base=pd.Series(base,index=origins);oldmodels=pd.read_pickle(RES/'tab_final.pkl')['models'];rows=[];calibration={};predictions={}
for h in HS:
    for reg in ['past','feed']:
        xx=[];cc=[]
        for t in origins:
            hist=H.loc[[t]];pp=plan.loc[t+pd.Timedelta(hours=1):t+pd.Timedelta(hours=24)] if reg=='feed' else None
            fv=feature_row(hist,h,pp,hist_cache[t]);xx.append(fv)
            if t in TE:
                parsed_plan=gui.csv_frame(pp.to_csv(float_format='%.17g'),['hsw_m3_h','ps_on_fraction','twas_m3_h']) if pp is not None else None
                cc.append(feature_row(hist,h,parsed_plan,csv_cache[t]))
        x=pd.DataFrame(xx,index=origins);csvx=pd.DataFrame(cc,index=TE)[x.columns]
        assert np.array_equal(x.loc[TE].values,csvx.values),f'Feature mismatch {h} {reg}'
        assert np.array_equal(base.loc[TE].values,csvbase)
        original=oldmodels[h,f'LGB-Δ [{reg}]'];model=type(original)(**original.get_params());y=target(h)
        model.fit(x.loc[TR],(y-base).loc[TR]);key=f'{reg}_{h}'
        (destination/'models'/f'{key}.txt').write_text(model.booster_.model_to_string(),encoding='utf8')
        prediction=model.predict(x.loc[TE])+base.loc[TE].values;csvprediction=model.predict(csvx)+csvbase
        residual=abs(model.predict(x.loc[CAL])+base.loc[CAL].values-y.loc[CAL].values)
        rank=int(np.ceil((len(residual)+1)*.9));q=np.sort(residual)[rank-1]*CFM
        calibration[key]={'halfwidth_m3_h':float(q),'n':len(residual),'method':'Independent calibration; finite order statistic','native_input_decimal_places':6}
        oldp=original.predict(features(h,reg).loc[TE])+H.Biogas.loc[TE].values
        rows.append(dict(h=h,regime=reg,n=len(TE),n_calibration=len(residual),max_CSV_difference_m3_h=float(abs(prediction-csvprediction).max()*CFM),max_change_from_archived_m3_h=float(abs(prediction-oldp).max()*CFM),**metric(y.loc[TE].values,prediction),coverage_90=float(np.mean(abs(prediction-y.loc[TE].values)*CFM<=q)),halfwidth_m3_h=q))
        predictions[h,reg]=prediction
        print('canonical interface',h,reg,'exact',flush=True)
(destination/'models/calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf8')
source=(destination/'predictor.py').read_text(encoding='utf8');start=source.index('def feature_row(');end=source.index('def predict(',start)
source=source[:start]+'from canonical_features import feature_row, canonical_history_features, current_anchor\n\n'+source[end:]
source=source.replace("history=read_history(history_text); origin=history.index[-1];plan=None","history=read_history(history_text); origin=history.index[-1];plan=None\n    historical=canonical_history_features(history)")
source=source.replace('feature_row(history,h,p)','feature_row(history,h,p,historical)').replace('history.Biogas.iloc[-1]*1.69901','current_anchor(history)*1.69901').replace('+history.Biogas.iloc[-1])','+current_anchor(history))')
source=source.replace('same local, timezone-free timestamps','same timezone-free timestamps').replace('Nominal 90% validation-calibrated interval','Nominal 90% independently calibrated interval')
source=source.replace("'model':'Persistence-anchored LightGBM (LGB-Δ)'","'model':'Persistence-anchored LightGBM (canonical six-decimal features)' ")
(destination/'predictor.py').write_text(source,encoding='utf8')
frame('gui_canonical_validation.csv',rows);pd.to_pickle(predictions,OUT/'gui_canonical_predictions.pkl')
print('CANONICAL INTERFACE COMPLETE',flush=True)
