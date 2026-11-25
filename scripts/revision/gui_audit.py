"""Round-trip every eligible origin through the actual SI-unit CSV parser.

Batch model calls are equivalent to independent predict calls; all history and
plan records use the same parser and feature function as the interface.
"""
from context import *
import importlib.util,json
from io import StringIO
spec=importlib.util.spec_from_file_location('interface_predictor',HERE/'reference_interface_source/predictor.py')
gui=importlib.util.module_from_spec(spec);spec.loader.exec_module(gui)
si=pd.DataFrame({col:H[v]*factor+offset for col,v,factor,offset in gui.SCHEMA},index=H.index)
si.index.name='timestamp'
plan=pd.DataFrame({'hsw_m3_h':(H['Q-HSW1_GPM']+H['Q-HSW2_GPM'])*.227125,'ps_on_fraction':H['Q-PS_on'],'twas_m3_h':H['Q-TWAS_GPM']*.227125},index=H.index)
plan.index.name='timestamp';native_models=pd.read_pickle(RES/'tab_final.pkl')['models']
future_cache={h:future_feed(h).loc[TE] for h in HS}
matrix={};native_matrix={};bases=[];decomp=[]
for i,t in enumerate(TE):
    hist=si.loc[t-pd.Timedelta(hours=167):t]
    parsed=gui.read_history(hist.to_csv(float_format='%.17g'))
    f=gui.feature_row(parsed,1);native=gui.feature_row(H.loc[hist.index],1)
    block={k:v for k,v in f.items() if '|L' in k};nblock={k:v for k,v in native.items() if '|L' in k}
    bases.append(parsed.Biogas.iloc[-1])
    pp=plan.loc[t+pd.Timedelta(hours=1):t+pd.Timedelta(hours=24)]
    parsed_plan=gui.csv_frame(pp.to_csv(float_format='%.17g'),['hsw_m3_h','ps_on_fraction','twas_m3_h'])
    for h in HS:
        cal=calendar(pd.DatetimeIndex([t]),h).iloc[0].to_dict()
        for reg in ['past','feed']:
            fv={**block,**cal};nv={**nblock,**cal}
            if reg=='feed':
                feeds={'Planned_HSW|F':float(parsed_plan.hsw_m3_h.iloc[:h].mean()/.227125),'Planned_PS|F':float(parsed_plan.ps_on_fraction.iloc[:h].mean()),'Planned_TWAS|F':float(parsed_plan.twas_m3_h.iloc[:h].mean()/.227125)}
                fv.update(feeds);nv.update({k:v for k,v in zip(feeds,future_cache[h].iloc[i].values)})
            matrix.setdefault((h,reg),[]).append(fv);native_matrix.setdefault((h,reg),[]).append(nv)
    if i%300==0:print('CSV origins',i,flush=True)
rows=[];preds={};feature_rows=[]
for h in HS:
    for reg in ['past','feed']:
        m=native_models[h,f'LGB-Δ [{reg}]'];names=[x.replace(' ','_') for x in features(h,reg).columns]
        reference=features(h,reg).loc[TE].values;native=np.array([[d[n] for n in names] for d in native_matrix[h,reg]])
        rebuilt=np.array([[d[n] for n in names] for d in matrix[h,reg]])
        reference_prediction=m.predict(reference)+H.Biogas.loc[TE].values
        native_prediction=m.predict(native)+H.Biogas.loc[TE].values
        csv_prediction=m.predict(rebuilt)+bases
        preds[h,reg]=dict(reference=reference_prediction,native_window=native_prediction,csv=csv_prediction)
        for label,p in [('local_window',native_prediction),('SI_CSV',csv_prediction)]:
            diff=abs(p-reference_prediction)*CFM
            rows.append(dict(h=h,regime=reg,stage=label,n=len(diff),max=diff.max(),median=np.median(diff),p95=np.quantile(diff,.95),p99=np.quantile(diff,.99),fraction_above_1e_8=np.mean(diff>1e-8),fraction_above_0_1=np.mean(diff>.1),RMSE_m3_h=metric(target(h).loc[TE].values,p)['RMSE']))
        for j,name in enumerate(names):
            delta=rebuilt[:,j]-reference[:,j]
            feature_rows.append(dict(h=h,regime=reg,feature=name,max_abs_difference=abs(delta).max(),native_local_max_abs_difference=abs(native[:,j]-reference[:,j]).max(),fraction_different=np.mean(delta!=0)))
        # Record every numerical split whose decision is changed by reconstruction.
        def visit(node):
            if 'split_feature' not in node:return
            j=node['split_feature'];threshold=node['threshold']
            if isinstance(threshold,(float,int)):
                crosses=(reference[:,j]<=threshold)!=(rebuilt[:,j]<=threshold)
                if crosses.any():decomp.append(dict(h=h,regime=reg,feature=names[j],threshold=threshold,affected_rows=int(crosses.sum()),max_feature_difference=float(abs(rebuilt[crosses,j]-reference[crosses,j]).max())))
            visit(node['left_child']);visit(node['right_child'])
        for tree in m.booster_.dump_model()['tree_info']:visit(tree['tree_structure'])
frame('gui_all_origin_validation.csv',rows);frame('gui_feature_roundoff.csv',feature_rows);frame('gui_crossed_thresholds.csv',decomp)
pd.to_pickle(preds,OUT/'gui_all_origin_predictions.pkl')
pd.to_pickle({'SI':matrix,'native_window':native_matrix,'base':bases},OUT/'gui_reconstructed_features.pkl')
print(pd.DataFrame(rows).to_string(index=False),flush=True)
print('GUI AUDIT COMPLETE',flush=True)
