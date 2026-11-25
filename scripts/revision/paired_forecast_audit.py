"""Benchmark loss dependence, paired errors and fully identified ablations."""
from context import *
from statistics_utils import *
fp=pd.read_pickle(OUT/'independent_final_preds.pkl')
old=pd.read_pickle(RES/'final_preds.pkl');tab=pd.read_pickle(RES/'tab_final.pkl')
tuned=pd.read_pickle(RES/'tune/preds_anchor.pkl')['P'];deep=pd.read_pickle(RES/'tune/preds_dl5.pkl')
proxies=pd.read_pickle(OUT/'available_feed_predictions.pkl')
TUNED=['Random forest','Extra trees','HistGB','LightGBM','XGBoost','CatBoost','SVR','KNN','LightGBM +feed','CatBoost +feed']
rows=[];tests=[];corr=[];metrics=[];bench={}
for h in HS:
    t=fp['pred'][h]['test'];y=t.y.values
    b={c:t[c].values for c in t if c!='y'}
    for name in TUNED:
        if (name,h) in tuned:b[name]=np.asarray(tuned[name,h]['te'])
    for name,value in deep.items():
        if 'PG-HQ' not in name:b[name]=value['te'][:,h-1]
    for method in ['Weekly feed','Four-week feed median']:b[method]=proxies[h,method]
    b['Seasonal naive (24 h)']=H.Biogas.shift(24-h).loc[TE].values
    bench[h]=b
    for model,p in b.items():metrics.append(dict(h=h,model=model,**metric(y,p),Skill=1-metric(y,p)['RMSE']/metric(y,b['Persistence'])['RMSE']))
    comparisons=[('PG-HQ ensemble [feed]','PG-HQ ensemble [past]'),('LGB-Δ [feed]','LGB-Δ [past]'),('Cat-Δ [feed]','Cat-Δ [past]'),('PG-HQ ensemble [feed]','Cat-Δ [feed]'),('Weekly feed','LGB-Δ [past]'),('Four-week feed median','LGB-Δ [past]')]
    for length in [24,72,168]:
        indices=block_indices(len(y),length)
        for candidate,reference in comparisons:
            rows.append(dict(h=h,candidate=candidate,reference=reference,block_h=length,repeats=2000,**rmse_difference(y,b[candidate],b[reference],indices)))
    if h in [1,6,24]:
        reference=b['PG-HQ ensemble [feed]']
        for model,p in b.items():
            if model=='PG-HQ ensemble [feed]':continue
            d=(reference-y)**2-(p-y)**2
            for lag,value in enumerate(acf(d)):corr.append(dict(h=h,baseline=model,lag=lag,autocorrelation=value))
            for bandwidth in sorted(set([h-1,24,72,168])):
                statistic,pvalue=hac_test(d,bandwidth)
                tests.append(dict(h=h,baseline=model,bandwidth_h=bandwidth,mean_loss_difference=float(d.mean()),statistic=statistic,p=pvalue))
        print('paired benchmark',h,flush=True)
test=pd.DataFrame(tests);test['p_Holm']=test.groupby(['h','bandwidth_h']).p.transform(lambda s:holm(s.values))
frame('paired_RMSE_uncertainty.csv',rows);frame('loss_difference_acf.csv',corr);test.to_csv(OUT/'HAC_benchmark_tests.csv',index=False)
frame('all_benchmark_metrics.csv',metrics);pd.to_pickle(bench,OUT/'benchmark_predictions.pkl')

raw_level=pd.read_csv(RES/'extra/ablation_rawlevel.csv');raw_anchor=pd.read_csv(RES/'ablation_features.csv')
abl=[]
for h in [1,6,24]:
    y=fp['pred'][h]['test'].y.values;prmse=metric(y,fp['pred'][h]['test'].Persistence.values)['RMSE']
    for identifier,representation,anchor,feed,n,rmse,r2 in [
        ('A1','Raw lags and rolling summaries',False,False,484,float(raw_level.loc[raw_level.h==h,'RMSE'].iloc[0])*CFM,float(raw_level.loc[raw_level.h==h,'R2'].iloc[0])),
        ('A2','Process-group lag bins',False,False,132,metric(y,tab['pred'][h]['test'].LightGBM.values)['RMSE'],metric(y,tab['pred'][h]['test'].LightGBM.values)['R2']),
        ('A3','Raw lags and rolling summaries',True,False,484,float(raw_anchor.loc[(raw_anchor.h==h)&(raw_anchor.n_features==484),'RMSE'].iloc[0])*CFM,float(raw_anchor.loc[(raw_anchor.h==h)&(raw_anchor.n_features==484),'R2'].iloc[0])),
        ('A4','Process-group lag bins',True,False,132,metric(y,fp['pred'][h]['test']['LGB-Δ [past]'].values)['RMSE'],metric(y,fp['pred'][h]['test']['LGB-Δ [past]'].values)['R2']),
        ('A5','Process-group lag bins',True,True,135,metric(y,fp['pred'][h]['test']['LGB-Δ [feed]'].values)['RMSE'],metric(y,fp['pred'][h]['test']['LGB-Δ [feed]'].values)['R2'])]:
        abl.append(dict(h=h,configuration=identifier,representation=representation,persistence_anchor=anchor,executed_feed=feed,n_features=n,RMSE_m3_h=rmse,R2=r2,Skill=1-rmse/prmse))
frame('ablation_configuration_register.csv',abl)
print('PAIRED FORECAST AUDIT COMPLETE',flush=True)
