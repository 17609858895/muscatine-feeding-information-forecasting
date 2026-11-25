"""Refit the selected network and ensemble weights without the calibration window."""
from context import *
import torch, pickle, time, json
from scipy.optimize import nnls
torch.set_num_threads(2)
import dl
mask=VA.isin(SELECT)
full_validation=[v.clone() for v in dl.D['va']]
dl.D['va']=[v[mask] for v in full_validation]
original=pd.read_pickle(RES/'final_preds.pkl')
output={'pred':{},'selection':SELECT,'calibration':CAL,'network_selection':[]}
checkpoint=OUT/'independent_network_predictions.pkl'
network=pd.read_pickle(checkpoint) if checkpoint.exists() else {}
selection_records=OUT/'independent_network_selection.csv'
if network and selection_records.exists():
    output['network_selection']=pd.read_csv(selection_records).to_dict('records')
for regime in ['past','feed']:
    if regime in network:continue
    pval=[];ptest=[]
    for seed in range(3):
        torch.manual_seed(SEED+seed);np.random.seed(SEED+seed)
        model=dl.PGHQ2(d=32,use_feed=regime=='feed')
        start=time.time()
        model,epoch=dl.train(model,seed=SEED+seed,lr=.001,epochs=80,patience=12)
        select_rmse=dl.evalp(model,'va')
        dl.D['va']=full_validation
        pval.append(dl.predict(model,'va'));ptest.append(dl.predict(model,'te'))
        dl.D['va']=[v[mask] for v in full_validation]
        torch.save(model.state_dict(),OUT/f'pghq_independent_{regime}_{seed}.pt')
        record=dict(regime=regime,seed=seed,epochs=epoch+1,selection_RMSE_m3_h=select_rmse*CFM,seconds=time.time()-start)
        output['network_selection'].append(record)
        print(record,flush=True)
    network[regime]={'val':np.mean(pval,axis=0),'test':np.mean(ptest,axis=0)}
    pd.to_pickle(network,checkpoint)
    frame('independent_network_selection.csv',output['network_selection'])
rows=[];weights=[]
for h in HS:
    source=original['pred'][h]
    output['pred'][h]={split:source[split].copy() for split in ['val','test']}
    for regime in ['past','feed']:
        cols=[f'LGB-Δ [{regime}]',f'Cat-Δ [{regime}]',f'PG-HQ [{regime}]']
        for split in ['val','test']:
            output['pred'][h][split][cols[-1]]=network[regime][split][:,h-1]
        val=output['pred'][h]['val']
        w,_=nnls(val.loc[SELECT,cols].values,val.loc[SELECT,'y'].values)
        assert w.sum()>0
        w=w/w.sum()
        key=f'PG-HQ ensemble [{regime}]'
        for split in ['val','test']:
            df=output['pred'][h][split];df[key]=df[cols].values@w
        for col,weight in zip(cols,w):weights.append(dict(h=h,regime=regime,component=col,weight=weight,n_selection=len(SELECT),n_calibration=len(CAL)))
    test=output['pred'][h]['test']
    test['Seasonal naive (168 h)']=H.Biogas.shift(168-h).loc[TE].values
    for model in ['Persistence','Seasonal naive (168 h)','LGB-Δ [past]','LGB-Δ [feed]','Cat-Δ [past]','Cat-Δ [feed]','PG-HQ ensemble [past]','PG-HQ ensemble [feed]']:
        vals=metric(test.y.values,test[model].values)
        vals['Skill']=1-vals['RMSE']/metric(test.y.values,test.Persistence.values)['RMSE']
        rows.append(dict(h=h,model=model,**vals))
pd.to_pickle(output,OUT/'independent_final_preds.pkl')
frame('independent_primary_metrics.csv',rows)
frame('independent_ensemble_weights.csv',weights)
frame('independent_network_selection.csv',output['network_selection'])
print('INDEPENDENT ENSEMBLES COMPLETE',flush=True)
