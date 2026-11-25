"""Publication figures from the explicitly separated selection/calibration run."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/'scripts'/'revision'))
from context import *
sys.path.insert(0,str(PKG/'figures_scripts'/'plotting'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator
from ml_publication_style import configure_publication_style,style_axis,numeric_ticks,export_subfigures,save_figure_with_subfigures
from scientific_palettes import get_palette
import string
FIG=_Path(__file__).resolve().parent/'figures_ready';FIG.mkdir(exist_ok=True)
DPI=int(os.environ.get('REVISION_FIGURE_DPI','600'))
configure_publication_style()
plt.rcParams.update({'font.weight':'normal','axes.labelweight':'normal','font.size':10,'axes.labelsize':10.5,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,'axes.linewidth':1.1,'xtick.major.width':1.1,'ytick.major.width':1.1,'xtick.major.size':4,'ytick.major.size':4})
C=get_palette('au_nimno_soft_8',n=8);PAST=C[0];FEED=C[1];STORE=C[2];GAS=C[3];INK='#222222'
groups=list(GROUPS)+['Calendar','Planned feed'];SHORT=['Gas','Storage','HSW','Sludge','Thermal','Hydraulics','Calendar','Future feed']
GC=[C[0],C[2],C[1],C[7],C[5],C[3],C[4],C[6]]
M=pd.read_csv(OUT/'all_benchmark_metrics.csv');P=pd.read_csv(OUT/'paired_RMSE_uncertainty.csv')
A=pd.read_csv(OUT/'attribution_consistency.csv');L=pd.read_csv(OUT/'loco_paired_uncertainty.csv')
HSW=pd.read_csv(OUT/'HSW_lag_stability.csv');F=pd.read_csv(OUT/'firm_supply_independent.csv')
CP=pd.read_csv(OUT/'commitment_paired_uncertainty.csv');CR=pd.read_csv(OUT/'commitment_matched_risk.csv')
CALDIAG=pd.read_csv(OUT/'calibration_sensitivity.csv');TRACES=pd.read_csv(OUT/'adaptive_state_traces.csv',parse_dates=['origin'])
IP=pd.read_pickle(OUT/'interval_paths.pkl');FP=pd.read_pickle(OUT/'independent_final_preds.pkl')
FEEDC=pd.read_csv(OUT/'available_feed_controls.csv');PERT=pd.read_csv(OUT/'feeding_perturbations.csv')
labels={'PG-HQ ensemble [past]':'Ensemble · history','PG-HQ ensemble [feed]':'Ensemble · recorded feed','LGB-Δ [past]':'LGB-Δ · history','LGB-Δ [feed]':'LGB-Δ · recorded feed','Cat-Δ [past]':'Cat-Δ · history','Cat-Δ [feed]':'Cat-Δ · recorded feed'}

def style(ax,x='',y='',title=None):
    x=x.replace('m³ h⁻¹','m$^{3}$ h$^{-1}$');y=y.replace('m³ h⁻¹','m$^{3}$ h$^{-1}$')
    numeric_ticks(ax,x=not isinstance(ax.xaxis.get_major_locator(),(FixedLocator,mdates.DateLocator)),y=not isinstance(ax.yaxis.get_major_locator(),FixedLocator),n=5)
    style_axis(ax,x,y,tick_size=9,label_size=10.5)
    for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontweight('normal')
    ax.xaxis.label.set_fontweight('normal');ax.yaxis.label.set_fontweight('normal')
    ax.tick_params(length=4,width=1.1);ax.spines['left'].set_linewidth(1.1);ax.spines['bottom'].set_linewidth(1.1)
    if title:ax.set_title(title,fontsize=10.5,fontweight='normal',pad=10)

def legend(ax,ncol=1):
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=ncol,fontsize=8,handlelength=1.8,columnspacing=.9,borderaxespad=0)

def line(ax,x,y,color,label,marker='o',ls='-'):
    ax.plot(x,y,color=color,lw=1.5,marker=marker,ms=4,mfc='white',mew=1,label=label,ls=ls)

def interval(ax,x,point,low,high,color,label=None):
    ax.errorbar(x,point,yerr=[np.asarray(point)-low,np.asarray(high)-point],fmt='o',color=color,ms=4,capsize=2,lw=1.15,label=label)

def save(stem,panels,height=6.4):
    fig,axes=plt.subplots(2,2,figsize=(7.7,height),layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.1,h_pad=.1,wspace=.12,hspace=.16)
    for i,(ax,func) in enumerate(zip(axes.flat,panels)):
        func(fig,ax);ax.text(-.17,1.12,string.ascii_lowercase[i],transform=ax.transAxes,fontweight='bold',fontsize=13,va='bottom')
    folder=FIG/stem;folder.mkdir(exist_ok=True)
    fig.savefig(folder/(stem+'_preview.png'),dpi=160,bbox_inches='tight')
    save_figure_with_subfigures(fig,folder,stem,dpi=DPI)
    # Render each panel independently, retaining its colourbar and all legends.
    for i,func in enumerate(panels):
        f,ax=plt.subplots(figsize=(4.5,3.6),layout='constrained');func(f,ax)
        ax.text(-.14,1.1,string.ascii_lowercase[i],transform=ax.transAxes,fontweight='bold',fontsize=13)
        for ext in ['png','pdf']:f.savefig(folder/'subfigures'/f'{stem}_{string.ascii_lowercase[i]}.{ext}',dpi=DPI,bbox_inches='tight')
        plt.close(f)
    print('figure',stem,flush=True)

def information_a(fig,ax):
    for g,c in [('Gas storage',STORE),('Co-substrate feed',FEED),('Calendar',C[4])]:
        d=A[(A.regime=='past')&(A.window=='all')&(A.mapping=='complete')&(A.group==g)]
        line(ax,d.h,d['mean'],c,{'Co-substrate feed':'HSW','Gas storage':'Storage','Calendar':'Calendar'}[g])
    style(ax,'Horizon (h)','Permutation ΔRMSE (m³ h⁻¹)');legend(ax,3);ax.set_xticks([1,6,12,18,24])
def information_b(fig,ax):
    for j,(g,c) in enumerate([('Gas storage',STORE),('Co-substrate feed',FEED),('Calendar',C[4])]):
        d=L[(L.group==g)&(L.block_h==72)&L.h.isin([1,6,24])]
        interval(ax,np.arange(3)+(j-1)*.2,d.delta_RMSE_m3_h,d.CI_low,d.CI_high,c,{'Gas storage':'Storage','Co-substrate feed':'HSW','Calendar':'Calendar'}[g])
    ax.axhline(0,color=INK,lw=.8,ls=':');ax.set_xticks(range(3),['1','6','24']);style(ax,'Horizon (h)','Refit ΔRMSE (m³ h⁻¹)');legend(ax,3)
def information_c(fig,ax):
    for name,c in [('LGB-Δ',FEED),('Cat-Δ',C[4]),('PG-HQ ensemble',PAST)]:
        d=P[(P.candidate==name+' [feed]')&(P.reference==name+' [past]')&(P.block_h==72)]
        line(ax,d.h,-d.delta_RMSE_m3_h,c,name.replace('PG-HQ ','').replace('ensemble','Ensemble'))
    style(ax,'Horizon (h)','RMSE reduction (m³ h⁻¹)');ax.axhline(0,color=INK,lw=.8,ls=':');legend(ax,3);ax.set_xticks([1,6,12,18,24])
def information_d(fig,ax):
    for mapping,c,label in [('correction',C[3],'Correction input only'),('complete',PAST,'Complete forecast')]:
        d=A[(A.regime=='past')&(A.window=='all')&(A.mapping==mapping)&(A.group=='Gas history')]
        line(ax,d.h,d['mean'],c,label)
    style(ax,'Horizon (h)','Gas-history ΔRMSE (m³ h⁻¹)');legend(ax);ax.set_xticks([1,6,12,18,24])

def benchmark_a(fig,ax):
    for prefix,c in [('LGB-Δ',PAST),('Cat-Δ',C[4]),('PG-HQ ensemble',FEED)]:
        for reg,ls,mk in [('past','--','o'),('feed','-','s')]:
            d=M[M.model==f'{prefix} [{reg}]'];line(ax,d.h,d.Skill,c,(prefix.replace('PG-HQ ','')+' · '+('history' if reg=='past' else 'feed')),mk,ls)
    style(ax,'Horizon (h)','Persistence-relative skill');ax.set_xticks([1,6,12,18,24]);legend(ax,2)
def benchmark_b(fig,ax):
    for j,(name,c) in enumerate([('LGB-Δ',PAST),('Cat-Δ',C[4]),('PG-HQ ensemble',FEED)]):
        d=P[(P.candidate==name+' [feed]')&(P.reference==name+' [past]')&(P.block_h==72)&P.h.isin([1,6,24])]
        interval(ax,np.arange(3)+(j-1)*.2,d.delta_RMSE_m3_h,d.CI_low,d.CI_high,c,name.replace('PG-HQ ','').replace('ensemble','Ensemble'))
    style(ax,'Horizon (h)','Feed − history RMSE (m³ h⁻¹)');ax.set_xticks(range(3),['1','6','24']);ax.axhline(0,color=INK,lw=.8,ls=':');legend(ax,3)
def benchmark_c(fig,ax):
    d=pd.read_csv(OUT/'ablation_configuration_register.csv');d=d[d.h==24]
    ax.bar(range(5),d.Skill,color=[C[3],C[0],C[2],C[4],C[1]],edgecolor=INK,lw=.4,width=.65)
    ax.set_xticks(range(5),['A1\nRaw\nlevel','A2\nBins\nlevel','A3\nRaw\ndelta','A4\nBins\ndelta','A5\nBins + feed\ndelta'],fontsize=8)
    style(ax,'','Day-ahead skill');ax.set_ylim(0,.5)
def benchmark_d(fig,ax):
    d=pd.read_csv(OUT/'matched_MLP_metrics.csv');d=d[d.h==24]
    for j,(name,c) in enumerate([('Matched LightGBM',PAST),('Matched MLP',FEED)]):
        values=[d[(d.model==name)&(d.regime==reg)&(d.anchor==anchor)].RMSE.iloc[0] for reg,anchor in [('past',False),('past',True),('feed',False),('feed',True)]]
        ax.bar(np.arange(4)+(j-.5)*.34,values,width=.32,color=c,edgecolor=INK,lw=.4,label=name.replace('Matched ',''))
    ax.set_xticks(range(4),['History\nlevel','History\ndelta','Feed\nlevel','Feed\ndelta']);style(ax,'','Day-ahead RMSE (m³ h⁻¹)');legend(ax,2)

def trajectory_a(fig,ax):
    d=FP['pred'][24]['test'];t=d.index+pd.Timedelta(hours=24)
    ax.plot(t,d.y*CFM,color=INK,lw=.65,label='Observed');ax.plot(t,d['PG-HQ ensemble [feed]']*CFM,color=FEED,lw=.65,label='Feed ensemble',alpha=.85)
    style(ax,'Target date','Outlet flow (m³ h⁻¹)');ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'));legend(ax,2)
def trajectory_b(fig,ax):
    d=IP[24,'Feed-conditioned ensemble'];t=TE+pd.Timedelta(hours=24);mask=(t>=pd.Timestamp('2023-01-18'))&(t<pd.Timestamp('2023-01-26'))
    ax.fill_between(t[mask],(d['p']-d['q'])[mask]*CFM,(d['p']+d['q'])[mask]*CFM,color=FEED,alpha=.25,label='Nominal 90% interval')
    ax.plot(t[mask],d['y'][mask]*CFM,color=INK,lw=.8,label='Observed');ax.plot(t[mask],d['p'][mask]*CFM,color=FEED,lw=1)
    style(ax,'Target date in January','Outlet flow (m³ h⁻¹)');ax.xaxis.set_major_locator(mdates.DayLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'));legend(ax)
def trajectory_c(fig,ax):
    d=pd.read_csv(OUT/'intervals_independent.csv')
    for name,c in [('PG-HQ ensemble [past]',PAST),('PG-HQ ensemble [feed]',FEED)]:
        z=d[d.model==name];line(ax,z.h,z.coverage*100,c,'History' if '[past]' in name else 'Feed')
    ax.axhline(90,color=INK,ls=':',lw=.9);style(ax,'Horizon (h)','Empirical coverage (%)');legend(ax,2);ax.set_xticks([1,6,12,18,24]);ax.set_ylim(86,96)
def trajectory_d(fig,ax):
    d=pd.read_csv(OUT/'intervals_independent.csv')
    for name,c in [('PG-HQ ensemble [past]',PAST),('PG-HQ ensemble [feed]',FEED)]:
        z=d[d.model==name];line(ax,z.h,z.mean_width,c,'History' if '[past]' in name else 'Feed')
    style(ax,'Horizon (h)','Mean interval width (m³ h⁻¹)');legend(ax,2);ax.set_xticks([1,6,12,18,24])

def group_heat(fig,ax,reg):
    d=A[(A.regime==reg)&(A.window=='all')&(A.mapping=='complete')].pivot(index='group',columns='h',values='mean')
    selected=[g for g in groups if g in d.index];z=d.reindex(selected)[HS]
    im=ax.imshow(z,aspect='auto',cmap=LinearSegmentedColormap.from_list('importance',['white',PAST]))
    ax.set_xticks(range(9),HS);ax.set_yticks(range(len(selected)),[SHORT[groups.index(g)] for g in selected]);style(ax,'Horizon (h)','',('History only' if reg=='past' else 'Recorded future feeding'))
    cb=fig.colorbar(im,ax=ax,shrink=.83,pad=.03);cb.ax.tick_params(labelsize=8);cb.set_label('ΔRMSE (m$^{3}$ h$^{-1}$)',fontsize=9)
def lag_c(fig,ax):
    z=HSW[HSW.window=='all'].pivot(index='h',columns='lag',values='mean').reindex([1,6,12,24])[BIN_LAB]
    im=ax.imshow(z,aspect='auto',cmap=LinearSegmentedColormap.from_list('lag',['white',FEED]))
    ax.set_xticks(range(8),[x.replace('-','–') for x in BIN_LAB],rotation=45,ha='right');ax.set_yticks(range(4),[1,6,12,24]);style(ax,'HSW lag window (h)','Horizon (h)')
    cb=fig.colorbar(im,ax=ax,shrink=.83,pad=.03);cb.ax.tick_params(labelsize=8);cb.set_label('ΔRMSE (m$^{3}$ h$^{-1}$)',fontsize=9)
def lag_d(fig,ax):
    for j,window in enumerate(['third-1','third-2','third-3']):
        d=HSW[(HSW.h==24)&(HSW.window==window)].set_index('lag').reindex(BIN_LAB)
        line(ax,np.arange(8),d['mean'],[PAST,FEED,STORE][j],f'Test third {j+1}')
    ax.set_xticks(range(8),[x.replace('-','–') for x in BIN_LAB],rotation=45,ha='right');style(ax,'HSW lag window (h)','Day-ahead ΔRMSE (m³ h⁻¹)');legend(ax)

def schedule_a(fig,ax):
    for name,c in [('Weekly feed',C[4]),('Four-week feed median',STORE)]:
        d=FEEDC[(FEEDC.method==name)&(FEEDC.variable=='HSW')];line(ax,d.h,d.window_WAPE*100,c,name)
    style(ax,'Horizon (h)','HSW window WAPE (%)');legend(ax);ax.set_xticks([1,6,12,18,24])
def schedule_b(fig,ax):
    for name,c,label in [('LGB-Δ [past]',PAST,'History only'),('Weekly feed',C[4],'Weekly feed'),('Four-week feed median',STORE,'Four-week median'),('LGB-Δ [feed]',FEED,'Recorded feed')]:
        d=M[M.model==name];line(ax,d.h,d.RMSE,c,label)
    style(ax,'Horizon (h)','LGB-Δ RMSE (m³ h⁻¹)');legend(ax,2);ax.set_xticks([1,6,12,18,24])
def schedule_c(fig,ax):
    d=PERT[(PERT.h==24)&(PERT.scenario=='hourly_noise')].groupby(['magnitude','shift_h']).RMSE.mean().unstack()
    im=ax.imshow(d,aspect='auto',cmap=LinearSegmentedColormap.from_list('noise',['white',FEED]))
    ax.set_xticks(range(len(d.columns)),d.columns.astype(int));ax.set_yticks(range(len(d)),(d.index*100).astype(int));style(ax,'Timing shift (h; + = delayed)','Hourly noise SD (%)')
    cb=fig.colorbar(im,ax=ax,shrink=.83,pad=.03);cb.ax.tick_params(labelsize=8);cb.set_label('24 h RMSE (m$^{3}$ h$^{-1}$)',fontsize=9)
def schedule_d(fig,ax):
    for scenario,c,mk,label in [('hourly_noise',C[3],'o','Hourly noise; no shift'),('sustained_bias',FEED,'s','Sustained bias'),('episode_cancellation',C[4],'^','Episode cancellation'),('episode_delay_6h',STORE,'D','Episode delay')]:
        d=PERT[(PERT.h==24)&(PERT.scenario==scenario)]
        if scenario=='hourly_noise':d=d[d.shift_h==0]
        d=d.groupby('magnitude')[['window_WAPE','RMSE']].mean().sort_values('window_WAPE')
        ax.scatter(d.window_WAPE*100,d.RMSE,color=c,marker=mk,s=32,edgecolor=INK,lw=.4,label=label)
    ax.axhline(M[(M.h==24)&(M.model=='LGB-Δ [past]')].RMSE.iloc[0],color=PAST,ls=':',lw=1)
    style(ax,'HSW window WAPE (%)','24 h RMSE (m³ h⁻¹)');legend(ax)

def commitment_a(fig,ax):
    for name,c in [('Persistence',C[4]),('Past-only ensemble',PAST),('Feed-conditioned ensemble',FEED)]:
        d=F[(F.h==24)&(F.method=='ACI')&(F.strategy==name)].sort_values('shortfall_rate')
        line(ax,d.shortfall_rate*100,d.firm_fraction*100,c,name.replace('-conditioned ensemble','').replace('-only ensemble',''))
    style(ax,'Observed shortfall frequency (%)','Fulfilled / metered flow (%)');legend(ax);ax.set_xlim(0,45)
def commitment_b(fig,ax):
    for name,c in [('Past-only ensemble',PAST),('Feed-conditioned ensemble',FEED)]:
        d=F[(F.h==24)&(F.method=='ACI')&(F.strategy==name)&(F.alpha<=.1)].sort_values('shortfall_rate')
        line(ax,d.shortfall_rate*100,d.firm_fraction*100,c,'History' if name.startswith('Past') else 'Feed')
        for r in d.itertuples():
            if r.alpha<=.02:
                offset=(-17,8) if name.startswith('Past') else (5,6) if r.alpha==.01 else (5,-12)
            else:offset=(3,4) if not name.startswith('Past') else (3,-12)
            ax.annotate(f'{int(r.alpha*100)}%',(r.shortfall_rate*100,r.firm_fraction*100),xytext=offset,textcoords='offset points',fontsize=8)
    style(ax,'Observed shortfall frequency (%)','Fulfilled / metered flow (%)');legend(ax,2);ax.set_xlim(0,13)
def commitment_c(fig,ax):
    d=CP[(CP.h==24)&(CP.block_h==72)]
    interval(ax,np.arange(len(d)),d.delta_fulfilled_fraction*100,d.CI_low*100,d.CI_high*100,FEED)
    ax.set_xticks(range(len(d)),(d.alpha*100).round().astype(int));ax.axhline(0,color=INK,ls=':',lw=.8)
    style(ax,'Nominal shortfall target (%)','Feed − history fulfilled share (pp)')
def commitment_d(fig,ax):
    d=CR[(CR.h==24)&(CR.block_h==72)]
    interval(ax,np.arange(len(d)),d.delta_fulfilled_fraction*100,d.CI_low*100,d.CI_high*100,FEED)
    ax.set_xticks(range(len(d)),(d.risk*100).round().astype(int));ax.axhline(0,color=INK,ls=':',lw=.8)
    style(ax,'Matched observed shortfall (%)','Feed − history fulfilled share (pp)')

def state_alpha(fig,ax,alpha):
    for strategy,c in [('Past-only ensemble',PAST),('Feed-conditioned ensemble',FEED)]:
        d=TRACES[(TRACES.h==24)&(TRACES.alpha==alpha)&(TRACES.boundary=='finite')&(TRACES.strategy==strategy)]
        ax.plot(d.origin,d.a,color=c,lw=.85,label='History' if strategy.startswith('Past') else 'Feed')
    ax.axhline(0,color=INK,ls=':',lw=.8);style(ax,'Origin date','Adaptive level a')
    ax.text(.97,.06,f'Nominal shortfall {int(alpha*100)}%',transform=ax.transAxes,ha='right',va='bottom',fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'));legend(ax,2)
def boundary_c(fig,ax):
    d=CALDIAG[(CALDIAG.h==24)&(CALDIAG.alpha==.01)&(CALDIAG.gamma==.005)&(CALDIAG.history=='expanding')&(CALDIAG.timing=='after')&(CALDIAG.calibration=='independent')&CALDIAG.strategy.isin(['Past-only ensemble','Feed-conditioned ensemble'])]
    for j,(boundary,c) in enumerate([('finite',STORE),('infinite',C[4])]):
        z=d[d.boundary==boundary].set_index('strategy').reindex(['Past-only ensemble','Feed-conditioned ensemble'])
        ax.bar(np.arange(2)+(j-.5)*.32,z.firm_fraction*100,width=.3,color=c,edgecolor=INK,lw=.4,label=boundary.capitalize()+' endpoints')
    ax.set_xticks([0,1],['History','Feed']);style(ax,'','Fulfilled / metered flow (%)');legend(ax)
def sensitivity_d(fig,ax):
    d=CALDIAG[(CALDIAG.h==24)&(CALDIAG.alpha==.01)&(CALDIAG.boundary=='finite')&(CALDIAG.timing=='after')&(CALDIAG.calibration=='independent')&CALDIAG.strategy.isin(['Past-only ensemble','Feed-conditioned ensemble'])]
    for j,history in enumerate(['168','360','720','expanding']):
        z=d[d.history==history].pivot(index='gamma',columns='strategy',values='firm_fraction')
        line(ax,z.index,(z['Feed-conditioned ensemble']-z['Past-only ensemble'])*100,[PAST,FEED,STORE,C[4]][j],history+' h' if history!='expanding' else 'Expanding')
    ax.axhline(0,color=INK,ls=':',lw=.8);style(ax,'Update step γ','Feed − history fulfilled share (pp)');ax.set_xticks([.001,.005,.01]);legend(ax,2)

def diagnostics_a(fig,ax):
    d=pd.read_csv(OUT/'loss_difference_acf.csv');d=d[d.baseline=='PG-HQ ensemble [past]']
    for h,c in [(1,PAST),(6,STORE),(24,FEED)]:
        z=d[d.h==h];ax.plot(z.lag,z.autocorrelation,color=c,lw=1.4,label=f'{h} h')
    style(ax,'Loss-difference lag (h)','Autocorrelation');legend(ax,3);ax.axhline(0,color=INK,ls=':',lw=.8);ax.set_xticks([0,24,72,120,168])
def diagnostics_b(fig,ax):
    d=pd.read_csv(OUT/'lag_bin_sensitivity.csv')
    for h,c in [(12,STORE),(24,FEED)]:
        z=d[d.h==h];line(ax,np.arange(3),z['mean'],c,f'{h} h')
    ax.set_xticks([0,1,2],['40–63','48–71','56–79']);style(ax,'HSW lag window (h)','Permutation ΔRMSE (m³ h⁻¹)');legend(ax,2)
def diagnostics_c(fig,ax):
    d=P[(P.h==24)&(P.candidate=='PG-HQ ensemble [feed]')&(P.reference=='PG-HQ ensemble [past]')]
    interval(ax,np.arange(3),d.delta_RMSE_m3_h,d.CI_low,d.CI_high,FEED);ax.set_xticks([0,1,2],d.block_h);style(ax,'Bootstrap block length (h)','Feed − history RMSE (m³ h⁻¹)');ax.axhline(0,color=INK,ls=':',lw=.8)
def diagnostics_d(fig,ax):
    d=L[(L.h==24)&(L.group=='Gas storage')]
    interval(ax,np.arange(3),d.delta_RMSE_m3_h,d.CI_low,d.CI_high,STORE);ax.set_xticks([0,1,2],d.block_h);style(ax,'Bootstrap block length (h)','Storage-removal ΔRMSE (m³ h⁻¹)');ax.axhline(0,color=INK,ls=':',lw=.8)

if __name__=='__main__':
    selections=sys.argv[1:]
    builds={
      'Fig4_information_value':[information_a,information_b,information_c,information_d],
      'Fig5_benchmark':[benchmark_a,benchmark_b,benchmark_c,benchmark_d],
      'Fig6_trajectories_uncertainty':[trajectory_a,trajectory_b,trajectory_c,trajectory_d],
      'Fig7_horizon_resolved_attribution':[lambda f,a:group_heat(f,a,'past'),lambda f,a:group_heat(f,a,'feed'),lag_c,lag_d],
      'Fig9_feed_schedule_value':[schedule_a,schedule_b,schedule_c,schedule_d],
      'Fig10_firm_supply':[commitment_a,commitment_b,commitment_c,commitment_d],
      'FigS6_uncertainty_and_state_errors':[lambda f,a:state_alpha(f,a,.01),lambda f,a:state_alpha(f,a,.05),boundary_c,sensitivity_d],
      'FigS10_dependence_and_stability':[diagnostics_a,diagnostics_b,diagnostics_c,diagnostics_d]}
    for stem,panels in builds.items():
        if not selections or stem in selections:save(stem,panels)
