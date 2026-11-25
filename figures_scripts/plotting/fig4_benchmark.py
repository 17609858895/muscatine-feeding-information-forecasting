from figcommon import *
def rankdata(x): return pd.Series(x).rank(method='average').to_numpy()
TUN=os.path.exists('extra/metrics_all_horizons_tuned.csv')   # baselines Optuna-tuned, deep models 5-seed means
M=pd.read_csv('extra/metrics_all_horizons_tuned.csv' if TUN else 'extra/metrics_all_horizons.csv'); HS=[1,2,3,4,6,9,12,18,24]
ROLL='extra/rolling_all_horizons_tuned.csv' if TUN else 'extra/rolling_all_horizons.csv'; DMF=os.environ['REVIEW_DM_FILE']
FAM={'Naive':['Persistence','Seasonal naive (24 h)'],'Linear':['Ridge','AR-Ridge (gas only)'],'Kernel / kNN':['SVR','KNN'],
     'Tree ensembles':['Random forest','Extra trees','HistGB','LightGBM','XGBoost','CatBoost'],
     'Deep sequence':['MLP','LSTM','GRU','TCN','Transformer','PatchTST','DLinear','N-BEATS','iTransformer'],
     'Proposed (past-only)':['LGB-Δ [past]','Cat-Δ [past]','PG-HQ [past]','PG-HQ ensemble [past]'],
     'Proposed (feed-conditioned)':['LGB-Δ [feed]','Cat-Δ [feed]','PG-HQ [feed]','PG-HQ ensemble [feed]']}
MK={'Naive':'s','Linear':'D','Kernel / kNN':'v','Tree ensembles':'^','Deep sequence':'P'}
NAME=lambda m: m.replace(' [past]','').replace(' [feed]','').replace('PG-HQ ensemble','PG-HQ ens.').replace('Seasonal naive (24 h)','Seasonal naive').replace('AR-Ridge (gas only)','AR-Ridge')
def piv(v='Skill'): return M.pivot_table(index='model',columns='h',values=v)
def pa(fig,ss):
    g=ss.subgridspec(1,3,width_ratios=[30,6.2,1],wspace=0.02); ax=fig.add_subplot(g[0]); S=piv()
    rows=[]; seps=[]; labs=[]
    for f,ms in FAM.items():
        for m in ms: rows.append(m); labs.append(NAME(m))
        seps.append(len(rows))
    A=S.loc[rows,HS].values; im=ax.pcolormesh(np.arange(len(HS)+1)-0.5,np.arange(len(rows)+1)-0.5,np.clip(A,-0.45,0.45),cmap=cmap_div(),vmin=-0.45,vmax=0.45,shading='flat',edgecolors='none',linewidth=0,antialiased=False,rasterized=False)
    ax.set_ylim(len(rows)-0.5,-0.5,auto=True)
    for i in range(len(rows)):
        for j in range(len(HS)):
            if getattr(fig,'_is_standalone',False) or HS[j] in [1,6,24]:
                ax.text(j,i,f'{A[i,j]:.2f}'.replace('0.','.'),ha='center',va='center',fontsize=10.3,color=INK)
    for s_ in seps[:-1]: ax.axhline(s_-0.5,color='white',lw=3)
    for i,m in enumerate(rows):
        if 'ensemble' in m: ax.add_patch(matplotlib.patches.Rectangle((-0.5,i-0.5),len(HS),1,fill=False,ec=PROP,lw=2.2))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labs); ax.set_xticks(range(len(HS))); ax.set_xticklabels(HS)
    style(ax,'Horizon (h)',None); [l.set_fontsize(11 if getattr(fig,'_is_standalone',False) else 10.5) for l in ax.get_yticklabels()]
    SH={'Naive':'Naive','Linear':'Linear','Kernel / kNN':'Kernel','Tree ensembles':'Trees','Deep sequence':'Deep','Proposed (past-only)':'Past','Proposed (feed-conditioned)':'Feed'}
    for k,(f,ms) in enumerate(FAM.items()):
        y0=(seps[k-1] if k else 0); ax.plot([len(HS)-0.3]*2,[y0-0.35,seps[k]-0.65],color=PROP if 'Proposed' in f else INK,lw=2,clip_on=False)
        ax.text(len(HS)-0.02,(y0+seps[k]-1)/2,SH[f],va='center',ha='left',fontsize=10,color=INK,clip_on=False)
    for sp in ax.spines.values(): sp.set_visible(True)
    cb=cbar(fig,im,fig.add_subplot(g[2]),'Skill vs persistence',size=11)
    g.figure=fig
def pb(fig,ss):
    ax=fig.add_subplot(ss); S=piv(); R=pd.read_csv(ROLL)
    r=R[R.model=='LGB-Δ [feed]'].groupby('h').Skill.agg(['min','max'])
    ax.fill_between(r.index,r['min'],r['max'],color=P['pink'],alpha=0.55,lw=0,label='LGB-Δ feed, fold range')
    for reg,lw in [('past',2.4),('feed',3.6)]:
        ax.plot(HS,S.loc[f'PG-HQ ensemble [{reg}]',HS],'-o',color=PROP,lw=lw,ms=6 if reg=='past' else 8,mfc='white' if reg=='past' else PROP,mew=2,label=f'PG-HQ {reg}',zorder=4)
    for f in ['Linear','Tree ensembles','Deep sequence']:
        best=S.loc[FAM[f],HS].mean(axis=1).idxmax()
        ax.plot(HS,S.loc[best,HS],'-'+MK[f],color=BASE,lw=1.8,ms=7,mfc='white',mew=1.8,label=NAME(best))
    ax.axhline(0,color=INK,lw=1.2,ls='--'); ax.set_xticks([1,6,12,18,24]); ax.set_ylim(-0.12,0.66)
    style(ax,'Horizon (h)','Skill vs persistence',ny=5); ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),fontsize=11.5,ncol=2,handlelength=1.8,columnspacing=0.9,borderaxespad=0)
def pc(fig,ss):
    ax=fig.add_subplot(ss); R=piv('RMSE')
    sel=['Persistence','AR-Ridge (gas only)','LightGBM','XGBoost','CatBoost','TCN','PatchTST','N-BEATS','LGB-Δ [past]','Cat-Δ [past]','PG-HQ ensemble [past]','LGB-Δ [feed]','Cat-Δ [feed]','PG-HQ ensemble [feed]']
    X=R.loc[sel,HS].values; rk=np.apply_along_axis(rankdata,0,X).mean(1); k=len(sel); N=len(HS)
    qa=3.354  # Studentized range / sqrt(2), alpha=0.05, k=14 (Demsar 2006)
    cd=qa*np.sqrt(k*(k+1)/(6*N)); order=np.argsort(rk)
    for i,j in enumerate(order):
        c=PROP if ('Δ' in sel[j] or 'PG-HQ' in sel[j]) else BASE
        ax.plot(rk[j],i,'o',ms=9,color=c,mfc=c if '[feed]' in sel[j] else 'white',mew=2.2)
        ax.text(rk[j]+0.45,i,NAME(sel[j])+(' (feed)' if '[feed]' in sel[j] else (' (past)' if '[past]' in sel[j] else '')),va='center',fontsize=10.5,color=INK)
    ax.plot([1,1+cd],[k+0.2,k+0.2],color=INK,lw=3); ax.text(1+cd/2,k-0.35,f'CD = {cd:.2f} (Nemenyi, α = 0.05)',ha='center',fontsize=11,color=INK)
    ax.set_ylim(k+1.2,-0.8); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(0.5,k+3.2)
    style(ax,'Mean RMSE rank',None,nx=6); ax.spines['left'].set_visible(False); ax.tick_params(left=False)
def pd_(fig,ss):
    g=ss.subgridspec(1,2,width_ratios=[25,1],wspace=0.04); ax=fig.add_subplot(g[0]); D=pd.read_csv(DMF)
    bl=['Persistence','AR-Ridge (gas only)','LightGBM','XGBoost','CatBoost','LSTM','TCN','PatchTST','N-BEATS','LightGBM +feed']
    cols=[(p,h) for p in ['PG-HQ ensemble [past]','PG-HQ ensemble [feed]'] for h in [1,6,24]]
    A=np.array([[D[(D.baseline==b)&(D.proposed==p)&(D.h==h)].DM.iloc[0] for p,h in cols] for b in bl]); Pv=np.array([[D[(D.baseline==b)&(D.proposed==p)&(D.h==h)].p_holm.iloc[0] for p,h in cols] for b in bl])
    cm=LinearSegmentedColormap.from_list('dm',[P['blue'],'#FFFFFF',P['red']])
    im=ax.pcolormesh(np.arange(len(cols)+1)-0.5,np.arange(len(bl)+1)-0.5,np.clip(A,-8,8),cmap=cm,vmin=-8,vmax=8,shading='flat',edgecolors='none',linewidth=0,antialiased=False,rasterized=False)
    ax.set_ylim(len(bl)-0.5,-0.5,auto=True)
    for i in range(len(bl)):
        for j in range(len(cols)):
            st='***' if Pv[i,j]<0.001 else '**' if Pv[i,j]<0.01 else '*' if Pv[i,j]<0.05 else 'ns'
            ax.text(j,i-0.17,f'{A[i,j]:.1f}',ha='center',va='center',fontsize=10.2,color=INK)
            ax.text(j,i+0.28,st,ha='center',va='center',fontsize=7.5,color=INK)
    ax.axvline(2.5,color='white',lw=4); ax.set_yticks(range(len(bl))); ax.set_yticklabels([NAME(b) for b in bl])
    ax.set_xticks(range(6)); ax.set_xticklabels(['1','6','24']*2); style(ax,'Horizon (h)',None); [l.set_fontsize(10.5) for l in ax.get_yticklabels()]
    ax.text(1,-0.85,'Past',ha='center',fontsize=11.5,color=INK); ax.text(4,-0.85,'Feed',ha='center',fontsize=11.5,color=INK)
    for sp in ax.spines.values(): sp.set_visible(True)
    cbar(fig,im,fig.add_subplot(g[1]),'DM statistic',size=11)
def pe(fig,ss):
    ax=fig.add_subplot(ss); R=pd.read_csv(ROLL)
    mods=[('Persistence',P['lavender']),('LightGBM',BASE),('LGB-Δ [past]',P['rose']),('LGB-Δ [feed]',PROP)]
    for i,h in enumerate([1,6,24]):
        for j,(m,c) in enumerate(mods):
            v=R[(R.model==m)&(R.h==h)].R2.values; x=i*5+j
            ax.boxplot(v,positions=[x],widths=0.7,patch_artist=True,boxprops=dict(fc=c,ec=INK,lw=1.2),medianprops=dict(color=INK,lw=2),whiskerprops=dict(color=INK,lw=1.2),capprops=dict(color=INK,lw=1.2),showfliers=False)
            ax.scatter(np.full(len(v),x)+np.linspace(-0.18,0.18,len(v)),v,s=18,c=c,edgecolors=INK,lw=0.9,zorder=3)
    ax.set_xticks([1.5,6.5,11.5]); ax.set_xticklabels(['1 h','6 h','24 h']); ax.axhline(0,color=INK,lw=1,ls=':')
    style(ax,'Horizon','R$^{2}$ (5 rolling folds)',ny=6)
    ax.legend(handles=[matplotlib.patches.Patch(fc=c,ec=INK,label=NAME(m)+(' (feed)' if 'feed' in m else ' (past)' if 'past' in m else '')) for m,c in mods],loc='lower left',bbox_to_anchor=(0,1.02),fontsize=10.5,ncol=2,borderaxespad=0)
def pf(fig,ss):
    ax=fig.add_subplot(ss); S=piv(); RL=pd.read_csv('extra/ablation_rawlevel.csv').set_index('h'); pers=piv('RMSE').loc['Persistence']
    S0=pd.read_csv('extra/metrics_all_horizons.csv').pivot_table(index='model',columns='h',values='Skill')   # ablation keeps the pre-specified LightGBM so every step shares one configuration
    steps=['Raw lags','+ PG-LB','+ Δ anchor','+ ensemble','+ feed plan']
    for h,c,mk in [(6,P['teal'],'s'),(24,PROP,'o')]:
        v=[1-RL.loc[h,'RMSE']/pers[h],S0.loc['LightGBM',h],S.loc['LGB-Δ [past]',h],S.loc['PG-HQ ensemble [past]',h],S.loc['PG-HQ ensemble [feed]',h]]
        ax.plot(range(5),v,'-'+mk,color=c,lw=2.6,ms=9,mfc='white',mew=2.2,label=f'{h} h',drawstyle='default')
        for i,vi in enumerate(v): ax.text(i,vi+0.022,f'{vi:.2f}',ha='center',fontsize=10.5,color=INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(steps,rotation=20,ha='right'); ax.set_ylim(-0.05,0.5)
    style(ax,None,'Skill vs persistence',ny=5); ax.legend(loc='upper left',fontsize=11.5)
def layout(fig):
    G=GridSpec(3,2,figure=fig,width_ratios=[1.05,1],height_ratios=[2.05,1.2,1],hspace=0.60,wspace=0.76)
    return {'a':G[0,0],'b':G[0,1],'c':G[1,0],'d':G[1,1],'e':G[2,0],'f':G[2,1]}
panels=[('a',pa,(8.5,9.5)),('b',pb,(8,6)),('c',pc,(8,5.5)),('d',pd_,(8,5.5)),('e',pe,(8,5)),('f',pf,(8,5))]
if __name__=='__main__': render('Fig5_benchmark','Fig5_benchmark',panels,(18,23),layout)
