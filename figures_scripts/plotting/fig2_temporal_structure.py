from figcommon import *
from scipy.stats import wasserstein_distance, spearmanr, gaussian_kde
from statsmodels.tsa.stattools import acf, pacf
from common import H, idx, o_tr, o_va, o_te, GROUPS, SPLIT_VAL, SPLIT_TEST
h=H.copy()
y=h['Biogas']*CFM; hsw=(h['Q-HSW1_GPM']+h['Q-HSW2_GPM'])*GPM
SHORT={'Biogas':'Biogas','H-Dig1_FT':'Cover D1','H-Dig2_FT':'Cover D2','Q-HSW1_GPM':'HSW→D1','Q-HSW2_GPM':'HSW→D2','H-HSW_ft':'HSW tank',
       'V-HSW-DIG2-yest_gal':'HSW D2 (d−1)','Q-PS_on':'PS pump on','Q-TWAS_GPM':'TWAS','D1_TEMPERATURE':'T D1','D2_TEMPERATURE':'T D2',
       'Q-influent_MGD':'Influent','Q-Aeration_SCFM':'Aeration','Q-RAS_MGD':'RAS','Q-WAS1_MGD':'WAS1','Q-WAS2_MGD':'WAS2'}
VARS=[v for vs in GROUPS.values() for v in vs]; VG={v:g for g,vs in GROUPS.items() for v in vs}
t0,t1=h.index.min(),h.index.max()
def shade(ax):
    for (a,b),nm in [((t0,pd.Timestamp(SPLIT_VAL)),'Train'),((pd.Timestamp(SPLIT_VAL),pd.Timestamp(SPLIT_TEST)),'Validation'),((pd.Timestamp(SPLIT_TEST),t1),'Test')]:
        ax.axvspan(a,b,color=SPLIT_COL[nm],alpha=0.28,lw=0)
def pa(fig,ss):
    g=ss.subgridspec(2,2,height_ratios=[4,0.55],width_ratios=[60,1],hspace=0.08,wspace=0.02)
    ax=fig.add_subplot(g[0,0]); shade(ax)
    ax.plot(y.index,y.values,color=OBS,lw=0.45,alpha=0.55)
    ax.plot(y.index,y.rolling(168,center=True,min_periods=84).mean(),color=OBS,lw=2.6)
    ymax=y.max()*1.12; ax.set_ylim(0,ymax); ax.set_xlim(t0,t1)
    for nm,x in [('Train',t0+(pd.Timestamp(SPLIT_VAL)-t0)/2),('Validation',pd.Timestamp(SPLIT_VAL)+(pd.Timestamp(SPLIT_TEST)-pd.Timestamp(SPLIT_VAL))/2),('Test',pd.Timestamp(SPLIT_TEST)+(t1-pd.Timestamp(SPLIT_TEST))/2)]:
        ax.text(x,ymax*0.97,nm,ha='center',va='top',fontsize=12.5,color=INK)
    style(ax,ylabel=BIOGAS_LAB,ny=5); ax.tick_params(labelbottom=False)
    ax2=fig.add_subplot(g[1,0],sharex=ax)
    d=hsw.resample('D').mean()
    # Flat vector cells retain the daily strip's original extent in PDF.
    edges=np.linspace(matplotlib.dates.date2num(d.index[0]),matplotlib.dates.date2num(d.index[-1]+pd.Timedelta(days=1)),len(d)+1)
    im=ax2.pcolormesh(edges,[0,1],d.values[None,:],cmap=cmap_seq(P['orange']),shading='flat',edgecolors='none',linewidth=0,antialiased=False,rasterized=False)
    ax2.set_yticks([0.5]); ax2.set_yticklabels(['HSW'])
    style(ax2,xlabel='Date'); ax2.xaxis_date(); ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y')); ax2.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[3,6,9,12]))
    bold_ticks(ax2)
    cax=fig.add_subplot(g[:,1]); cbar(fig,im,cax,'HSW (m$^{3}$ h$^{-1}$)')
def pb(fig,ss):
    ax=fig.add_subplot(ss); yy=y.interpolate()
    a=acf(yy,nlags=168,fft=True); p=pacf(yy,nlags=48)
    ax.fill_between(range(169),0,a,color=OBS,alpha=0.25,lw=0); ax.plot(range(169),a,color=OBS,lw=2.2,label='ACF')
    ax.vlines(range(1,49),0,p[1:],color=P['sky'],lw=2.0,label='PACF')
    ci=1.96/np.sqrt(len(yy)); ax.axhspan(-ci,ci,color=P['lavender'],alpha=0.6,lw=0)
    for L,yt in [(1,0.93),(6,0.81),(24,0.69)]:
        ax.plot(L,a[L],'o',ms=8,mfc='white',mec=PROP,mew=2.2,zorder=5)
        ax.text(0.10,yt,f'{L} h: {a[L]:.2f}',transform=ax.transAxes,fontsize=11,color=INK)
    ax.axhline(0,color=INK,lw=1); ax.set_xlim(0,168); ax.set_ylim(-0.25,1.08); ax.set_xticks([0,48,96,144])
    style(ax,'Lag (h)','Autocorrelation',ny=5); ax.legend(loc='upper right',fontsize=11,handlelength=1.2)
def pc(fig,ss):
    # Keep the left 23 h tick distinct from the right 0 h tick in both exports.
    g=ss.subgridspec(2,2,height_ratios=[18,1],hspace=1.25,wspace=.35)
    days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for j,(s,c,lab) in enumerate([(y,P['blue'],BIOGAS_LAB),(hsw,P['orange'],'HSW (m$^{3}$ h$^{-1}$)')]):
        M=s.groupby([s.index.dayofweek,s.index.hour]).mean().unstack().values
        ax=fig.add_subplot(g[0,j]); im=ax.pcolormesh(np.arange(M.shape[1]+1)-0.5,np.arange(M.shape[0]+1)-0.5,M,cmap=cmap_seq(c),shading='flat',edgecolors='none',linewidth=0,antialiased=False,rasterized=False)
        ax.set_ylim(M.shape[0]-0.5,-0.5,auto=True)
        ax.set_xticks([0,6,12,18,23]); ax.set_yticks(range(7)); ax.set_yticklabels(days if j==0 else [])
        style(ax,'Hour of day',None); [sp.set_visible(True) for sp in ax.spines.values()]
        cax=fig.add_subplot(g[1,j]); cb=cbar(fig,im,cax,lab,orientation='horizontal',size=11)
        cb.locator=MaxNLocator(3); cb.update_ticks(); bold_ticks(cb.ax,size=10)
def pd_(fig,ss):
    ax=fig.add_subplot(ss); lags=np.arange(-24,97)
    def dt(s): s=s.interpolate(); return s-s.rolling(168,center=True,min_periods=84).mean()
    yb=dt(y)
    for s,c,lab in [(dt(hsw),P['orange'],'HSW feed'),(dt(h['H-Dig1_FT']+h['H-Dig2_FT']),P['teal'],'Cover height')]:
        # Line pattern distinguishes HSW from cover height without colour alone.
        r=[s.corr(yb.shift(-k)) for k in lags]; ax.plot(lags,r,color=c,lw=2.6,ls='-' if c==P['orange'] else '--',label=lab)
        k=lags[np.nanargmax(r)]; ax.plot(k,np.nanmax(r),'o',ms=8,mfc='white',mec=c,mew=2.2); ax.annotate(f'{k} h',(k,np.nanmax(r)),(k+7,np.nanmax(r)+0.03) if c==P['orange'] else (k-24,np.nanmax(r)+0.03),fontsize=12,color=INK)
    ax.axhline(0,color=INK,lw=1); ax.axvline(0,color=INK,lw=1,ls=':')
    ax.set_xticks([-24,0,24,48,72,96]); style(ax,'Lag of biogas (h)','Cross-correlation',ny=5); ax.legend(loc='upper right',fontsize=10.5,handlelength=1.2,borderaxespad=0.3)
def pe(fig,ss):
    g=ss.subgridspec(1,2,width_ratios=[2.1,1],wspace=0.55)
    ax=fig.add_subplot(g[0]); tr=h.loc[h.index<SPLIT_VAL]; te=h.loc[h.index>=SPLIT_TEST]
    w={v:wasserstein_distance((tr[v].dropna()-tr[v].mean())/tr[v].std(),(te[v].dropna()-tr[v].mean())/tr[v].std()) for v in VARS}
    order=sorted(VARS,key=lambda v:w[v])
    ax.barh(range(len(order)),[w[v] for v in order],color=[GROUP_COL[VG[v]] for v in order],edgecolor=INK,lw=0.8,height=0.72)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[v] for v in order])
    style(ax,'Wasserstein distance',None,nx=4); [l.set_fontsize(11.5) for l in ax.get_yticklabels()]
    ax2=fig.add_subplot(g[1]); xs=np.linspace(0,y.max(),300)
    for s,c,lab,ls in [(y[y.index<SPLIT_VAL],SPLIT_COL['Train'],'Train','-'),(y[y.index>=SPLIT_TEST],SPLIT_COL['Test'],'Test','--')]:
        k=gaussian_kde(s.dropna()); ax2.fill_between(xs,1e3*k(xs),color=c,alpha=0.45,lw=0); ax2.plot(xs,1e3*k(xs),color=c,lw=2.4,ls=ls,label=lab)
    style(ax2,BIOGAS_LAB,'Density (×10$^{-3}$)',nx=3,ny=4)
    ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.02),fontsize=11.5,handlelength=1.2,ncol=2,columnspacing=0.7)
def pf(fig,ss):
    g=ss.subgridspec(1,2,width_ratios=[30,1],wspace=0.04)
    ax=fig.add_subplot(g[0]); D=h[VARS].dropna(); n=len(VARS)
    R=np.zeros((n,n)); Pv=np.zeros((n,n))
    for i in range(n):
        for j in range(n): r=spearmanr(D[VARS[i]],D[VARS[j]]); R[i,j]=r.correlation; Pv[i,j]=r.pvalue
    M=np.ma.array(R,mask=np.triu(np.ones_like(R,bool),1))
    im=ax.pcolormesh(np.arange(n+1)-0.5,np.arange(n+1)-0.5,M,cmap=cmap_div(),vmin=-1,vmax=1,shading='flat',edgecolors='none',linewidth=0,antialiased=False,rasterized=False)
    ax.set_ylim(n-0.5,-0.5,auto=True)
    for i in range(n):
        for j in range(i+1):
            if i!=j and abs(R[i,j])>=0.3: ax.text(j,i,f'{R[i,j]:.1f}'.replace('0.','.'),ha='center',va='center',fontsize=10,color=INK)
    corr_labels={'H-Dig1_FT':'Cover D1','H-Dig2_FT':'Cover D2','H-HSW_ft':'HSW tank',
                 'V-HSW-DIG2-yest_gal':'HSW d−1','Q-PS_on':'PS on','D1_TEMPERATURE':'T D1','D2_TEMPERATURE':'T D2'}
    labels=[corr_labels.get(v,SHORT[v]) for v in VARS]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels,rotation=90); ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    style(ax); ts=10.5 if getattr(fig,'_is_standalone',False) else 10.5
    [l.set_fontsize(ts) for l in ax.get_xticklabels()+ax.get_yticklabels()]
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    # group brackets: coloured squares along diagonal
    k=0
    for gname,vs in GROUPS.items():
        ax.add_patch(matplotlib.patches.Rectangle((k-0.5,k-0.5),len(vs),len(vs),fill=False,ec=GROUP_COL[gname],lw=2.4)); k+=len(vs)
    for lab,v in zip(ax.get_yticklabels(),VARS): lab.set_color(INK)
    cbar(fig,im,fig.add_subplot(g[1]),'Spearman ρ')
def layout(fig):
    G=GridSpec(3,1,figure=fig,height_ratios=[1.05,1,1.25],hspace=0.82)
    r2=G[1].subgridspec(1,3,width_ratios=[1,1.35,1],wspace=0.60); r3=G[2].subgridspec(1,2,width_ratios=[1.25,1],wspace=0.45)
    return {'a':G[0],'b':r2[0],'c':r2[1],'d':r2[2],'e':r3[0],'f':r3[1]}
panels=[('a',pa,(14,4.4)),('b',pb,(6.2,4.8)),('c',pc,(8,4.8)),('d',pd_,(6.2,4.8)),('e',pe,(9,6)),('f',pf,(8,7.4))]
if __name__=='__main__': render('Fig3_temporal_structure','Fig3_temporal_structure',panels,(18,19),layout)
