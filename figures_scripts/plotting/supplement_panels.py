from pathlib import Path
"""Scientific panel builders backed by the stored study results."""
import sys, pickle
from figcommon import *
from common import H, idx, o_tr, o_va, o_te, GROUPS, SPLIT_VAL, SPLIT_TEST, build, calendar, target, metrics
from scipy.stats import spearmanr
VARS=[v for vs in GROUPS.values() for v in vs]; VG={v:g for g,vs in GROUPS.items() for v in vs}
CONV={'Biogas':(CFM,'Biogas (m$^{3}$ h$^{-1}$)'),'H-Dig1_FT':(0.3048,'Cover D1 (m)'),'H-Dig2_FT':(0.3048,'Cover D2 (m)'),'Q-HSW1_GPM':(GPM,'HSW→D1 (m$^{3}$ h$^{-1}$)'),'Q-HSW2_GPM':(GPM,'HSW→D2 (m$^{3}$ h$^{-1}$)'),
      'H-HSW_ft':(0.3048,'HSW tank (m)'),'V-HSW-DIG2-yest_gal':(0.003785,'HSW D2, d−1 (m$^{3}$)'),'Q-PS_on':(100,'PS pump on (%)'),'Q-TWAS_GPM':(GPM,'TWAS (m$^{3}$ h$^{-1}$)'),
      'D1_TEMPERATURE':(None,'T D1 (°C)'),'D2_TEMPERATURE':(None,'T D2 (°C)'),'Q-influent_MGD':(3785.41/1000,'Influent (10$^{3}$ m$^{3}$ d$^{-1}$)'),'Q-Aeration_SCFM':(CFM/1000,'Aeration (10$^{3}$ m$^{3}$ h$^{-1}$)'),
      'Q-RAS_MGD':(3785.41/1000,'RAS (10$^{3}$ m$^{3}$ d$^{-1}$)'),'Q-WAS1_MGD':(3785.41,'WAS1 (m$^{3}$ d$^{-1}$)'),'Q-WAS2_MGD':(3785.41,'WAS2 (m$^{3}$ d$^{-1}$)')}
def conv(v,s): f,_=CONV[v]; return (s-32)*5/9 if f is None else s*f
def shade(ax,t0=None,t1=None):
    t0=t0 or H.index.min(); t1=t1 or H.index.max()
    for (a,b),nm in [((t0,pd.Timestamp(SPLIT_VAL)),'Train'),((pd.Timestamp(SPLIT_VAL),pd.Timestamp(SPLIT_TEST)),'Validation'),((pd.Timestamp(SPLIT_TEST),t1),'Test')]: ax.axvspan(a,b,color=SPLIT_COL[nm],alpha=0.28,lw=0)
def datefmt(ax,months=(3,6,9,12),fmt='%b %y',size=TICK):
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=list(months))); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(fmt)); bold_ticks(ax,size)
HS=[1,2,3,4,6,9,12,18,24]
def standalone(fig):
    """Shared export sets this flag for independently drawn panels."""
    return bool(getattr(fig, '_is_standalone', False))
def outside_legend(ax,ncol=2,fontsize=11,**kwargs):
    return ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.02),ncol=ncol,
                     fontsize=fontsize,handlelength=1.8,columnspacing=1.1,**kwargs)
def shared_legend(fig,ax,ncol=3,fontsize=11):
    """A figure key is separate from the calibrated panel footprints."""
    h,l=ax.get_legend_handles_labels()
    return fig.legend(h,l,loc='upper center',bbox_to_anchor=(0.5,-0.06),
                      ncol=ncol,fontsize=fontsize,handlelength=1.8,columnspacing=1.4)
# ---------------- S1 data QC ----------------
_MIN=None
def minute():
    global _MIN
    if _MIN is None:
        p=os.environ.get('SCADA_CSV',str(Path(__file__).resolve().parents[2]/'data/SCADA-raw.csv'))
        _MIN=pd.read_csv(p,parse_dates=['Timestamp'],usecols=['Timestamp','Q-PS_MGD','V-burner_FT3','V-Boiler_FT3','Biogas','Biogas_burner']).set_index('Timestamp').sort_index()
    return _MIN
def s1a(fig,ss):
    ax=fig.add_subplot(ss); m=minute(); c=m['Biogas'].resample('D').count(); c=c.iloc[1:-1]
    ax.plot(c.index,c.values,'-',color=P['blue'],lw=1.8); low=c[c<1430]
    ax.plot(low.index,low.values,'o',ms=7,mfc='white',mec=PROP,mew=2,label=f'Days < 1,430 records (n = {len(low)})')
    ax.axhline(1440,color=INK,lw=1.2,ls='--'); ax.set_ylim(900,1500); style(ax,'Date','Records per day',ny=4); datefmt(ax); ax.legend(loc='lower center',fontsize=11)
def s1b(fig,ss):
    ax=fig.add_subplot(ss); m=minute(); d=m.loc['2022-10-12 00:00':'2022-10-12 23:59','Q-PS_MGD']
    ax.plot(d.index,d.values,color=P['mustard'],lw=1.0,label='1-min signal')
    on=(d>1).resample('1h').mean()*100; ax2=ax.twinx()
    # The final hourly value applies through midnight; duplicate only its boundary.
    end=on.index[-1]+pd.Timedelta(hours=1); step_idx=on.index.append(pd.DatetimeIndex([end])); step_values=np.r_[on.values,on.iloc[-1]]
    ax2.step(step_idx,step_values,where='post',color=P['orange'],lw=2.6,label='Hourly on-fraction'); ax.set_xlim(on.index[0],end)
    style(ax,'Time of day','PS flow signal (raw)',ny=4); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M')); ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,6,12,18])); bold_ticks(ax)
    ax2.set_ylim(0,max(on.max()*1.6,5)); ax2.set_ylabel('Pump on (% of hour)',fontsize=LABEL,fontweight='bold'); ax2.tick_params(labelsize=TICK,width=1.5,length=7); bold_ticks(ax2)
    for s in ['top']: ax2.spines[s].set_visible(False)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,loc='lower center',bbox_to_anchor=(0.5,1.02),ncol=2,fontsize=11)
def s1c(fig,ss):
    ax=fig.add_subplot(ss); m=minute(); d=m.loc['2022-10-10':'2022-10-12']
    cum=d['Biogas_burner'].groupby(d.index.date).cumsum()   # cfm x 1 min = ft3
    ax.plot(d.index,d['V-burner_FT3']/1e5,color=P['slate'],lw=2.2,label='Burner counter')
    ax.plot(d.index,cum/1e5,color=P['red'],lw=1.6,ls='--',label='Integrated flow')
    style(ax,'Date','Cumulative volume (10$^{5}$ ft$^{3}$)',ny=4); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b')); ax.xaxis.set_major_locator(matplotlib.dates.DayLocator()); bold_ticks(ax)
    ax.axhline(1,color=INK,lw=1,ls=':'); ax.text(d.index[60],1.03,'counter ceiling',fontsize=10.5,fontweight='bold',color=INK,va='bottom'); ax.set_ylim(0,1.9); outside_legend(ax,ncol=2,fontsize=10.5)
def s1d(fig,ss):
    ax=fig.add_subplot(ss); m=minute()['Biogas']*CFM; hh=H['Biogas'].dropna()*CFM
    b=np.linspace(0,480,49); w=np.full(len(m),1e3/len(m)/(b[1]-b[0])); ax.hist(m,bins=b,weights=w,color=P['sky'],alpha=0.6,ec='white',lw=0.4,label='1-min')
    ax.hist(hh,bins=b,weights=np.full(len(hh),1e3/len(hh)/(b[1]-b[0])),histtype='step',color=P['blue'],lw=2.4,label='Hourly mean')
    ax.text(0.97,0.62,f'Near-zero\nminutes: {100*(minute()["Biogas"]<1).mean():.1f}%',transform=ax.transAxes,ha='right',va='top',fontsize=11.5,fontweight='bold',color=INK)
    style(ax,BIOGAS_LAB,'Density (×10$^{-3}$)',nx=5,ny=4); outside_legend(ax,ncol=2,fontsize=11)

def grid(nr,nc,letters,**kw):
    def lay(fig):
        G=GridSpec(nr,nc,figure=fig,**kw); return {l:G[i//nc,i%nc] for i,l in enumerate(letters)}
    return lay
L_=letters()
# ---------------- S2 all inputs ----------------
def s2_one(v,last_row):
    def f(fig,ss):
        ax=fig.add_subplot(ss); s=conv(v,H[v]); shade(ax)
        ax.plot(s.index,s.values,color=GROUP_COL[VG[v]],lw=0.6,alpha=0.70); ax.plot(s.index,s.rolling(168,center=True,min_periods=84).mean(),color=GROUP_COL[VG[v]],lw=2.6)
        lo,hi=np.nanmin(s),np.nanmax(s); pad=max((hi-lo)*0.06,0.01); ax.set_ylim(lo-pad,hi+pad); ax.set_xlim(H.index.min(),H.index.max())
        style(ax,None,CONV[v][1],ny=3,tick=11,label=11.5); datefmt(ax,(4,8,12),'%b',11)
        if not last_row and not standalone(fig): ax.tick_params(labelbottom=False)
        if standalone(fig):
            ax.set_xlabel('Date',fontsize=12)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor=SPLIT_COL[nm],alpha=0.28,label=nm) for nm in SPLIT_COL],
                      loc='lower center',bbox_to_anchor=(0.5,1.02),ncol=3,fontsize=10.5,handlelength=1.3,columnspacing=0.8)
    return f
# ---------------- S3 lab data ----------------
def labs(): return pd.read_pickle('supp/labs.pkl')
def s3_one(cols,lab,leg):
    def f(fig,ss):
        L=labs(); ax=fig.add_subplot(ss); ax.axvspan(H.index.min(),H.index.max(),color=P['lavender'],alpha=0.35,lw=0)
        for c,col in zip(cols,[P['blue'],P['red']]):
            s=L[c]*(0.003785 if 'gal' in c else 1); nm='D1' if 'Dig1' in c else ('D2' if 'Dig2' in c else None)
            ax.plot(s.index,s,color=col,lw=0.7,alpha=0.55); ax.plot(s.index,s.rolling(7,min_periods=3).mean(),color=col,lw=2.2,label=nm)
        style(ax,'Year',lab,ny=4); datefmt(ax,(1,),'%Y')
        if standalone(fig) and len(cols)>1: outside_legend(ax,fontsize=11.5)
        elif leg: shared_legend(fig,ax,ncol=2,fontsize=11.5)
    return f
def s3_scatter(fig,ss):
    L=labs(); ax=fig.add_subplot(ss); d=(H['Biogas']*CFM).resample('D').mean(); l=L['Biogas']*CFM; j=pd.concat([l,d],axis=1,keys=['lab','scada']).dropna()
    ax.scatter(j.scada,j.lab,s=26,facecolors='none',edgecolors=P['blue'],lw=1.2); lim=[j.min().min()*0.9,j.max().max()*1.05]; ax.plot(lim,lim,color=INK,ls='--',lw=1.3)
    ax.text(0.05,0.94,f"r = {j.corr().iloc[0,1]:.3f}\nn = {len(j)} d",transform=ax.transAxes,va='top',fontsize=12,fontweight='bold',color=INK)
    style(ax,'SCADA daily (m$^{3}$ h$^{-1}$)','Lab daily (m$^{3}$ h$^{-1}$)',nx=4,ny=4)
# ---------------- S4 error vs FOS/TAC ----------------
def s4_one(h,leg):
    def f(fig,ss):
        E=pd.read_csv('supp/error_by_fostac.csv'); ax=fig.add_subplot(ss); d=E[E.h==h]; x=np.arange(len(d)); w=0.38
        ax.bar(x-w/2,d.mae_past*CFM,w*0.92,color=P['rose'],ec=INK,lw=0.8,label='Past-only'); ax.bar(x+w/2,d.mae_feed*CFM,w*0.92,color=PROP,ec=INK,lw=0.8,label='Feed-conditioned')
        ax.set_xticks(x); ax.set_xticklabels(list(d.quartile))
        style(ax,'FOS/TAC quartile','MAE (m$^{3}$ h$^{-1}$)',ny=4); [l.set_fontsize(11) for l in ax.get_xticklabels()]
        ax.text(0.04,0.96,f'h = {h} h',transform=ax.transAxes,va='top',fontsize=12.5,fontweight='bold',color=INK)
        ax.set_ylim(0,max(d.mae_past.max(),d.mae_feed.max())*CFM*1.3)
        if standalone(fig): outside_legend(ax,fontsize=11)
        elif leg: shared_legend(fig,ax,ncol=2,fontsize=11)
    return f
# ---------------- S5 optuna ----------------
def s5_trace(m,h,leg):
    def f(fig,ss):
        T=pd.read_csv('tune/trials.csv' if os.path.exists('tune/trials.csv') else 'supp/optuna_trials.csv'); ax=fig.add_subplot(ss); d=T[(T.model==m)&(T.h==h)].sort_values('trial')
        ax.scatter(d.trial,d.value*CFM,s=24,facecolors='none',edgecolors=P['slate'],lw=1.1,label='Trial')
        ax.plot(d.trial,d.value.cummin()*CFM,color=PROP,lw=2.4,label='Best so far')
        lo,hi=np.min(d.value*CFM),np.max(d.value*CFM); span=max(hi-lo,0.1); ax.set_ylim(lo-0.08*span,hi+0.18*span)
        style(ax,'Trial','RMSE (m$^{3}$ h$^{-1}$)',nx=4,ny=4)
        ax.scatter([0],[d.value.iloc[0]*CFM],marker='*',s=150,color=P['mustard'],ec=INK,lw=0.8,zorder=4,label='Pre-specified')
        ax.text(0.96,0.95,f'{m.replace(" [past]"," (past)").replace(" [feed]"," (feed)")}, {h} h',transform=ax.transAxes,ha='right',va='top',fontsize=11,fontweight='normal',color=INK)
        if leg or standalone(fig): outside_legend(ax,ncol=3 if standalone(fig) else 1,fontsize=10.5)
    return f
def s5_imp(fig,ss):
    ax=fig.add_subplot(ss); imp=pickle.load(open('supp/optuna_LGB-Δ_24.pkl','rb'))['imp']; ks=list(imp)[::-1]
    ax.barh(range(len(ks)),[imp[k] for k in ks],color=P['blue'],ec=INK,lw=0.8,height=0.65); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
    style(ax,'fANOVA importance (LGB-Δ, 24 h)',None,nx=4); [l.set_fontsize(11.5) for l in ax.get_yticklabels()]
def s5_gain(fig,ss):
    ax=fig.add_subplot(ss); A=pd.read_csv('tune/tuned_anchor.csv'); A['gain']=(1-A.val_RMSE_tuned/A.val_RMSE_default)*100
    mods=list(dict.fromkeys(A.model)); 
    for j,(h,mk,c) in enumerate([(1,'s',P['slate']),(6,'D',P['teal']),(24,'o',PROP)]):
        d=A[A.h==h].set_index('model').loc[mods]
        ax.scatter(d.gain,np.arange(len(mods))+(j-1)*0.24,marker=mk,s=40,facecolors='white',edgecolors=c,lw=1.7,label=f'{h} h',zorder=3)
    ax.axvline(0,color=INK,lw=1.1,ls='--'); ax.set_yticks(range(len(mods))); ax.set_yticklabels([m.replace(' [past]',' (past)').replace(' [feed]',' (feed)') for m in mods]); ax.invert_yaxis()
    style(ax,'RMSE reduction (%)',None,nx=5); [l.set_fontsize(10.5) for l in ax.get_yticklabels()]
    outside_legend(ax,ncol=3,fontsize=10.5)
def s7_seed5(fig,ss):
    ax=fig.add_subplot(ss); B=pd.read_csv('tune/dl_seeds.csv'); mods=list(dict.fromkeys(B.model))
    for i,mn in enumerate(mods):
        for j,(h,mk) in enumerate([(1,'s'),(6,'D'),(24,'o')]):
            v=B[(B.model==mn)&(B.h==h)].R2.values; c=[P['slate'],P['teal'],PROP][j]
            ax.scatter(v,np.full(len(v),i)+(j-1)*0.24,marker=mk,s=30,facecolors='white',edgecolors=c,lw=1.4,label=f'{h} h' if i==0 else None,zorder=3)
            ax.plot([v.mean()]*2,[i+(j-1)*0.24-0.1,i+(j-1)*0.24+0.1],color=c,lw=2.4,zorder=4)
    ax.set_yticks(range(len(mods))); ax.set_yticklabels([m.replace(' [past]',' (past)').replace(' [feed]',' (feed)') for m in mods]); ax.invert_yaxis()
    style(ax,'Test R$^{2}$ (5 seeds)',None,nx=5); [l.set_fontsize(11) for l in ax.get_yticklabels()]; ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3,fontsize=11)
# ---------------- S6 learning curves ----------------
def s6_one(h,leg):
    def f(fig,ss):
        Lc=pd.read_csv('supp/learning_curves.csv'); ax=fig.add_subplot(ss); d=Lc[Lc.h==h]
        ax.plot(d.months,d.R2_train,'-o',color=TRAIN,lw=2.4,ms=7,label='Training'); ax.plot(d.months,d.R2_val,'--s',color=SPLIT_COL['Validation'],lw=2.4,ms=7,mfc='white',mew=2,label='Validation'); ax.plot(d.months,d.R2_test,':D',color=TEST,lw=2.4,ms=6,mfc='white',mew=2,label='Test')
        lo=min(d.R2_train.min(),d.R2_val.min(),d.R2_test.min()); ax.set_ylim(lo-0.12,1.02); ax.set_xticks([1,2,4,6,8]); style(ax,'Training length (months)','R$^{2}$',ny=5)
        ax.text(0.96,0.06,f'h = {h} h',transform=ax.transAxes,ha='right',fontsize=12.5,fontweight='bold',color=INK)
        if standalone(fig): outside_legend(ax,ncol=3,fontsize=11)
        elif leg: shared_legend(fig,ax,ncol=3,fontsize=11)
    return f
# ---------------- S7 DL training ----------------
DLC={'PG-HQ (past)':P['rose'],'PG-HQ (feed)':PROP,'DLinear':P['sky'],'TCN':P['teal'],'PatchTST':P['slate']}
def s7_train(fig,ss):
    Hh=pickle.load(open('supp/dl_histories.pkl','rb')); ax=fig.add_subplot(ss)
    for nm,hst in Hh.items(): e,tl,vl=zip(*hst); ax.plot(e,tl,color=DLC[nm],lw=2.2,label=nm)
    style(ax,'Epoch','Training loss (Huber)',ny=5); outside_legend(ax,ncol=2,fontsize=10.5)
def s7_val(fig,ss):
    Hh=pickle.load(open('supp/dl_histories.pkl','rb')); ax=fig.add_subplot(ss)
    for nm,hst in Hh.items():
        e,tl,vl=zip(*hst); ax.plot(e,np.array(vl)*CFM,color=DLC[nm],lw=2.2,label=nm); k=int(np.argmin(vl)); ax.plot(e[k],vl[k]*CFM,'o',ms=8,mfc='white',mec=DLC[nm],mew=2)
    style(ax,'Epoch','Validation RMSE (m$^{3}$ h$^{-1}$)',ny=5)
    if standalone(fig): outside_legend(ax,ncol=2,fontsize=10.5)
def s7_seed(fig,ss):
    ax=fig.add_subplot(ss); B=pd.read_csv('bench_dl_all.csv'); mods=list(B.model.unique())
    for i,mn in enumerate(mods):
        for j,(h,mk) in enumerate([(1,'s'),(6,'D'),(24,'o')]):
            v=B[(B.model==mn)&(B.h==h)].R2.values; ax.scatter(v,np.full(len(v),i)+(j-1)*0.22,marker=mk,s=36,facecolors='white',edgecolors=[P['slate'],P['teal'],PROP][j],lw=1.6,label=f'{h} h' if i==0 else None)
    ax.set_yticks(range(len(mods))); ax.set_yticklabels([m.replace(' (proposed, past-only)',' (past)').replace(' (proposed, feed-conditioned)',' (feed)') for m in mods]); ax.invert_yaxis()
    style(ax,'Test R$^{2}$ (3 seeds)',None,nx=5); [l.set_fontsize(11) for l in ax.get_yticklabels()]; outside_legend(ax,ncol=3,fontsize=11)
# ---------------- S8 SHAP beeswarm ----------------
SH=None
def shapd():
    global SH
    if SH is None: SH=pickle.load(open('shap.pkl','rb'))
    return SH
def pretty(f):
    v,b=(f.split('|')+[''])[:2]
    nm={'Biogas':'Biogas','H-Dig1_FT':'Cover D1','H-Dig2_FT':'Cover D2','Q-HSW1_GPM':'HSW→D1','Q-HSW2_GPM':'HSW→D2','H-HSW_ft':'HSW tank','V-HSW-DIG2-yest_gal':'HSW D2 d−1','Q-PS_on':'PS on','Q-TWAS_GPM':'TWAS','D1_TEMPERATURE':'T D1','D2_TEMPERATURE':'T D2','Q-influent_MGD':'Influent','Q-Aeration_SCFM':'Aeration','Q-RAS_MGD':'RAS','Q-WAS1_MGD':'WAS1','Q-WAS2_MGD':'WAS2'}.get(v,v)
    return f'{nm} [{b[1:]} h]' if b else {'hod_sin':'Hour (sin)','hod_cos':'Hour (cos)','dow_sin':'Weekday (sin)','dow_cos':'Weekday (cos)'}.get(v,v)
def s8_one(h):
    def f(fig,ss):
        ax=fig.add_subplot(ss); rng=np.random.default_rng(0); d=shapd()[h]; sv=d['sv']*CFM; X=d['X']; top=np.argsort(-np.abs(sv).mean(0))[:12]
        for r,j in enumerate(top):
            x=sv[:,j]; vr=pd.Series(X.iloc[:,j].values).rank(pct=True).values; yj=r+rng.uniform(-0.32,0.32,len(x)); sel=rng.choice(len(x),min(900,len(x)),replace=False)
            sc=ax.scatter(x[sel],yj[sel],c=vr[sel],cmap=cmap_div(),vmin=0,vmax=1,s=9,lw=0,alpha=0.85)
        ax.set_yticks(range(12)); ax.set_yticklabels([pretty(X.columns[j]) for j in top]); ax.invert_yaxis(); ax.axvline(0,color=INK,lw=1)
        style(ax,'SHAP (m$^{3}$ h$^{-1}$)',None,nx=4); [l.set_fontsize(11) for l in ax.get_yticklabels()]
        ax.text(0.97,0.03,f'h = {h} h',transform=ax.transAxes,ha='right',fontsize=12.5,fontweight='bold',color=INK)
        cb=cbar(fig,sc,ax.inset_axes([1.04,0.0,0.035,1.0]),'Feature percentile',size=11.5); cb.set_ticks([0,1]); cb.set_ticklabels(['0','100']); bold_ticks(cb.ax,11)
    return f
# ---------------- S9 attribution sensitivity ----------------
GR9=list(GROUPS)+['Calendar']
def s9_one(h,leg,ylab):
    def f(fig,ss):
        C=pd.read_csv('supp/conditional_permutation.csv'); Sg=pd.read_csv('shap_group_by_horizon.csv',index_col=0); ax=fig.add_subplot(ss); y=np.arange(len(GR9)); w=0.26
        m=C[(C.h==h)&(C['mode']=='marginal')].set_index('group').dRMSE.clip(lower=0); c=C[(C.h==h)&(C['mode']=='conditional')].set_index('group').dRMSE.clip(lower=0); s=Sg.loc[h]
        for j,(v,lab,col) in enumerate([(m/m.sum()*100,'Marginal permutation',P['blue']),(c/c.sum()*100,'Conditional permutation',P['orange']),(s/s.sum()*100,'Grouped TreeSHAP',P['teal'])]):
            ax.barh(y+(j-1)*w,[v.get(gr,0) for gr in GR9],w*0.92,color=col,ec=INK,lw=0.7,label=lab)
        ax.set_yticks(y); ax.set_yticklabels([GROUP_SHORT.get(gr,gr) for gr in GR9] if ylab else []); ax.invert_yaxis(); ax.set_xlim(0,100)
        style(ax,'Share (%)',None,nx=5)
        rc=spearmanr([m.get(gr,0) for gr in GR9],[c.get(gr,0) for gr in GR9]).correlation
        ax.text(0.96,0.06,f'h = {h} h\nρ = {rc:.2f}',transform=ax.transAxes,ha='right',va='bottom',fontsize=12,fontweight='bold',color=INK)
        if standalone(fig): ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.25),ncol=1,fontsize=11)
        elif leg: shared_legend(fig,ax,ncol=3,fontsize=11)
    return f
# ---------------- S10 lag-bin design ----------------
def s10(fig,ss):
    D=pd.read_csv('supp/lagbin_design.csv'); ax=fig.add_subplot(ss); des=['Raw lags + rolling statistics','Equal-width bins (8 × 21 h)','Log-spaced bins (PG-LB)']; nf={des[0]:484,des[1]:132,des[2]:132}
    x=np.arange(3); w=0.26
    for j,(dn,c) in enumerate(zip(des,[P['teal'],P['blue'],PROP])):
        v=[D[(D.design==dn)&(D.h==h)].R2.iloc[0] for h in [1,6,24]]; ax.bar(x+(j-1)*w,v,w*0.92,color=c,ec=INK,lw=0.8,label=f'{dn} ({nf[dn]})')
        for xi,vi in zip(x+(j-1)*w,v): ax.text(xi,vi+0.015,f'{vi:.3f}',ha='center',fontsize=10,fontweight='bold',color=INK,va='bottom')
    ax.set_xticks(x); ax.set_xticklabels(['1 h','6 h','24 h']); ax.set_ylim(0,1.04); ax.set_yticks([0,0.25,0.5,0.75,1.0]); style(ax,'Horizon','Test R$^{2}$ (LGB-Δ, past-only)'); outside_legend(ax,ncol=1,fontsize=11,title='Design (features)',title_fontsize=11)
# ---------------- S11 attention vs attribution ----------------
def s11_heat(fig,ss):
    A=np.load('pghq_attention.npy'); ax=fig.add_subplot(ss); M=A.reshape(24,-1); cells=M.T
    # Match imshow's half-cell extent and upper origin with vector quadrilaterals.
    rows,cols=cells.shape; im=ax.pcolormesh(np.arange(cols+1)-0.5,np.arange(rows+1)-0.5,cells,shading='flat',cmap=cmap_seq(P['blue']),linewidth=0,antialiased=False,rasterized=False)
    ax.set_xlim(-0.5,cols-0.5); ax.set_ylim(rows-0.5,-0.5); ax.set_aspect('auto')
    for k in range(1,6): ax.axhline(k*8-0.5,color='white',lw=2)
    ax.set_yticks([k*8+3.5 for k in range(6)]); ax.set_yticklabels([GROUP_SHORT[g_] for g_ in GROUPS]); ax.set_xticks([0,5,11,17,23]); ax.set_xticklabels([1,6,12,18,24])
    style(ax,'Horizon (h)',None); [sp.set_visible(True) for sp in ax.spines.values()]
    cbar(fig,im,ax.inset_axes([1.04,0,0.035,1]),'Attention weight',size=11.5)
def s11_rho(fig,ss):
    A=np.load('pghq_attention.npy'); hr=pickle.load(open('hrpga.pkl','rb')); ax=fig.add_subplot(ss); HS_=hr['HS']
    rho=[spearmanr(A[h-1].sum(1),hr['past']['Ag'][i,:6]).correlation for i,h in enumerate(HS_)]
    ax.plot(HS_,rho,'-o',color=P['slate'],lw=2.4,ms=8,mfc='white',mew=2); ax.axhline(0,color=INK,lw=1); ax.set_ylim(-1,1); ax.set_xticks([1,6,12,18,24])
    style(ax,'Horizon (h)','Spearman ρ',ny=5)
# ---------------- S12 next-24-h mean ----------------
def s12(fig,ss):
    B=pd.read_csv('supp/next24_mean_benchmark.csv').sort_values('R2'); ax=fig.add_subplot(ss)
    c=[PROP if 'Δ' in m else (P['lavender'] if 'Persistence' in m else BASE) for m in B.model]
    ax.barh(range(len(B)),B.R2,color=c,ec=INK,lw=0.8,height=0.66); ax.set_yticks(range(len(B))); ax.set_yticklabels(B.model)
    for i,v in enumerate(B.R2): ax.text(max(v,0)+0.01,i,f'{v:.3f}',va='center',fontsize=11.5,fontweight='bold',color=INK)
    ax.set_xlim(0,1.0); style(ax,'Test R$^{2}$, mean over next 24 h',None,nx=5)
# ---------------- S14 conformal methods ----------------
CC={'Split':P['blue'],'Normalised':P['teal'],'ACI':P['orange'],'CQR':P['rose']}; CM={'Split':'o','Normalised':'s','ACI':'D','CQR':'^'}
def s14_cov(fig,ss):
    C=pd.read_csv('supp/conformal_methods.csv'); ax=fig.add_subplot(ss)
    for m in CC: d=C[C.method==m]; ax.plot(d.h,d.coverage,('--' if m=='CQR' else '-')+CM[m],color=CC[m],lw=2.3,ms=7,mfc='white',mew=2,label=m)
    ax.axhline(0.9,color=INK,lw=1.3,ls='--'); ax.set_ylim(0.75,1.0); ax.set_xticks([1,6,12,18,24]); style(ax,'Horizon (h)','Coverage (nominal 0.90)',ny=5)
    if standalone(fig): outside_legend(ax,ncol=2,fontsize=11)
    else: shared_legend(fig,ax,ncol=4,fontsize=11)
def s14_wid(fig,ss):
    C=pd.read_csv('supp/conformal_methods.csv'); ax=fig.add_subplot(ss)
    for m in CC: d=C[C.method==m]; ax.plot(d.h,d.width*CFM,('--' if m=='CQR' else '-')+CM[m],color=CC[m],lw=2.3,ms=7,mfc='white',mew=2,label=m)
    ax.set_xticks([1,6,12,18,24]); style(ax,'Horizon (h)','Mean width (m$^{3}$ h$^{-1}$)',ny=5)
    if standalone(fig): outside_legend(ax,ncol=2,fontsize=11)
# ---------------- S15 PDP vs ALE ----------------
def pdp(m,X,f,grid,n=800):
    S=X.sample(min(n,len(X)),random_state=0); out=[]
    for v in grid: Z=S.copy(); Z[f]=v; out.append(m._Booster.predict(Z).mean())
    out=np.array(out); return out-out.mean()
_O=None
def s15_one(f_,h,lab,u,col,ylab):
    def f(fig,ss):
        global _O
        if _O is None: _O=(pickle.load(open('tab_final.pkl','rb')),pickle.load(open('ale.pkl','rb')),build()[0])
        o,ALE,X0=_O; g=ss.subgridspec(2,1,hspace=0.12); m=o['models'][(h,'LGB-Δ [past]')]; X=pd.concat([X0,calendar(X0.index,h)],axis=1).loc[idx[o_tr]]
        q,a=ALE[(h,f_)]; lo,hi=np.percentile(X[f_],[1,99]); keep=(q>=lo)&(q<=hi); gr=q[keep]; pv=pdp(m,X,f_,gr)
        for r,(yv,name,ls) in enumerate([(pv,'PDP','--'),(a[keep]-np.mean(a[keep]),'ALE','-')]):
            ax=fig.add_subplot(g[r]); ax.plot(gr*u,yv*CFM,ls,color=col,lw=2.6); ax.axhline(0,color=INK,lw=0.9,ls=':')
            style(ax,lab if r==1 else None,f'{name} (m$^{{3}}$ h$^{{-1}}$)' if ylab else None,nx=4,ny=4,tick=12,label=13.5)
            if r==0: ax.tick_params(labelbottom=False); ax.text(0.96 if f_=='Q-HSW1_GPM|L48-71' else 0.04,0.94,f'h = {h} h',transform=ax.transAxes,ha='right' if f_=='Q-HSW1_GPM|L48-71' else 'left',va='top',fontsize=12,fontweight='bold',color=INK)
    return f
# ---------------- S16 SHAP dependence ----------------
def s16_one(fe,lab,u,leg):
    def f(fig,ss):
        ax=fig.add_subplot(ss)
        for h,col in [(1,P['slate']),(6,P['teal']),(24,PROP)]:
            d=shapd()[h]; j=list(d['X'].columns).index(fe); x=d['X'][fe].values*u; y=d['sv'][:,j]*CFM
            lo,hi=np.percentile(x,[1,99]); s=(x>=lo)&(x<=hi); ax.scatter(x[s],y[s],s=9,facecolors='none',edgecolors=col,lw=0.6,alpha=0.45)
            b=pd.Series(y[s]).groupby(pd.qcut(x[s],12,duplicates='drop'),observed=True).mean(); ax.plot([iv.mid for iv in b.index],b.values,'-',color=col,lw=2.6,label=f'{h} h')
        ax.axhline(0,color=INK,lw=0.9,ls=':'); style(ax,lab,'SHAP (m$^{3}$ h$^{-1}$)',nx=4,ny=5)
        if standalone(fig): outside_legend(ax,ncol=3,fontsize=11.5)
        elif leg: shared_legend(fig,ax,ncol=3,fontsize=11.5)
    return f
# ---------------- registry ----------------
S3I=[(['Dig1-VFA_mgL','Dig2-VFA_mgL'],'VFA (mg L$^{-1}$)'),(['Dig1-alk_mgL','Dig2-alk_mgL'],'Alkalinity (mg L$^{-1}$)'),(['Dig1-FOS-TAC','Dig2-FOS-TAC'],'FOS/TAC'),(['Dig1-pH','Dig2-pH'],'pH'),(['V-HSW_gal'],'HSW fed (m$^{3}$ d$^{-1}$)')]
FE16=[('Q-HSW2_GPM|L0','Current HSW to D2\n(m$^{3}$ h$^{-1}$)',GPM),('Q-HSW1_GPM|L48-71','HSW to D1, 48–71 h ago\n(m$^{3}$ h$^{-1}$)',GPM),('H-Dig1_FT|L0','Current cover height, D1 (m)',0.3048)]
C15=[('Q-HSW2_GPM|L0',6,'Current HSW to D2\n(m$^{3}$ h$^{-1}$)',GPM,P['orange']),('Q-HSW1_GPM|L48-71',24,'HSW to D1, 48–71 h ago\n(m$^{3}$ h$^{-1}$)',GPM,P['orange']),('H-Dig1_FT|L0',1,'Current cover height, D1 (m)',0.3048,P['teal'])]
