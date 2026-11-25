from figcommon import *
import pickle
from common import metrics
fp=pickle.load(open('final_preds.pkl','rb')); TR=pickle.load(open('extra/train_preds_feed_ens.pkl','rb')); C=pd.read_csv('conformal.csv')
HS=[1,2,3,4,6,9,12,18,24]; ENS='PG-HQ ensemble [feed]'; ENSP='PG-HQ ensemble [past]'
from common import H
from aci import aci_halfwidths
_Q={}
def qaci(h,key,alpha=0.1):
    k=(h,key,alpha)
    if k not in _Q:
        V=fp['pred'][h]['val']; T=fp['pred'][h]['test']; _Q[k]=aci_halfwidths(V.y.values-V[key].values,T.y.values-T[key].values,h,alpha=alpha)
    return _Q[k]
hsw=(H['Q-HSW1_GPM']+H['Q-HSW2_GPM'])
def q90(h,reg='feed'): return C[(C.h==h)&(C.regime==reg)&(C.nominal==0.9)].width_split.iloc[0]/2
te_idx=fp['pred'][1]['test'].index
d=hsw.loc[te_idx.min():te_idx.max()].resample('D').mean()
roll=d.rolling(10).mean().dropna()
W=[(roll.idxmin()-pd.Timedelta(days=9),roll.idxmin()+pd.Timedelta(days=1)),(roll.idxmax()-pd.Timedelta(days=9),roll.idxmax()+pd.Timedelta(days=1))]
def pa(fig,ss):
    g=ss.subgridspec(3,2,hspace=.07,wspace=.05)
    for c,(a,b) in enumerate(W):
        for r,h in enumerate([1,6,24]):
            ax=fig.add_subplot(g[r,c]); T=fp['pred'][h]['test']; t=T.index+pd.Timedelta(hours=h); m=(t>=a)&(t<b)
            yy=T.y[m]*CFM; pp=T[ENS][m]*CFM; q=qaci(h,ENS)[np.asarray(m)]*CFM
            ax.fill_between(t[m],pp-q,pp+q,color=PROP,alpha=0.16,lw=0)
            ax.plot(t[m],yy,color=OBS,lw=1.8,label='Observed'); ax.plot(t[m],pp,color=PROP,lw=1.9,label='Forecast')
            ax.set_ylim(0,460); ax.set_xlim(a,b)
            style(ax,None,BIOGAS_LAB if (c==0 and r==1) else None,ny=3,tick=11.5)
            ax.text(0.015,0.94,f'h = {h} h',transform=ax.transAxes,va='top',fontsize=11.5,fontweight='bold',color=INK)
            if c==1: ax.tick_params(labelleft=False)
            if r<2: ax.tick_params(labelbottom=False)
            else:
                ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b')); ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3)); bold_ticks(ax,11.5)
            if r==0: ax.text(0.5,1.04,'Low HSW loading' if c==0 else 'High HSW loading',transform=ax.transAxes,ha='center',fontsize=12.5,color=INK)
            if r==0 and c==1: ax.legend(loc='upper right',fontsize=11.5,ncol=2,bbox_to_anchor=(1.0,.99),borderaxespad=.2)
def joint(fig,ss,h,show_y):
    g=ss.subgridspec(1,2,width_ratios=[5,1],wspace=0.05); ax=fig.add_subplot(g[0]); axr=fig.add_subplot(g[1])
    T=fp['pred'][h]['test']; tr=TR[h]
    ax.scatter(tr.y*CFM,tr.p*CFM,marker='^',s=24,c=TRAIN,edgecolors=INK,linewidths=0.25,alpha=0.65,label='Train')
    ax.scatter(T.y*CFM,T[ENS]*CFM,marker='o',s=21,facecolors='none',edgecolors=TEST,linewidths=0.9,alpha=0.8,label='Test')
    lim=[0,430]; ax.plot(lim,lim,color=INK,lw=1.4,ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
    m=metrics(T.y.values,T[ENS].values); mt=metrics(tr.y.values,tr.p.values)
    ax.text(0.04,0.96,f"h = {h} h\nTrain R$^{{2}}$ = {mt['R2']:.3f}\nTest R$^{{2}}$ = {m['R2']:.3f}\nRMSE = {m['RMSE']*CFM:.1f}",transform=ax.transAxes,va='top',fontsize=11.5,fontweight='normal',color=INK,bbox=dict(fc='white',ec='none',alpha=0.85))
    style(ax,'Observed\n(m$^{3}$ h$^{-1}$)','Predicted (m$^{3}$ h$^{-1}$)' if show_y else None,nx=4,ny=4)
    if h==24: lg=ax.legend(loc='lower right',fontsize=10.5,markerscale=1.8); [hd.set_alpha(1) for hd in lg.legend_handles]
    res=(T[ENS]-T.y)*CFM; axr.hist(res,bins=36,orientation='horizontal',color=TEST,alpha=1,ec='white',lw=0.3); axr.axhline(0,color=INK,lw=1.2); axr.set_ylim(-180,180)
    style(axr,'Count',None,tick=10,label=11.5,nx=2,ny=5); axr.tick_params(labelleft=False)
def pb(fig,ss):
    g=ss.subgridspec(1,3,wspace=.34)
    for k,h in enumerate([1,6,24]): joint(fig,g[k],h,k==0)
def pc(fig,ss):
    ax=fig.add_subplot(ss); h=6; T=fp['pred'][h]['test']; feeding=(hsw.reindex(T.index)>1).values
    data=[]; pos=[]; cols=[]
    for i,(nm,cond) in enumerate([('HSW feeding',feeding),('No HSW feeding',~feeding)]):
        for j,(key,c) in enumerate([(ENSP,P['rose']),(ENS,PROP)]):
            e=np.abs(T[key]-T.y).values[cond]*CFM; data.append(e); pos.append(i*3+j); cols.append(c)
    vp=ax.violinplot(data,positions=pos,widths=0.85,showextrema=False)
    for b,c in zip(vp['bodies'],cols): b.set_facecolor(c); b.set_edgecolor(INK); b.set_alpha(1); b.set_linewidth(1)
    for p_,e in zip(pos,data):
        ax.plot([p_-0.18,p_+0.18],[np.median(e)]*2,color=INK,lw=2.4); ax.text(p_+0.20,np.median(e),f'{np.median(e):.1f}',va='center',fontsize=10.5,fontweight='bold',color=INK)
    ax.set_xticks([0.5,3.5]); ax.set_xticklabels([f'HSW fed\n(n = {feeding.sum():,})',f'Not fed\n(n = {(~feeding).sum():,})']); ax.set_ylim(0,130)
    style(ax,None,'|Error|, 6 h (m$^{3}$ h$^{-1}$)',ny=5)
    ax.legend(handles=[matplotlib.patches.Patch(fc=P['rose'],ec=INK,label='Past-only'),matplotlib.patches.Patch(fc=PROP,ec=INK,label='Feed-conditioned')],loc='lower left',ncol=1,fontsize=10.5,bbox_to_anchor=(0.02,1.01),borderaxespad=0.2)
def pd_(fig,ss):
    ax=fig.add_subplot(ss); noms=np.array([0.5,0.6,0.7,0.8,0.9,0.95])
    for h,c,mk in [(1,P['sky'],'s'),(6,P['blue'],'D'),(24,PROP,'o')]:
        T=fp['pred'][h]['test']; rt=np.abs(T.y-T[ENS]).values; V=fp['pred'][h]['val']; rv=np.abs(V.y-V[ENS]).values; n=len(rv)
        cov=[np.mean(rt<=qaci(h,ENS,1-a)) for a in noms]; covs=[np.mean(rt<=np.quantile(rv,min(1,np.ceil((n+1)*a)/n))) for a in noms]
        ax.plot(noms,cov,'-'+mk,color=c,lw=2.4,ms=7,mfc='white',mew=2,label=f'{h} h, ACI')
        ax.plot(noms,covs,':',color=c,lw=1.6)
    ax.plot([0.45,1],[0.45,1],color=INK,lw=1.3,ls='--'); ax.set_xlim(0.45,1); ax.set_ylim(0.45,1)
    style(ax,'Nominal coverage','Test coverage',nx=5,ny=5)
    hs,ls_=ax.get_legend_handles_labels(); hs.append(matplotlib.lines.Line2D([],[],color=INK,ls=':',lw=1.6)); ls_.append('Split conformal')
    ax.legend(hs,ls_,loc='lower right',fontsize=10,handlelength=1.6,borderaxespad=0.2)
def pe(fig,ss):
    ax=fig.add_subplot(ss); HS_=[1,2,3,4,6,9,12,18,24]; W={}
    for key,lab,c,ls in [(ENSP,'Past-only',P['rose'],'--'),(ENS,'Feed-conditioned',PROP,'-')]:
        W[key]=[2*qaci(h,key).mean()*CFM for h in HS_]; ax.plot(HS_,W[key],ls,marker='o',color=c,lw=2.6,ms=7,mfc='white',mew=2,label=lab)
    # Place the 24 h reduction label below the forecast-width curve, leaving
    # both the last segment and its endpoint visible.
    r=1-W[ENS][-1]/W[ENSP][-1]; ax.annotate(f'−{100*r:.0f}%',(24,W[ENS][-1]),(18.2,W[ENS][-1]-50),fontsize=12,color=INK,arrowprops=dict(arrowstyle='-',color=INK,lw=1.2))
    ax.set_xticks([1,6,12,18,24]); style(ax,'Horizon (h)','ACI width (m$^{3}$ h$^{-1}$)',ny=5)
    legend_font=11.5 if getattr(fig,'_is_standalone',False) else 10.5
    ax.legend(loc='lower right',fontsize=legend_font,bbox_to_anchor=(1,0.02),
              handlelength=0.8,handletextpad=0.4,labelspacing=0.35,borderaxespad=0.35)
def layout(fig):
    G=GridSpec(3,1,figure=fig,height_ratios=[1.35,0.95,0.95],hspace=0.95)
    # Move only row a to halve the previous 18.42 mm gap to b (~9.21 mm),
    # retaining all axes sizes and the b/c/d/e row positions.
    a_box=G[0].get_position(fig); a_shift=0.06
    a=GridSpec(1,1,figure=fig,left=a_box.x0,right=a_box.x1,
               bottom=a_box.y0-a_shift,top=a_box.y1-a_shift)
    r3=G[2].subgridspec(1,3,wspace=.34)
    return {'a':a[0],'b':G[1],'c':r3[0],'d':r3[1],'e':r3[2]}
panels=[('a',pa,(14,7)),('b',pb,(14,4.6)),('c',pc,(6,5)),('d',pd_,(6,5)),('e',pe,(6,5))]
if __name__=='__main__': render('Fig6_trajectories_uncertainty','Fig6_trajectories_uncertainty',panels,(18,19),layout)
