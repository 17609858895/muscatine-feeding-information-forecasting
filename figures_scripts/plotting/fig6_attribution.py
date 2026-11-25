from figcommon import *
import pickle
from scipy.stats import spearmanr
hr=pickle.load(open('hrpga.pkl','rb')); HS=hr['HS']; blocks=hr['blocks']; groups6=list(GROUP_COL)[:6]
BL=['0','1-2','3-5','6-11','12-23','24-47','48-71','72-167']; MID=np.array([0.5,1.5,4,8.5,17.5,35.5,59.5,119.5])
def block_mat(reg,h):
    i=HS.index(h); A=np.zeros((6,8))
    for j,(g,b) in enumerate(blocks):
        if g in groups6: A[groups6.index(g),BL.index(b)]=hr[reg]['Ab'][i,j]
    return A*CFM
def shares(reg):
    G=pd.DataFrame(hr[reg]['Ag'],index=HS,columns=hr['groups']).clip(lower=0)
    if reg=='past': G=G.drop(columns='Planned feed')
    return G.div(G.sum(1),axis=0)*100
def pa(fig,ss):
    g=ss.subgridspec(2,2,height_ratios=[1,0.18],hspace=.65,wspace=.12)
    for k,reg in enumerate(['past','feed']):
        ax=fig.add_subplot(g[0,k]); S=shares(reg); cols=[c for c in GROUP_COL if c in S.columns]
        ax.stackplot(HS,[S[c].values for c in cols],colors=[GROUP_COL[c] for c in cols],edgecolor='white',lw=0.6,alpha=1)
        ax.set_xlim(1,24); ax.set_ylim(0,100); ax.set_xticks([1,6,12,18,24])
        style(ax,'Horizon (h)','HR-PGA\nshare (%)' if k==0 else None,ny=5)
        if k==1: ax.tick_params(labelleft=False)
        ax.text(0.5,1.035,'Past-only' if reg=='past' else 'Feed-conditioned',transform=ax.transAxes,ha='center',va='bottom',fontsize=12,color=INK)
    # A dedicated row protects the common legend from both x-axis labels when
    # the combined and standalone canvases undergo footprint calibration.
    legend_ax=fig.add_subplot(g[1,:]); legend_ax.set_axis_off()
    h_=[matplotlib.patches.Patch(fc=GROUP_COL[c],ec='white',label=GROUP_SHORT[c]) for c in GROUP_COL]
    legend_ax.legend(handles=h_,loc='center',ncol=3,fontsize=10.5,frameon=False,handlelength=1.0,handletextpad=0.4,columnspacing=0.6)
def pb(fig,ss):
    g=ss.subgridspec(1,4,width_ratios=[1,1,1,0.07],wspace=0.12)
    for k,h in enumerate([1,6,24]):
        ax=fig.add_subplot(g[k]); A=np.clip(block_mat('past',h),0,None); A=A/np.where(A.max(1,keepdims=True)>0,A.max(1,keepdims=True),1)
        im=ax.pcolormesh(np.arange(A.shape[1]+1)-0.5,np.arange(A.shape[0]+1)-0.5,A,cmap=cmap_seq(P['blue']),vmin=0,vmax=1,shading='flat',rasterized=False)
        ax.set_xlim(-0.5,A.shape[1]-0.5); ax.set_ylim(A.shape[0]-0.5,-0.5)
        for r in range(6):
            c=np.argmax(A[r])
            ax.add_patch(matplotlib.patches.Rectangle((c-0.46,r-0.44),0.92,0.88,fill=False,ec=GROUP_COL[groups6[r]],lw=2.4))
        # All eight lag bins remain plotted; the compact combined figure labels
        # four representative ticks. Standalone panels retain every lag label.
        ticks=list(range(8)) if getattr(fig,'_is_standalone',False) else [0,2,4,7]
        ax.set_xticks(ticks); ax.set_xticklabels([BL[i] for i in ticks],rotation=60 if getattr(fig,'_is_standalone',False) else 90); ax.set_yticks(range(6)); ax.set_yticklabels([GROUP_SHORT[x] for x in groups6] if k==0 else [])
        style(ax,'Lag bin (h)' if k==1 else ' ',None,tick=11.5); [l.set_fontsize(11) for l in ax.get_xticklabels()]; [sp.set_visible(True) for sp in ax.spines.values()]
        # Facet identity belongs above the matrix, so no hydraulic cells are
        # covered by a text/background patch.
        ax.text(0.5,1.025,f'h = {h} h',transform=ax.transAxes,ha='center',va='bottom',fontsize=12.5,color=INK)
    cbar(fig,im,fig.add_subplot(g[3]),'ΔRMSE / group maximum',size=11)
def pc(fig,ss):
    ax=fig.add_subplot(ss)
    for r,gname in enumerate(groups6):
        if gname not in ('Gas history','Gas storage','Co-substrate feed'): continue
        cen=[]
        for h in HS:
            w=np.clip(block_mat('past',h)[r],0,None); cen.append(np.sum(w*MID)/w.sum() if w.sum()>0 else np.nan)
        mk={'Gas history':'o','Gas storage':'s','Co-substrate feed':'^'}[gname]
        ax.plot(HS,cen,'-'+mk,color=GROUP_COL[gname],lw=3,ms=8,mfc='white',mew=2.2,label=GROUP_SHORT[gname])
    ax.set_yscale('log'); ax.set_yticks([1,3,10,30,100]); ax.set_yticklabels(['1','3','10','30','100']); ax.set_xticks([1,6,12,18,24]); ax.set_xlim(0,25)
    style(ax,'Horizon (h)','Weighted lag (h)')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,fontsize=11,
              handlelength=1.4,handletextpad=0.45,columnspacing=0.65,borderaxespad=0)
def pd_(fig,ss):
    ax=fig.add_subplot(ss); x=np.arange(3); w=0.26
    for k,(key,lab,c) in enumerate([('random','Random order',P['lavender']),('glob','Global ranking',BASE),('specific','Horizon-specific ranking',PROP)]):
        v=[np.mean(hr['faith'][h][key][1:]-hr['faith'][h][key][0])*CFM for h in [1,6,24]]
        ax.bar(x+(k-1)*w,v,w*0.92,color=c,ec=INK,lw=0.8,label=lab)
        for xi,vi in zip(x+(k-1)*w,v):
            # Put the middle-series values inside their bars; adjacent 10.9/
            # 10.4 and 6.1/6.6 labels otherwise collide on compact canvases.
            ax.text(xi,vi-0.5 if k==1 else vi+0.4,f'{vi:.1f}',ha='center',va='top' if k==1 else 'bottom',fontsize=10.5,color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['1 h','6 h','24 h'])
    ax.set_ylim(0,ax.get_ylim()[1]*1.3); style(ax,'Horizon','Mean ΔRMSE (m$^{3}$ h$^{-1}$)',ny=5); ax.legend(loc='upper left',fontsize=11,handlelength=1.2)
def pe(fig,ss):
    g=ss.subgridspec(1,2,width_ratios=[1,0.045],wspace=0.08); ax=fig.add_subplot(g[0]); cax=fig.add_subplot(g[1])
    A=np.nan_to_num(hr['past']['Ab']); n=len(HS); R=np.array([[spearmanr(A[i],A[j]).correlation for j in range(n)] for i in range(n)])
    im=ax.pcolormesh(np.arange(n+1)-0.5,np.arange(n+1)-0.5,R,cmap=cmap_seq(P['blue']),vmin=min(0.5,R.min()),vmax=1,shading='flat',rasterized=False)
    ax.set_xlim(-0.5,n-0.5); ax.set_ylim(n-0.5,-0.5)
    for i in range(n):
        for j in range(n): ax.text(j,i,'1' if i==j else (f'{R[i,j]:.2f}'.replace('0.','.')),ha='center',va='center',fontsize=10.8,color=INK)
    ax.set_xticks(range(n)); ax.set_xticklabels(HS); ax.set_yticks(range(n)); ax.set_yticklabels(HS)
    style(ax,'Horizon (h)','Horizon (h)'); [sp.set_visible(True) for sp in ax.spines.values()]
    try:
        nl=pd.read_csv('extra/erank_null.csv'); obs=nl.observed.iloc[0]; pv=(np.sum(nl.null_erank<=obs)+1)/(len(nl)+1)
        txt=f'Effective rank {obs:.2f}\nnull mean {nl.null_erank.mean():.2f}, p {"< 0.001" if pv<0.001 else "= %.3f"%pv}'
    except Exception: txt=f"Effective rank {hr['erank']['past']:.2f}"
    ax.text(1.0,1.02,txt,transform=ax.transAxes,ha='right',va='bottom',fontsize=11,fontweight='bold',color=INK)
    cbar(fig,im,cax,'Spearman ρ',size=11)
def pf(fig,ss):
    ax=fig.add_subplot(ss); G=pd.read_csv('shap_group_by_horizon.csv',index_col=0); G=G.div(G.sum(1),axis=0)*100
    cols=[c for c in GROUP_COL if c in G.columns]; left=np.zeros(len(G)); yy=np.arange(len(G))
    for c in cols:
        ax.barh(yy,G[c].values,left=left,color=GROUP_COL[c],ec='white',lw=0.8,height=0.62); 
        for i,v in enumerate(G[c].values):
            if v>=9: ax.text(left[i]+v/2,yy[i],f'{v:.0f}',ha='center',va='center',fontsize=10.5,color=INK)
        left+=G[c].values
    rho=[]
    for h in G.index:
        a=pd.Series(hr['past']['Ag'][HS.index(int(h))],index=hr['groups']).drop('Planned feed'); rho.append(spearmanr(G.loc[h,a.index],a).correlation)
    ax.set_yticks(yy); ax.set_yticklabels([f'{h} h' for h in G.index]); ax.set_xlim(0,100); ax.invert_yaxis()
    for i,r in enumerate(rho): ax.text(101.5,yy[i],f'ρ = {r:.2f}',va='center',fontsize=11.5,fontweight='bold',color=INK,clip_on=False)
    style(ax,'Share of mean |SHAP| (%)',None,nx=5)
def layout(fig):
    G=GridSpec(3,2,figure=fig,height_ratios=[1,1,1.04],hspace=0.86,wspace=0.82)
    return {'a':G[0,0],'b':G[0,1],'c':G[1,0],'d':G[1,1],'e':G[2,0],'f':G[2,1]}
panels=[('a',pa,(8.5,5.6)),('b',pb,(9.5,4.8)),('c',pc,(7,5)),('d',pd_,(7,5)),('e',pe,(6.8,6)),('f',pf,(7.5,4))]
if __name__=='__main__': render('Fig7_horizon_resolved_attribution','Fig7_horizon_resolved_attribution',panels,(19,17.5),layout)
