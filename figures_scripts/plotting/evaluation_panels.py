from figcommon import *
import pickle
from common import metrics
LA=pd.read_csv('leakage_audit.csv'); LH=pd.read_csv('extra/leakage_all_horizons.csv'); LP=pickle.load(open('extra/leakage_preds24.pkl','rb'))
CHR=BASE; RND=PROP
def pa(fig,ss):
    ax=fig.add_subplot(ss); mk={'A':('o',CHR,'Chronological'),'B':('o',RND,'Random 80/20'),'C':('D',RND,'Random + target counters'),'D':('D',CHR,'Chronological + target counters')}
    for i,h in enumerate([1,6,24]):
        d=LA[LA.h==h].set_index(LA[LA.h==h].setting.str[0]); y=2-i
        ax.plot([d.loc['A','R2'],d.loc['B','R2']],[y,y],color=INK,lw=1.6,zorder=1)
        for k,(m,c,lab) in mk.items():
            off={'A':0.0,'B':0.0,'C':0.16,'D':-0.16}[k]
            ax.scatter(d.loc[k,'R2'],y+off,marker=m,s=120,c='white' if k in 'CD' else c,edgecolors=c,linewidths=2.2,zorder=3,label=lab if i==0 else None)
        ax.annotate(f"+{d.loc['B','R2']-d.loc['A','R2']:.2f}",((d.loc['A','R2']+d.loc['B','R2'])/2,y),(0,13),textcoords='offset points',ha='center',fontsize=11.5,color=INK)
    ax.set_yticks([2,1,0]); ax.set_yticklabels(['1 h','6 h','24 h']); ax.set_ylim(-0.5,5.4); ax.set_xlim(0.35,1.0)
    style(ax,'Test R$^{2}$','Horizon',nx=5); ax.legend(loc='upper left',fontsize=10.5,handletextpad=0.3,borderaxespad=0.2,ncol=1)
def pb(fig,ss):
    ax=fig.add_subplot(ss); c=LH[LH.split=='chronological'].set_index('h').R2; r=LH[LH.split=='random'].set_index('h').R2
    ax.fill_between(c.index,c.values,r.values,color=RND,alpha=0.22,lw=0)
    ax.plot(r.index,r.values,'-o',color=RND,lw=2.6,ms=7,mfc='white',mew=2,label='Random 80/20')
    ax.plot(c.index,c.values,'-o',color=CHR,lw=2.6,ms=7,label='Chronological')
    ax.text(15,(c.loc[18]+r.loc[18])/2,'ΔR$^{2}$',fontsize=12.5,color=INK,ha='center')
    ax.set_xticks([1,6,12,18,24]); ax.set_ylim(0.3,1.0); style(ax,'Horizon (h)','Test R$^{2}$',ny=5); ax.legend(loc='lower left',fontsize=11.5)
def scatter_panel(fig,ss,key,lab):
    g=ss.subgridspec(1,2,width_ratios=[5,1.0],wspace=0.04); ax=fig.add_subplot(g[0]); axr=fig.add_subplot(g[1])
    te=LP[key]; tr=LP[key+'_train']; y,p=te.y*CFM,te.p*CFM
    ax.scatter(tr.y*CFM,tr.p*CFM,marker='^',s=22,c=TRAIN,alpha=0.62,edgecolors=INK,linewidths=0.2,label='Train',zorder=2)
    ax.scatter(y,p,marker='o',s=20,facecolors='none',edgecolors=TEST,linewidths=0.8,alpha=0.78,label='Test',zorder=3)
    lim=[0,430]; ax.plot(lim,lim,color=INK,lw=1.4,ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
    m=metrics(te.y.values,te.p.values)
    ax.text(0.035,0.96,f"{lab}\nR$^{{2}}$ {m['R2']:.3f}\nRMSE {m['RMSE']*CFM:.1f}",transform=ax.transAxes,va='top',fontsize=10.5,color=INK,bbox=dict(fc='white',ec='none',alpha=0.85))
    style(ax,'Observed (m$^{3}$ h$^{-1}$)','Predicted (m$^{3}$ h$^{-1}$)',nx=4,ny=5); lg=ax.legend(loc='lower right',fontsize=11,markerscale=1.4,handletextpad=0.4); [hd.set_alpha(1) for hd in lg.legend_handles]
    res=(p-y); axr.hist(res,bins=40,orientation='horizontal',color=TEST,alpha=0.6,ec='white',lw=0.3); axr.axhline(0,color=INK,lw=1.2)
    axr.set_ylim(-200,200); style(axr,None,None,tick=10.5,nx=2,ny=5); axr.set_xticks([]); axr.spines['bottom'].set_visible(False)
    axr.yaxis.set_label_position('right'); axr.yaxis.tick_right(); axr.spines['right'].set_visible(True); axr.spines['left'].set_visible(False)
    axr.tick_params(axis='y',right=True,left=False); axr.set_ylabel('Residual\n(m$^{3}$ h$^{-1}$)',fontsize=11.5,labelpad=2)
def pc(fig,ss):
    g=ss.subgridspec(1,2,wspace=1.2); scatter_panel(fig,g[0],'random','Random'); scatter_panel(fig,g[1],'chronological','Chronological')
def pd_(fig,ss):
    ax=fig.add_subplot(ss); d=LA[LA.h==24].set_index(LA[LA.h==24].setting.str[0]).R2
    vals={('Hourly value at t+24 h','Chronological'):d['A'],('Hourly value at t+24 h','Random'):d['B'],('Mean over next 24 h','Chronological'):d['E'],('Mean over next 24 h','Random'):d['F']}
    x=np.arange(2); w=0.36
    for k,(sp,c) in enumerate([('Chronological',CHR),('Random',RND)]):
        v=[vals[(t,sp)] for t in ['Hourly value at t+24 h','Mean over next 24 h']]
        ax.bar(x+(k-0.5)*w,v,w*0.92,color=c,ec=INK,lw=0.9,label=sp)
        for xi,vi in zip(x+(k-0.5)*w,v): ax.text(xi,vi+0.015,f'{vi:.2f}',ha='center',va='bottom',fontsize=11.5,color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['Value at t + 24 h','Mean over next 24 h']); ax.set_ylim(0,1.3)
    style(ax,'Target definition','Test R$^{2}$',ny=5); ax.legend(loc='upper center',fontsize=11.5,ncol=2)
