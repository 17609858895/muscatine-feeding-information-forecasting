"""Draw unchanged commitment metrics with explicit outlet-flow and nominal-coverage labels."""
from figcommon import *
R=pd.read_csv(DATA/'firm_supply.csv');R=R[R.method=='ACI']
STY={'Persistence':dict(color=P['blue'],marker='D',mfc='white',lw=1.8,ms=7),
     'Past-only ensemble':dict(color=PROP,marker='o',mfc='white',lw=1.8,ms=7),
     'Feed-conditioned ensemble':dict(color=PROP,marker='o',mfc=PROP,lw=3.2,ms=9)}
LAB={'Persistence':'Persistence','Past-only ensemble':'Past-only','Feed-conditioned ensemble':'Feed-conditioned'}

def pa(fig,ss):
    ax=fig.add_subplot(ss)
    for s in STY:
        g=R[(R.alpha==.05)&(R.strategy==s)].sort_values('h')
        ax.plot(g.h,100*g.firm_fraction,label=LAB[s],**STY[s])
    ax.set_xticks([1,6,12,18,24]);ax.set_xlim(0,25);ax.set_ylim(30,90)
    style(ax,'Horizon (h)','Fulfilled commitment\n(% of metered flow)')
    ax.set_yticks([30,40,50,60,70,80,90])
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=3,handlelength=2.2)
    ax.text(.985,.94,'Nominal 95% coverage (α = 0.05)',transform=ax.transAxes,ha='right',va='top',fontsize=LEG)

def tail(fig,ss,h):
    ax=fig.add_subplot(ss)
    for s in STY:
        g=R[(R.h==h)&(R.strategy==s)].sort_values('alpha')
        ax.plot(100*g.shortfall_rate,100*g.firm_fraction,**STY[s])
        p=g[g.alpha==.05].iloc[0]
        ax.plot(100*p.shortfall_rate,100*p.firm_fraction,ls='none',marker='s',ms=11,mfc='none',mec=INK,mew=1.2,zorder=5)
    ax.set_xscale('log');ax.set_xlim(.6,70);ax.set_ylim(10,100)
    style(ax,'Shortfall hours (%)','Fulfilled commitment (%)')
    ax.set_xticks([1,2,5,10,20,50]);ax.set_xticklabels(['1','2','5','10','20','50'])
    ax.set_yticks([20,40,60,80,100])
    ax.text(.04,.95,f'h = {h} h',transform=ax.transAxes,ha='left',va='top',fontsize=LEG+1)
    if getattr(fig,'_is_standalone',False):
        ax.legend(handles=[Line2D([],[],label=LAB[s],**STY[s]) for s in STY],loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,fontsize=10)

def pb(fig,ss):tail(fig,ss,6)
def pc(fig,ss):tail(fig,ss,24)
def layout(fig):
    g=GridSpec(2,2,figure=fig,height_ratios=[1,1.05],hspace=.57,wspace=.48,left=.13,right=.97,top=.89,bottom=.10)
    return {'a':g[0,:],'b':g[1,0],'c':g[1,1]}
panels=[('a',pa,(13,6)),('b',pb,(8,6)),('c',pc,(8,6))]
if __name__=='__main__':render('Fig10_firm_supply','Fig10_firm_supply',panels,(16.4,15),layout)
