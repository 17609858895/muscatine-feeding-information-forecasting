"""Current BRT figure edits, using retained native plotting functions and data."""
from pathlib import Path
import sys, os, json, importlib.util, string
HERE=Path(__file__).resolve().parent;WORKSPACE=HERE.parents[1]  # package root
sys.dont_write_bytecode=True
sys.path.insert(0,str(WORKSPACE/'scripts/revision'))
import context as ctx
import numpy as np
import pandas as pd
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
layout=module('current_brt_layout',WORKSPACE/'figures_scripts/layout_20261014/make_figures.py')
plt=layout.plt;C=layout.C
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
layout.OUT=OUT;layout.PREVIEW=False;layout.DPI=600
from matplotlib.patches import Rectangle,Patch
from matplotlib.colors import LinearSegmentedColormap
intervals=pd.read_csv(ctx.RES/'figure_intervals/forecast_pointwise_intervals.csv')
response=pd.read_pickle(ctx.RES/'figure_intervals/response_pointwise_intervals.pkl')
def ribbon(ax,model,quantity,color,alpha=.15):
    d=intervals[(intervals.model==model)&(intervals.quantity==quantity)].sort_values('h')
    ax.fill_between(d.h,d.lower,d.upper,color=color,alpha=alpha,lw=0,zorder=1)
    for l in ax.lines:l.set_zorder(3)
def skill(fig,ax):
    layout.skill(fig,ax)
    for name,c in [('LGB-Δ',C[0]),('Cat-Δ',C[4]),('PG-HQ ensemble',C[1])]:
        for reg in ['past','feed']:ribbon(ax,f'{name} [{reg}]','Skill',c,.10)
    ax.margins(y=.10)
def feed_forecast(fig,ax):
    layout.feed_forecast(fig,ax)
    for line,name in zip(ax.lines,['LGB-Δ [past]','Weekly feed','Four-week feed median','LGB-Δ [feed]']):
        ribbon(ax,name,'RMSE',line.get_color(),.17)
    ax.margins(y=.12)
def figure4():
    layout.save_figure_with_subfigures('Fig_4',[skill,layout.paired,layout.ablation,
                                              layout.matched,layout.feed_wape,feed_forecast])

FEATS=['Q-HSW2_GPM|L0','Q-HSW1_GPM|L48-71','H-Dig1_FT|L0']
RAW=[ctx.H['Q-HSW2_GPM'],ctx.H['Q-HSW1_GPM'].shift(48).rolling(24).mean(),ctx.H['H-Dig1_FT']]
UNITS=[.227125,.227125,.3048]
XLABS=['HSW to D2, current\n(m$^3$ h$^{-1}$)','HSW to D1, 48–71 h\n(m$^3$ h$^{-1}$)','Cover height, D1\n(m)']
def first_order(k):
    def panel(fig,ax):
        raw=RAW[k].dropna().to_numpy();xmin,xmax=np.percentile(raw,[1,99]);yrange=[]
        for h,c,ls in [(1,C[0],'-'),(6,C[2],'--'),(24,C[1],':')]:
            d=response[h,FEATS[k]];q=d['q'];keep=(q>=xmin-1e-9)&(q<=xmax+1e-9)
            x=q[keep]*UNITS[k];lo=d['lower'][keep]*ctx.CFM;hi=d['upper'][keep]*ctx.CFM
            ax.fill_between(x,lo,hi,color=c,alpha=.23,lw=0)
            ax.plot(x,d['point'][keep]*ctx.CFM,ls,color=c,lw=1.7,label=f'{h} h')
            yrange.extend([*lo,*hi])
        ymin=min(-14,min(yrange)*1.15);ymax=max(14,max(yrange)*1.15)
        vv=raw[::12];vv=vv[(vv>=xmin)&(vv<=xmax)]*UNITS[k]
        ax.scatter(vv,np.full(len(vv),ymin+.035*(ymax-ymin)),marker='|',color=C[3],alpha=.75,s=14,lw=.65)
        ax.axhline(0,color=layout.INK,lw=.8,ls=':');ax.set_xlim(xmin*UNITS[k],xmax*UNITS[k]);ax.set_ylim(ymin,ymax)
        layout.style(ax,XLABS[k],'ALE (m$^3$ h$^{-1}$)')
        layout.legend(ax,3,loc='lower left',anchor=(0,1.01),size=9)
    return panel
def mixed(h):
    def panel(fig,ax):
        d=response[h,'mixed'];a=d['point']*ctx.CFM;nr,nc=a.shape
        vm=max(np.abs(response[j,'mixed']['point']).max() for j in [6,24])*ctx.CFM
        cmap=LinearSegmentedColormap.from_list('mixed_diverging',[C[0],'white',C[1]])
        im=ax.pcolormesh(np.arange(nc+1)-.5,np.arange(nr+1)-.5,a,cmap=cmap,vmin=-vm,vmax=vm)
        for i in range(nr):
            for j in range(nc):
                hatch='xx' if not d['supported'][i,j] else ('///' if d['lower'][i,j]<=0<=d['upper'][i,j] else None)
                if hatch:
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='none',edgecolor='#555555',lw=0,hatch=hatch,alpha=.30))
        ax.set_xlim(-.5,nc-.5);ax.set_ylim(-.5,nr-.5)
        ax.set_xticks(np.arange(nc)[::2],labels=[f'{v:.2f}' for v in (d['q2'][1:]*.3048)[::2]])
        ax.set_yticks(np.arange(nr)[::2],labels=[f'{v:.1f}' for v in (d['q1'][1:]*.227125)[::2]])
        layout.style(ax,'Cover height, D1 (m)','HSW to D2 (m$^3$ h$^{-1}$)')
        cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.025)
        cb.set_label('Accumulated mixed difference\n(m$^3$ h$^{-1}$)',fontsize=12)
        cb.ax.tick_params(labelsize=11)
        ax.text(.98,.97,f'{h} h',transform=ax.transAxes,ha='right',va='top',fontsize=13,
                bbox=dict(facecolor='white',edgecolor='none',pad=2,alpha=.9))
        keys=[Patch(facecolor='white',edgecolor='#555555',hatch='///',lw=.4,label='Interval includes 0'),
              Patch(facecolor='white',edgecolor='#555555',hatch='xx',lw=.4,label='Interval unavailable')]
        ax.legend(handles=keys,ncol=2,loc='lower left',bbox_to_anchor=(0,1.015),fontsize=10.5,
                  frameon=False,handlelength=1.5,columnspacing=1,borderaxespad=0)
    return panel
def figure5():
    funcs=[first_order(k) for k in range(3)]+[mixed(6),mixed(24)]
    fig=plt.figure(figsize=(12.3,7.5))
    cells=[(.008+i*.329,.525,.319,.465) for i in range(3)]+[(.008+i*.495,.015,.485,.485) for i in range(2)]
    groups=[];bounds=[]
    for i,(fn,cell) in enumerate(zip(funcs,cells)):
        _,axes,bbox=layout.draw_panel(fig,fn,cell,string.ascii_lowercase[i]);groups.append(axes);bounds.append(bbox)
    palette=layout.inspect_artists(fig,groups)
    folder=OUT/'Fig_5';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'Fig_5_preview.png',dpi=180,facecolor='white')
    fig.savefig(folder/'Fig_5.png',dpi=600,facecolor='white')
    fig.savefig(folder/'Fig_5.pdf',facecolor='white',metadata={'Author':'Anonymous','Creator':'Anonymous'})
    plt.close(fig);sub=folder/'subfigures';sub.mkdir(exist_ok=True)
    for i,fn in enumerate(funcs):
        f=plt.figure(figsize=(4.7,3.8) if i<3 else (6.4,4.7))
        _,axes,_=layout.draw_panel(f,fn,(.02,.02,.96,.96),string.ascii_lowercase[i],standalone=True)
        layout.inspect_artists(f,[axes])
        for ext in ['png','pdf']:
            opts={'metadata':{'Author':'Anonymous','Creator':'Anonymous'}} if ext=='pdf' else {}
            f.savefig(sub/f'Fig_5_{string.ascii_lowercase[i]}.{ext}',dpi=600,facecolor='white',**opts)
        plt.close(f)
    layout.AUDIT['Fig_5']=dict(panel_count=5,outer_bounds=bounds,rendered_series_colors=palette,
                               palette='au_nimno_soft_8',canvas_inches=[12.3,7.5])
    print('Fig_5 complete',flush=True)
def figure6():
    corrected=module('corrected_feeding',WORKSPACE/'figures_scripts/feeding_20261014/rebuild_figure8.py')
    lay=corrected.layout;lay.OUT=OUT
    def commitments(fig,ax):
        lay.source.commitment_a(fig,ax)
        for line in ax.lines:
            if line.get_label()=='Past':line.set_label('History')
        ax.set_xlabel('Observed shortfall (%)');ax.set_ylabel('Fulfilled flow (%)')
        lay.legend(ax,loc='lower right',size=9.3)
    lay.save_figure_with_subfigures('Fig_6',[lay.scenario_heat,corrected.scenario_scatter,
                                           commitments,lay.strict,lay.nominal,lay.realized])
    layout.AUDIT.update(lay.AUDIT)
def figureS4():
    os.environ['REVIEW_PROJECT_ROOT']=str(WORKSPACE)
    os.environ['REVIEW_FIGURE_OUTPUT']=str(OUT)
    os.environ['REVIEW_FIGURE_QA']=str(HERE/'layout_QA')
    code=WORKSPACE/'figures_scripts/plotting';sys.path.insert(0,str(code))
    import figcommon as fc
    import figS_supplement as sf
    _,panels,_,_=sf.FIGS['S5']
    core=panels[0][1]
    def renamed(fig,ss):
        n=len(fig.axes);core(fig,ss)
        for ax in fig.axes[n:]:
            for obj in [*ax.collections,*ax.lines]:
                if obj.get_label()=='Pre-specified':obj.set_label('Initial configuration')
            legend=ax.get_legend()
            if legend:
                for t in legend.get_texts():
                    if t.get_text()=='Pre-specified':t.set_text('Initial configuration')
    panels=[('a',renamed,panels[0][2]),*panels[1:]]
    def grid(fig):
        g=fc.GridSpec(2,2,figure=fig,left=.08,right=.96,bottom=.09,top=.91,
                      hspace=.80,wspace=.78,height_ratios=[1.12,1])
        return {l:g[i//2,i%2] for i,l in enumerate('abcd')}
    fc.render('Fig_S4','Fig_S4',panels,(21,13),grid)
    print('Fig_S4 complete',flush=True)
if __name__=='__main__':
    for name in sys.argv[1:] or ['4','5','6','S4']:
        {'4':figure4,'5':figure5,'6':figure6,'S4':figureS4}[name]()
    (HERE/'figure_layout_audit.json').write_text(json.dumps(layout.AUDIT,indent=2),encoding='utf8')
