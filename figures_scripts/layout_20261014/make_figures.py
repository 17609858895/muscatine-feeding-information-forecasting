"""Recompose retained biogas results into compact, three-column scientific plates."""
from pathlib import Path
import sys,os,json,string,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];WORKSPACE=ROOT.parent
sys.path.insert(0,str(ROOT/'figures_scripts/revision_20261013'))
import make_figures as source
# The legacy module changes its process cwd while reading retained research data.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import AsinhNorm,LinearSegmentedColormap,to_hex,to_rgba
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection,PolyCollection,LineCollection
from matplotlib.ticker import FixedLocator,FixedFormatter
from ml_publication_style import configure_publication_style,style_axis,numeric_ticks
from scientific_palettes import get_palette
OUT=HERE/'figures_ready';OUT.mkdir(exist_ok=True)
DPI=int(os.environ.get('LAYOUT_DPI','600'))
PREVIEW='--preview'in sys.argv
C=get_palette('au_nimno_soft_8',n=8);INK='#222222'
configure_publication_style()
plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.labelweight':'normal','xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':12,'axes.linewidth':1.15,'xtick.major.width':1.1,'ytick.major.width':1.1,'xtick.major.size':4.5,'ytick.major.size':4.5,'pdf.fonttype':42})

def style(ax,x='',y='',title=None):
    numeric_ticks(ax,x=not isinstance(ax.xaxis.get_major_locator(),(FixedLocator,mdates.DateLocator)),y=not isinstance(ax.yaxis.get_major_locator(),FixedLocator),n=4)
    style_axis(ax,x.replace('m³ h⁻¹','m$^{3}$ h$^{-1}$'),y.replace('m³ h⁻¹','m$^{3}$ h$^{-1}$'),tick_size=14,label_size=16)
    ax.xaxis.label.set_fontweight('normal');ax.yaxis.label.set_fontweight('normal')
    ax.xaxis.labelpad=5;ax.yaxis.labelpad=5
    ax.tick_params(length=4.5,width=1.1)
    for side in ['left','bottom']:ax.spines[side].set_linewidth(1.15)
    ax.set_title('')

def legend(ax,ncol=1,loc='upper left',anchor=None,size=9.5):
    kw=dict(loc=loc,ncol=ncol,fontsize=size*1.3,handlelength=1.4,columnspacing=.65,handletextpad=.35,borderaxespad=.3,labelspacing=.3,frameon=False)
    if anchor is not None:kw['bbox_to_anchor']=anchor
    return ax.legend(**kw)

source.style=style
source.legend=lambda ax,ncol=1:legend(ax,ncol,loc='lower left',anchor=(0,1.01))
source.line=lambda ax,x,y,color,label,marker='o',ls='-':ax.plot(x,y,color=color,lw=1.7,marker=marker,ms=4.8,mfc='white',mew=1.2,label=label,ls=ls)

NORM=AsinhNorm(linear_width=1,vmin=-1,vmax=80)
CMAP=LinearSegmentedColormap.from_list('signed_loss',[(0,C[1]),(float(NORM(0)),'white'),(1,C[0])])

def heat(fig,ax,reg):
    d=source.A[(source.A.regime==reg)&(source.A.window=='all')&(source.A.mapping=='complete')].pivot(index='group',columns='h',values='mean')
    gs=[g for g in source.groups if g in d.index];z=d.reindex(gs)[source.HS]
    im=ax.imshow(z,aspect='auto',cmap=CMAP,norm=NORM,interpolation='nearest')
    ax.set_xticks(range(9),source.HS);ax.set_yticks(range(len(gs)),[source.SHORT[source.groups.index(g)]for g in gs]);style(ax,'Horizon (h)','')
    ax.tick_params(axis='x',labelsize=12.5,labelrotation=45)
    ticks=[-1,0,1,5,20,80]
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.035)
    cb.locator=FixedLocator(ticks);cb.formatter=FixedFormatter([str(t)for t in ticks]);cb.update_ticks()
    cb.set_label('ΔRMSE (m$^3$ h$^{-1}$)',fontsize=13)
    cb.ax.tick_params(labelsize=11.7,length=3,width=.8)

def refit(fig,ax):
    source.information_b(fig,ax);legend(ax,3,loc='lower left',anchor=(0,1.01),size=9)

def anchor(fig,ax):
    source.information_d(fig,ax);ax.set_ylabel('Gas-history ΔRMSE\n(m$^3$ h$^{-1}$)');legend(ax,loc='upper right',size=9)

def lag_heat(fig,ax):
    source.lag_c(fig,ax)
    for a in fig.axes:
        if a is not ax and a.get_label()=='<colorbar>':
            a.tick_params(labelsize=11.7,length=3);a.yaxis.label.set_size(13)

def lag_time(fig,ax):
    source.lag_d(fig,ax)
    ax.set_ylabel('Day-ahead ΔRMSE\n(m$^3$ h$^{-1}$)')
    legend(ax,loc='upper left',anchor=(.01,.93),size=9.5)

def skill(fig,ax):
    source.benchmark_a(fig,ax)
    handles=[Line2D([],[],color=c,lw=1.7,marker='o',mfc='white',label=n)for n,c in [('LGB-Δ',C[0]),('Cat-Δ',C[4]),('Ensemble',C[1])]]
    ax.legend(handles=handles,loc='lower right',fontsize=12.1,frameon=False,handlelength=1.5,borderaxespad=.4,labelspacing=.3)

def paired(fig,ax):
    source.benchmark_b(fig,ax);ax.set_ylabel('Feed − history RMSE\n(m$^3$ h$^{-1}$)');legend(ax,loc='lower left',size=9.3)

def ablation(fig,ax):
    d=source.pd.read_csv(source.OUT/'ablation_configuration_register.csv');d=d[d.h==24]
    colors=[C[3],C[0],C[2],C[4],C[1]]
    ax.vlines(np.arange(5),0,d.Skill,colors=colors,lw=3)
    ax.scatter(np.arange(5),d.Skill,c=colors,s=64,edgecolor=INK,lw=.55,zorder=3)
    ax.set_xticks(range(5),['A1','A2','A3','A4','A5']);style(ax,'Configuration','Day-ahead skill');ax.set_ylim(0,.46);ax.set_xlim(-.45,4.45)

def matched(fig,ax):
    d=source.pd.read_csv(source.OUT/'matched_MLP_metrics.csv');d=d[d.h==24]
    selections=[('past',False),('past',True),('feed',False),('feed',True)]
    values=[[d[(d.model==m)&(d.regime==r)&(d.anchor==a)].RMSE.iloc[0]for r,a in selections]for m in ['Matched LightGBM','Matched MLP']]
    for j,(v,w)in enumerate(zip(*values)):ax.plot([v,w],[j,j],color=C[3],lw=2,zorder=1)
    for v,c,label,mk in zip(values,[C[0],C[1]],['LightGBM','MLP'],['o','D']):ax.scatter(v,range(4),c=c,s=48,marker=mk,edgecolor=INK,lw=.45,label=label,zorder=3)
    ax.set_yticks(range(4),['History · level','History · delta','Feed · level','Feed · delta']);ax.invert_yaxis()
    style(ax,'Day-ahead RMSE (m³ h⁻¹)','');legend(ax,2,loc='lower left',anchor=(0,1.01),size=9.3)

def feed_wape(fig,ax):
    source.schedule_a(fig,ax)
    for l,label in zip(ax.lines,['Weekly','Four-week median']):l.set_label(label)
    legend(ax,loc='upper right',size=9)

def feed_forecast(fig,ax):
    source.schedule_b(fig,ax)
    for l,label in zip(ax.lines,['History','Weekly','Four-week median','Recorded']):l.set_label(label)
    legend(ax,2,loc='upper left',size=8.8)

def trajectory(fig,ax):
    d=source.FP['pred'][24]['test'];t=d.index+source.pd.Timedelta(hours=24)
    ax.plot(t,d.y*source.CFM,color=C[0],lw=1.15,label='Observed')
    ax.plot(t,d['PG-HQ ensemble [feed]']*source.CFM,color=C[1],lw=1.05,label='Feed ensemble')
    style(ax,'Target date','Outlet flow (m³ h⁻¹)');ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set_ylim(-8,445);ax.margins(x=.005);legend(ax,2,loc='upper right',size=10)

def trajectory_zoom(fig,ax):
    source.trajectory_b(fig,ax)
    for l in ax.lines:
        if l.get_label()=='Observed':l.set_color(C[0]);l.set_linewidth(1.3)
    # All three visual objects have explicit legend keys.
    ax.lines[-1].set_label('Forecast');ax.collections[0].set_label('90% interval')
    ax.set_ylim(100,515)
    legend(ax,2,loc='upper left',size=8.7)

def coverage(fig,ax):
    source.trajectory_c(fig,ax);legend(ax,2,loc='upper left',size=9.5)

def width(fig,ax):
    source.trajectory_d(fig,ax);ax.set_ylabel('Mean interval width\n(m$^3$ h$^{-1}$)');legend(ax,2,loc='upper left',size=9.5)

def scenario_heat(fig,ax):
    source.schedule_c(fig,ax);ax.set_xlabel('Timing shift (h)',fontsize=16)
    for a in fig.axes:
        if a is not ax and a.get_label()=='<colorbar>':
            a.tick_params(labelsize=11.7,length=3);a.yaxis.label.set_size(13)

def scenario_scatter(fig,ax):
    source.schedule_d(fig,ax)
    for c,label in zip(ax.collections,['Hourly noise','Sustained bias','Cancellation','6 h delay']):c.set_label(label)
    ax.set_ylim(44,67);legend(ax,2,loc='upper left',size=9)

def commitments(fig,ax):
    source.commitment_a(fig,ax);ax.set_xlabel('Observed shortfall (%)');ax.set_ylabel('Fulfilled flow (%)');legend(ax,loc='lower right',size=9.3)

def strict(fig,ax):
    source.commitment_b(fig,ax);ax.set_xlabel('Observed shortfall (%)');ax.set_ylabel('Fulfilled flow (%)');legend(ax,2,loc='lower left',anchor=(0,1.01),size=9.3)
    for t in ax.texts:t.set_fontsize(11.5)

def nominal(fig,ax):
    source.commitment_c(fig,ax);ax.set_xlabel('Nominal shortfall target (%)');ax.set_ylabel('Feed − history (pp)')

def realized(fig,ax):
    source.commitment_d(fig,ax);ax.set_xlabel('Matched observed shortfall (%)');ax.set_ylabel('Feed − history (pp)')

PLATES={
 'Fig4_information_value':[lambda f,a:heat(f,a,'past'),lambda f,a:heat(f,a,'feed'),refit,anchor,lag_heat,lag_time],
 'Fig5_benchmark':[skill,paired,ablation,matched,feed_wape,feed_forecast],
 'Fig6_trajectories_uncertainty':[trajectory,trajectory_zoom,coverage,width],
 'Fig8_feeding_and_supply':[scenario_heat,scenario_scatter,commitments,strict,nominal,realized],
}
AUDIT={}

def fit_group(fig,axes,box):
    """Align each panel's complete rendered footprint, including bars and legends."""
    for a in axes:
        a.set_axes_locator(None)
        if a.get_label()=='<colorbar>':a.set_box_aspect(None)
    target=Bbox.from_bounds(*box)
    for _ in range(12):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        actual=Bbox.union([a.get_tightbbox(renderer)for a in axes]).transformed(fig.transFigure.inverted())
        if max(abs(np.array(actual.extents)-np.array(target.extents)))<.0006:break
        sx,sy=target.width/actual.width,target.height/actual.height
        for a in axes:
            p=a.get_position();a.set_position([target.x0+(p.x0-actual.x0)*sx,target.y0+(p.y0-actual.y0)*sy,p.width*sx,p.height*sy])
    return list(actual.extents)

def draw_panel(fig,func,cell,letter,standalone=False):
    x,y,w,h=cell
    before=set(fig.axes)
    ax=fig.add_axes([x+.15*w,y+.20*h,w*.70,h*.66])
    func(fig,ax)
    axes=[a for a in fig.axes if a not in before]
    for a in axes:
        for t in a.get_xticklabels()+a.get_yticklabels():t.set_fontweight('bold')
    top=.34/fig.get_figheight();pad=.045/fig.get_figwidth()
    bounds=fit_group(fig,axes,[x+pad,y+.025/fig.get_figheight(),w-2*pad,h-top-.025/fig.get_figheight()])
    label=fig.text(x+.015/fig.get_figwidth(),y+h,letter,fontsize=22 if standalone else 26,fontweight='bold',va='top',ha='left')
    return ax,axes,bounds

def inspect_artists(fig,groups):
    colors=[];allowed={to_hex(c).lower()for c in C};structural={to_hex(INK),'#333943','#000000'}
    for group in groups:
        for a in group:
            if a.get_label()=='<colorbar>':continue
            for l in a.lines:
                c=to_hex(l.get_color()).lower()
                if c in structural:
                    # Only horizontal zero/coverage/reference lines may use neutral ink.
                    yy=np.asarray(l.get_ydata());assert len(yy)<=2 and np.all(yy==yy[0]),(c,l.get_label())
                else:assert c in allowed,(c,l.get_label());colors.append(c)
            for c in a.collections:
                if isinstance(c,LineCollection):
                    for rgba in c.get_edgecolors():
                        hx=to_hex(rgba).lower()
                        if rgba[3]>0:assert hx in allowed,hx;colors.append(hx)
                if isinstance(c,(PathCollection,PolyCollection)):
                    for rgba in c.get_facecolors():
                        hx=to_hex(rgba).lower()
                        if rgba[3]>0 and hx!='#ffffff':assert hx in allowed,hx;colors.append(hx)
            for p in a.patches:
                hx=to_hex(p.get_facecolor()).lower()
                if p.get_facecolor()[3]>0 and hx!='#ffffff':assert hx in allowed,hx;colors.append(hx)
            leg=a.get_legend()
            if leg is not None:
                for handle in leg.legend_handles:
                    if isinstance(handle,Line2D):
                        hx=to_hex(handle.get_color()).lower();assert hx in allowed,hx
    return sorted(set(colors))

def save_figure_with_subfigures(stem,funcs):
    """Adapted publication helper: combined exports plus independently drawn panels."""
    fig=plt.figure(figsize=(12.3,6.9));w,h=fig.get_size_inches()
    left,right,bottom,top=.10/w,.09/w,.08/h,.06/h
    gx,gy=.18/w,.16/h
    cw=(1-left-right-2*gx)/3;ch=(1-bottom-top-gy)/2
    if len(funcs)==4:
        cells=[(left,bottom+ch+gy,1-left-right,ch)]+[(left+i*(cw+gx),bottom,cw,ch)for i in range(3)]
    else:cells=[(left+(i%3)*(cw+gx),bottom+(1-i//3)*(ch+gy),cw,ch)for i in range(6)]
    groups=[];bounds=[]
    for i,(func,cell)in enumerate(zip(funcs,cells)):
        ax,aa,bb=draw_panel(fig,func,cell,string.ascii_lowercase[i]);groups.append(aa);bounds.append(bb)
    palette=inspect_artists(fig,groups)
    folder=OUT/stem;folder.mkdir(exist_ok=True)
    fig.savefig(folder/(stem+'_preview.png'),dpi=180,facecolor='white')
    if not PREVIEW:
        fig.savefig(folder/(stem+'.png'),dpi=DPI,facecolor='white')
        fig.savefig(folder/(stem+'.pdf'),metadata={'Author':'Anonymous','Creator':'Anonymous'},facecolor='white')
        sub=folder/'subfigures';sub.mkdir(exist_ok=True)
        for i,func in enumerate(funcs):
            size=(9.2,3.8)if stem=='Fig6_trajectories_uncertainty'and i==0 else (4.7,3.8)
            f=plt.figure(figsize=size);_,panel_axes,_=draw_panel(f,func,(.025,.025,.95,.95),string.ascii_lowercase[i],standalone=True)
            inspect_artists(f,[panel_axes])
            for ext in ['png','pdf']:
                kw={'metadata':{'Author':'Anonymous','Creator':'Anonymous'}}if ext=='pdf'else {}
                f.savefig(sub/f'{stem}_{string.ascii_lowercase[i]}.{ext}',dpi=DPI,facecolor='white',**kw)
            plt.close(f)
        (sub/'README.md').write_text('Each panel is rendered independently from the same retained data and plotting function as the combined figure, including its legend and colorbar.\n',encoding='utf8')
    # Every row/column shares an explicitly recalibrated outer footprint.
    AUDIT[stem]=dict(panel_count=len(funcs),canvas_inches=[w,h],panel_label_pt=26,outer_bounds=bounds,rendered_series_colors=palette,palette='au_nimno_soft_8',layout='wide top + three columns'if len(funcs)==4 else 'two rows by three columns')
    plt.close(fig);print(stem,'preview'if PREVIEW else 'complete',flush=True)

if __name__=='__main__':
    keys=[s for s in sys.argv[1:]if not s.startswith('--')]or list(PLATES)
    for key in keys:save_figure_with_subfigures(key,PLATES[key])
    auditfile=HERE/('preview_layout_audit.json'if PREVIEW else 'figure_layout_audit.json')
    previous=json.loads(auditfile.read_text(encoding='utf8'))if auditfile.exists()else {}
    auditfile.write_text(json.dumps(previous|AUDIT,ensure_ascii=False,indent=2),encoding='utf8')
