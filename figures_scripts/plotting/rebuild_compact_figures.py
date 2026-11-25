"""Restyle existing, unchanged scientific panels and recombine them compactly.

No model fitting is performed. Figure 4 stars use the verified full-family Holm
values. Figure 1 is copied unchanged only when the final package is assembled.
"""
from pathlib import Path
import os,sys,shutil,importlib.util,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('REVIEW_PROJECT_ROOT',HERE.parents[1]))  # package root
OUT=Path(os.environ.get('REVIEW_FIGURE_OUTPUT',ROOT/'figures_output'));CODE=HERE
os.environ['REVIEW_PROJECT_ROOT']=str(ROOT)
os.environ['REVIEW_FIGURE_OUTPUT']=str(OUT)
os.environ['REVIEW_FIGURE_QA']=os.environ.get('REVIEW_FIGURE_QA',str(ROOT/'figures_output'/'QA'))
os.environ['REVIEW_DM_FILE']=os.environ.get('REVIEW_DM_FILE',str(HERE/'dm_tests_holm_verified.csv'))
sys.path.insert(0,str(CODE))
import figcommon as fc
from figcommon import GridSpec

prepare_original=fc.prepare_artists
def prepare_supported_units(fig):
    for text in fig.findobj(fc.Text):
        text.set_text(text.get_text().replace('m³ h⁻¹','m$^{3}$ h$^{-1}$'))
    prepare_original(fig)
fc.prepare_artists=prepare_supported_units

footprint_original=fc.panel_footprint
def footprint_with_shared_bar(fig,axes,renderer):
    return footprint_original(fig,[a for a in axes if a.get_in_layout()],renderer)
fc.panel_footprint=footprint_with_shared_bar

def module(name):
    spec=importlib.util.spec_from_file_location(name,CODE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def grid(rows,cols,letters,hs=.42,ws=.45,heights=None,widths=None):
    def layout(fig):
        g=GridSpec(rows,cols,figure=fig,left=.08,right=.96,bottom=.09,top=.91,
                   hspace=hs,wspace=ws,height_ratios=heights,width_ratios=widths)
        return {l:g[i//cols,i%cols] for i,l in enumerate(letters)}
    return layout

def f2_layout(fig):
    g=GridSpec(3,1,figure=fig,height_ratios=[.85,1,1.4],hspace=.72,left=.08,right=.96,bottom=.08,top=.94)
    mid=g[1].subgridspec(1,3,width_ratios=[1,1.25,1],wspace=.65)
    last=g[2].subgridspec(1,2,width_ratios=[1.12,1],wspace=.50)
    return {'a':g[0],'b':mid[0],'c':mid[1],'d':mid[2],'e':last[0],'f':last[1]}

def f3_layout(fig):return grid(2,3,'abcdef',hs=.64,ws=1.05)(fig)

def f4_layout(fig):
    g=GridSpec(3,2,figure=fig,height_ratios=[1.65,1.05,.95],hspace=.62,wspace=.68,
               left=.12,right=.96,bottom=.09,top=.93)
    return {l:g[i//2,i%2] for i,l in enumerate('abcdef')}

def f5_layout(fig):
    g=GridSpec(3,1,figure=fig,height_ratios=[1.2,1.10,.95],hspace=.74,left=.08,right=.97,bottom=.09,top=.92)
    box=g[0].get_position(fig)
    top=GridSpec(1,1,figure=fig,left=box.x0,right=box.x1,bottom=box.y0-.045,top=box.y1-.045)
    last=g[2].subgridspec(1,3,wspace=.55)
    return {'a':top[0],'b':g[1],'c':last[0],'d':last[1],'e':last[2]}

def f7_layout(fig):
    g=GridSpec(2,6,figure=fig,height_ratios=[1,1.05],hspace=.80,wspace=.75,left=.08,right=.96,bottom=.23,top=.88)
    return {'a':g[0,:2],'b':g[0,2:4],'c':g[0,4:],'d':g[1,:3],'e':g[1,3:]}

def f8_layout(fig):
    g=GridSpec(2,3,figure=fig,height_ratios=[.70,1.1],hspace=.65,wspace=.65,left=.08,right=.95,bottom=.11,top=.89)
    return {'a':g[0,:],'b':g[1,:2],'c':g[1,2]}

def f9_layout(fig):
    g=GridSpec(2,2,figure=fig,height_ratios=[.95,1],hspace=.38,wspace=.45,left=.10,right=.97,bottom=.10,top=.90)
    return {'a':g[0,:],'b':g[1,0],'c':g[1,1]}

def shared_top(builder,ncol=3):
    def panel(fig,ss):
        n=len(fig.legends);builder(fig,ss)
        if not getattr(fig,'_is_standalone',False):
            for lg in list(fig.legends)[n:]:
                lg.set_bbox_to_anchor((.5,1.055),transform=fig.transFigure)
    return panel

selection=sys.argv[1:] or ['2','3','4','5','6','7','8','9','S1','S2','S3','S4','S5','S6','S7','S8','S9']
configs=[]
if '2' in selection:
    m=module('fig2_temporal_structure');configs.append(('Fig3_temporal_structure',m.panels,(19.5,18),f2_layout))
if any(k in selection for k in ['3','7','8']):
    b=module('current_figure_builders');sp=module('supplement_panels')
    if '3' in selection:
        def comparison(h,legend):
            core=sp.s9_one(h,legend,True)
            def panel(fig,ss):
                before=len(fig.legends)
                core(fig,ss)
                for lg in list(fig.legends)[before:]:lg.set_bbox_to_anchor((.5,.015),transform=fig.transFigure)
            return panel
        def process_line(core,shared=False):
            def panel(fig,ss):
                n=len(fig.axes);core(fig,ss);ax=fig.axes[n]
                if not getattr(fig,'_is_standalone',False):
                    handles,labels=ax.get_legend_handles_labels();ax.get_legend().remove()
                    if shared:fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,fontsize=10.5)
            return panel
        panels=[('a',process_line(b.a3,True),(7,5)),('b',process_line(b.b3),(7,5)),('c',b.c3,(7,5))]+[(l,comparison(h,l=='e'),(7,6)) for l,h in zip('def',[1,6,24])]
        configs.append(('Fig4_information_value',panels,(21,12.8),f3_layout))
    if '7' in selection:
        def first_ale(k):
            core=b.ale_feature(k)
            def panel(fig,ss):
                n=len(fig.axes);core(fig,ss);ax=fig.axes[n]
                ax.set_xlabel(['HSW to D2, current\n(m³ h⁻¹)','HSW to D1, 48–71 h\n(m³ h⁻¹)','Cover height, D1\n(m)'][k],fontsize=14)
                ax.set_ylabel('ALE (m³ h⁻¹)',fontsize=14)
            return panel
        def interaction(h,last=False):
            def panel(fig,ss):
                standalone=getattr(fig,'_is_standalone',False)
                if standalone:
                    g=ss.subgridspec(1,2,width_ratios=[1,.035],wspace=.12);ax=fig.add_subplot(g[0])
                else:ax=fig.add_subplot(ss)
                data=b.R2D[h];A=data['A'][1:,1:]*fc.CFM
                vm=max(np.abs(b.R2D[j]['A']).max() for j in [6,24])*fc.CFM
                im=ax.pcolormesh(np.arange(A.shape[1]+1)-.5,np.arange(A.shape[0]+1)-.5,A,cmap=fc.cmap_div(),vmin=-vm,vmax=vm,shading='flat',rasterized=False)
                ax.set_xlim(-.5,A.shape[1]-.5);ax.set_ylim(-.5,A.shape[0]-.5)
                ax.set_xticks(np.arange(A.shape[1])[::2]);ax.set_xticklabels([f'{v:.2f}' for v in (data['q2'][1:]*b.FT)[::2]],rotation=30,ha='right')
                ax.set_yticks(np.arange(A.shape[0])[::2]);ax.set_yticklabels([f'{v:.1f}' for v in (data['q1'][1:]*fc.GPM)[::2]])
                fc.style(ax,'Cover height, D1 (m)','HSW to D2 (m³ h⁻¹)',label=14,tick=12)
                for s in ax.spines.values():s.set_visible(True)
                ax.text(.98,1.025,f'h = {h} h',transform=ax.transAxes,ha='right',va='bottom',fontsize=12)
                if standalone:fc.cbar(fig,im,fig.add_subplot(g[1]),'2-D ALE (m³ h⁻¹)',size=11)
                elif last:
                    cax=fig.add_axes([.29,.055,.45,.017]);cax.set_in_layout(False)
                    cb=fc.cbar(fig,im,cax,'2-D ALE (m³ h⁻¹)',orientation='horizontal',size=11);cb.set_ticks([-.5,0,.5])
            return panel
        panels=[(l,first_ale(k),(8,5)) for k,l in enumerate('abc')]+[(l,interaction(h,l=='e'),(8,5)) for l,h in zip('de',[6,24])]
        configs.append(('Fig8_storage_feed_response',panels,(20,13),f7_layout))
    if '8' in selection:
        def detection(fig,ss):
            n=len(fig.axes);b.pd8(fig,ss);ax=fig.axes[n]
            ax.set_xlabel('High-flow detection, 24 h',fontsize=16)
            for k,text in enumerate(ax.texts):
                value=float(text.get_text());series=k//3
                text.set_y(value-.045 if series==1 else value+(.035 if series==0 else .07))
                text.set_va('top' if series==1 else 'bottom')
                text.set_fontsize(8.5 if getattr(fig,'_is_standalone',False) else 10)
            h,l=ax.get_legend_handles_labels();l=[x.replace('Feed-conditioned','Feed').replace('Past-only','Past') for x in l]
            ax.legend(h,l,loc='lower center',bbox_to_anchor=(.5,1.025),ncol=3,fontsize=9,frameon=False)
        panels=[('a',b.pa8,(8,4.5)),('b',b.pb8,(9,5)),('c',detection,(6,5))]
        configs.append(('Fig9_feed_schedule_value',panels,(21,12),f8_layout))
if '4' in selection:
    m=module('fig4_benchmark')
    def compact_ranks(fig,ss,_model=m):
        m=_model
        ax=fig.add_subplot(ss);R=m.piv('RMSE')
        selected=['Persistence','AR-Ridge (gas only)','LightGBM','XGBoost','CatBoost','TCN','PatchTST','N-BEATS','LGB-Δ [past]','Cat-Δ [past]','PG-HQ ensemble [past]','LGB-Δ [feed]','Cat-Δ [feed]','PG-HQ ensemble [feed]']
        X=R.loc[selected,m.HS].values;ranks=np.apply_along_axis(m.rankdata,0,X).mean(1);order=np.argsort(ranks)
        short=lambda name:name.replace('PG-HQ ensemble','Ensemble').replace('AR-Ridge (gas only)','AR-Ridge').replace(' [past]',' (past)').replace(' [feed]',' (+feed)')
        for y,j in enumerate(order):
            c=fc.PROP if ('Δ' in selected[j] or 'PG-HQ' in selected[j]) else fc.BASE
            ax.plot([1,ranks[j]],[y,y],color=c,lw=1.2,alpha=.65)
            ax.plot(ranks[j],y,'o',ms=6.5,mfc=c if '[feed]' in selected[j] else 'white',mec=c,mew=1.6)
        ax.set_yticks(range(len(order)));ax.set_yticklabels([short(selected[j]) for j in order],fontsize=9.5)
        ax.set_ylim(len(order)-.5,-.5);ax.set_xlim(.5,14.5)
        fc.style(ax,'Mean RMSE rank',None,nx=4,tick=10,label=13.5)
        for t in ax.get_yticklabels():t.set_fontsize(9.5)
        cd=3.354*np.sqrt(14*15/(6*9))
        ax.text(.99,.96,f'Nominal CD = {cd:.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=9.5)
    panels=[(l,compact_ranks if l=='c' else builder,size) for l,builder,size in m.panels]
    configs.append(('Fig5_benchmark',panels,(19,21),f4_layout))
if '5' in selection:
    m=module('fig5_trajectories');configs.append(('Fig6_trajectories_uncertainty',m.panels,(19,19),f5_layout))
if '6' in selection:
    m=module('fig6_attribution')
    def shares(fig,ss,_model=m):
        m=_model
        g=ss.subgridspec(1,2,wspace=.12)
        for k,reg in enumerate(['past','feed']):
            ax=fig.add_subplot(g[k]);S=m.shares(reg);cols=[c for c in fc.GROUP_COL if c in S.columns]
            ax.stackplot(m.HS,[S[c].values for c in cols],colors=[fc.GROUP_COL[c] for c in cols],edgecolor='white',lw=.6)
            ax.set_xlim(1,24);ax.set_ylim(0,100);ax.set_xticks([1,6,12,18,24])
            fc.style(ax,'Horizon (h)','HR-PGA share (%)' if k==0 else None,ny=5)
            if k==1:ax.tick_params(labelleft=False)
            ax.text(.5,1.035,'Past-only' if reg=='past' else 'Feed-conditioned',transform=ax.transAxes,ha='center',va='bottom',fontsize=12)
        handles=[fc.Patch(fc=fc.GROUP_COL[c],ec='white',label=fc.GROUP_SHORT[c]) for c in fc.GROUP_COL]
        if getattr(fig,'_is_standalone',False):fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.025),ncol=4,fontsize=9.5)
        else:fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.045),ncol=4,fontsize=10)
    def faithful(fig,ss,_model=m):
        m=_model
        n=len(fig.axes);m.pd_(fig,ss);ax=fig.axes[n];handles,labels=ax.get_legend_handles_labels()
        ax.legend(handles,['Random','Global','Horizon-specific'],loc='lower center',bbox_to_anchor=(.5,1.025),ncol=3,fontsize=10)
    def lag(fig,ss,_model=m):
        m=_model
        from matplotlib.ticker import NullLocator
        n=len(fig.axes);m.pc(fig,ss);fig.axes[n].yaxis.set_minor_locator(NullLocator())
    def shap_shares(fig,ss,_model=m):
        _model.pf(fig,ss)
        if getattr(fig,'_is_standalone',False):
            handles=[fc.Patch(fc=fc.GROUP_COL[c],ec='white',label=fc.GROUP_SHORT[c]) for c in fc.GROUP_COL if c!='Planned feed']
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.025),ncol=4,fontsize=9.5)
    panels=[(l,shares if l=='a' else faithful if l=='d' else lag if l=='c' else shap_shares if l=='f' else builder,size) for l,builder,size in m.panels]
    configs.append(('Fig7_horizon_resolved_attribution',panels,(20,18),grid(3,2,'abcdef',.76,.70,heights=[1,1,1.15])))
if '9' in selection:
    m=module('fig9_firm_supply');configs.append(('Fig10_firm_supply',m.panels,(18,12.5),f9_layout))
if any(k.startswith('S') for k in selection):
    sf=module('figS_supplement');sp=sf.sp
    settings={
      'S1':((18,10.5),grid(2,2,'abcd',.47,.35)),
      'S2':((21,15),grid(4,4,'abcdefghijklmnop',.60,.56)),
      'S3':((20,9.5),grid(2,3,'abcdef',.42,.50)),
      'S4':((20,13),grid(2,2,'abcd',.52,.68)),
      'S5':((21,13),grid(2,2,'abcd',.80,.78,heights=[1.12,1])),
      'S6':((20,5.5),grid(1,3,'abc',.3,.43)),
      'S8':((20,7.3),grid(1,3,'abc',.3,.46)),
      'S9':((23,7.5),grid(1,3,'abc',.3,1.08))
    }
    def uncertainty_layout(fig):
        g=GridSpec(2,6,figure=fig,height_ratios=[1,1],hspace=.44,wspace=.88,left=.08,right=.96,bottom=.09,top=.90)
        return {'a':g[0,:3],'b':g[0,3:],'c':g[1,:2],'d':g[1,2:4],'e':g[1,4:]}
    settings['S7']=((20,11),uncertainty_layout)
    for key in [k for k in selection if k.startswith('S')]:
        stem,panels,_,_=sf.FIGS[key];size,layout=settings[key]
        if key in ['S3','S6']:
            panels=[(l,shared_top(builder),s) for l,builder,s in panels]
        configs.append((stem,panels,size,layout))
for stem,panels,size,layout in configs:
    print('Rendering',stem,flush=True)
    fc.render(stem,stem,panels,size,layout)
(Path(os.environ['REVIEW_FIGURE_QA'])/'redraw_configuration.json').write_text(json.dumps({
 'palette':fc.PALETTE_NAME,'font_scale':fc.FS,'panel_letter_pt':fc.LETTER*fc.FS,
 'figures':[{'stem':stem,'panels':[l for l,_,_ in panels],'canvas_inches':[v*fc.K for v in size]} for stem,panels,size,_ in configs],
 'fig1':'preserved unchanged','data':'existing results; no training'},ensure_ascii=False,indent=2),encoding='utf8')
