"""Replot the original Fig. 3 with compact panel spacing; preserve all data."""
from pathlib import Path
import sys,os,importlib.util,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1]  # package root
for q in [R/'figures_scripts/plotting']:
    sys.path.insert(0,str(q))
os.environ.update(REVIEW_PROJECT_ROOT=str(R),REVIEW_FIGURE_OUTPUT=str(P/'compact_figures'),REVIEW_FIGURE_QA=str(P/'compact_figure_QA'))
import figcommon as fc
import numpy as np
from ml_publication_style import save_figure_with_subfigures
source=R/'figures_scripts/plotting/fig2_temporal_structure.py'
spec=importlib.util.spec_from_file_location('temporal_source',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_pb=m.pb;original_pc=m.pc;original_pe=m.pe
def pb(fig,ss):
    n=len(fig.axes);original_pb(fig,ss);ax=fig.axes[n]
    if not getattr(fig,'_is_standalone',False):
        box=ax.get_position().frozen();ax.set_position([box.x0,box.y0,box.width+.030,box.height]);ax.set_in_layout(True)
def pc(fig,ss):
    # Reuse the exact data builder; reposition its two colorbars above the maps.
    n=len(fig.axes);original_pc(fig,ss);axes=fig.axes[n:]
    slot=ss.get_position(fig)
    for ax,cb in [(axes[0],axes[1]),(axes[2],axes[3])]:
        box=ax.get_position().frozen()
        cb_height=6/(72*fig.get_figheight());gap=8/(72*fig.get_figheight())
        ax.set_position([box.x0,slot.y0,box.width,slot.height-cb_height-gap]);ax.set_in_layout(True)
        cb.set_position([box.x0,slot.y1-cb_height,box.width,cb_height]);cb.set_in_layout(True)
        cb.xaxis.set_ticks_position('top');cb.xaxis.set_label_position('top')
        cb.xaxis.labelpad=3;cb.tick_params(labelsize=10,length=3,pad=2)
def pe(fig,ss):
    # The large internal gap is layout-only; unchanged original artist arrays.
    n=len(fig.axes);original_pe(fig,ss);a,b=fig.axes[n:n+2]
    aa,bb=a.get_position().frozen(),b.get_position().frozen();gap=.055
    b.set_position([aa.x1+gap,bb.y0,bb.width*1.5,bb.height]);b.set_in_layout(True)
    a.xaxis.labelpad=4;b.xaxis.labelpad=4
def layout(fig):
    g=fc.GridSpec(3,1,figure=fig,height_ratios=[.80,1.08,1.40],hspace=.55,left=.08,right=.96,bottom=.08,top=.94)
    mid=g[1].subgridspec(1,3,width_ratios=[1,1.32,1],wspace=.39)
    last=g[2].subgridspec(1,2,width_ratios=[1.06,1],wspace=.32)
    return {'a':g[0],'b':mid[0],'c':mid[1],'d':mid[2],'e':last[0],'f':last[1]}
panels=[(key,pb if key=='b' else pc if key=='c' else pe if key=='e' else fn,size) for key,fn,size in m.panels]
# Compare scientific artist values before/after the revised layout on fresh figures.
def signature(fig):
    values=[]
    for ax in fig.axes:
        for line in ax.lines:
            for a in [line.get_xdata(orig=False),line.get_ydata(orig=False)]:values.append(np.asarray(a).tobytes())
        for c in ax.collections:
            if c.get_array() is not None:values.append(np.asarray(c.get_array()).tobytes())
            for path in c.get_paths():values.append(np.asarray(path.vertices).tobytes())
    return hashlib.sha256(b''.join(sorted(values))).hexdigest()
fig=fc.plt.figure(figsize=(10.725,9.9));fig._is_standalone=False;slots=layout(fig)
for key,fn,_ in m.panels:fn(fig,slots[key])
before=signature(fig);fc.plt.close(fig)
fig=fc.plt.figure(figsize=(10.725,9.9));fig._is_standalone=False;slots=layout(fig)
for key,fn,_ in panels:fn(fig,slots[key])
after=signature(fig);fc.plt.close(fig);assert before==after,'Data artists changed'
original_save=fc.save_export
def save(fig,folder,stem,preview=False):
    original_save(fig,folder,stem,preview)
    # Shared publication exporter is also used for a native master and axes export.
    if preview:save_figure_with_subfigures(fig,P/'compact_figures'/'native_export',stem,dpi=600,close=False)
fc.save_export=save
fc.render('Fig2_temporal_structure','Fig2_temporal_structure',panels,(19.5,18),layout)
(P/'compact_figure_data_check.json').write_text(json.dumps({'source':source.name,'before':before,'after':after,'scientific_artist_values_unchanged':True},indent=2),encoding='utf8')
