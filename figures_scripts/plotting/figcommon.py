"""Shared regular-weight soft_8 style, scientific roles and independently rendered panels.

Publication helpers are adapted from the nature-research-figures skill. Figure
geometry is calibrated from rendered outer footprints on every fresh run.
"""
import os, sys, string, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, to_rgba, to_hex
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from row_spacing import pack_rows
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import Collection
import matplotlib.patheffects as pe
from scientific_palettes import get_palette, STYLE_COLORS
from ml_publication_style import configure_publication_style
from audit_palette_usage import validate_series_colors, audit_series_artists

ROOT=Path(os.environ['REVIEW_PROJECT_ROOT'])
CODE=DATA=ROOT
if (ROOT/'scripts'/'pipeline'/'common.py').exists():
    CODE,DATA=ROOT/'scripts'/'pipeline',Path(os.environ.get('MUSCATINE_RESULTS',ROOT/'results'))
sys.path.insert(0,str(CODE)); os.chdir(DATA)
OUT=Path(os.environ['REVIEW_FIGURE_OUTPUT'])
QA=Path(os.environ['REVIEW_FIGURE_QA'])
QA.mkdir(parents=True,exist_ok=True)
INK=STYLE_COLORS['ink']
PALETTE_NAME='au_nimno_soft_8'
SOFT=get_palette(PALETTE_NAME,n=8)
# Explicit role mapping. Model colours and process colours are reused only in
# different comparison contexts; no automatic cycling or invented ninth hue.
P={'blue':SOFT[0],'red':SOFT[1],'orange':SOFT[1],'teal':SOFT[2],
   'sky':SOFT[3],'lavender':SOFT[4],'rose':SOFT[5],'pink':SOFT[6],
   'mustard':SOFT[7],'slate':SOFT[0]}
GROUP_COL={'Gas history':P['blue'],'Gas storage':P['teal'],
 'Co-substrate feed':P['orange'],'Sludge feed':P['mustard'],
 'Digester thermal':P['rose'],'Plant hydraulics':P['sky'],
 'Calendar':P['lavender'],'Planned feed':P['pink']}
GROUP_SHORT={'Gas history':'Gas history','Gas storage':'Gas storage',
 'Co-substrate feed':'HSW feed','Sludge feed':'Sludge feed',
 'Digester thermal':'Thermal','Plant hydraulics':'Hydraulics',
 'Calendar':'Calendar','Planned feed':'Planned feed'}
PROP=P['red']; BASE=P['slate']; OBS=P['blue']; TRAIN=P['teal']; TEST=P['red']
SPLIT_COL={'Train':TRAIN,'Validation':P['mustard'],'Test':TEST}
ROLE_MAP={**GROUP_COL,'Proposed':PROP,'Baseline':BASE,'Observed':OBS,
          'Train':TRAIN,'Validation':SPLIT_COL['Validation'],'Test':TEST}
validate_series_colors(PALETTE_NAME,ROLE_MAP)
(QA/'series_role_map.json').write_text(json.dumps(
 {'palette':PALETTE_NAME,'colours':ROLE_MAP},indent=2),encoding='utf-8')

CFM=1.69901
GPM=0.227125
BIOGAS_LAB='Biogas (m$^{3}$ h$^{-1}$)'
def cmap_seq(c,name='seq'):
    return LinearSegmentedColormap.from_list(name,['#FFFFFF',c])
def cmap_div():
    return LinearSegmentedColormap.from_list('soft8_div',[P['blue'],'#FFFFFF',P['red']])
LABEL=16; TICK=13; LEG=12; LETTER=20
# Scale objects once at export. Avoid recursive FontProperties monkeypatches.
K,FS,LW,MS=0.55,0.86,0.85,0.90
def configure():
    configure_publication_style()
    plt.rcParams.update({'font.family':'sans-serif',
      'font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
      'axes.unicode_minus':False,'figure.dpi':110,'savefig.dpi':600,
      'font.weight':'normal','axes.labelsize':LABEL,'axes.labelweight':'normal',
      'xtick.labelsize':TICK,'ytick.labelsize':TICK,'legend.fontsize':LEG,
      'axes.linewidth':1.55,'xtick.major.size':7.2,'ytick.major.size':7.2,
      'xtick.major.width':1.55,'ytick.major.width':1.55,
      'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,
      'mathtext.default':'regular','axes.prop_cycle':matplotlib.cycler(color=SOFT)})
configure()

def style(ax,xlabel=None,ylabel=None,tick=TICK,label=LABEL,nx=None,ny=None,keep_top=False):
    ax.grid(False)
    ax.xaxis.labelpad=7; ax.yaxis.labelpad=7
    for s in ['top','right']: ax.spines[s].set_visible(keep_top)
    for s in ['bottom','left']:
        ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(1.55)
    ax.tick_params(axis='both',which='major',colors=INK,width=1.55,length=7.2,
                   direction='out',labelsize=tick,bottom=True,left=True,top=False,right=False)
    if nx: ax.xaxis.set_major_locator(MaxNLocator(nx))
    if ny: ax.yaxis.set_major_locator(MaxNLocator(ny))
    bold_ticks(ax,tick)
    if xlabel is not None: ax.set_xlabel(xlabel,fontsize=label,fontweight='normal',labelpad=5)
    if ylabel is not None: ax.set_ylabel(ylabel,fontsize=label,fontweight='normal',labelpad=5)
def bold_ticks(ax,size=TICK):
    for l in ax.get_xticklabels()+ax.get_yticklabels():
        l.set_fontweight('normal'); l.set_fontsize(size)
def cbar(fig,mappable,cax,label,orientation='vertical',size=11.5):
    cb=fig.colorbar(mappable,cax=cax,orientation=orientation)
    cb.set_label(label,fontsize=size+0.5,fontweight='normal')
    cb.ax.tick_params(labelsize=max(11,size-1),width=1.0,length=4,direction='out')
    cb.outline.set_linewidth(0.9)
    if cb.solids is not None: cb.solids.set_rasterized(False)
    return cb
def letters(): return list(string.ascii_lowercase)

def prepare_artists(fig):
    """Regular-weight scientific labels and ticks; pale data keep exact base hues."""
    allowed={c.lower() for c in SOFT}
    for text in fig.findobj(Text):
        if not getattr(text,'_soft8_scaled',False):
            text.set_fontsize(text.get_fontsize()*FS)
            text._soft8_scaled=True
        text.set_fontweight('normal')
        numeric=text.get_text().replace('.','').replace('−','').replace('-','').isdigit()
        if text.get_text() and text.get_text() not in ['*','**','***','ns'] and text.get_fontsize()>=10 and not (numeric and text.get_fontsize()<11):
            text.set_fontsize(max(9.0,text.get_fontsize()))
        if not getattr(fig,'_is_standalone',False) and text.get_text() in list('abcdefghijklmnop'): text.set_fontsize(LETTER*FS)
        try:
            if to_hex(text.get_color()).lower() in allowed: text.set_color(INK)
        except (ValueError,TypeError): pass
    for ax in fig.findobj(matplotlib.axes.Axes):
        for text in ax.get_xticklabels()+ax.get_yticklabels():
            text.set_fontweight('normal'); text.set_color(INK)
        for line in ax.lines:
            if not getattr(line,'_soft8_scaled',False):
                line.set_linewidth(line.get_linewidth()*LW)
                line.set_markersize(line.get_markersize()*MS)
                line.set_markeredgewidth(line.get_markeredgewidth()*LW)
                line._soft8_scaled=True
            try:
                rgb=np.array(to_rgba(line.get_color())[:3])
                if to_hex(rgb).lower() in allowed and np.dot(rgb,[.2126,.7152,.0722])>.74:
                    line.set_path_effects([pe.Stroke(linewidth=line.get_linewidth()+.65,
                                                   foreground=INK,alpha=.4),pe.Normal()])
            except (ValueError,TypeError): pass
        for patch in ax.patches:
            if not getattr(patch,'_soft8_scaled',False):
                patch.set_linewidth(patch.get_linewidth()*LW)
                patch._soft8_scaled=True

def panel_footprint(fig,axes,renderer):
    boxes=[]
    for ax in axes:
        if not ax.get_visible(): continue
        try:
            box=ax.get_tightbbox(renderer)
            if box is not None and np.isfinite(box.extents).all(): boxes.append(box)
        except (ValueError,TypeError): pass
        for child in ax.child_axes:
            if child.get_visible():
                box=child.get_tightbbox(renderer)
                if box is not None: boxes.append(box)
    if not boxes: return None
    return Bbox.union(boxes).transformed(fig.transFigure.inverted())

def align_footprints(fig,groups,tol=.002,iterations=20):
    """Fresh four-boundary calibration; includes labels, legends and colourbars."""
    keys=[spec.get_position(fig) for spec,_ in groups]
    before=None
    targets=None
    original_unions=[Bbox.union([a.get_position().frozen() for a in axes
                     if a.get_visible() and a.get_in_layout()]) for _,axes in groups]
    for iteration in range(iterations):
        fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        bounds=[panel_footprint(fig,axes,renderer) for _,axes in groups]
        if before is None: before=[None if b is None else list(b.extents) for b in bounds]
        if targets is None:
            # Fix the target once. Recomputing an intersection every iteration
            # squeezed plots indefinitely when labels had a fixed physical width.
            targets=[]
            for i,(slot,b) in enumerate(zip(keys,bounds)):
                if b is None: targets.append(None); continue
                values=[]
                for edge in ['x0','y0','x1','y1']:
                    cohort=[j for j,k in enumerate(keys)
                            if abs(getattr(k,edge)-getattr(slot,edge))<.012 and bounds[j] is not None]
                    vals=[getattr(bounds[j],edge) for j in cohort]
                    values.append((min(vals) if edge in ['x0','y0'] else max(vals)) if len(vals)>1
                                  else getattr(b,edge))
                targets.append(Bbox.from_extents(*values))
        # Twin axes share position state. Snapshot all axes before mutation so
        # moving one twin cannot apply the same transform twice to its sibling.
        positions={a:a.get_position().frozen() for _,axes in groups for a in axes}
        worst=0
        for (spec,axes),current,target,original in zip(groups,bounds,targets,original_unions):
            if current is None or target is None: continue
            worst=max(worst,max(abs(current.extents-target.extents)))
            visible=[a for a in axes if a.get_visible() and a.get_in_layout()]
            if not visible: continue
            union=Bbox.union([positions[a] for a in visible])
            x0=union.x0+target.x0-current.x0
            y0=union.y0+target.y0-current.y0
            x1=union.x1+target.x1-current.x1
            y1=union.y1+target.y1-current.y1
            if x1-x0<max(.035,.75*original.width) or y1-y0<max(.035,.75*original.height): continue
            sx=(x1-x0)/union.width; sy=(y1-y0)/union.height
            for ax in visible:
                p=positions[ax]
                ax.set_position([x0+(p.x0-union.x0)*sx,y0+(p.y0-union.y0)*sy,
                                 p.width*sx,p.height*sy],which='both')
                # Matplotlib set_position disables layout participation; retain
                # the authored group for subsequent calibration iterations.
                ax.set_in_layout(True)
        if worst<tol: break
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    after=[panel_footprint(fig,a,renderer) for _,a in groups]
    residuals=[]
    for edge in ['x0','x1','y0','y1']:
        for i,slot in enumerate(keys):
            cohort=[j for j,k in enumerate(keys)
                    if abs(getattr(k,edge)-getattr(slot,edge))<.012 and after[j] is not None]
            if len(cohort)>1 and i==min(cohort):
                v=[getattr(after[j],edge) for j in cohort]
                residuals.append({'edge':edge,'panels':cohort,'spread':max(v)-min(v)})
    overlaps=[]
    for i in range(len(after)):
        for j in range(i+1,len(after)):
            if after[i] is None or after[j] is None: continue
            inter=Bbox.intersection(after[i],after[j])
            if inter is not None and inter.width>.001 and inter.height>.001:
                overlaps.append({'panels':[i,j],'intersection':list(inter.extents)})
    return {'fresh_calibration':True,'iterations':iteration+1,'tolerance':tol,
            'before':before,'after':[None if b is None else list(b.extents) for b in after],
            'edge_residuals':residuals,'footprint_overlaps':overlaps,
            'passed':all(x['spread']<=tol for x in residuals) and not overlaps}

def align_columns(fig,groups,tol=.02):
    return align_footprints(fig,groups)

def panel_letters(fig,groups,keys,tol=.012):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    info=[]
    for (spec,axes),key in zip(groups,keys):
        b=panel_footprint(fig,axes,renderer)
        if b is not None: info.append((spec.get_position(fig),b,key))
    for slot,b,key in info:
        left=min(c.x0 for s,c,k in info if abs(s.x0-slot.x0)<tol)
        top=max(c.y1 for s,c,k in info if abs(s.y1-slot.y1)<tol)
        fig.text(left,top+.004,key,ha='left',va='bottom',fontsize=LETTER*FS,
                 color=INK,fontweight='normal')

def palette_audit(fig):
    """Audit categorical marks and legend proxies; numeric maps listed separately."""
    allowed={c.lower() for c in SOFT}
    series={}; unknown=[]; numeric=[]
    neutral={'#ffffff','#000000',INK.lower(),STYLE_COLORS['muted'].lower(),
             STYLE_COLORS['light'].lower(),'#808080','#b0b0b0'}
    def include(artist,colours,tag):
        colours=[to_rgba(c) for c in colours if c is not None]
        visible=[c for c in colours if c[3]>0 and to_hex(c).lower() not in neutral]
        if not visible: return
        hexes={to_hex(c).lower() for c in visible}
        if hexes<=allowed: series[tag]=artist
        else: unknown.append({'artist':tag,'colours':sorted(hexes)})
    for ai,ax in enumerate(fig.axes):
        for j,line in enumerate(ax.lines):
            cols=[]
            if line.get_linestyle() not in ['None','none','',' ']: cols.append(line.get_color())
            if line.get_marker() not in ['None','none','',' ',None]:
                fc=line.get_markerfacecolor()
                cols.append(line.get_markeredgecolor() if to_hex(fc).lower()=='#ffffff' else fc)
            try: include(line,cols,f'ax{ai}:line{j}')
            except (ValueError,TypeError): pass
        for j,patch in enumerate(ax.patches):
            if patch.get_fill(): include(patch,[patch.get_facecolor()],f'ax{ai}:patch{j}')
        for j,collection in enumerate(ax.collections):
            if collection.get_array() is not None:
                numeric.append({'artist':f'ax{ai}:collection{j}','cmap':collection.cmap.name})
            else:
                face=collection.get_facecolors()
                colours=list(face) if len(face) else list(collection.get_edgecolors())
                include(collection,colours,f'ax{ai}:collection{j}')
        for j,image in enumerate(ax.images):
            numeric.append({'artist':f'ax{ai}:image{j}','cmap':image.cmap.name})
        legend=ax.get_legend()
        if legend is not None:
            for j,a in enumerate(legend.legend_handles):
                if isinstance(a,Line2D):
                    cols=[a.get_color()] if a.get_linestyle() not in ['None','none','',' '] else []
                    if a.get_marker() not in ['None','none','',' ',None]:
                        fc=a.get_markerfacecolor()
                        cols.append(a.get_markeredgecolor() if to_hex(fc).lower()=='#ffffff' else fc)
                    try: include(a,cols,f'ax{ai}:legend{j}')
                    except (ValueError,TypeError): pass
                elif isinstance(a,Patch): include(a,[a.get_facecolor()],f'ax{ai}:legend{j}')
    for li,legend in enumerate(fig.legends):
        for j,a in enumerate(legend.legend_handles):
            if isinstance(a,Line2D):
                cols=[a.get_color()] if a.get_linestyle() not in ['None','none','',' '] else []
                if a.get_marker() not in ['None','none','',' ',None]:
                    fc=a.get_markerfacecolor()
                    cols.append(a.get_markeredgecolor() if to_hex(fc).lower()=='#ffffff' else fc)
                try: include(a,cols,f'figurelegend{li}:handle{j}')
                except (ValueError,TypeError): pass
            elif isinstance(a,Patch): include(a,[a.get_facecolor()],f'figurelegend{li}:handle{j}')
    result=audit_series_artists(PALETTE_NAME,series) if series else {'passed':True,'checks':[]}
    result['unregistered_non_neutral_colours']=unknown
    result['numeric_maps']=numeric
    result['passed']=result['passed'] and not unknown
    return result

def save_export(fig,folder,stem,preview=False):
    meta={'Author':'Anonymous','Creator':'Anonymous','Title':stem}
    fig.savefig(folder/f'{stem}.png',dpi=(160 if os.environ.get('REVIEW_PREVIEW_ONLY')=='1' else 600),bbox_inches='tight',facecolor='white',metadata=meta)
    if os.environ.get('REVIEW_PREVIEW_ONLY')!='1': fig.savefig(folder/f'{stem}.pdf',bbox_inches='tight',facecolor='white',metadata=meta)
    if preview: fig.savefig(folder/f'{stem}_preview.png',dpi=130,bbox_inches='tight',
                             facecolor='white',metadata=meta)

def render(stem,folder,panels,figsize,layout,align=True,letters=True):
    """Independent subplot rendering uses the same scientifically unchanged builder."""
    folder=OUT/folder; (folder/'subfigures').mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(figsize[0]*K,figsize[1]*K))
    fig._is_standalone=False; specs=layout(fig); groups=[]
    for key,builder,_ in panels:
        n0=len(fig.axes); builder(fig,specs[key]); groups.append((specs[key],fig.axes[n0:]))
    prepare_artists(fig)
    alignment=align_footprints(fig,groups) if align else {'fresh_calibration':False,'exempt':'schematic'}
    if letters=='inside':
        for (spec,axes),(key,_,_) in zip(groups,panels):
            axes[0].text(0,1.03,key,transform=axes[0].transAxes,ha='left',va='bottom',
                         fontsize=LETTER*FS,fontweight='normal',color=INK)
    elif letters and len(panels)>1: panel_letters(fig,groups,[k for k,_,_ in panels])
    row_spacing=pack_rows(fig,groups,panel_footprint,alignment)
    typography=[{'text':t.get_text(),'size':t.get_fontsize(),'weight':t.get_fontweight()} for t in fig.findobj(Text) if t.get_text()]
    audit=palette_audit(fig); save_export(fig,folder,stem,preview=True); plt.close(fig)
    lines=['# Subfigures','','Each panel is rendered independently from the same final panel builder, not cropped.','']
    standalone=[]
    for key,builder,size in panels:
        fs=plt.figure(figsize=(size[0]*K,size[1]*K)); fs._is_standalone=True
        gs=GridSpec(1,1,figure=fs,left=.16,right=.95,bottom=.15,top=.95)
        builder(fs,gs[0]); prepare_artists(fs); sa=palette_audit(fs)
        save_export(fs,folder/'subfigures',f'{stem}_{key}')
        plt.close(fs); lines.append(f'- {key}: {stem}_{key}.png, {stem}_{key}.pdf')
        standalone.append({'panel':key,'palette_audit':sa})
    (folder/'subfigures'/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (QA/f'{stem}.json').write_text(json.dumps({'figure':stem,'palette':PALETTE_NAME,
       'alignment':alignment,'row_spacing':row_spacing,'palette_audit':audit,'standalone':standalone,'typography':typography},indent=2),encoding='utf-8')
    print('saved',folder/stem,'alignment:',alignment.get('passed','exempt'),'palette:',audit['passed'])
