"""Ten complementary supplementary figures, in manuscript citation order."""
import sys,shutil
from pathlib import Path
import figcommon as fc
from figcommon import *
import supplement_panels as sp
import evaluation_panels as ev

def legend_panel(builder,ncol=2):
    def panel(fig,ss):
        n=len(fig.axes);before=len(fig.legends);builder(fig,ss)
        for lg in list(fig.legends)[before:]:lg.remove()
        ax=fig.axes[n];handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.025),ncol=ncol,fontsize=11.5)
    return panel

def eval_comparison(fig,ss):
    n=len(fig.axes);ev.pa(fig,ss);ax=fig.axes[n]
    ax.set_ylim(-.5,2.6)
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=1,fontsize=11)

def mixed_grid(fig):
    g=GridSpec(2,6,figure=fig,hspace=.7,wspace=1.45,height_ratios=[1,1])
    return {'a':g[0,:3],'b':g[0,3:],'c':g[1,:2],'d':g[1,2:4],'e':g[1,4:]}

def eval_grid(fig):
    g=GridSpec(3,2,figure=fig,hspace=.65,wspace=.8)
    return {'a':g[0,0],'b':g[0,1],'c':g[1,:],'d':g[2,:]}

def mae_panel(h,legend=False):
    def panel(fig,ss):
        n=len(fig.axes);sp.s4_one(h,False)(fig,ss)
        if not sp.standalone(fig) and legend:
            sp.shared_legend(fig,fig.axes[n],ncol=2,fontsize=11.5)
            fig.legends[-1].set_bbox_to_anchor((.5,.015),transform=fig.transFigure)
    return panel

L=list('abcdefghijklmnop')
FIGS={
 'S1':('FigS1_data_quality',[('a',sp.s1a,(8,4.6)),('b',sp.s1b,(8,4.6)),('c',sp.s1c,(8,4.6)),('d',sp.s1d,(8,4.6))],(17,12),sp.grid(2,2,'abcd',hspace=.90,wspace=.36)),
 'S2':('Fig2_input_series',[(L[k],sp.s2_one(v,k>=12),(5,3)) for k,v in enumerate(sp.VARS)],(18,18),sp.grid(4,4,L,hspace=.82,wspace=.74)),
 'S3':('FigS2_lab_indicators',[(L[k],sp.s3_one(c,l,k==0),(6.5,4.5)) for k,(c,l) in enumerate(sp.S3I)]+[('f',sp.s3_scatter,(5.5,5))],(18,10),sp.grid(2,3,'abcdef',hspace=.62,wspace=.70)),
 'S4':('FigS3_evaluation_design',[('a',eval_comparison,(7,6)),('b',ev.pb,(7,5)),('c',ev.pd_,(8,5)),('d',sp.s12,(9,6))],(16,16),eval_grid),
 'S5':('FigS4_training_diagnostics',[('a',sp.s5_trace('LGB-Δ [feed]',6,True),(7,5)),('b',sp.s5_gain,(9,6)),('c',legend_panel(sp.s7_val,2),(8,5)),('d',sp.s7_seed5,(8,6))],(16,15),sp.grid(2,2,'abcd',hspace=.80,wspace=1.45)),
 'S6':('FigS5_learning_curves',[(L[k],sp.s6_one(h,k==0),(6,5)) for k,h in enumerate([1,6,24])],(19.5,5.8),sp.grid(1,3,'abc',wspace=.60)),
 'S7':('FigS6_uncertainty_and_state_errors',[('a',legend_panel(sp.s14_cov,2),(8,5)),('b',legend_panel(sp.s14_wid,2),(8,5))]+[(l,mae_panel(h,l=='c'),(6,5)) for l,h in zip('cde',[1,6,24])],(18,13),mixed_grid),
 'S8':('FigS7_PDP_vs_ALE',[(L[k],sp.s15_one(*c,True),(6,6.5)) for k,c in enumerate(sp.C15)],(19,9.5),sp.grid(1,3,'abc',wspace=.65)),
 'S9':('FigS8_SHAP_beeswarm',[(L[k],sp.s8_one(h),(7,6)) for k,h in enumerate([1,6,24])],(22,7.5),sp.grid(1,3,'abc',wspace=1.55))
}

def main(keys=None):
    for key in keys or list(FIGS):
        if key=='S10':continue
        stem,panels,size,layout=FIGS[key]
        fc.render(stem,stem,panels,size,layout,letters=True)
    if keys is None or 'S10' in keys:
        dest=fc.OUT/'FigS9_prediction_interface';dest.mkdir(parents=True,exist_ok=True)
        for ext in ['png','pdf']:
            candidates=[fc.OUT/f'assets/prediction_interface_native.{ext}',fc.ROOT/f'figures/assets/prediction_interface_native.{ext}']
            source=next(p for p in candidates if p.exists())
            shutil.copy2(source,dest/f'FigS9_prediction_interface.{ext}')
        sub=dest/'subfigures';sub.mkdir(exist_ok=True)
        for ext in ['png','pdf']:
            shutil.copy2(dest/('FigS9_prediction_interface.'+ext),sub/('FigS9_prediction_interface_a.'+ext))
        (sub/'README.md').write_text('Single interface screenshot; the standalone export retains the full native view.\n',encoding='utf8')

if __name__=='__main__':main(sys.argv[1:] or None)
