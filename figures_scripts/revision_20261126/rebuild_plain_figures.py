"""Rebuild Figs. 3-6 and S11 with plain (non-bold), horizontal text; data, layout and colors unchanged.

Uses the earlier builders in this package:
  Fig. 3, S11 : figures_scripts/layout_20261014/make_figures.py (plates Fig4_information_value, Fig6_trajectories_uncertainty)
  Fig. 4, 6   : figures_scripts/submission_20261018/rebuild_figures.py
  Fig. 5      : the same builder with the segment-respecting response intervals (as in redraw_fig5.py)
Changes: tick, axis, legend and annotation text set to normal weight; x tick labels horizontal; axis labels
'Outlet flow' -> 'Metered flow' and 'Fulfilled flow' -> 'Fulfilled fraction'. Panel letters stay bold.
Run from the package root: python figures_scripts/revision_20261126/rebuild_plain_figures.py [3 4 5 6 S11]
"""
from pathlib import Path
import sys, json, string, importlib.util
import matplotlib
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
W = HERE.parents[1]  # package root
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
FU = W / 'figures_scripts/submission_20261018'
RELABEL = {'Outlet flow': 'Metered flow', 'Fulfilled flow': 'Fulfilled fraction'}
REPORT = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def plain(fig, axes, stem):
    """Normal-weight, horizontal text; shrink overlapping horizontal x tick labels."""
    renderer = None
    for a in axes:
        for t in a.findobj(Text):
            t.set_fontweight('normal')
        for lab in (a.xaxis.label, a.yaxis.label):
            s = lab.get_text()
            for old, new in RELABEL.items():
                if old in s:
                    lab.set_text(s.replace(old, new))
        rotated = any(abs(t.get_rotation()) > 1 for t in a.get_xticklabels())
        if rotated:
            a.tick_params(axis='x', labelrotation=0)
            for t in a.get_xticklabels():
                t.set_rotation(0); t.set_ha('center'); t.set_rotation_mode('default')
            fig.canvas.draw(); renderer = fig.canvas.get_renderer()
            labels = [t for t in a.get_xticklabels() if t.get_text()]
            size = labels[0].get_fontsize() if labels else 12
            boxes = [t.get_window_extent(renderer) for t in labels]
            gaps = [boxes[i + 1].x0 - boxes[i].x1 for i in range(len(boxes) - 1)]
            mode = 'horizontal'
            if min(gaps, default=10) < 2:
                # Long categorical labels (HSW lag windows): two staggered rows instead of tilted text.
                ticks = a.get_xticks(); texts = [t.get_text() for t in a.get_xticklabels()]
                a.set_xticks(ticks, [('\n' if i % 2 else '') + s for i, s in enumerate(texts)])
                size = min(size, 12)
                a.tick_params(axis='x', labelsize=size, labelrotation=0)
                mode = 'staggered'
            REPORT.setdefault(stem, []).append({'ticks': [t.get_text() for t in labels], 'fontsize': size, 'mode': mode})


def patch(layout, stem_hint=''):
    fit_group = layout.fit_group

    def draw_panel(fig, func, cell, letter, standalone=False):
        x, y, w, h = cell
        before = set(fig.axes)
        ax = fig.add_axes([x + .15 * w, y + .20 * h, w * .70, h * .66])
        func(fig, ax)
        axes = [a for a in fig.axes if a not in before]
        plain(fig, axes, stem_hint or 'panel')
        top = .34 / fig.get_figheight(); pad = .045 / fig.get_figwidth()
        bounds = fit_group(fig, axes, [x + pad, y + .025 / fig.get_figheight(), w - 2 * pad, h - top - .025 / fig.get_figheight()])
        fig.text(x + .015 / fig.get_figwidth(), y + h, letter, fontsize=22 if standalone else 26, fontweight='bold', va='top', ha='left')
        return ax, axes, bounds

    layout.draw_panel = draw_panel
    layout.OUT = OUT
    return layout


rf = load('rebuild_figures', FU / 'rebuild_figures.py')
rf.OUT = OUT
patch(rf.layout, 'layout')
_module = rf.module
def module(name, path):
    m = _module(name, path)
    if hasattr(m, 'layout'):
        patch(m.layout, name)
    return m
rf.module = module

targets = sys.argv[1:] or ['3', '4', '5', '6', 'S11']
for t in targets:
    if t == '3':
        rf.layout.save_figure_with_subfigures('Fig4_information_value', rf.layout.PLATES['Fig4_information_value'])
    elif t == 'S11':
        rf.layout.save_figure_with_subfigures('Fig6_trajectories_uncertainty', rf.layout.PLATES['Fig6_trajectories_uncertainty'])
    elif t == '4':
        rf.figure4()
    elif t == '5':
        rf.response = rf.pd.read_pickle(rf.ctx.RES / 'figure_intervals_segmented/response_pointwise_intervals.pkl')
        rf.figure5()
    elif t == '6':
        rf.figure6()
    print('built', t, flush=True)
(HERE / 'plain_text_report.json').write_text(json.dumps(REPORT, indent=1, ensure_ascii=False), encoding='utf-8')
