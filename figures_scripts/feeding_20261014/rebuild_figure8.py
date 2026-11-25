"""Rebuild Fig. 8 from the corrected common-support perturbation results.

The fixed models and the commitment results are unchanged. This figure-only
adapter reuses the existing publication style, panel builders and export helper.
"""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
STEM = 'Fig8_feeding_and_supply'
RESULTS = Path(os.environ.get('MUSCATINE_RESULTS', ROOT / 'results'))
PERT_FILE = RESULTS / 'feeding_complete_support/feeding_perturbations_complete_support.csv'
SUMMARY_FILE = RESULTS / 'feeding_complete_support/feeding_correction_summary.json'
LAYOUT_FILE = ROOT / 'figures_scripts/layout_20261014/make_figures.py'
OUT = HERE / 'figures_ready'
OUT.mkdir(exist_ok=True)
os.environ['LAYOUT_DPI'] = '600'
sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location('retained_figure_layout', LAYOUT_FILE)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
layout.OUT = OUT
layout.DPI = 600
layout.PREVIEW = False

summary = json.loads(SUMMARY_FILE.read_text(encoding='utf8'))
layout.source.PERT = layout.source.pd.read_csv(PERT_FILE)
assert len(layout.source.PERT) == summary['scenario_rows'] == 1270
assert set(layout.source.PERT.n) == {summary['n_common']} == {1791}
assert not layout.source.PERT.isna().any().any()

mask = (layout.source.M.h == 24) & (layout.source.M.model == 'LGB-Δ [past]')
assert int(mask.sum()) == 1
baseline = next(row for row in summary['baselines']
                if row['h'] == 24 and row['model'] == 'LGB-Δ [past]')['RMSE']
assert baseline == 58.090255533812844
layout.source.M.loc[mask, 'RMSE'] = baseline


def scenario_scatter(fig, ax):
    layout.source.schedule_d(fig, ax)
    for artist, label in zip(ax.collections, ['Hourly noise', 'Sustained bias',
                                             'Missing episodes', 'Delayed entries']):
        artist.set_label(label)
    ax.set_ylim(44, 67)
    layout.legend(ax, 2, loc='lower left', anchor=(0, 1.01), size=9)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_notebook(folder):
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'metadata': {}, 'source': [
                '# Fig8_feeding_and_supply\n',
                'Panels a–b use the corrected evaluation on 1,791 common origins; c–f retain the commitment analysis. Rebuild through the current numbered entry point.\n']},
            {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
                'from pathlib import Path\n',
                'import sys, subprocess\n',
                "package = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'figures_scripts/feeding_20261014/rebuild_figure8.py').exists())\n",
                "entry = package / 'figures_scripts/feeding_20261014/rebuild_figure8.py'\n",
                "subprocess.run([sys.executable, str(entry)], check=True)\n"]},
        ],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}},
        'nbformat': 4, 'nbformat_minor': 5,
    }
    (folder / f'{STEM}_jupyter_code.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf8')


def build():
    funcs = [layout.scenario_heat, scenario_scatter, layout.commitments,
             layout.strict, layout.nominal, layout.realized]
    # Retain the existing combined/export helper and its panel geometry.
    layout.save_figure_with_subfigures(STEM, funcs)
    folder = OUT / STEM
    from PIL import Image
    from pypdf import PdfReader
    for letter in 'abcdef':
        path = folder / 'subfigures' / f'{STEM}_{letter}.png'
        with Image.open(path) as image:
            image.thumbnail((1128, 912), Image.Resampling.LANCZOS)
            image.save(path.with_name(path.stem + '_preview.png'))
    # Keep the figure input and provenance adjacent to the exports.
    shutil.copy2(PERT_FILE, folder / PERT_FILE.name)
    shutil.copy2(SUMMARY_FILE, folder / SUMMARY_FILE.name)
    write_notebook(folder)
    figure_record = {
        **layout.AUDIT[STEM],
        'source_perturbation_csv': str(PERT_FILE),
        'source_perturbation_sha256': sha256(PERT_FILE),
        'source_layout': str(LAYOUT_FILE),
        'source_layout_sha256': sha256(LAYOUT_FILE),
        'panels_a_b_common_origins': summary['n_common'],
        'panel_b_history_rmse_m3_h': baseline,
        'panel_b_legend': ['Hourly noise', 'Sustained bias', 'Missing episodes', 'Delayed entries'],
        'panels_c_f': 'Original commitment results and plotting functions retained without changes.',
        'commitment_inputs': {name: sha256(layout.source.OUT / name) for name in
                              ['firm_supply_independent.csv', 'commitment_paired_uncertainty.csv',
                               'commitment_matched_risk.csv']},
        'commitment_builder_sha256': {name: hashlib.sha256(inspect.getsource(getattr(layout.source, name)).encode()).hexdigest()
                                     for name in ['commitment_a', 'commitment_b', 'commitment_c', 'commitment_d']},
        'combined_label_pt': 26,
        'standalone_label_pt': 22,
        'png_dpi': 600,
        'exports': [],
    }
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        item = {'file': str(path.relative_to(folder)), 'sha256': sha256(path)}
        if path.suffix == '.png':
            with Image.open(path) as image:
                item.update(size_px=list(image.size), dpi=image.info.get('dpi'))
        elif path.suffix == '.pdf':
            pdf = PdfReader(path)
            item.update(pages=len(pdf.pages), author=pdf.metadata.author,
                        creator=pdf.metadata.creator)
            assert pdf.metadata.author == pdf.metadata.creator == 'Anonymous'
        figure_record['exports'].append(item)
    (HERE / 'figure8_rebuild_audit.json').write_text(json.dumps(figure_record, ensure_ascii=False, indent=2), encoding='utf8')
    (folder / 'README.md').write_text(
        'Fig. 8: feeding perturbations and fulfilled-supply comparisons.\n\n'
        'Rebuild: figures_scripts/feeding_20261014/rebuild_figure8.py.\n'
        'Panels a–b use the 1,791 common-origin perturbation evaluation; c–f retain the original commitment evaluation.\n'
        'Combined and independent panel PNG/PDF exports use the same plotting functions and inputs.\n', encoding='utf8')
    print(folder, flush=True)


if __name__ == '__main__':
    build()
