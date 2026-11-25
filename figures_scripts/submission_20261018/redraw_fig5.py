from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
source=HERE/'rebuild_figures.py'
spec=importlib.util.spec_from_file_location('figures',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=HERE/'figures_fig5_segmented';m.OUT.mkdir(exist_ok=True);m.layout.OUT=m.OUT
m.response=m.pd.read_pickle(m.ctx.RES/'figure_intervals_segmented/response_pointwise_intervals.pkl')
m.figure5()
(m.OUT/'figure_layout_audit.json').write_text(json.dumps(m.layout.AUDIT,indent=2),encoding='utf8')
