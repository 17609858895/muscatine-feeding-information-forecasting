# Figure scripts

The submitted figures were produced in several layout passes that reuse one plotting library. The scripts read saved results only; they do not fit models. Run them from the package root after the analyses in `scripts/` have filled `results/` (or point `MUSCATINE_RESULTS` to such a folder). `SCADA_CSV` is needed for Fig. S1.

| Folder | Content |
|---|---|
| `plotting/` | Shared style, palettes and panel builders (`figcommon.py`, `ml_publication_style.py`, `scientific_palettes.py`, `supplement_panels.py`, ...) and `rebuild_compact_figures.py`, which renders the supplementary figures that were not redrawn later. Its figure keys follow an earlier numbering (table below). |
| `revision_20261013/` | `make_figures.py`: plotting functions for the revision analyses; renders Figs. S6 and S10. Imported by the later layouts. |
| `layout_20261014/` | `make_figures.py`: compact three-column plates; renders Fig. 3 and Fig. S11 and provides panel functions for Figs. 4 and 6. |
| `feeding_20261014/` | `rebuild_figure8.py`: feeding-perturbation panels on the 1,791 complete-window origins (Fig. 6a, b). |
| `submission_20261018/` | `compact_temporal_figure.py` (Fig. 2), `rebuild_figures.py` (Figs. 4, 5, 6 and S4 with bootstrap bands and current labels), `redraw_fig5.py` (final Fig. 5 with segment-respecting intervals). |

Outputs are written next to each script (`figures_ready/`, `figures/`, `compact_figures/`) or to `figures_output/`; these folders are ignored by git. PDF metadata use the author string `Anonymous`.

The builders check panel alignment after layout. These checks were calibrated with Matplotlib 3.10.8 (the version used for the article); with Matplotlib 3.11.2 the Fig. S1 builder stops at the assertion `Packing broke calibrated outer boundaries`, whereas it passes with 3.10.8. Fonts default to Arial.

## Submitted figure → command

| Figure | Data produced by | Command (from the package root) |
|---|---|---|
| Fig. 1 | Conceptual framework illustration (edited with an image-generation tool, as disclosed in the article); no plotting script | – |
| Fig. 2 | `prep.py` (hourly data) | `python figures_scripts/submission_20261018/compact_temporal_figure.py` |
| Fig. 3 | `attribution_audit.py`, `group_refits.py` | `python figures_scripts/layout_20261014/make_figures.py Fig4_information_value` |
| Fig. 4 | `paired_forecast_audit.py`, `feeding_audit.py`, `matched_mlp.py`, `compute_intervals.py` | `python figures_scripts/submission_20261018/rebuild_figures.py 4` |
| Fig. 5 | `ale_dml.py`, `compute_extra.py` (2-D grid), `compute_intervals.py`, `recompute_response_intervals.py` | `python figures_scripts/submission_20261018/redraw_fig5.py` |
| Fig. 6 | `recheck_feeding_common_support.py` (a, b), `calibration_audit.py` (c–f) | `python figures_scripts/submission_20261018/rebuild_figures.py 6` |
| Fig. S1 | raw SCADA minutes, `prep.py` | `python figures_scripts/plotting/rebuild_compact_figures.py S1` |
| Fig. S2 | `compute_supp.py` (laboratory records) | `... rebuild_compact_figures.py S3` |
| Fig. S3 | `leak_audit.py`, `compute_extra.py`, `compute_supp.py` | `... rebuild_compact_figures.py S4` |
| Fig. S4 | `tune_all.py`, `dl_tune.py`, `compute_supp.py` | `python figures_scripts/submission_20261018/rebuild_figures.py S4` |
| Fig. S5 | `compute_supp.py` (training-window curves) | `... rebuild_compact_figures.py S6` |
| Fig. S6 | `calibration_audit.py` | `python figures_scripts/revision_20261013/make_figures.py FigS6_uncertainty_and_state_errors` |
| Fig. S7 | `tab_final.py`, `ale_dml.py` (PDP computed in the builder) | `... rebuild_compact_figures.py S8` |
| Fig. S8 | `shap_run.py` | `... rebuild_compact_figures.py S9` |
| Fig. S9 | Screenshot of the running interface (`interface/`) | – |
| Fig. S10 | `paired_forecast_audit.py`, `attribution_audit.py` | `python figures_scripts/revision_20261013/make_figures.py FigS10_dependence_and_stability` |
| Fig. S11 | `calibration_audit.py` | `python figures_scripts/layout_20261014/make_figures.py Fig6_trajectories_uncertainty` |
| Fig. S12 | `prep.py` (hourly inputs) | `... rebuild_compact_figures.py S2` |

The graphical abstract is not included.
