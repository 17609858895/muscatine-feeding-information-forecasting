# Horizon-dependent value of feeding information for hourly biogas forecasting in full-scale anaerobic co-digestion

Analysis code, prediction outputs, result tables and a local prediction interface for the article of the same title (Bioresource Technology).

## Study summary

Using eleven months of public SCADA and laboratory records from the Muscatine Water Resource Recovery Facility (Iowa, USA), the study asks how much advance feeding information improves hourly forecasts of metered biogas flow (total flow to the boiler and waste-gas burner) at horizons of 1–24 h, and whether accuracy gains translate into larger low-risk supply commitments. Matched persistence-anchored LightGBM and CatBoost models (LGB-Δ, Cat-Δ), a process-group horizon-query network (PG-HQ) and their NNLS ensemble received either historical inputs only or historical inputs plus the feeding recorded during the forecast window (high-strength waste, primary sludge, thickened waste activated sludge), and were evaluated on 1,797 chronological hourly test origins (15 December 2022 – 27 February 2023). Recorded future feeding reduced the day-ahead ensemble RMSE from 57.48 to 45.52 m³ h⁻¹ (LGB-Δ: 58.17 to 46.03 m³ h⁻¹), whereas the 1 h difference was near zero. Origin-available feeding proxies and perturbed schedules tested how this gain depends on information quality; group-removal refits and permutation attribution distinguished incremental information from fitted-model reliance; and adaptively calibrated lower bounds evaluated hypothetical gas-supply commitments at matched shortfall risk.

**Retrospective evaluation.** Test-period scores were inspected during exploratory architecture selection, so the test-period comparisons are retrospective rather than confirmatory. Recorded future feeding is executed feeding, an idealized reference for schedules issued in advance; issued schedules were not archived. Bootstrap intervals condition on the fitted models (and, for commitments, on the realized calibration paths).

## Data

The raw data are not redistributed here. Download `SCADA-raw.csv` and `LABS-raw.csv` from

> Schroer, H.W., Just, C.L., 2024. Laboratory, operational, and SCADA datasets for anaerobic co-digestion at the Muscatine, IA Water Resource Recovery Facility. https://doi.org/10.25820/data.006715

and place them in `data/` (or set `SCADA_CSV` and `LABS_CSV`; see `data/README.md`).

## Repository layout

```
data/              raw-data instructions (CSV files are not included)
scripts/pipeline/  original modelling pipeline (run with results/ as working directory)
scripts/revision/  analyses behind the reported results (independent ensembles, intervals, commitments, ...)
figures_scripts/   figure builders for the submitted figures
interface/         local prediction interface (Section S8): code, fitted LightGBM models, examples, validation
outputs/           retained predictions (CSV), result tables, fitted LGB-Δ/Cat-Δ models, PG-HQ network weights
results/           empty working directory for regenerated intermediate files
```

Details: `scripts/README.md`, `figures_scripts/README.md`, `outputs/README.md`, `interface/README.md`.

## Quick check without rerunning the models

```python
import numpy as np, pandas as pd
d = pd.read_csv('outputs/predictions/test_predictions.csv')
d24 = d[d.horizon_h == 24]
rmse = d24.groupby(['model', 'regime']).apply(lambda g: np.sqrt(((g.prediction - g.observed) ** 2).mean()))
print(rmse.round(2))
# Ensemble history 57.48, feed 45.52; LGB-Δ history 58.17, feed 46.03; Cat-Δ 57.20 / 45.23; persistence 76.99 (m3 h-1, n = 1,797)
```

## Output conventions

- Flows are metered biogas flow in m³ h⁻¹ (SCADA native ft³ min⁻¹ × 1.69901).
- Timestamps are hourly interval-start labels without timezone (source clock; no daylight-saving correction). The hour labelled *t* covers [*t*, *t* + 1 h) and is available at *t* + 1 h.
- `origin_time` is the last complete hour *t* at issuance; `target_time` = *t* + *h*. For example, a forecast issued at 09:00 (*t* = 08:00) targets 09:00–10:00 at h = 1 and 08:00–09:00 on the following day at h = 24.
- `regime`: `history` (historical inputs only), `feed` (plus recorded future feeding), `feed_proxy_weekly` / `feed_proxy_4week` (origin-available feeding proxies in the feed-conditioned LGB-Δ model).
- Column-level descriptions of every file are in `outputs/README.md`.

## Environment

Python 3.12. The scientific analyses used (Supporting Information, Section S7): NumPy 2.5.3, pandas 3.0.3, SciPy 1.18.1, scikit-learn 1.9.1, LightGBM 4.7.0, CatBoost 1.2.10, PyTorch 2.13.0 (CPU build) and Matplotlib 3.10.8, with two CPU cores and two PyTorch threads. Individual scripts also need XGBoost, Optuna, SHAP, statsmodels and openpyxl; `requirements.txt` lists the versions found in the analysis environment.

The interface runs with `interface/requirements.txt` (NumPy 2.2.6, pandas 3.0.3, LightGBM 4.6.0). The smoke tests of this package used that interpreter (Python 3.12.14, NumPy 2.2.6, pandas 3.0.3, SciPy 1.15.3, scikit-learn 1.7.2, Matplotlib 3.11.2) with LightGBM 4.7.0, CatBoost 1.2.10 and statsmodels 0.15.0 added; PyTorch was not needed for those tests. The pickled intermediate files are written with NumPy 2.x.

## Reproducing the analyses

1. Put the raw CSV files in `data/` and install `requirements.txt`.
2. Run the pipeline in `results/`, then the revision analyses, in the order given in `scripts/README.md` (several steps, e.g. tuning and five-seed neural benchmarks, take hours on two CPU cores).
3. Rebuild figures with the commands in `figures_scripts/README.md`.

`scripts/revision/derive_findings.py` prints the main reported numbers from the regenerated tables.

Checks run for this release (with the retained intermediate files and the interface interpreter listed above): `prep.py` regenerated `hourly.pkl` identically from `SCADA-raw.csv` (8,349 hours; 5,421/696/1,797 origins); `calibration_audit.py`, `feeding_audit.py`, `paired_forecast_audit.py` and `group_refits.py` reproduced the saved result tables byte for byte, except HAC p values that differed by less than 10⁻¹⁵ (relative); the builders of Figs. 2–6, S6 and S12, and of Fig. S1 with Matplotlib 3.10.8, ran from this layout; the interface reproduced its stored example output exactly. The full pipeline (model fitting, tuning, neural training) was not rerun for this release.

## Manuscript items and code

| Item | Analysis scripts | Retained output | Figure builder |
|---|---|---|---|
| Section 2.1–2.2, S1: hourly data, filled values, routing shares | `prep.py`, `filled_value_sensitivity.py`, `gas_use_split.py` | `tables/revision_20261013/filled_*.csv`, `timestamp_audit.json` | – |
| Section 2.1, 3.6, 4.1, S11; Tables S23–S25: feeding by digester, HSW-associated biogas, energy value of flared gas | `revision_feeding_energy.py`, `energy_equivalents.py` | `tables/revision_20261126/` | – |
| Fig. 1 (framework) | conceptual illustration | – | – |
| Fig. 2 (temporal structure) | `prep.py` | – | `submission_20261018/compact_temporal_figure.py` |
| Fig. 3; Tables S4, S19; Section 3.2, 3.5, S5 | `attribution_audit.py`, `group_refits.py` | `tables/revision_20261013/attribution_consistency.csv`, `HSW_lag_stability.csv`, `loco_paired_uncertainty.csv`; `predictions/group_refit_test_predictions.csv` | `layout_20261014/make_figures.py Fig4_information_value` |
| Fig. 4; Tables S13, S14, S16, S17; Section 3.3 | `tab_final.py`, `dl.py`, `fit_independent_ensemble.py`, `paired_forecast_audit.py`, `feeding_audit.py`, `matched_mlp.py`, `compute_intervals.py` | `predictions/test_predictions.csv`, `tables/revision_20261013/independent_*.csv`, `all_benchmark_metrics.csv`, `paired_RMSE_uncertainty.csv`, `ablation_configuration_register.csv`, `available_feed_controls.csv`, `matched_MLP_*.csv`; `tables/figure_intervals_20261018/forecast_pointwise_intervals.csv` | `submission_20261018/rebuild_figures.py 4` |
| Fig. 5; Section 3.6, S6 (ALE responses) | `ale_dml.py`, `compute_extra.py`, `compute_intervals.py`, `recompute_response_intervals.py` | `tables/figure_intervals_20261018/ale_*.csv` | `submission_20261018/redraw_fig5.py` |
| Fig. 6a, b; Section 3.7, S2 (proxies, perturbations) | `feeding_audit.py`, `recheck_feeding_common_support.py`, `discussion_evidence.py` | `tables/feeding_complete_support_20261014/`, `tables/discussion_20261013/` | `submission_20261018/rebuild_figures.py 6` |
| Fig. 6c–f; Section 3.4, 3.8, S4; Tables S9, S20–S22; Figs. S6, S11 | `calibration_audit.py`, `discussion_evidence.py` | `tables/revision_20261013/firm_supply_independent.csv`, `commitment_*.csv`, `calibration_sensitivity.csv`, `intervals_independent.csv`, `adaptive_state_traces.csv.gz` | `rebuild_figures.py 6`; `revision_20261013/make_figures.py`; `layout_20261014/make_figures.py Fig6_trajectories_uncertainty` |
| Table S6, Fig. S10 (HAC tests, dependence) | `paired_forecast_audit.py`, `attribution_audit.py` | `HAC_benchmark_tests.csv`, `loss_difference_acf.csv`, `lag_bin_sensitivity.csv` | `revision_20261013/make_figures.py FigS10_dependence_and_stability` |
| Tables S5, S18 (expanding windows) | `rolling.py`, `compute_extra.py`, `tune_all.py`, `post_tune.py` | `tables/original_pipeline/extra/rolling_all_horizons*.csv` | – |
| Table S2, Fig. S3 (evaluation designs) | `leak_audit.py`, `compute_extra.py`, `compute_supp.py` | `tables/original_pipeline/leakage_audit.csv`, `extra/leakage_all_horizons.csv`, `supp/next24_mean_benchmark.csv` | `plotting/rebuild_compact_figures.py S4` |
| Tables S3, S7, S15; Figs. S4, S5 (tuning, training) | `tune_all.py`, `dl_tune.py`, `compute_supp.py`, `matched_mlp.py` | `tables/original_pipeline/tune/`, `supp/` | `rebuild_figures.py S4`; `rebuild_compact_figures.py S6` |
| Table S10 (matched MLP) | `matched_mlp.py` | `predictions/matched_mlp_test_predictions.csv`, `matched_MLP_metrics.csv` | – |
| Table S8, Fig. S9, Section S8 (interface) | `rebuild_interface.py`, `verify_interface.py`, `gui_audit.py`, `runtime_full_check.py` | `interface/validation/` | screenshot |
| Figs. S1, S2, S7, S8, S12 | `prep.py`, `compute_supp.py`, `ale_dml.py`, `shap_run.py` | `tables/original_pipeline/` | `rebuild_compact_figures.py S1, S3, S8, S9, S2` |
| Table S1, S11, S12 | variable definitions, architectures (`dl.py`), literature | `tables/supplementary_workbook/` | – |

Supplementary-table values as assembled for the SI are also exported sheet by sheet in `outputs/tables/supplementary_workbook/` (see `outputs/README.md` for the sheet-to-table mapping).

## Prediction interface

`interface/` contains the local forecasting interface described in Section S8 (persistence-anchored LightGBM, nine horizons, optional 24 h feeding plan, static 90% intervals). An older online demonstration is available at https://github.com/17609858895/Muscatine-Biogas-Streamlit; it predates this version and was not updated.

## Citation

Please cite the article: *Horizon-dependent value of feeding information for hourly biogas forecasting in full-scale anaerobic co-digestion*, Bioresource Technology. DOI: to be added on publication.

Please also cite the dataset (Schroer and Just, 2024; https://doi.org/10.25820/data.006715) when using the data.
