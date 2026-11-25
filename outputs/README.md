# Outputs

Retained predictions and result tables from the analysis runs reported in the article. All files are plain CSV (UTF-8) unless stated otherwise.

## Conventions

- **Flow units.** Every flow in `predictions/` is metered biogas flow in m³ h⁻¹ (total flow to the boiler and the waste-gas burner). The SCADA records are in ft³ min⁻¹; the scripts work in these native units and convert with 1 ft³ min⁻¹ = 1.69901 m³ h⁻¹. Values are rounded to four decimals. Some files in `tables/original_pipeline/` keep native units (see below).
- **Time stamps.** Hourly labels are interval-start labels without timezone (`YYYY-MM-DD HH:MM`, source clock, no daylight-saving correction). The value labelled *t* is the mean over [*t*, *t* + 1 h) and becomes available at *t* + 1 h.
- **Origins and horizons.** `origin_time` is the label *t* of the last complete hour at forecast issuance. A horizon-*h* forecast targets the hour labelled `target_time` = *t* + *h*, i.e. h = 1 is the next complete hour and h = 24 is the hour beginning 23 h after issuance (issued at *t* + 1 h). Horizons: 1, 2, 3, 4, 6, 9, 12, 18, 24 h.
- **Periods.** Test: 1,797 consecutive origins, 15 December 2022 00:00 – 27 February 2023 20:00. Development: 696 origins, 15 November – 13 December 2022 (312 selection origins, a 24-origin gap, and 360 calibration origins). Training (not exported): 5,421 origins, 25 March – 13 November 2022.
- **Regimes.** `history`: historical inputs only (labelled *past* in the code). `feed`: historical inputs plus the recorded (executed) mean HSW flow, PS on-time fraction and TWAS flow over labels *t* + 1 … *t* + *h*; this is executed feeding, not a schedule known at issuance. `feed_proxy_weekly` and `feed_proxy_4week`: the fitted feed-conditioned LGB-Δ model supplied with origin-available proxies (same hour one week earlier; median of the same hour 1–4 weeks earlier).
- The test period was inspected during exploratory model development, so all test-period comparisons are retrospective.

## predictions/

| File | Rows | Content |
|---|---|---|
| `test_predictions.csv` | 194,076 | Principal forecasts on the 1,797 test origins at all nine horizons (tidy). |
| `development_predictions.csv` | 56,376 | The same principal models on the 696 development origins, with `period` = `selection`, `gap` or `calibration`; the calibration origins initialise the adaptive intervals and commitments. |
| `benchmark_test_predictions_wide.csv` | 16,173 | All 35 forecasts compared in the article (persistence, seasonal baselines, kernel/tree/linear and neural benchmarks, matched-feed variants, components, ensembles, feeding proxies); one column per model, keyed by `origin_time`, `horizon_h`; `observed`. Blank cells: model not evaluated at that horizon. |
| `group_refit_test_predictions.csv` | 48,519 | Historical-input LGB-Δ refitted without storage histories, historical HSW inputs or calendar terms (`removed_group`), 9 horizons (Table S4). |
| `matched_mlp_test_predictions.csv` | 21,564 | Matched scalar-output MLP at 1, 6 and 24 h for level and increment targets, with the selected learning rate (Fig. 4d; Tables S10, S15). |

Columns of `test_predictions.csv`: `origin_time`, `target_time`, `horizon_h`, `regime`, `model`, `prediction`, `observed`.

| `model` | Description | Regimes |
|---|---|---|
| `Persistence` | y(t) | history |
| `Seasonal naive (168 h)` | y(t + h − 168) | history |
| `LGB-Δ` | Persistence-anchored LightGBM (increment model) | history, feed, feed_proxy_weekly, feed_proxy_4week |
| `Cat-Δ` | Persistence-anchored CatBoost | history, feed |
| `PG-HQ` | Process-group horizon-query network, mean of three seeds | history, feed |
| `Ensemble` | NNLS-weighted LGB-Δ, Cat-Δ and PG-HQ (weights from the 312 selection origins; Table S14). Called `PG-HQ ensemble` in the code. | history, feed |

Check (Python):

```python
import pandas as pd, numpy as np
d = pd.read_csv('outputs/predictions/test_predictions.csv')
d24 = d[d.horizon_h == 24]
print(d24.groupby(['model', 'regime']).apply(lambda g: np.sqrt(((g.prediction - g.observed) ** 2).mean())))
# Ensemble: history 57.48, feed 45.52; LGB-Δ: history 58.17, feed 46.03 m3 h-1 (1,797 origins)
```

## tables/

| Folder | Content | Produced by |
|---|---|---|
| `revision_20261013/` | Current analyses: independent ensembles (metrics, weights, network selection), paired block-bootstrap RMSE differences, HAC tests and loss autocorrelation, representation/anchoring register (A1–A5), feeding proxies and schedule perturbations, matched MLP, attribution under both anchor mappings and HSW lag stability, group-refit intervals, calibration/commitment results, events and sensitivities, filled-value sensitivity. `adaptive_state_traces.csv.gz` holds the hourly adaptive-calibration states (gzip). `runtime_versions.json` lists the package versions of these runs. | `scripts/revision/` |
| `feeding_complete_support_20261014/` | Perturbation scores re-evaluated on the 1,791 origins with complete windows under every shift (Fig. 6a, b; Section 3.7). | `recheck_feeding_common_support.py` |
| `figure_intervals_20261018/` | Pointwise 95% block-bootstrap intervals for skill and RMSE (Fig. 4a, f) and for the ALE responses and accumulated mixed differences (Fig. 5), converted to m³ h⁻¹, plus their audit records. | `compute_intervals.py`, `recompute_response_intervals.py` |
| `discussion_20261013/` | Error-tail structure, the 11 January 2023 shortfall event and perturbation summaries (Sections 3.7, 3.8). | `discussion_evidence.py` |
| `group_refits/` | Refit-minus-full RMSE for the 27 group-removal refits. | `group_refits.py` |
| `revision_20261126/` | Descriptive analyses prepared for the reviewer responses: feeding by digester, laboratory loads and HRT, daily HSW–biogas association, HSW lag summary, indicative energy value of flared gas. | `scripts/pipeline/revision_feeding_energy.py` |
| `original_pipeline/` | CSV outputs of the original exploratory pipeline (`results/`, `extra/`, `supp/`, `tune/`) and `tune/best_parameters.json`. Several files are superseded for ensemble-dependent results: `all_metrics.csv`, `summary_1_6_24.csv`, `conformal.csv`, `firm_supply.csv`, `dm_tests*.csv` use the earlier ensemble whose weights and calibration shared all 696 development origins; the article uses `revision_20261013/`. `bench_tab.csv` is an early benchmark fitted on training plus development data. Most RMSE/MAE values in these files are in native ft³ min⁻¹ (multiply by 1.69901). | `scripts/pipeline/` |
| `supplementary_workbook/` | Each sheet of the supplementary-table workbook used to assemble the SI tables, exported as CSV (`_index.csv` lists the sheets). Sheet numbering follows an earlier draft: current Tables S1–S3 and S5–S11 correspond to the same-numbered `Table_S*` sheets; S4 ← `Table_S4_Group_refits`; S13 ← `Table_S5_Benchmark` (test rows); S14 ← `S3_Ensemble_weights`; S15 ← `S3_Matched_MLP_selection` and `Table_S10_Matched_models`; S16 ← `S5_Ablation_register`; S17 ← `S5_Paired_RMSE`; S18 ← `S5_fold_scores`; S19 ← `S5_Group_permutation`; S20 ← `Table_S9_Firm_supply`; S21 ← `S9_Paired_uncertainty`, `S9_Matched_risk`; S22 ← `S9_Calibration_sensitivity`. | assembled from the tables above |

## model_weights/

PyTorch `state_dict` files for the PG-HQ network (`PGHQ2` in `scripts/pipeline/dl.py`).

- `principal_ensemble/pghq_independent_{past,feed}_{0,1,2}.pt`: the three seeds per regime used in the principal ensembles (early-stopped on the 312 selection origins; `fit_independent_ensemble.py`).
- `exploratory_benchmark/`: earlier exploratory PG-HQ weights (`dl.py`, `dl_tune.py`).

Fitted LightGBM and CatBoost research models (`tab_final.pkl`, 36 MB) and the TreeSHAP values (`shap.pkl`) are not distributed; they are regenerated by `tab_final.py` and `shap_run.py`. LightGBM models refitted for the prediction interface are in `interface/models/`.
