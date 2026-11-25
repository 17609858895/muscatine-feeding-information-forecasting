# Analysis scripts

The scripts are the code used for the article, copied with path changes only (default data and output locations; the original logic is unchanged). Two groups:

- `pipeline/`: the original modelling pipeline. Each script is run **with `results/` as the working directory** and reads/writes files there by relative name (`hourly.pkl`, `tab_final.pkl`, ...). `feats.py` and `common.py` are shared modules (process groups, lag-bin features, splits, targets).
- `revision/`: analyses that produce the results reported in the current article. `context.py` sets the paths (`results/` as `RES`, `results/revision_analysis/` as `OUT`), changes into `RES`, and exposes the splits used throughout: 5,421 training, 312 selection, 360 calibration and 1,797 test origins. These scripts can be run from any directory.

Native flow units are ft³ min⁻¹; results are converted with 1.69901 m³ h⁻¹ per ft³ min⁻¹. Most scripts use two CPU threads; neural training is CPU-only.

## Run order

```bash
mkdir -p results && cd results
python ../scripts/pipeline/prep.py            # needs data/SCADA-raw.csv or SCADA_CSV
python ../scripts/pipeline/tab_final.py
python ../scripts/pipeline/dl.py
python ../scripts/pipeline/ens_conf.py
python ../scripts/pipeline/hrpga.py
python ../scripts/pipeline/ablation_feat.py
python ../scripts/pipeline/leak_audit.py
python ../scripts/pipeline/rolling.py
python ../scripts/pipeline/feed_robust.py
python ../scripts/pipeline/ale_dml.py
python ../scripts/pipeline/shap_run.py
python ../scripts/pipeline/dm_att.py
python ../scripts/pipeline/compute_extra.py
python ../scripts/pipeline/tune_all.py
python ../scripts/pipeline/dl_tune.py
python ../scripts/pipeline/post_tune.py
python ../scripts/pipeline/compute_supp.py     # needs data/LABS-raw.csv or LABS_CSV
cd ..
python scripts/revision/fit_independent_ensemble.py
python scripts/revision/feeding_audit.py
python scripts/revision/paired_forecast_audit.py
python scripts/revision/group_refits.py
python scripts/revision/attribution_audit.py
python scripts/revision/calibration_audit.py
python scripts/revision/matched_mlp.py
python scripts/revision/filled_value_sensitivity.py
python scripts/revision/recheck_feeding_common_support.py
python scripts/revision/discussion_evidence.py results results/discussion
python scripts/revision/compute_intervals.py
python scripts/revision/recompute_response_intervals.py
python scripts/revision/gui_audit.py
python scripts/revision/rebuild_interface.py
python scripts/revision/verify_interface.py
python scripts/revision/prepare_runtime_check.py
python scripts/revision/runtime_full_check.py   # run with the interface environment
python scripts/revision/derive_findings.py      # prints the key numbers
python scripts/pipeline/gas_use_split.py
python scripts/pipeline/energy_equivalents.py
python scripts/pipeline/revision_feeding_energy.py
```

The order follows the data dependencies (e.g. `compute_extra.py` needs `final_preds.pkl` from `ens_conf.py`; `post_tune.py` needs `extra/` and `tune/`; `paired_forecast_audit.py` needs `tune/`, `extra/ablation_rawlevel.csv`, `ablation_features.csv` and the feeding proxies from `feeding_audit.py`; `attribution_audit.py` needs `group_refits/`; `compute_intervals.py` needs `ale.pkl`, `extra/ale2d.pkl` and `benchmark_predictions.pkl`). Some steps (Optuna tuning, five-seed neural benchmarks, block bootstraps) take hours on two CPU cores. `fit_independent_ensemble.py` and `matched_mlp.py` reuse saved checkpoints in `results/revision_analysis/` when present.

## pipeline/

| Script | Output (in `results/`) | Used for |
|---|---|---|
| `prep.py` | `hourly.pkl` | Hourly resampling, low-count screening (Section 2.1; S1) |
| `feats.py`, `common.py` | – | Process groups, PG-LB lag bins, calendar terms, future-feeding means, chronological origins |
| `tab_final.py` | `tab_final.pkl`, `feature_meta.csv` | LGB-Δ and Cat-Δ for both regimes at all horizons; 1/6/24 h kernel, tree and linear benchmarks (Section 2.3) |
| `dl.py` | `preds_dl_all.pkl`, `bench_dl_all.csv`, `pghq_*.pt` | Neural benchmarks and the PG-HQ network (`PGHQ2`; Table S11) |
| `ens_conf.py`, `aci.py` | `final_preds.pkl`, `all_metrics.csv`, `conformal.csv` | Earlier ensemble with weights on all 696 development origins; retained as the container for component predictions (superseded by `fit_independent_ensemble.py`) |
| `hrpga.py` | `hrpga.pkl`, `hrpga_*.csv` | Horizon-resolved block permutation (earlier version of Fig. 3 inputs) |
| `ablation_feat.py` | `ablation_features.csv` | Raw-lag vs. lag-bin anchored models (A3, A4; Table S16) |
| `leak_audit.py` | `leakage_audit.csv` | Random split and target-derived counters (Fig. S3; Table S2) |
| `rolling.py` | `rolling_origin.csv` | Expanding-window folds (Tables S5, S18) |
| `feed_robust.py` | `feed_schedule_robustness.csv` | Early schedule-noise test (superseded by `feeding_audit.py`) |
| `ale_dml.py` | `ale.pkl`, `ale_curves.csv`, `dml_*.csv` | First-order ALE (Fig. 5a–c). The DML estimates in the same script are not reported in the article |
| `shap_run.py` | `shap.pkl`, `shap_group_by_horizon.csv` | TreeSHAP (Fig. S8; Table S19 context) |
| `dm_att.py` | `dm_tests.csv`, `pghq_attention.npy` | Early DM tests; PG-HQ attention agreement (Section S5) |
| `compute_extra.py` | `extra/` | All-horizon benchmark metrics, leakage at all horizons, expanding windows, 2-D ALE grid (Fig. 5d, e), raw-lag direct model (A1), training-period predictions |
| `compute_supp.py` | `supp/` | Optuna diagnostics, training-window curves (Fig. S5), neural training histories (Fig. S4c), following-24-h mean target (Fig. S3), laboratory records and FOS/TAC (Fig. S2), conformal variants |
| `tune_all.py`, `tune_pghq.py`, `dl_tune.py`, `post_tune.py` | `tune/`, `extra/*_tuned.csv`, `dm_tests_tuned.csv` | Exploratory tuning on the development block (Table S3; Fig. S4), five-seed neural benchmarks (Table S7), tuned expanding-window control |
| `bench_tab.py` | `bench_tab.csv`, `preds_tab.pkl` | Early exploratory benchmark fitted on training plus development data; not used in the article |
| `firm_supply.py` | `firm_supply.csv` | Earlier commitment analysis on `final_preds.pkl` (superseded by `calibration_audit.py`) |
| `make_tables.py` | `tables/` | Legacy table/workbook export (needs most outputs above) |
| `gas_use_split.py` | `gas_use_split.txt` | Burner/boiler routing shares (Section 2.1); reads the raw SCADA file |
| `energy_equivalents.py` | `energy_equivalents.csv` | Indicative energy equivalents of metered flows (assumed methane fraction) |
| `revision_feeding_energy.py` | `revision_20261126/` | Feeding by digester, laboratory loads, daily HSW association, lag summary, indicative value of flared gas (reviewer responses) |

## revision/

| Script | Output (in `results/revision_analysis/` unless noted) | Article items |
|---|---|---|
| `context.py`, `statistics_utils.py` | – | Paths and splits; circular block bootstrap, HAC tests, Holm adjustment, event runs |
| `fit_independent_ensemble.py` | `independent_final_preds.pkl`, `independent_primary_metrics.csv`, `independent_ensemble_weights.csv`, `pghq_independent_*.pt` | Principal ensembles: three PG-HQ seeds early-stopped on 312 selection origins, NNLS weights on the same block (Sections 2.3, 3.3; Tables S13, S14) |
| `feeding_audit.py` | `available_feed_controls.csv`, `available_feed_predictions.pkl`, `feeding_perturbations.csv` | Weekly and four-week feeding proxies (Fig. 4e, f; Section 3.7); schedule perturbations (Section S2) |
| `paired_forecast_audit.py` | `all_benchmark_metrics.csv`, `benchmark_predictions.pkl`, `paired_RMSE_uncertainty.csv`, `HAC_benchmark_tests.csv`, `loss_difference_acf.csv`, `ablation_configuration_register.csv` | Fig. 4a–c; Tables S6, S13, S16, S17; Fig. S10a, c |
| `group_refits.py` | `results/group_refits/` | 27 group-removal refits (Table S4) |
| `attribution_audit.py` | `attribution_consistency.csv`, `HSW_lag_stability.csv`, `loco_paired_uncertainty.csv`, `lag_bin_sensitivity.csv` | Fig. 3; Table S4, S19; Fig. S10b, d |
| `calibration_audit.py` | `firm_supply_independent.csv`, `intervals_independent.csv`, `calibration_sensitivity.csv`, `adaptive_state_traces.csv`, `commitment_*.csv`, `calibration_paths.pkl`, `interval_paths.pkl` | Section 3.4, 3.8; Fig. 6c–f; Figs. S6, S11; Tables S9, S20–S22 |
| `matched_mlp.py` | `matched_MLP_metrics.csv`, `matched_MLP_selection.csv`, `matched_mlp_predictions.pkl` | Fig. 4d; Tables S10, S15 |
| `filled_value_sensitivity.py` | `filled_target_sensitivity.csv`, `filled_input_inventory.csv` | Section 2.1; S1 |
| `recheck_feeding_common_support.py` | `results/feeding_complete_support/` | Fig. 6a, b; Section 3.7 (1,791 complete-window origins) |
| `discussion_evidence.py` | output folder given as argument | Error quantiles, abrupt declines, 11 January 2023 event, perturbation Spearman correlation, HSW episodes (Sections 3.7, 3.8) |
| `compute_intervals.py` | `results/figure_intervals/` | Pointwise 95% block-bootstrap bands for Fig. 4a, f (and first-pass Fig. 5 bands) |
| `recompute_response_intervals.py` | `results/figure_intervals_segmented/` | Fig. 5 bands resampled within the two uninterrupted training segments (Section S6) |
| `gui_audit.py` | `gui_all_origin_validation.csv`, `gui_feature_roundoff.csv`, `gui_crossed_thresholds.csv` | Diagnosis of the earlier interface (Table S8; uses `reference_interface_source/predictor.py`) |
| `rebuild_interface.py`, `verify_interface.py`, `prepare_runtime_check.py`, `runtime_full_check.py`, `canonical_features.py` | `results/interface_rebuild/`, `gui_canonical_*`, `gui_full_predict_validation.csv` | Interface models and CSV-path equivalence (Section S8; Table S8) |
| `derive_findings.py` | printed summary | Key numbers in Sections 3.2–3.8 |

`reference_interface_source/` holds the code of the earlier interface version (its fitted models are not included); `rebuild_interface.py` starts from it.
