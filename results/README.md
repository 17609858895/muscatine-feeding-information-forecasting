# Working directory for regenerated results

This folder is empty in the repository (its contents are ignored by git). The scripts in `scripts/pipeline/` are run with this folder as the working directory and write their intermediate files here (`hourly.pkl`, `tab_final.pkl`, `final_preds.pkl`, `extra/`, `supp/`, `tune/`, ...). The scripts in `scripts/revision/` read these files and write to the subfolders below.

| Subfolder | Written by |
|---|---|
| `revision_analysis/` | `scripts/revision/*.py` via `context.py` (independent ensembles, paired intervals, calibration, attribution, matched MLP, interface checks) |
| `group_refits/` | `scripts/revision/group_refits.py` |
| `feeding_complete_support/` | `scripts/revision/recheck_feeding_common_support.py` |
| `figure_intervals/` | `scripts/revision/compute_intervals.py` |
| `figure_intervals_segmented/` | `scripts/revision/recompute_response_intervals.py` |
| `interface_rebuild/`, `interface_runtime_check/` | `rebuild_interface.py`, `verify_interface.py`, `prepare_runtime_check.py`, `runtime_full_check.py` |
| `revision_20261126/` | `scripts/pipeline/revision_feeding_energy.py` |

Set `MUSCATINE_RESULTS` to use a different working directory, and `MUSCATINE_REVISION_OUT` to redirect `revision_analysis/`.

The retained outputs of the original runs are distributed as CSV files in `outputs/`.
