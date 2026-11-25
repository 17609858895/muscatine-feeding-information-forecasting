# Local prediction interface (Supporting Information, Section S8)

A local, offline research prototype that returns hourly metered biogas forecasts (flow to the boiler and waste-gas burner, m³ h⁻¹) at horizons 1, 2, 3, 4, 6, 9, 12, 18 and 24 h from persistence-anchored LightGBM models (LGB-Δ), with and without a supplied 24 h feeding plan. It is not connected to plant SCADA and has not been evaluated by operators.

## Run

Python 3.12:

```bash
cd interface
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python server.py                               # then open http://127.0.0.1:8765/
```

On Windows, `install_runtime.ps1` creates the `.venv` and `start_gui.ps1` (or `start_gui.cmd`) starts the server and opens the browser. The server listens on 127.0.0.1 only; use `server.py --port 8766` if the port is taken.

Programmatic use:

```python
import predictor
result = predictor.predict(open('examples/history_168h.csv').read(),
                           open('examples/recorded_feed_oracle_24h.csv').read())   # plan is optional
```

## Inputs

- **History CSV**: at least 168 consecutive hourly rows ending at the last complete hour *t*, column `timestamp` plus the 16 SI-unit columns of `examples/history_168h.csv` (Table S8a). Timestamps are interval-start labels without timezone; values must be finite, flows non-negative and the PS on-time fraction within [0, 1]. Nothing is filled automatically.
- **Plan CSV (optional)**: exactly 24 rows labelled *t* + 1 … *t* + 24 with `hsw_m3_h` (total HSW to both digesters), `ps_on_fraction` and `twas_m3_h`. The example `recorded_feed_oracle_24h.csv` contains executed feeding, not a schedule that was available at issuance; an edited plan is a conditional input scenario, not evidence of the process response to a changed feeding policy.

## Models and intervals

- `models/{past,feed}_{h}.txt`: 18 LightGBM boosters refitted with the fixed LGB-Δ configuration on the original training origins, using the shared feature function in `canonical_features.py` (native-unit inputs and lag-bin means rounded to six decimals, no full-series rolling state). They differ slightly from the research models: day-ahead test RMSE is 58.63 (history) and 46.10 m³ h⁻¹ (with plan), versus 58.17 and 46.03 for the research LGB-Δ models and 57.48 and 45.52 for the ensembles reported in the article.
- `models/calibration.json`: static nominal 90% half-widths from absolute errors on the separate 360 calibration origins (order statistic ⌈(n + 1)·0.9⌉). These differ from the adaptive intervals used for the ensembles; lower bounds are not truncated and can be negative.
- `models/history_bounds.json`: training ranges used to flag out-of-range inputs.

## Validation

`validation/` contains the checks reported in Table S8 and Section S8:

- `gui_canonical_validation.csv`: test RMSE, MAE, R², coverage and half-width for each horizon and regime.
- `gui_full_predict_validation.csv`: all 1,797 test origins × 9 horizons × 2 regimes passed through the CSV parser and `predict()`; maximum difference from offline predictions 0.
- `installed_runtime_audit.json`: the same check under NumPy 2.2.6, pandas 3.0.3 and LightGBM 4.6.0.
- `gui_all_origin_validation.csv`, `gui_feature_roundoff.csv`, `gui_crossed_thresholds.csv`: diagnosis of the earlier interface version (CSV-path deviations up to 8.53 m³ h⁻¹ caused by window-mean round-off crossing tree thresholds), which motivated the shared six-decimal feature function.
- `interface_example.json`: output for the example files.

The interface is rebuilt and verified by `scripts/revision/rebuild_interface.py`, `verify_interface.py`, `prepare_runtime_check.py` and `runtime_full_check.py`.

An older online demonstration (Streamlit) is available at https://github.com/17609858895/Muscatine-Biogas-Streamlit; it predates this version and has not been updated.
