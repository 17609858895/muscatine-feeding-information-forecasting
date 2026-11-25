# Fitted research models (LGB-Δ and Cat-Δ)

The 36 persistence-anchored tree models used in the article: LightGBM (LGB-Δ) and CatBoost (Cat-Δ) for nine horizons
(1, 2, 3, 4, 6, 9, 12, 18 and 24 h) and two information regimes (`past` = historical inputs only, 132 predictors;
`feed` = historical inputs plus recorded future feeding, 135 predictors). They were fitted on the 5,421 training origins
with the fixed configurations reported in the Supporting Information and are the tree components of the ensembles.

| File | Format | Load with |
|---|---|---|
| `lgb_delta_{past,feed}_h{h}.txt.gz` | LightGBM text model, gzip-compressed | `lightgbm.Booster(model_str=gzip.open(path, 'rt').read())` |
| `cat_delta_{past,feed}_h{h}.cbm` | CatBoost binary model | `catboost.CatBoostRegressor().load_model(path)` |
| `index.json` | horizon, regime, file names and the ordered feature names of each model | `json.load` |

Inputs are the process-group lag-bin (PG-LB) features built by `scripts/pipeline/feats.py` from hourly SCADA records in
their native units (ft³ min⁻¹, US gal min⁻¹, ft, °F, MGD), followed by the four calendar terms and, for `feed` models,
the three future-feeding window means (`Planned_HSW|F`, `Planned_PS|F`, `Planned_TWAS|F`). Feature order must follow
`index.json`.

Each model predicts the increment y(t + h) − y(t) of metered biogas flow in ft³ min⁻¹. The forecast is the current flow
y(t) plus this increment; multiply by 1.69901 to obtain m³ h⁻¹. The forecasts in `outputs/predictions/` were produced by
these models before export.
