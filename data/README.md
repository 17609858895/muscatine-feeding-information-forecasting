# Raw data

The raw records are not redistributed in this repository. Download them from the public dataset:

> Schroer, H.W., Just, C.L., 2024. Laboratory, operational, and SCADA datasets for anaerobic co-digestion at the Muscatine, IA Water Resource Recovery Facility. https://doi.org/10.25820/data.006715

Place the two files used here in this folder:

| File | Content | Used by |
|---|---|---|
| `SCADA-raw.csv` | One-minute SCADA records, 18 March 2022 – 28 February 2023 (500,400 rows) | `scripts/pipeline/prep.py`, `gas_use_split.py`, `energy_equivalents.py`, `revision_feeding_energy.py`, Fig. S1 builder |
| `LABS-raw.csv` | Daily laboratory records | `scripts/pipeline/compute_supp.py` (Fig. S2), `revision_feeding_energy.py` |

Alternatively, keep the files elsewhere and set environment variables with their full paths:

```bash
export SCADA_CSV=/path/to/SCADA-raw.csv
export LABS_CSV=/path/to/LABS-raw.csv
```

Source timestamps carry no timezone or UTC-offset information. The scripts do not convert time zones or correct daylight-saving transitions.
