# Adaptive Smart BMS: drift-gated online recalibration for EV battery packs

Simulation code and data for the paper "Internal Resistance Drift Detection and Adaptive Online Recalibration for Electric Vehicle Battery Management Systems" (Amrita School of Engineering, Coimbatore).

A software-only Battery Management System for a simulated 400 V EV pack. The BMS estimates State of Charge (SOC) and State of Health (SOH), detects when the battery starts behaving differently (ageing), recalibrates its own model parameters online, and explains its predictions, all shown in a live dashboard.

## How it works

```
Simulink 98S30P pack (Molicel P45B)  ──  Eco / Normal / Sport drive cycles, 1800 s
        │  CSV export  /  live UDP stream (MATLAB → Python, 127.0.0.1:5005)
        ▼
Python adaptive BMS
   Coulomb counting + OCV correction  →  SOC
   Mahalanobis drift detection        →  "battery has changed" alarm
   FFRLS recalibration (R0, capacity) →  updated model, SOH
        │
        ├── Ageing test: R0 ×2, capacity −20 % injected at t = 300 s
        ├── KNN imputation of missing sensor data
        ├── ML SOC (LG HG2) and ML SOH (NASA) with SHAP / LIME explanations
        ▼
Streamlit dashboard (7 tabs, including live UDP view)
```

## Key results

| Experiment | Result |
|---|---|
| Adaptive vs static BMS on aged pack (SOC RMSE) | Eco 0.118 → 0.070 %, Normal 0.285 → 0.175 %, Sport 2.034 → 0.483 % (39–76 % lower) |
| Drift detection | 0 false alarms on fresh pack; detected 0–250 s after ageing onset |
| R0 tracking (true 20 mΩ) | 20.93 / 19.94 / 20.34 mΩ (Eco / Normal / Sport) |
| ML SOC, LG HG2 (Random Forest) | RMSE 1.72 %; averaged voltage dominates (SHAP) |
| ML SOH, NASA, leave-one-battery-out | Linear model on EVDT: mean RMSE 2.35 %, R² 0.939 |
| Sensor-dropout imputation | Gap-aware hybrid KNN: 15.7 % lower RMSE than linear on burst current dropouts, never worse elsewhere |
| MATLAB → Python UDP bridge | 181/181 packets, 0 % loss, p95 latency 0.79 ms, BMS step 0.61 ms |

## Dashboard v2 (three packs, live UDP)

```
pip install streamlit plotly pandas numpy scipy scikit-learn
streamlit run dashboard_v2.py
```
Opens in the browser. Pick the pack (P45B / NCR / LFP), drive mode and condition in the sidebar.
Tabs: Overview · Playback (animated BMS run) · Live UDP · Ageing (30-seed results) · Detector ·
Compare packs · Imputation · SOC & SOH models.

**Live UDP**, with the dashboard listening on 127.0.0.1:5005 and running the selected pack's BMS
on every packet:
1. Live UDP tab → **Start listening**.
2. Either press **Send demo stream** (no MATLAB needed; or `python live_udp.py send --pack NCR --aged`),
   or run `udp_stream_pack.m` in MATLAB with the same pack (run `python live_udp.py export` once
   first to create the NCR/LFP stream files; P45B uses the original files).

Packets are the same 7 doubles as `udp_stream.m`. Do not run `udp_receiver.py` at the same time;
both need port 5005. Dashboard runs are saved to `bms/results/live_dashboard/`, so the original
live-run results in `bms/results/live/` are never overwritten. The old `dashboard.py` is unchanged.

## Additional battery packs (NCR and LFP)

The same BMS was rerun, unchanged, on a Panasonic NCR18650BD 98S45P pack and an A123 ANR26650
LFP 108S57P pack. Models are `ESSSimulation_NCR.slx` and `ESSSimulation_LFP.slx`; data, results,
figures and tables are under [`packs/`](packs/README.md), and the scripts under `tools/`.
In short: the method carries over to NCR (deep-discharge SOC error down 65–88 %, 2.2 % false
alarms), but the flat LFP voltage curve defeats the SOC correction and the drift test in Sport.

## Repository layout

**Current scripts (use these)**

| File | Purpose |
|---|---|
| `ESSSimulation.slx`, `P45B_98S30P_Library_lib.slx`, `+P45B_98S30P_Library/` | Simulink pack model and cell library |
| `setup_battery_component.m` | Rebuilds the 7S30P module component |
| `*_profile.m`, `make_profiles.m`, `run_mode.m`, `export_timeseries.m` | Drive profiles, simulation runs, CSV export |
| `bms_pipeline_v2.py` | Adaptive BMS: Fresh / Aged-static / Aged-adaptive comparison |
| `shap_lime.py` | ML SOC on LG HG2 + SHAP + LIME |
| `knn_imputation.py` | Sensor-dropout benchmark (linear, KNN, KNN+context, hybrid) |
| `soh_nasa.py` | ML SOH on NASA cells + SHAP |
| `export_streams.py`, `udp_stream.m`, `udp_receiver.py` | Live MATLAB → Python UDP bridge |
| `dashboard.py` | Streamlit dashboard |

**Legacy (kept for history, do not use for results)**
`bms/bms_pipeline.py`, `aged_pipeline.py`, `Eco_bms.csv`, `Normal_bms.csv`, `Sport_bms.csv`, `bms/*_aged_bms.csv`. The first pipeline fed pack-level voltage into a cell-level model, so its SOC estimate stayed at 100 % and its drift detector never activated. `bms_pipeline_v2.py` replaces it.

## How to run

Requirements: Python 3.12, MATLAB R2025a (Simulink, Simscape).

```
pip install numpy pandas scipy scikit-learn matplotlib shap lime streamlit plotly
```

The `datasets/` folder is not in the repo (over 1 GB). Download the LG 18650HG2 dataset (McMaster University) and NASA PCoE battery dataset and place them under `C:\ESS\datasets\datasets\`.

```
python bms_pipeline_v2.py      # adaptive BMS results
python shap_lime.py            # ML SOC + explainability
python knn_imputation.py       # imputation benchmark
python soh_nasa.py             # ML SOH
streamlit run dashboard.py     # dashboard
```

Live demo: `streamlit run dashboard.py` (open the Live UDP tab) → `python export_streams.py` (once) → `python udp_receiver.py` → in MATLAB: `cd C:\ESS`, `udp_stream`.

## Limitations

- The BMS model closely matches the simulated cell, so SOC errors are optimistic compared with real hardware.
- Drive cycles cover a narrow SOC window (Eco 100 → 98 %, Sport 100 → 85 %); capacity is only re-estimated after ≥ 3 % SOC change.
- Simulated temperature stays at 25 °C, so thermal effects are not exercised.
- Ageing is injected as a step change at t = 300 s rather than gradual degradation.
- The SOH model is trained on four NASA cells; B0006, which degrades below the training range, has the largest error (4.4 % RMSE).