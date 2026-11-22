# Extra battery packs: Panasonic NCR18650BD and A123 ANR26650 (LFP)

The adaptive BMS of the paper was rerun, with no retuning, on two more 400 V-class packs
built with the Simscape Battery Builder. Same vehicle, drive cycles, noise seeds (30),
ageing cases and BMS settings as the P45B study.

| | P45B (repo root) | NCR | LFP |
|---|---|---|---|
| Cell | Molicel P45B 21700 | Panasonic NCR18650BD | A123 ANR26650 (LFP) |
| Pack | 98S30P | 98S45P | 108S57P |
| Simulink model | `ESSSimulation.slx` | `ESSSimulation_NCR.slx` | `ESSSimulation_LFP.slx` |
| BMS nominal R0 | 10 mOhm | 47 mOhm | 15 mOhm |
| OCV fit in the BMS | cubic (eq. 5) | cubic (eq. 5) | monotone PCHIP |
| Thermal model | yes | no (25 C) | no (25 C) |

## Folders
- `NCR/`, `LFP/`
  - `Eco|Normal|Sport_timeseries.csv` – Simulink exports from `run_pack.m` (inputs to the Python BMS)
  - `<PACK>_mode_comparison.xlsx` – MATLAB summary per mode
  - `bms/results/` – everything the pipeline writes (seed-42 runs, `robustness/`, `revision2/`,
    `paper_figures/`, `imputation_can/`)
  - `report/` – `figures/`, `collages/`, `<PACK>_results_tables.xlsx` (LFP/NCR versions of the
    paper tables II–XI with the P45B rerun alongside), `table_pack_response.csv`
- `LFP_polynomial_OCV/` – the LFP pack run with the paper's polynomial OCV fit, kept to show why
  it fails on LFP (degenerates to a straight line; SOC RMSE 1.3–3.4 % even on a healthy pack)
- `cross_pack.json`, `cross_pack_summary.png` – numbers and chart comparing all three packs

## Key results (30 seeds)
| | P45B | NCR | LFP |
|---|---|---|---|
| Matched plant, step ageing, Sport SOC RMSE static -> gated (%) | 2.004 -> 0.552 | 3.284 -> 1.390 | 39.56 -> 10.55 |
| Table plant, step ageing, reduction vs static (Eco/Normal/Sport) | 22/40/69 % | 65/74/88 % | 9/48/28 % |
| Capacity estimate error, Sport step | +0.2 % | +0.2 % | +11.5 % |
| Detector 2-of-3, tau 3.5: false alarm / detection | 2.2 / 96.7 % | 2.2 / 100 % | 81.1 / 66.7 % |

The method carries over to NCR; the flat LFP OCV curve breaks both the SOC correction and the
drift detector in Sport. See the paper's Section VII for the discussion.

## Reproduce
MATLAB (R2025a, Simscape Battery), from the repo root:
```
run_pack('NCR')   % writes NCR_<Mode>_timeseries.csv -> copy to packs/NCR/<Mode>_timeseries.csv
run_pack('LFP')
```
Python, from the repo root (about 20 min per pack):
```
tools/run_pack_analysis.sh NCR poly
tools/run_pack_analysis.sh LFP pchip
tools/run_pack_analysis.sh LFP poly      # optional: packs/LFP_polynomial_OCV
python tools/pack_response_table.py NCR    # and LFP
python tools/make_pack_tables.py NCR       # and LFP
python tools/make_ncr_report.py
python tools/make_lfp_report.py
python tools/cross_pack.py && python tools/cross_pack_figure.py
```
The pack is selected with the `ESS_PACK` environment variable (`P45B` default, `NCR`, `LFP`) and
the OCV fit with `ESS_OCV` (`poly` default, `pchip`). With defaults, the scripts reproduce the
P45B results of the paper exactly.

## MATLAB files at the repo root
`build_new_packs.m`, `build_lfp_pack.m`, `apply_cell_part.m` build the packs and copy the
manufacturer cell data into them; `run_pack.m` runs the three modes and exports;
`export_LFP_results_SAVE_FIGURES.m` saves scope figures. The generated Battery Builder libraries
(`*_Library_lib.slx`, `*_Library.slx`, `*_Library.mat`, `+*_Library/` packages) are committed so
the models open without rebuilding.
