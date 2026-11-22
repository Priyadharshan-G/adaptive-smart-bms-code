"""Table II (pack response) for an extra pack, from the MATLAB run_pack.m outputs.
Usage: python tools/pack_response_table.py LFP|NCR  -> packs/<PACK>/report/table_pack_response.csv"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = sys.argv[1]; Q = {"LFP": 2.341, "NCR": 3.153}[PACK]
D = os.path.join(ROOT, "packs", PACK)
x = pd.read_excel(os.path.join(D, f"{PACK}_mode_comparison.xlsx")).set_index("Mode")
rows = []
for m in ["Eco", "Normal", "Sport"]:
    d = pd.read_csv(os.path.join(D, f"{m}_timeseries.csv")); p = pd.read_csv(os.path.join(ROOT, f"{m}_timeseries.csv")); r = x.loc[m]
    rows.append({"Mode": m, "Max speed (km/h)": d.Speed_ms.max() * 3.6, "Peak acceleration (m/s2)": (np.diff(d.Speed_ms) / 10).max(),
                 "Peak battery power (kW)": r.PeakBattPower_kW, "Peak pack current (A)": r.PeakCurrent_A, "Peak cell current (A)": r.PeakCellCurrent_A,
                 "Initial voltage (V)": r.Vinit_V, "Final voltage (V)": r.Vfinal_V, "Minimum voltage (V)": r.Vmin_V, "Final SOC (%)": r.SOCfinal_pct,
                 "SOC used (%)": r.SOCused_pct, "Energy used (kWh)": r.Energy_kWh, "Charge used (Ah)": r.Charge_Ah,
                 "Peak C-rate (cell)": r.PeakCellCurrent_A / Q, "Max voltage sag (V)": (d.Voltage_V.iloc[0] - d.Voltage_V).max(),
                 "P45B peak current (A)": p.Current_A.max(), "P45B final SOC (%)": p.SOC_pct.iloc[-1]})
os.makedirs(os.path.join(D, "report"), exist_ok=True)
pd.DataFrame(rows).set_index("Mode").T.to_csv(os.path.join(D, "report", "table_pack_response.csv"))
print(pd.DataFrame(rows).set_index("Mode").T.round(3))
