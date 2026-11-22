"""
export_streams.py - prepares the pack data that MATLAB streams over UDP.
Writes C:/ESS/<Mode>_fresh_stream.csv and C:/ESS/<Mode>_aged_stream.csv
(pack level, with sensor noise; aged = R0 x2, capacity -20 % from t = 300 s).
Run once before udp_stream.m.
"""
import os
import numpy as np
import pandas as pd
import bms_pipeline_v2 as bp

fresh = {m: bp.load_mode(m) for m in bp.MODES}
ocv = bp.identify_ocv(fresh)
rng = np.random.default_rng(bp.SEED)
for m in bp.MODES:
    for cond, d in [("fresh", fresh[m]), ("aged", bp.make_aged(fresh[m], ocv))]:
        dn = bp.add_noise(d, rng)
        out = pd.DataFrame(dict(Time_s=dn["t"], Voltage_V=dn["V_meas"] * bp.N_S,
                                Current_A=dn["I_meas"] * bp.N_P, Temp_degC=dn["T"],
                                SOC_pct=dn["soc"] * 100))
        path = os.path.join(bp.BASE, f"{m}_{cond}_stream.csv")
        out.to_csv(path, index=False)
        print(f"Saved {path} ({len(out)} samples)")