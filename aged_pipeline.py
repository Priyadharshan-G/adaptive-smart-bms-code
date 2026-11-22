# aged_pipeline.py
# Aged pack simulation: injects R0×2 and capacity -20% at t=300s
# Shows drift detection + FFRLS recalibration

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = r"C:\ESS\bms"
CSV_DIR = r"C:\ESS"

def load_csv(mode):
    path = os.path.join(CSV_DIR, f"{mode}_bms.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {mode}: {len(df)} rows, cols: {list(df.columns)}")
    return df

def inject_ageing(df, time_col="Time", r0_multiplier=2.0, capacity_factor=0.80):
    df = df.copy()
    aged_mask = df[time_col] >= 300

    if "R0" in df.columns:
        df.loc[aged_mask, "R0"] = df.loc[aged_mask, "R0"] * r0_multiplier
    else:
        df["R0"] = 0.003
        df.loc[aged_mask, "R0"] = 0.006

    if "SoC" in df.columns:
        pre_300_final_soc = df.loc[~aged_mask, "SoC"].iloc[-1] if (~aged_mask).any() else 100.0
        soc_aged = df.loc[aged_mask, "SoC"].values
        soc_aged_scaled = pre_300_final_soc - (pre_300_final_soc - soc_aged) / capacity_factor
        df.loc[aged_mask, "SoC_aged"] = soc_aged_scaled
        df.loc[~aged_mask, "SoC_aged"] = df.loc[~aged_mask, "SoC"]
    else:
        df["SoC_aged"] = df.get("SoC", 80.0)

    df["aged"] = aged_mask.astype(int)
    return df

def ffrls_recalibrate(df, time_col="Time", forgetting_factor=0.98):
    r0_col = "R0" if "R0" in df.columns else None
    if r0_col is None:
        print("  WARNING: No R0 column found, skipping FFRLS")
        df["R0_ffrls"] = np.nan
        return df

    r0_meas = df[r0_col].values
    n = len(r0_meas)
    theta = r0_meas[0]
    P = 1.0
    lam = forgetting_factor
    r0_est = np.zeros(n)

    for k in range(n):
        y = r0_meas[k]
        phi = 1.0
        K = P * phi / (lam + phi * P * phi)
        theta = theta + K * (y - phi * theta)
        P = (1 / lam) * (P - K * phi * P)
        r0_est[k] = theta

    df["R0_ffrls"] = r0_est
    return df

def detect_drift(df, time_col="Time", threshold_r0=0.0015, threshold_soc=5.0):
    drifts = []

    if "R0_ffrls" in df.columns and "R0" in df.columns:
        r0_drift = (df["R0_ffrls"] - df["R0"]).abs()
        r0_flags = r0_drift > threshold_r0
        n_r0 = r0_flags.sum()
        drifts.append(f"R0 drift events: {n_r0}")
        df["drift_R0"] = r0_flags.astype(int)

    if "SoC" in df.columns and "SoC_aged" in df.columns:
        soc_diff = (df["SoC_aged"] - df["SoC"]).abs()
        soc_flags = soc_diff > threshold_soc
        n_soc = soc_flags.sum()
        drifts.append(f"SoC drift events: {n_soc}")
        df["drift_SoC"] = soc_flags.astype(int)

    for d in drifts:
        print(f"  {d}")

    return df

if __name__ == "__main__":
    modes = ["Eco", "Normal", "Sport"]

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"  Processing: {mode} (AGED)")
        print(f"{'='*50}")

        try:
            df = load_csv(mode)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        time_col = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
        print(f"  Time column: '{time_col}'")

        print("  Injecting ageing (R0x2, capacity -20%) at t=300s...")
        df = inject_ageing(df, time_col=time_col)

        print("  Running FFRLS recalibration...")
        df = ffrls_recalibrate(df, time_col=time_col)

        print("  Detecting drift...")
        df = detect_drift(df, time_col=time_col)

        out_path = os.path.join(OUTPUT_DIR, f"{mode}_aged_bms.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")

    print("\n\nAll modes done. Check C:\\ESS\\bms\\ for *_aged_bms.csv files.")