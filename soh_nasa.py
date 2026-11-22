"""
soh_nasa.py - Objective 2b: data-driven SOH estimation (NASA Ames 18650 cells)

Health indicators are taken from PARTIAL windows a BMS can see in normal use
(no full-discharge capacity measurement needed):
  - EVDT      : time for voltage to fall 3.8 V -> 3.5 V during discharge (s)
  - V_mean    : mean voltage in the first 1000 s of discharge (V)
  - dT_win    : temperature rise inside the 3.8 -> 3.5 V window (degC)
  - CC_time   : constant-current charge time until 4.2 V in the previous charge (s)
Target: SOH = measured capacity / 2.0 Ah x 100 %
Validation: leave-one-battery-out (train on 3 cells, test on the unseen 4th)
Outputs -> C:/ESS/bms/results/soh/
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("ESS_BASE", r"C:\ESS")
NASA = os.path.join(BASE, "datasets", "datasets", "nasa")
OUT = os.path.join(BASE, "bms", "results", "soh")
CELLS = ["B0005", "B0006", "B0007", "B0018"]
Q_NOM = 2.0
FEATURES = ["EVDT", "V_mean", "dT_win", "CC_time"]


def time_at(t, v, level):
    """First time the (falling) voltage crosses below `level`."""
    idx = np.where(v <= level)[0]
    if len(idx) == 0 or idx[0] == 0:
        return np.nan
    k = idx[0]
    return t[k - 1] + (level - v[k - 1]) * (t[k] - t[k - 1]) / (v[k] - v[k - 1])


def cc_charge_time(c):
    t, v = np.asarray(c["Time"], float), np.asarray(c["Voltage_measured"], float)
    idx = np.where(v >= 4.2)[0]
    return t[idx[0]] - t[0] if len(idx) else np.nan


def extract(cell):
    m = sio.loadmat(os.path.join(NASA, f"{cell}.mat"), simplify_cells=True)
    cycles = m[cell]["cycle"]
    rows, last_cc, n = [], np.nan, 0
    for c in cycles:
        if c["type"] == "charge":
            last_cc = cc_charge_time(c["data"])
        elif c["type"] == "discharge":
            d = c["data"]
            t = np.asarray(d["Time"], float)
            v = np.asarray(d["Voltage_measured"], float)
            T = np.asarray(d["Temperature_measured"], float)
            cap = float(np.atleast_1d(d["Capacity"])[0])
            n += 1
            t38, t35 = time_at(t, v, 3.8), time_at(t, v, 3.5)
            win = (t >= t38) & (t <= t35) if np.isfinite(t38) and np.isfinite(t35) else np.zeros_like(t, bool)
            rows.append(dict(
                Cell=cell, Cycle=n, Capacity_Ah=cap, SOH=cap / Q_NOM * 100,
                EVDT=t35 - t38,
                V_mean=v[t <= 1000].mean(),
                dT_win=T[win].max() - T[win].min() if win.sum() > 1 else np.nan,
                CC_time=last_cc if last_cc >= 100 else np.nan,   # < 100 s = aborted/partial charge
            ))
    return pd.DataFrame(rows)


def score(y, p):
    return dict(RMSE=np.sqrt(mean_squared_error(y, p)), MAE=mean_absolute_error(y, p),
                MaxAE=np.max(np.abs(y - p)), R2=r2_score(y, p))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    print("[1/4] Extracting health indicators from NASA cells...")
    df = pd.concat([extract(c) for c in CELLS], ignore_index=True)
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["SOH"])
    df.to_csv(os.path.join(OUT, "nasa_features.csv"), index=False)
    for c in CELLS:
        s = df[df.Cell == c]
        print(f"  {c}: {len(s)} discharge cycles, SOH {s.SOH.iloc[0]:.1f} -> {s.SOH.iloc[-1]:.1f} %")
    print(f"  ({n0 - len(df)} cycles dropped for missing data)")

    print("[2/4] Leave-one-battery-out validation...")
    models = {
        "Linear (EVDT only)": (lambda: LinearRegression(), ["EVDT"]),
        "Linear (all HIs)": (lambda: LinearRegression(), FEATURES),
        "RandomForest (all HIs)": (lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                                 random_state=42, n_jobs=-1), FEATURES),
    }
    rows, preds = [], []
    for test in CELLS:
        tr, te = df[df.Cell != test], df[df.Cell == test].copy()
        for name, (make, feats) in models.items():
            p = make().fit(tr[feats], tr.SOH).predict(te[feats])
            te[name] = p
            rows.append(dict(Test_cell=test, Model=name, **score(te.SOH.values, p)))
        preds.append(te)
    res = pd.DataFrame(rows)
    mean = res.groupby("Model")[["RMSE", "MAE", "MaxAE", "R2"]].mean().reset_index()
    mean.insert(0, "Test_cell", "MEAN")
    res = pd.concat([res, mean], ignore_index=True).round(3)
    res.to_csv(os.path.join(OUT, "soh_results.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(OUT, "soh_predictions.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))

    print("[3/4] SHAP on Random Forest (trained on all cells)...")
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(df[FEATURES], df.SOH)
    sv = shap.TreeExplainer(rf).shap_values(df[FEATURES])
    imp = pd.DataFrame({"Feature": FEATURES, "Mean_abs_SHAP": np.abs(sv).mean(axis=0)}) \
            .sort_values("Mean_abs_SHAP", ascending=False).round(3)
    imp.to_csv(os.path.join(OUT, "soh_shap_importance.csv"), index=False)
    print(imp.to_string(index=False))
    shap.summary_plot(sv, df[FEATURES], show=False)
    plt.savefig(os.path.join(OUT, "soh_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("[4/4] Saving plots...")
    P = pd.concat(preds)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, c in zip(axs.ravel(), CELLS):
        s = P[P.Cell == c]
        r = res[(res.Test_cell == c) & (res.Model == "Linear (EVDT only)")].iloc[0]
        ax.plot(s.Cycle, s.SOH, "k-", lw=2, label="Measured SOH")
        ax.plot(s.Cycle, s["Linear (EVDT only)"], "b.", ms=5, label="Linear (EVDT only)")
        ax.plot(s.Cycle, s["RandomForest (all HIs)"], "r--", lw=1, label="Random Forest (all HIs)")
        ax.axhline(70, color="grey", ls=":", label="EOL (70 %)")
        ax.set_title(f"{c} (unseen in training) - RMSE {r.RMSE:.2f} %")
        ax.set_xlabel("Discharge cycle"); ax.set_ylabel("SOH (%)"); ax.grid(alpha=.3)
    axs[0, 0].legend()
    fig.suptitle("SOH estimation, leave-one-battery-out (NASA 18650 cells)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "soh_predictions.png"), dpi=150)
    plt.close(fig)

    print(f"\nDone. Results saved in {OUT}")    