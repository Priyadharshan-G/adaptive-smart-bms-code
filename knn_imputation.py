"""
knn_imputation.py - Objective 1: missing sensor data reconstruction

Injects sensor dropouts into the Simulink pack data (random single samples and
bursts, 5/10/20 %), then reconstructs them with:
  - Linear interpolation (baseline)
  - KNN (k = 5, values of other channels at the same time step)
  - KNN + temporal context (also uses neighbouring samples, +/- 2 steps)
  - Hybrid (linear by default; KNN + context only for gaps of >= 3 missing
            samples in pack current, the spiky signal SOC counting depends on)
Outputs -> C:/ESS/bms/results/imputation/
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("ESS_BASE", r"C:\ESS")
MODES = ["Eco", "Normal", "Sport"]
# --sensors: benchmark only the channels a BMS actually measures (voltage, current)
# --can: voltage and current plus vehicle speed, which a BMS can read from the CAN bus
SENSORS_ONLY = "--sensors" in sys.argv
WITH_CAN = "--can" in sys.argv
if SENSORS_ONLY:
    CHANNELS, OUT = ["Voltage_V", "Current_A"], os.path.join(BASE, "bms", "results", "imputation_sensors")
elif WITH_CAN:
    CHANNELS, OUT = ["Voltage_V", "Current_A", "Speed_ms"], os.path.join(BASE, "bms", "results", "imputation_can")
else:
    CHANNELS, OUT = ["Voltage_V", "Current_A", "SOC_pct", "Speed_ms"], os.path.join(BASE, "bms", "results", "imputation")
FAST = ["Current_A"]                      # spiky signal -> KNN + context for long gaps
MIN_GAP = 3                               # gaps of >= 3 missing samples use KNN
RATES = [0.05, 0.10, 0.20]
PATTERNS = ["random", "burst"]
TRIALS = 20
K = 5
LAGS = (1, 2)


def inject_dropouts(X, rate, pattern, rng):
    v = X.values.astype(float).copy()
    mask = np.zeros(v.shape, dtype=bool)
    n = len(v)
    for j in range(v.shape[1]):
        if pattern == "random":
            mask[rng.choice(np.arange(1, n - 1), int(rate * n), replace=False), j] = True
        else:                               # bursts of 3-6 consecutive samples
            while mask[:, j].mean() < rate:
                s = rng.integers(1, n - 7)
                mask[s:s + rng.integers(3, 7), j] = True
    v[mask] = np.nan
    return pd.DataFrame(v, columns=X.columns), mask


def linear(M, X):
    return M.interpolate(limit_direction="both").values


def knn(M, X, lags=()):
    lo, sc = X.min().values, (X.max() - X.min()).replace(0, 1).values
    Z = (M - lo) / sc
    feats = [Z]
    for l in lags:
        feats += [Z.shift(l).add_suffix(f"_m{l}"), Z.shift(-l).add_suffix(f"_p{l}")]
    F = pd.concat(feats, axis=1)
    out = KNNImputer(n_neighbors=K, weights="distance").fit_transform(F)[:, :X.shape[1]]
    return out * sc + lo


def knn_context(M, X):
    return knn(M, X, LAGS)


def gap_lengths(miss):
    """Length of the missing-run each sample belongs to (0 if present)."""
    out = np.zeros(len(miss), dtype=int)
    i = 0
    while i < len(miss):
        if miss[i]:
            j = i
            while j < len(miss) and miss[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def hybrid(M, X):
    """Linear everywhere, except long gaps (>= MIN_GAP) in spiky signals -> KNN + context."""
    out = linear(M, X).copy()
    kc = knn_context(M, X)
    for c in FAST:
        j = list(X.columns).index(c)
        use = gap_lengths(M[c].isna().values) >= MIN_GAP
        out[use, j] = kc[use, j]
    return out


METHODS = {"Linear": linear, "KNN": knn, "KNN+context": knn_context, "Hybrid": hybrid}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []

    print("[1/3] Running imputation benchmark...")
    for mode in MODES:
        X = pd.read_csv(os.path.join(BASE, f"{mode}_timeseries.csv"))[CHANNELS]
        for pattern in PATTERNS:
            for rate in RATES:
                for _ in range(TRIALS):
                    M, mask = inject_dropouts(X, rate, pattern, rng)
                    for name, f in METHODS.items():
                        P = f(M, X)
                        for j, c in enumerate(CHANNELS):
                            e = P[mask[:, j], j] - X[c].values[mask[:, j]]
                            rows.append(dict(Mode=mode, Pattern=pattern, Rate=int(rate * 100),
                                             Method=name, Channel=c,
                                             RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e))))
        print(f"  {mode} done")

    df = pd.DataFrame(rows)
    detail = (df.groupby(["Mode", "Pattern", "Rate", "Channel", "Method"])[["RMSE", "MAE"]]
                .mean().round(4).reset_index())
    detail.to_csv(os.path.join(OUT, "imputation_detail.csv"), index=False)

    print("[2/3] Summary (RMSE averaged over modes, rates and trials)")
    summ = df.groupby(["Pattern", "Channel", "Method"])["RMSE"].mean().unstack("Method")
    summ = summ[list(METHODS)].round(4)
    summ["Best"] = summ.idxmin(axis=1)
    summ["Hybrid_vs_Linear_%"] = ((1 - summ["Hybrid"] / summ["Linear"]) * 100).round(1)
    summ.to_csv(os.path.join(OUT, "imputation_summary.csv"))
    pd.set_option("display.width", 200)
    print(summ.to_string())

    print("[3/3] Saving plots...")
    units = {"Voltage_V": "V (pack)", "Current_A": "A (pack)", "SOC_pct": "%", "Speed_ms": "m/s"}
    fig, axs = plt.subplots(2, len(CHANNELS), figsize=(4 * len(CHANNELS), 7), squeeze=False)
    for r, pattern in enumerate(PATTERNS):
        for j, c in enumerate(CHANNELS):
            s = summ.loc[(pattern, c), list(METHODS)]
            axs[r, j].bar(s.index, s.values, color=["grey", "orange", "royalblue", "green"])
            axs[r, j].set_title(f"{c} - {pattern} dropouts", fontsize=10)
            axs[r, j].set_ylabel(f"RMSE ({units[c]})")
            axs[r, j].tick_params(axis="x", rotation=30, labelsize=8)
            axs[r, j].grid(alpha=.3, axis="y")
    fig.suptitle("Imputation RMSE by method (avg. over Eco/Normal/Sport, 5-20 % dropouts)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "imputation_rmse.png"), dpi=150)
    plt.close(fig)

    # Example reconstruction: Sport current, 20 % burst dropouts
    X = pd.read_csv(os.path.join(BASE, "Sport_timeseries.csv"))
    M, mask = inject_dropouts(X[CHANNELS], 0.20, "burst", np.random.default_rng(7))
    j = CHANNELS.index("Current_A")
    t = X["Time_s"].values
    lin_c, hyb_c = linear(M, X[CHANNELS])[:, j], hybrid(M, X[CHANNELS])[:, j]
    plt.figure(figsize=(12, 4))
    plt.plot(t, X["Current_A"], "k-", lw=2, label="True")
    plt.plot(t, lin_c, "--", color="grey", label="Linear interpolation")
    plt.plot(t, hyb_c, "-", color="green", label="Hybrid (gap-aware KNN)")
    plt.scatter(t[mask[:, j]], X["Current_A"].values[mask[:, j]], c="red", s=12, zorder=5, label="Dropped samples")
    plt.xlabel("Time (s)"); plt.ylabel("Pack current (A)")
    plt.title("Sport mode: reconstructing 20 % burst dropouts in pack current")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "imputation_example_sport_current.png"), dpi=150)
    plt.close()

    print(f"\nDone. Results saved in {OUT}")