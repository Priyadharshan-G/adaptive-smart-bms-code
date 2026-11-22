"""
make_extra_figures.py - per-mode versions of the table-plant and imputation figures,
so Eco and Normal are shown alongside Sport.
Outputs -> <ESS_BASE>/bms/results/paper_figures/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bms_pipeline_v2 as bp
import bms_robustness as br

OUT = os.path.join(bp.BASE, "bms", "results", "paper_figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 10,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

# ---------------------------------------------------------------- table plant
tp = br.TablePlant(br.load_tables())
pocv = br.PWLOCV(tp)
for mode in bp.MODES:
    t, p = br.chained_power(mode)
    rr, qr = br.ageing_profile(t, "ramp")
    aged = tp.simulate(t, p, rf=rr, qf=qr)
    # reproduce the noise draw order of bms_robustness: fresh, step-aged, ramp-aged
    rng = np.random.default_rng(5000)
    br.noisy(tp.simulate(t, p), rng, br.SIGMA_T)
    rs, qs = br.ageing_profile(t, "step")
    br.noisy(tp.simulate(t, p, rf=rs, qf=qs), rng, br.SIGMA_T)
    an = br.noisy(aged, rng, br.SIGMA_T)
    ex = {c: br.run(an, pocv, c, use_temp=False) for c in ["Static", "Ungated", "Gated"]}
    e = ex["Gated"]
    fig, ax = plt.subplots(3, 1, figsize=(6, 7.2), sharex=True)
    ax[0].plot(e.t / 60, e.SOC_true, "k-", lw=2.2, label="True")
    ax[0].plot(ex["Static"].t / 60, ex["Static"].SOC_est, "r--", lw=1.4, label="Static")
    ax[0].plot(ex["Ungated"].t / 60, ex["Ungated"].SOC_est, ":", color="orange", lw=1.6, label="Ungated")
    ax[0].plot(e.t / 60, e.SOC_est, "b-", lw=1.4, label="Gated")
    ax[0].set_ylabel("SOC (%)"); ax[0].legend(loc="lower left", ncol=2)
    ax[1].plot(e.t / 60, e.R0_true_mOhm, "k-", lw=2.2, label="True")
    ax[1].plot(e.t / 60, e.R0_est_mOhm, "b-", lw=1.6, label="Gated FFRLS")
    ax[1].set_ylabel("$R_0$ (m$\\Omega$)"); ax[1].legend(loc="lower right")
    ax[2].plot(e.t / 60, e.Q_true_Ah, "k-", lw=2.2, label="True")
    ax[2].plot(e.t / 60, e.Q_est_Ah, "b-", lw=1.6, label="Rest-point estimate")
    ax[2].set_ylabel("Capacity (Ah)"); ax[2].set_xlabel("Time (min)"); ax[2].legend(loc="lower left")
    for x in ax:
        x.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"table_plant_{mode.lower()}_ramp.png"), dpi=250)
    plt.close(fig)
    print(f"table plant {mode}: SOC {e.SOC_true.iloc[-1]:.1f}% end, "
          f"Q est {e.Q_est_Ah.iloc[-1]:.3f} vs true {e.Q_true_Ah.iloc[-1]:.3f} Ah")

# ------------------------------------------------------------- imputation
sys.argv.append("--can")                      # voltage, current, CAN speed
import knn_imputation as ki                   # module-level guard keeps the main block from running

fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
for ax, mode in zip(axs, bp.MODES):
    X = pd.read_csv(os.path.join(bp.BASE, f"{mode}_timeseries.csv"))
    M, mask = ki.inject_dropouts(X[ki.CHANNELS], 0.20, "burst", np.random.default_rng(7))
    j = ki.CHANNELS.index("Current_A")
    t = X["Time_s"].values
    lin = ki.linear(M, X[ki.CHANNELS])[:, j]
    hyb = ki.hybrid(M, X[ki.CHANNELS])[:, j]
    true = X["Current_A"].values
    ax.plot(t, true, "k-", lw=1.8, label="True")
    ax.plot(t, lin, "--", color="grey", lw=1.4, label="Linear")
    ax.plot(t, hyb, "-", color="green", lw=1.4, label="Hybrid (gap-aware KNN)")
    ax.scatter(t[mask[:, j]], true[mask[:, j]], c="red", s=14, zorder=5, label="Dropped")
    e_lin = np.sqrt(np.mean((lin[mask[:, j]] - true[mask[:, j]]) ** 2))
    e_hyb = np.sqrt(np.mean((hyb[mask[:, j]] - true[mask[:, j]]) ** 2))
    ax.set_ylabel("Pack current (A)")
    ax.set_title(f"{mode}: linear {e_lin:.1f} A, hybrid {e_hyb:.1f} A RMSE", fontsize=12)
    ax.grid(alpha=.3)
    print(f"imputation {mode}: linear {e_lin:.2f} A, hybrid {e_hyb:.2f} A")
axs[0].legend(loc="upper left", ncol=2, fontsize=9)
axs[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "imputation_all_modes.png"), dpi=250)
plt.close(fig)
print("figures saved in", OUT)
