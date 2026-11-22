"""
make_paper_figures.py - print-ready figures for the manuscript (light theme, large fonts).
Reads only files already in the repository:
  <Mode>_timeseries.csv, bms/results/<Mode>_<Case>.csv, bms/results/robustness/*.csv
Outputs -> <ESS_BASE>/bms/results/paper_figures/
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("ESS_BASE", r"C:\ESS")
RES = os.path.join(BASE, "bms", "results")
OUT = os.path.join(RES, "paper_figures")
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})
os.makedirs(OUT, exist_ok=True)

# Drive cycles (10 s samples exported from Simulink)
fig, ax = plt.subplots(figsize=(8, 3.6))
for m, c in [("Eco", "tab:blue"), ("Normal", "tab:orange"), ("Sport", "tab:green")]:
    d = pd.read_csv(os.path.join(BASE, f"{m}_timeseries.csv"))
    ax.plot(d.Time_s, d.Speed_ms * 3.6, color=c, lw=1.6, label=m)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (km/h)"); ax.grid(alpha=.3); ax.legend(ncol=3, loc="upper center")
ax.set_xlim(0, 1800); ax.set_ylim(0, 150)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "drive_cycles.png"), dpi=250); plt.close(fig)

# Aged-pack comparison, one figure per mode, larger fonts
for m in ["Eco", "Normal", "Sport"]:
    a = pd.read_csv(os.path.join(RES, f"{m}_Aged-adaptive.csv"))
    s = pd.read_csv(os.path.join(RES, f"{m}_Aged-static.csv"))
    fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    ax[0].plot(a.t, a.SOC_true, "k-", lw=2.2, label="True")
    ax[0].plot(s.t, s.SOC_est, "r--", lw=1.6, label="Static")
    ax[0].plot(a.t, a.SOC_est, "b-", lw=1.6, label="Adaptive")
    ax[0].set_ylabel("SOC (%)"); ax[0].legend(loc="upper right")
    ax[1].plot(a.t, a.DM, "b-", lw=1.2)
    ax[1].axhline(3.5, color="r", ls="--", lw=1.4)
    ax[1].scatter(a.t[a.drift == 1], a.DM[a.drift == 1], c="r", s=22, zorder=5)
    ax[1].set_ylabel("$D_M$ (–)")
    ax[2].plot(a.t, a.R0_true_mOhm, "k-", lw=2.2, label="True")
    ax[2].plot(a.t, a.R0_est_mOhm, "b-", lw=1.6, label="FFRLS")
    ax[2].set_ylabel("$R_0$ (m$\\Omega$)"); ax[2].set_xlabel("Time (s)"); ax[2].legend(loc="lower right")
    for x in ax:
        x.axvline(300, color="grey", ls=":"); x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{m.lower()}_aged_comparison.png"), dpi=250); plt.close(fig)

# Threshold sweep
ts = pd.read_csv(os.path.join(RES, "robustness", "tau_sweep_summary.csv"))
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
for m, c in [("Eco", "tab:blue"), ("Normal", "tab:orange"), ("Sport", "tab:green")]:
    q = ts[ts.Mode == m]
    ax[0].plot(q.Tau, 100 * q.FalseAlarmRate, "o-", color=c, label=m)
    ax[1].plot(q.Tau, 100 * q.DetectionRate, "o-", color=c, label=m)
ax[0].set_xlabel(r"Threshold $\tau$ (–)"); ax[0].set_ylabel("False-alarm rate (%)")
ax[1].set_xlabel(r"Threshold $\tau$ (–)"); ax[1].set_ylabel("Detection rate (%)")
for x in ax:
    x.grid(alpha=.3); x.set_ylim(-5, 105)
ax[1].legend(loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tau_sweep.png"), dpi=250); plt.close(fig)
print("Figures saved in", OUT)

# Table plant example (Sport x4, gradual ageing, seed 0), recomputed with bms_robustness
import numpy as np
import bms_robustness as br
tp = br.TablePlant(br.load_tables())
pocv = br.PWLOCV(tp)
t, p = br.chained_power("Sport")
rf, qf = br.ageing_profile(t, "ramp")
aged = tp.simulate(t, p, rf=rf, qf=qf)
rng = np.random.default_rng(5000)
# same noise draw order as bms_robustness: fresh, step-aged, then ramp-aged
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
ax[0].set_ylabel("SOC (%)"); ax[0].legend(loc="upper right", ncol=2)
ax[1].plot(e.t / 60, e.R0_true_mOhm, "k-", lw=2.2, label="True")
ax[1].plot(e.t / 60, e.R0_est_mOhm, "b-", lw=1.6, label="Gated FFRLS")
ax[1].set_ylabel("$R_0$ (m$\\Omega$)"); ax[1].legend(loc="lower right")
ax[2].plot(e.t / 60, e.Q_true_Ah, "k-", lw=2.2, label="True")
ax[2].plot(e.t / 60, e.Q_est_Ah, "b-", lw=1.6, label="Rest-point estimate")
ax[2].set_ylabel("Capacity (Ah)"); ax[2].set_xlabel("Time (min)"); ax[2].legend(loc="lower left")
for x in ax:
    x.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "table_plant_sport_ramp.png"), dpi=250); plt.close(fig)
print("Table-plant figure saved")
