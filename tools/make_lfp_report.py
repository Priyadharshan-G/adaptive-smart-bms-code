"""Figures and collages for the LFP pack -> packs/LFP/report/. Run after tools/run_pack_analysis.sh."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def pack_dir(p):
    """P45B lives at the repo root; extra packs under packs/<name>."""
    return ROOT if p == "P45B" else os.path.join(ROOT, "packs", {"LFP_poly": "LFP_polynomial_OCV"}.get(p, p))
L, P = pack_dir("LFP"), pack_dir("P45B")
OUT = os.path.join(L, "report"); FIG = f"{OUT}/figures"; COL = f"{OUT}/collages"
for d in (FIG, COL): os.makedirs(d, exist_ok=True)
MODES = ["Eco", "Normal", "Sport"]; C = {"Eco": "tab:blue", "Normal": "tab:orange", "Sport": "tab:green"}
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10})


def stitch(files, out, ncol, labels=None, title=None):
    ims = [Image.open(f).convert("RGB") for f in files]
    w = max(i.width for i in ims); h = max(i.height for i in ims)
    ims = [i.resize((w, int(i.height * w / i.width))) for i in ims]
    nrow = int(np.ceil(len(ims) / ncol))
    rowh = [max(i.height for i in ims[r * ncol:(r + 1) * ncol]) for r in range(nrow)]
    top = 90 if title else 0; lab = 70 if labels else 0
    ys = np.cumsum([top] + [h_ + lab for h_ in rowh])
    canvas = Image.new("RGB", (ncol * w, int(ys[-1])), "white")
    from PIL import ImageDraw, ImageFont
    dr = ImageDraw.Draw(canvas)
    try:
        fnt = ImageFont.truetype("DejaVuSans.ttf", 44); fntt = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)
    except Exception:
        fnt = fntt = ImageFont.load_default()
    if title:
        dr.text((ncol * w / 2, 45), title, fill="black", font=fntt, anchor="mm")
    for k, im in enumerate(ims):
        r, c = divmod(k, ncol)
        x, y = c * w, int(ys[r])
        canvas.paste(im, (x, y))
        if labels:
            dr.text((x + w / 2, y + rowh[r] + lab / 2), labels[k], fill="black", font=fnt, anchor="mm")
    canvas.save(out, dpi=(300, 300))


# 1 Simulink pack response, all modes (from MATLAB export)
fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
for m in MODES:
    d = pd.read_csv(f"{L}/{m}_timeseries.csv")
    a = np.r_[0, np.diff(d.Speed_ms) / 10]
    P_ = d.Voltage_V * d.Current_A / 1e3
    for axx, y, lab in [(ax[0, 0], d.Speed_ms * 3.6, "Speed (km/h)"), (ax[0, 1], a, "Acceleration (m/s$^2$)"),
                        (ax[1, 0], d.Current_A, "Pack current (A)"), (ax[1, 1], P_, "Battery power (kW)"),
                        (ax[2, 0], d.Voltage_V, "Pack voltage (V)"), (ax[2, 1], d.SOC_pct, "SOC (%)")]:
        axx.plot(d.Time_s, y, color=C[m], lw=1.4, label=m); axx.set_ylabel(lab); axx.grid(alpha=.3)
ax[0, 0].legend(ncol=3); ax[2, 0].set_xlabel("Time (s)"); ax[2, 1].set_xlabel("Time (s)")
fig.suptitle("LFP pack (A123 ANR26650, 108S57P) – Simulink response, 1800 s per mode", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/LFP_pack_response.png", dpi=300); plt.close(fig)

# 2 OCV curve: Simulink table vs paper polynomial vs PCHIP
import sys; sys.path.insert(0, ROOT)
os.environ.update(ESS_PACK="LFP", ESS_BASE=L)
import bms_pipeline_v2 as bp, bms_robustness as br
tab = br.load_tables()
fresh = {m: bp.load_mode(m) for m in MODES}
soc = np.concatenate([d.soc for d in fresh.values()]); ob = np.concatenate([(d.V + d.I * bp.R0_CELL) for d in fresh.values()])
poly, pch = bp.OCVCurve(soc, ob), bp.PchipOCV(soc, ob)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(tab["soc"] * 100, tab["V"][:, 0], "k-", lw=2, label="Simulink OCV table (101 pts)")
a2 = ax[0].twinx(); a2.plot(tab["soc"] * 100, tab["R"][:, 0] * 1000, "m:", lw=1.5); a2.set_ylabel("R$_0$ (mΩ, dotted)", color="m")
ax[0].axvspan(soc.min() * 100, 100, color="gold", alpha=.2, label="SOC window of the 1800 s runs")
ax[0].set_xlabel("SOC (%)"); ax[0].set_ylabel("Cell voltage (V)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[0].set_title("A123 ANR26650 cell tables")
s = np.linspace(soc.min(), 1, 300)
ax[1].plot(soc * 100, ob, ".", color="grey", ms=3, label="Observed V + I·R$_{0,n}$")
ax[1].plot(tab["soc"] * 100, tab["V"][:, 0], "k-", lw=1.5, label="True OCV table")
ax[1].plot(s * 100, poly(s), "r--", lw=2, label=f"Eq. (5) polynomial (degree {poly.deg})")
ax[1].plot(s * 100, pch(s), "b-", lw=2, label="Monotone PCHIP (1 % bins)")
ax[1].set_xlim(soc.min() * 100 - 1, 100.5); ax[1].set_ylim(3.3, 3.9)
ax[1].set_xlabel("SOC (%)"); ax[1].set_ylabel("OCV (V/cell)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
ax[1].set_title("OCV identification on the LFP pack")
fig.tight_layout(); fig.savefig(f"{FIG}/LFP_ocv_identification.png", dpi=300); plt.close(fig)

# 3 LFP vs P45B pack response comparison
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for m in MODES:
    l = pd.read_csv(f"{L}/{m}_timeseries.csv"); p = pd.read_csv(f"{P}/{m}_timeseries.csv")
    ax[0].plot(l.Time_s, l.Voltage_V, color=C[m], lw=1.4, label=f"LFP {m}"); ax[0].plot(p.Time_s, p.Voltage_V, "--", color=C[m], lw=1, label=f"P45B {m}")
    ax[1].plot(l.Time_s, l.SOC_pct, color=C[m], lw=1.4); ax[1].plot(p.Time_s, p.SOC_pct, "--", color=C[m], lw=1)
    ax[2].plot(l.Time_s, l.Current_A, color=C[m], lw=1.2); ax[2].plot(p.Time_s, p.Current_A, "--", color=C[m], lw=.9)
for a_, y in zip(ax, ["Pack voltage (V)", "SOC (%)", "Pack current (A)"]):
    a_.set_ylabel(y); a_.set_xlabel("Time (s)"); a_.grid(alpha=.3)
ax[0].legend(fontsize=8, ncol=2)
fig.suptitle("LFP 108S57P (solid) vs Molicel P45B 98S30P (dashed)", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/LFP_vs_P45B_pack.png", dpi=300); plt.close(fig)

# copy the paper-style figures with LFP names
import shutil
pf = f"{L}/bms/results/paper_figures"
copies = {f"{pf}/drive_cycles.png": "LFP_fig3_drive_cycles.png",
          f"{pf}/tau_sweep.png": "LFP_fig5_tau_sweep.png",
          f"{L}/bms/results/revision2/detector_rules.png": "LFP_fig6_detector_rules.png",
          f"{pf}/imputation_all_modes.png": "LFP_fig10_imputation.png"}
for m in MODES:
    copies[f"{pf}/{m.lower()}_aged_comparison.png"] = f"LFP_fig4_{m.lower()}_aged_matched.png"
    copies[f"{pf}/table_plant_{m.lower()}_ramp.png"] = f"LFP_fig7_{m.lower()}_table_plant.png"
for s_, d_ in copies.items():
    shutil.copy(s_, f"{FIG}/{d_}")

# collages
stitch([f"{FIG}/LFP_fig4_{m.lower()}_aged_matched.png" for m in MODES], f"{COL}/LFP_Fig4_matched_plant_step_ageing.png", 3,
       ["(a) Eco", "(b) Normal", "(c) Sport"], "LFP: matched plant, step ageing at 300 s (seed 42)")
stitch([f"{FIG}/LFP_fig7_{m.lower()}_table_plant.png" for m in MODES], f"{COL}/LFP_Fig7_table_plant_gradual_ageing.png", 3,
       ["(a) Eco", "(b) Normal", "(c) Sport"], "LFP: table plant, 4 chained cycles, gradual ageing (seed 0)")
stitch([f"{FIG}/LFP_fig5_tau_sweep.png", f"{FIG}/LFP_fig6_detector_rules.png"], f"{COL}/LFP_Fig5-6_drift_detector.png", 2,
       ["(a) Threshold sweep, matched plant", "(b) Persistence rules, table plant"], "LFP: drift detector")
stitch([f"{FIG}/LFP_pack_response.png", f"{FIG}/LFP_ocv_identification.png", f"{FIG}/LFP_vs_P45B_pack.png"],
       f"{COL}/LFP_Simulink_results_overview.png", 1, None, None)
stitch([f"{FIG}/{f}" for f in ["LFP_pack_response.png", "LFP_ocv_identification.png",
                                "LFP_fig5_tau_sweep.png", "LFP_fig6_detector_rules.png",
                                "LFP_fig3_drive_cycles.png", "LFP_fig10_imputation.png"]],
       f"{COL}/LFP_all_results_A.png", 2, ["Pack response", "OCV identification", "Threshold sweep", "Detector rules",
                                           "Drive cycles", "Sensor-dropout imputation"], "LFP 108S57P – results overview")
stitch([f"{FIG}/LFP_fig4_{m.lower()}_aged_matched.png" for m in MODES] +
       [f"{FIG}/LFP_fig7_{m.lower()}_table_plant.png" for m in MODES],
       f"{COL}/LFP_all_results_B.png", 3, ["Matched – Eco", "Matched – Normal", "Matched – Sport",
                                          "Table plant – Eco", "Table plant – Normal", "Table plant – Sport"],
       "LFP 108S57P – adaptive BMS under ageing")
print("figures + collages done")
