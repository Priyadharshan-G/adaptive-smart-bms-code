"""
bms_pipeline_v2.py - Adaptive Smart BMS (corrected)

Pipeline:
  1. Load Simulink pack data (98S30P Molicel P45B), convert pack -> average cell
  2. Identify the cell OCV-SOC curve from the fresh simulation (R0 = 10 mOhm from model)
  3. Build aged-pack data: R0 x2 and capacity -20 % injected at t = 300 s
  4. Run the BMS: Coulomb counting + Luenberger OCV correction,
     Mahalanobis drift detection, drift-gated FFRLS (R0) + capacity recalibration,
     monotonic SOH
  5. Compare: Fresh / Aged-static (no adaptation) / Aged-adaptive

Outputs -> C:\\ESS\\bms\\results\\
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
BASE = os.environ.get("ESS_BASE", r"C:\ESS")
OUT = os.path.join(BASE, "bms", "results")
MODES = ["Eco", "Normal", "Sport"]

# Pack selection (ESS_PACK env var). P45B = original paper pack.
PACKS = {
    "P45B": dict(N_S=98, N_P=30, Q_CELL=4.493, R0_CELL=0.010, SLX="ESSSimulation.slx"),
    # A123 ANR26650 LFP, 108S57P (Battery Builder, R0 and OCV vs SOC only, no thermal model).
    # BMS nominal R0 = mean of the cell R0 table over 30-100 % SOC (14.8 mOhm) -> 15 mOhm.
    # Panasonic NCR18650BD, 98S45P (Battery Builder, R0/OCV vs SOC only, no thermal model).
    # BMS nominal R0 = mean of the cell R0 table over 30-100 % SOC (47.1 mOhm) -> 47 mOhm.
    "NCR":  dict(N_S=98, N_P=45, Q_CELL=3.153, R0_CELL=0.047, SLX="ESSSimulation_NCR.slx"),
    "LFP":  dict(N_S=108, N_P=57, Q_CELL=2.341, R0_CELL=0.015, SLX="ESSSimulation_LFP.slx"),
}
PACK = os.environ.get("ESS_PACK", "P45B")
_pc = PACKS[PACK]
N_S, N_P = _pc["N_S"], _pc["N_P"]   # pack configuration
Q_CELL = _pc["Q_CELL"]             # Ah per cell
R0_CELL = _pc["R0_CELL"]           # Ohm, BMS nominal series resistance
SLX = _pc["SLX"]

T_AGE = 300                # s, ageing injected here
R0_FACTOR = 2.0            # resistance x2
Q_FACTOR = 0.80            # capacity -20 %

N_BASE = 18                # baseline samples for Mahalanobis (first 180 s at 10 s steps)
TAU_DM = 3.5               # Mahalanobis threshold
SIGMA_V = 0.002            # V, cell voltage sensor noise
SIGMA_I = 0.02             # A, cell current sensor noise
SEED = 42


# ── Data ──────────────────────────────────────────────────────────────────────
def load_mode(mode):
    df = pd.read_csv(os.path.join(BASE, f"{mode}_timeseries.csv"))
    return pd.DataFrame(dict(
        t=df["Time_s"].values.astype(float),
        V=df["Voltage_V"].values / N_S,      # average cell voltage
        I=df["Current_A"].values / N_P,      # cell current (discharge +)
        T=df["Temp_degC"].values.astype(float),
        soc=df["SOC_pct"].values / 100.0,
    ))


class OCVCurve:
    """Polynomial OCV(SOC) fitted on the simulated range, linear outside it."""
    def __init__(self, soc, v):
        self.lo, self.hi = float(soc.min()), float(soc.max())
        for deg in (3, 2, 1):
            c = np.polyfit(soc, v, deg)
            g = np.linspace(self.lo, self.hi, 400)
            if np.all(np.diff(np.polyval(c, g)) > 0):   # must be monotonic
                break
        self.c, self.deg = c, deg
        self.dc = np.polyder(c) if deg > 1 else np.array([c[0]])
        self.grid = np.linspace(0.0, 1.0, 4001)
        self.vgrid = self(self.grid)

    def __call__(self, s):
        s = np.asarray(s, dtype=float)
        sc = np.clip(s, self.lo, self.hi)
        v = np.polyval(self.c, sc)
        v = v + np.where(s < self.lo, (s - self.lo) * np.polyval(self.dc, self.lo), 0.0)
        v = v + np.where(s > self.hi, (s - self.hi) * np.polyval(self.dc, self.hi), 0.0)
        return v

    def inverse(self, v):
        return float(np.interp(v, self.vgrid, self.grid))


class PchipOCV:
    """Monotone piecewise-cubic (PCHIP) OCV(SOC) through 1 %-SOC bin medians of the
    observed OCV, linear outside the fitted range. Used for flat/kinked curves (LFP)
    where no monotonic polynomial of degree <= 3 fits."""
    def __init__(self, soc, v, width=0.01):
        from scipy.interpolate import PchipInterpolator
        self.lo, self.hi = float(soc.min()), float(soc.max())
        edges = np.arange(self.lo, self.hi + width, width)
        idx = np.clip(np.digitize(soc, edges) - 1, 0, len(edges) - 2)
        xs, ys = [], []
        for b in np.unique(idx):
            m = idx == b
            xs.append(np.median(soc[m])); ys.append(np.median(v[m]))
        xs, ys = np.array(xs), np.maximum.accumulate(np.array(ys))
        ys = ys + np.arange(len(ys)) * 1e-6            # strictly increasing
        self.f = PchipInterpolator(xs, ys, extrapolate=False)
        self.df = self.f.derivative()
        self.xlo, self.xhi = xs[0], xs[-1]
        self.deg = "pchip"
        self.grid = np.linspace(0.0, 1.0, 4001)
        self.vgrid = np.maximum.accumulate(self(self.grid))

    def slope(self, s):
        return float(self.df(np.clip(s, self.xlo, self.xhi)))

    def __call__(self, s):
        s = np.asarray(s, dtype=float)
        sc = np.clip(s, self.xlo, self.xhi)
        v = self.f(sc)
        v = v + np.where(s < self.xlo, (s - self.xlo) * self.df(self.xlo), 0.0)
        v = v + np.where(s > self.xhi, (s - self.xhi) * self.df(self.xhi), 0.0)
        return v

    def inverse(self, v):
        return float(np.interp(v, self.vgrid, self.grid))


OCV_METHOD = os.environ.get("ESS_OCV", "poly")      # "poly" (paper, eq. 5) or "pchip"


def identify_ocv(fresh):
    soc = np.concatenate([d["soc"].values for d in fresh.values()])
    ocv_obs = np.concatenate([(d["V"] + d["I"] * R0_CELL).values for d in fresh.values()])
    return PchipOCV(soc, ocv_obs) if OCV_METHOD == "pchip" else OCVCurve(soc, ocv_obs)


def make_aged(d, ocv):
    """Re-generate the plant with R0 x2 and Q x0.8 from t = T_AGE."""
    d = d.copy()
    t, I = d["t"].values, d["I"].values
    Q = np.where(t >= T_AGE, Q_CELL * Q_FACTOR, Q_CELL)
    R0 = np.where(t >= T_AGE, R0_CELL * R0_FACTOR, R0_CELL)
    resid = d["V"].values - (ocv(d["soc"].values) - I * R0_CELL)   # keep model residual
    soc = np.empty(len(t))
    soc[0] = d["soc"].iloc[0]
    for k in range(1, len(t)):
        soc[k] = soc[k - 1] - I[k - 1] * (t[k] - t[k - 1]) / (3600 * Q[k])
    d["soc"] = soc
    d["V"] = ocv(soc) - I * R0 + resid
    d["R0_true"] = R0
    d["Q_true"] = Q
    return d


def add_noise(d, rng):
    d = d.copy()
    d["V_meas"] = d["V"] + rng.normal(0, SIGMA_V, len(d))
    d["I_meas"] = d["I"] + rng.normal(0, SIGMA_I, len(d))
    return d


# ── BMS ───────────────────────────────────────────────────────────────────────
class AdaptiveBMS:
    def __init__(self, ocv, adaptive=True, R1=0.001, C1=2000.0, L=0.2, lam=0.95):
        self.ocv, self.adaptive = ocv, adaptive
        self.R0_nom, self.Q_nom = R0_CELL, Q_CELL
        self.R0_eff, self.Q_eff = R0_CELL, Q_CELL
        self.R1, self.tau1 = R1, R1 * C1
        self.L, self.lam = L, lam
        self.soc, self.Vp, self.soh = None, 0.0, 1.0
        # FFRLS: y = V - OCV(soc) = bias - R0 * I
        self.theta = np.array([0.0, R0_CELL])
        self.P = np.eye(2) * 10.0
        # Mahalanobis
        self.baseline, self.mu0, self.S0inv = [], None, None
        # Capacity estimation from rest points after drift
        self.ah = 0.0
        self.rest_pts = []
        self.drift_seen = False
        self.log = []

    def init_soc(self, V_rest):
        self.soc = self.ocv.inverse(V_rest)

    def step(self, t, V, I, I_prev, T, dt):
        # 1. Coulomb counting (previous-interval current) + Luenberger correction
        self.ah += I_prev * dt / 3600
        soc_prior = self.soc - I_prev * dt / (3600 * self.Q_eff)
        V_pred = self.ocv(soc_prior) - I * self.R0_eff - self.Vp
        ev = V - V_pred
        slope = max(self.ocv.slope(soc_prior) if hasattr(self.ocv, "slope") else
                    float(np.polyval(self.ocv.dc, np.clip(soc_prior, self.ocv.lo, self.ocv.hi))), 0.1)
        self.soc = float(np.clip(soc_prior + self.L * ev / slope, 0, 1))   # ev/slope = implied SOC error
        a = np.exp(-dt / self.tau1)
        self.Vp = a * self.Vp + self.R1 * (1 - a) * I

        # 2. FFRLS tracking of R0 (runs every step)
        y = V - self.ocv(self.soc) + self.Vp
        phi = np.array([1.0, -I])
        K = self.P @ phi / (self.lam + phi @ self.P @ phi)
        self.theta = self.theta + K * (y - phi @ self.theta)
        self.P = (self.P - np.outer(K, phi) @ self.P) / self.lam

        # 3. Mahalanobis drift detection on [voltage residual, temperature deviation]
        x = np.array([ev, T - 25.0])
        DM, drift = 0.0, False
        if len(self.baseline) < N_BASE:
            self.baseline.append(x)
        else:
            if self.mu0 is None:
                B = np.array(self.baseline)
                self.mu0 = B.mean(axis=0)
                self.S0inv = np.linalg.inv(np.cov(B.T) + np.eye(2) * 1e-8)
            d = x - self.mu0
            DM = float(np.sqrt(d @ self.S0inv @ d))
            drift = DM >= TAU_DM

        # 4. Drift-triggered recalibration (R0 from FFRLS, capacity from rest points)
        if drift:
            self.drift_seen = True
        if self.adaptive and self.drift_seen and self.theta[1] > 0:
            # once drift has been detected, keep recalibrating R0 from FFRLS
            self.R0_eff = 0.7 * self.R0_eff + 0.3 * float(self.theta[1])
        if self.adaptive and self.drift_seen and abs(I) < 0.05:
            self.rest_pts.append((self.ah, self.ocv.inverse(V + I * self.R0_eff)))
            self._update_capacity()

        # 5. Monotonic SOH
        soh_raw = 0.70 * min(self.Q_eff / self.Q_nom, 1.0) + 0.30 * min(self.R0_nom / self.R0_eff, 1.0)
        self.soh = min(self.soh, soh_raw)

        self.log.append(dict(t=t, V_cell=V, I_cell=I, T=T, SOC_est=self.soc * 100,
                             R0_est_mOhm=self.R0_eff * 1000, Q_est_Ah=self.Q_eff,
                             residual_mV=ev * 1000, DM=DM, drift=int(drift), SOH=self.soh * 100))

    def _update_capacity(self):
        pts = np.array(self.rest_pts)
        if len(pts) < 5 or np.ptp(pts[:, 1]) < 0.03:
            return
        slope = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]     # dSOC / dAh
        if slope < 0:
            q = float(np.clip(-1.0 / slope, 0.5 * self.Q_nom, 1.1 * self.Q_nom))
            self.Q_eff = 0.8 * self.Q_eff + 0.2 * q


def run_bms(d, ocv, adaptive):
    bms = AdaptiveBMS(ocv, adaptive=adaptive)
    bms.init_soc(d["V_meas"].iloc[0] + d["I_meas"].iloc[0] * R0_CELL)
    t, V, I, T = (d[c].values for c in ("t", "V_meas", "I_meas", "T"))
    for k in range(1, len(d)):
        bms.step(t[k], V[k], I[k], I[k - 1], T[k], t[k] - t[k - 1])
    r = pd.DataFrame(bms.log)
    r["SOC_true"] = d["soc"].values[1:] * 100
    r["R0_true_mOhm"] = d["R0_true"].values[1:] * 1000 if "R0_true" in d else R0_CELL * 1000
    r["Q_true_Ah"] = d["Q_true"].values[1:] if "Q_true" in d else Q_CELL
    return r


def metrics(r, mode, case):
    err = r["SOC_est"] - r["SOC_true"]
    pre = r[r["t"] < T_AGE]
    post = r[r["t"] >= T_AGE]
    first = post.loc[post["drift"] == 1, "t"]
    return dict(
        Mode=mode, Case=case,
        SOC_RMSE_pct=round(float(np.sqrt((err ** 2).mean())), 3),
        SOC_MaxErr_pct=round(float(err.abs().max()), 3),
        Drifts_before_300s=int(pre["drift"].sum()),
        Drifts_after_300s=int(post["drift"].sum()),   # for Fresh these are false alarms
        Detection_delay_s=(float(first.iloc[0] - T_AGE) if (len(first) and case != "Fresh") else None),
        Final_R0_est_mOhm=round(float(r["R0_est_mOhm"].iloc[-1]), 2),
        Final_Q_est_Ah=round(float(r["Q_est_Ah"].iloc[-1]), 3),
        Final_SOH_pct=round(float(r["SOH"].iloc[-1]), 1),
    )


def plot_mode(mode, runs):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    a = runs["Aged-adaptive"]
    ax[0].plot(a["t"], a["SOC_true"], "k-", lw=2, label="True SOC (aged pack)")
    ax[0].plot(runs["Aged-static"]["t"], runs["Aged-static"]["SOC_est"], "r--", label="Static BMS")
    ax[0].plot(a["t"], a["SOC_est"], "b-", label="Adaptive BMS")
    ax[0].set_ylabel("SOC (%)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(a["t"], a["DM"], "b-", label="Mahalanobis distance")
    ax[1].axhline(TAU_DM, color="r", ls="--", label=f"Threshold = {TAU_DM}")
    ax[1].scatter(a.loc[a.drift == 1, "t"], a.loc[a.drift == 1, "DM"], c="r", s=12, label="Drift flagged")
    ax[1].set_ylabel("DM"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(a["t"], a["R0_true_mOhm"], "k-", lw=2, label="True R0")
    ax[2].plot(a["t"], a["R0_est_mOhm"], "b-", label="FFRLS estimate")
    ax[2].set_ylabel("R0 (mOhm)"); ax[2].set_xlabel("Time (s)"); ax[2].legend(); ax[2].grid(alpha=.3)
    for x in ax:
        x.axvline(T_AGE, color="grey", ls=":")
    fig.suptitle(f"{mode} mode: ageing injected at t = {T_AGE} s (R0 x{R0_FACTOR}, capacity -{round((1-Q_FACTOR)*100)} %)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{mode}_aged_comparison.png"), dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[1/4] Loading pack data and converting to cell level...")
    fresh = {m: load_mode(m) for m in MODES}
    for m, d in fresh.items():
        print(f"  {m}: {len(d)} samples, true SOC {d.soc.iloc[0]*100:.1f} -> {d.soc.iloc[-1]*100:.1f} %")

    print("[2/4] Identifying OCV-SOC curve from fresh simulation...")
    ocv = identify_ocv(fresh)
    print(f"  Polynomial degree {ocv.deg}, SOC range {ocv.lo*100:.1f}-{ocv.hi*100:.1f} %")
    s = np.linspace(ocv.lo, ocv.hi, 100)
    plt.figure(figsize=(7, 4))
    for m, d in fresh.items():
        plt.plot(d.soc * 100, d.V + d.I * R0_CELL, ".", ms=3, label=f"{m} data")
    plt.plot(s * 100, ocv(s), "k-", lw=2, label="Fitted OCV")
    plt.xlabel("SOC (%)"); plt.ylabel("OCV (V/cell)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "ocv_fit.png"), dpi=150); plt.close()

    print("[3/4] Running BMS: Fresh / Aged-static / Aged-adaptive...")
    rows = []
    for m in MODES:
        fresh_n = add_noise(fresh[m], rng)
        aged = make_aged(fresh[m], ocv)
        aged_n = add_noise(aged, rng)
        runs = {
            "Fresh": run_bms(fresh_n, ocv, adaptive=True),
            "Aged-static": run_bms(aged_n, ocv, adaptive=False),
            "Aged-adaptive": run_bms(aged_n, ocv, adaptive=True),
        }
        for case, r in runs.items():
            r.to_csv(os.path.join(OUT, f"{m}_{case}.csv"), index=False)
            rows.append(metrics(r, m, case))
        plot_mode(m, runs)

    print("[4/4] Summary")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(f"\nDone. Results, CSVs and plots saved in {OUT}")
