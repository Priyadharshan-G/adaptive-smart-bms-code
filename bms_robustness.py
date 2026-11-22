"""
bms_robustness.py - revision experiments for the adaptive BMS (reviewer comments)

Uses the BMS of bms_pipeline_v2.py and adds:
  R1  Multi-seed statistics (N_SEEDS noise seeds) for the original scenario
      (bms_pipeline_v2 plant, step ageing at 300 s), including an ablation
      with ungated (always-on) FFRLS recalibration.
  R2  Gradual ageing: R0 x1 -> x2 and capacity 1 -> 0.8 as a linear ramp from 300 s.
  R3  Drift-threshold sweep: false-alarm rate on the fresh pack and
      detection rate / delay on the aged pack versus tau.
  R4  Current-sensor offset (bias) robustness.
  R5  Independent "table plant": the exact cell equations and the 101-point
      OCV and SOC/temperature-dependent R0 tables stored in ESSSimulation.slx,
      with the adiabatic thermal equation of the Simscape module. Each drive
      cycle is chained 4x (7200 s) so the SOC window is deep. The BMS keeps a
      constant R0, an RC branch the plant does not have, and an OCV
      characterised at 5 % SOC steps, so the BMS model does not match the plant.
  R6  Sensitivity of the SOH weights.
Outputs -> <ESS_BASE>/bms/results/robustness/
"""

import os
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bms_pipeline_v2 as bp

OUT = os.path.join(bp.BASE, "bms", "results", "robustness")
N_SEEDS = 30
TAUS = [2.5, 3.0, 3.5, 4.0, 5.0]
OFFSETS = [0.0, 0.02, 0.05]           # A per cell (0.6 A / 1.5 A at pack level)
N_CHAIN = 4
SIGMA_T = 0.1                          # degC, temperature sensor noise (table plant)
SOH_WEIGHTS = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]


# ── BMS (same algorithm as bp.AdaptiveBMS, with switches for the ablations) ───
class BMS:
    def __init__(self, ocv, adaptive=True, gated=True, use_temp=True, tau=bp.TAU_DM,
                 R0=bp.R0_CELL, Q=bp.Q_CELL, R1=0.001, C1=2000.0, L=0.2, lam=0.95,
                 persist=(1, 1), q_span=0.03, q_lam=1.0):
        self.ocv, self.adaptive, self.gated, self.use_temp, self.tau = ocv, adaptive, gated, use_temp, tau
        # persistence rule: declare drift when k of the last n samples exceed tau
        self.pk, self.pn = persist
        self.recent = []
        self.q_span, self.q_lam = q_span, q_lam   # capacity guard: min SOC span, regression forgetting
        self.R0_nom, self.Q_nom, self.R0_eff, self.Q_eff = R0, Q, R0, Q
        self.R1, self.tau1, self.L, self.lam = R1, R1 * C1, L, lam
        self.soc, self.Vp, self.soh = None, 0.0, 1.0
        self.theta, self.P = np.array([0.0, R0]), np.eye(2) * 10.0
        self.baseline, self.mu0, self.S0inv = [], None, None
        self.ah, self.rest_pts, self.q_railed = 0.0, [], False
        self.drift_seen = not gated            # ungated: recalibrate from the start
        self.log = []

    def slope(self, s):
        return self.ocv.slope(s) if hasattr(self.ocv, "slope") else \
            float(np.polyval(self.ocv.dc, np.clip(s, self.ocv.lo, self.ocv.hi)))

    def step(self, t, V, I, I_prev, T, dt, T0):
        self.ah += I_prev * dt / 3600
        soc_prior = self.soc - I_prev * dt / (3600 * self.Q_eff)
        ev = V - (self.ocv(soc_prior) - I * self.R0_eff - self.Vp)
        self.soc = float(np.clip(soc_prior + self.L * ev / max(self.slope(soc_prior), 0.1), 0, 1))
        a = np.exp(-dt / self.tau1)
        self.Vp = a * self.Vp + self.R1 * (1 - a) * I
        y = V - self.ocv(self.soc) + self.Vp
        phi = np.array([1.0, -I])
        K = self.P @ phi / (self.lam + phi @ self.P @ phi)
        self.theta = self.theta + K * (y - phi @ self.theta)
        self.P = (self.P - np.outer(K, phi) @ self.P) / self.lam
        x = np.array([ev, T - T0]) if self.use_temp else np.array([ev])
        DM, drift = 0.0, False
        if len(self.baseline) < bp.N_BASE:
            self.baseline.append(x)
        else:
            if self.mu0 is None:
                B = np.array(self.baseline)
                self.mu0 = B.mean(axis=0)
                C = np.atleast_2d(np.cov(B.T)) + np.eye(len(x)) * 1e-8
                self.S0inv = np.linalg.inv(C)
            d = x - self.mu0
            DM = float(np.sqrt(d @ self.S0inv @ d))
            self.recent.append(DM >= self.tau)
            self.recent = self.recent[-self.pn:]
            drift = sum(self.recent) >= self.pk
        if drift:
            self.drift_seen = True
        if self.adaptive and self.drift_seen and self.theta[1] > 0:
            self.R0_eff = 0.7 * self.R0_eff + 0.3 * float(self.theta[1])
        if self.adaptive and self.drift_seen and abs(I) < 0.05:
            self.rest_pts.append((self.ah, self.ocv.inverse(V + I * self.R0_eff)))
            pts = np.array(self.rest_pts)
            if len(pts) >= 5 and np.ptp(pts[:, 1]) >= self.q_span:
                w = self.q_lam ** np.arange(len(pts) - 1, -1, -1)   # q_lam<1 = forget old rest points
                sl = np.polyfit(pts[:, 0], pts[:, 1], 1, w=w)[0]
                if sl < 0:
                    q = -1.0 / sl
                    self.q_railed = not (0.5 * self.Q_nom <= q <= 1.1 * self.Q_nom)
                    q = float(np.clip(q, 0.5 * self.Q_nom, 1.1 * self.Q_nom))
                    self.Q_eff = 0.8 * self.Q_eff + 0.2 * q
        soh = 0.70 * min(self.Q_eff / self.Q_nom, 1.0) + 0.30 * min(self.R0_nom / self.R0_eff, 1.0)
        self.soh = min(self.soh, soh)
        self.log.append((t, self.soc * 100, self.R0_eff * 1000, self.Q_eff, DM, int(drift), self.soh * 100))


def run(d, ocv, case, tau=bp.TAU_DM, offset=0.0, use_temp=True, R0=bp.R0_CELL, T0=25.0, **kw):
    adaptive = case != "Static"
    bms = BMS(ocv, adaptive=adaptive, gated=(case != "Ungated"), use_temp=use_temp, tau=tau, R0=R0, **kw)
    t, V, T = d["t"].values, d["V_meas"].values, d["T_meas"].values if "T_meas" in d else d["T"].values
    I = d["I_meas"].values + offset
    bms.soc = bms.ocv.inverse(V[0] + I[0] * R0)
    for k in range(1, len(d)):
        bms.step(t[k], V[k], I[k], I[k - 1], T[k], t[k] - t[k - 1], T0)
    r = pd.DataFrame(bms.log, columns=["t", "SOC_est", "R0_est_mOhm", "Q_est_Ah", "DM", "drift", "SOH"])
    r["SOC_true"] = d["soc"].values[1:] * 100
    r["R0_true_mOhm"] = (d["R0_true"].values[1:] if "R0_true" in d else np.full(len(r), np.nan)) * 1000
    r["Q_true_Ah"] = d["Q_true"].values[1:] if "Q_true" in d else bp.Q_CELL
    r.attrs["q_railed"] = bms.q_railed
    return r


def metrics(r, t_age=bp.T_AGE):
    e = r.SOC_est - r.SOC_true
    post = r[r.t >= t_age]
    first = post.loc[post.drift == 1, "t"]
    return dict(SOC_RMSE=float(np.sqrt((e ** 2).mean())), SOC_MaxErr=float(e.abs().max()),
                Flags_pre=int(r.loc[r.t < t_age, "drift"].sum()), Flags_post=int(post.drift.sum()),
                Detected=int(len(first) > 0), Delay=float(first.iloc[0] - t_age) if len(first) else np.nan,
                R0_final=float(r.R0_est_mOhm.iloc[-1]), Q_final=float(r.Q_est_Ah.iloc[-1]),
                SOH_final=float(r.SOH.iloc[-1]))


def noisy(d, rng, sigma_T=0.0):
    d = bp.add_noise(d, rng)
    d["T_meas"] = d["T"] + (rng.normal(0, sigma_T, len(d)) if sigma_T > 0 else 0.0)
    return d


# ── Plants ────────────────────────────────────────────────────────────────────
def make_aged_ramp(d, ocv):
    """bms_pipeline_v2 plant, but R0 and Q change linearly from T_AGE to the end."""
    d = d.copy()
    t, I = d["t"].values, d["I"].values
    f = np.clip((t - bp.T_AGE) / (t[-1] - bp.T_AGE), 0, 1)
    Q = bp.Q_CELL * (1 - (1 - bp.Q_FACTOR) * f)
    R0 = bp.R0_CELL * (1 + (bp.R0_FACTOR - 1) * f)
    resid = d["V"].values - (ocv(d["soc"].values) - I * bp.R0_CELL)
    soc = np.empty(len(t)); soc[0] = d["soc"].iloc[0]
    for k in range(1, len(t)):
        soc[k] = soc[k - 1] - I[k - 1] * (t[k] - t[k - 1]) / (3600 * Q[k])
    d["soc"], d["V"], d["R0_true"], d["Q_true"] = soc, ocv(soc) - I * R0 + resid, R0, Q
    return d


def load_tables():
    """OCV and R0 tables of the cell, read directly from the pack's Simulink model.
    P45B: 2-D tables vs SOC and temperature + thermal mass.
    Battery Builder packs (LFP): 1-D tables vs SOC, no thermal model -> isothermal."""
    slx = os.path.join(bp.BASE, bp.SLX)
    if not os.path.isfile(slx):                        # models live at the repo root
        slx = os.path.join(os.path.dirname(os.path.abspath(__file__)), bp.SLX)
    with zipfile.ZipFile(slx) as z:
        names = [n for n in z.namelist() if re.match(r"simulink/systems/system_\d+\.xml", n)]
        for n in names:
            txt = z.read(n).decode("utf-8", "ignore")
            def par(key):
                s = re.search(r'Name="%s">([^<]*)<' % key, txt).group(1)
                return np.array([[float(v) for v in row.replace(",", " ").split()]
                                 for row in s.strip("[]").split(";")])
            if 'Name="V0_matCell"' in txt:
                return dict(soc=par("SOC_vecCell").ravel(), T=par("T_vecCell").ravel(),
                            V=par("V0_matCell"), R=par("R0_matCell"),
                            AH=float(par("AHCell")[0, 0]), mth=float(par("thermal_massCell")[0, 0]))
            if 'Name="V0_vecCell"' in txt:
                return dict(soc=par("SOC_vecCell").ravel(), T=np.array([25.0]),
                            V=par("V0_vecCell").reshape(-1, 1), R=par("R0_vecCell").reshape(-1, 1),
                            AH=float(par("AHCell")[0, 0]), mth=np.inf)
    raise RuntimeError("cell tables not found in " + bp.SLX)


class TablePlant:
    def __init__(self, tab):
        self.tab = tab
        self.Tg = tab["T"]

    def ocv(self, z):
        return float(np.interp(z, self.tab["soc"], self.tab["V"][:, 0]))

    def r0(self, z, T):
        col = np.array([np.interp(z, self.tab["soc"], self.tab["R"][:, j]) for j in range(len(self.Tg))])
        return float(np.interp(T, self.Tg, col))

    def simulate(self, t, Pcell, z0=1.0, T0=25.0, rf=None, qf=None):
        """Battery power per cell (W) -> I, V, SOC, T, solving I = P / V each step (Simulink loop)."""
        n = len(t)
        rf = np.ones(n) if rf is None else rf
        qf = np.ones(n) if qf is None else qf
        z, T = z0, T0
        out = np.zeros((n, 6))
        for k in range(n):
            v0, r = self.ocv(z), self.r0(z, T) * rf[k]
            p = Pcell[k]
            I = 0.0 if p <= 0 else (v0 - np.sqrt(v0 * v0 - 4 * r * p)) / (2 * r)
            out[k] = (I, v0 - I * r, z, T, r, self.tab["AH"] * qf[k])
            if k < n - 1:
                dt = t[k + 1] - t[k]
                z -= I * dt / (3600 * self.tab["AH"] * qf[k])
                T += 0.0 if np.isinf(self.tab["mth"]) else I * I * r * dt / self.tab["mth"]
        return pd.DataFrame(out, columns=["I", "V", "soc", "T", "R0_true", "Q_true"]).assign(t=t)


class PWLOCV:
    """OCV characterised at 5 % SOC steps (what the BMS knows), piecewise linear."""
    def __init__(self, plant, step=0.05):
        self.g = np.round(np.arange(0, 1 + 1e-9, step), 4)
        self.v = np.array([plant.ocv(s) for s in self.g])
        self.fine = np.linspace(0, 1, 4001)
        self.vf = np.interp(self.fine, self.g, self.v)

    def __call__(self, s):
        return np.interp(s, self.g, self.v)

    def inverse(self, v):
        return float(np.interp(v, self.vf, self.fine))

    def slope(self, s):
        i = int(np.clip(np.searchsorted(self.g, s) - 1, 0, len(self.g) - 2))
        return (self.v[i + 1] - self.v[i]) / (self.g[i + 1] - self.g[i])


def chained_power(mode, n=N_CHAIN):
    d = pd.read_csv(os.path.join(bp.BASE, f"{mode}_timeseries.csv"))
    p = (d.Voltage_V * d.Current_A).values / (bp.N_S * bp.N_P)
    t = d.Time_s.values
    T_cyc = t[-1] + (t[1] - t[0])
    return np.concatenate([t + i * T_cyc for i in range(n)]), np.tile(p, n)


def ageing_profile(t, kind):
    if kind == "step":
        f = (t >= bp.T_AGE).astype(float)
    else:
        f = np.clip((t - bp.T_AGE) / (t[-1] - bp.T_AGE), 0, 1)
    return 1 + (bp.R0_FACTOR - 1) * f, 1 - (1 - bp.Q_FACTOR) * f


def mstd(df, cols, by):
    g = df.groupby(by)[cols]
    m, s = g.mean(), g.std()
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(m[c], s[c])]
    return out.reset_index()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fresh = {m: bp.load_mode(m) for m in bp.MODES}
    ocv = bp.identify_ocv(fresh)

    # Check: the re-implemented BMS reproduces summary.csv for the original seed sequence
    rng = np.random.default_rng(bp.SEED)
    chk = []
    for m in bp.MODES:
        fn, an = bp.add_noise(fresh[m], rng), bp.add_noise(bp.make_aged(fresh[m], ocv), rng)
        for case, d in [("Fresh", fn), ("Static", an), ("Gated", an)]:
            chk.append(dict(Mode=m, Case=case, **metrics(run(d, ocv, case))))
    pd.DataFrame(chk).round(3).to_csv(os.path.join(OUT, "check_seed42.csv"), index=False)
    print("[check] seed-42 reproduction:\n", pd.DataFrame(chk)[["Mode", "Case", "SOC_RMSE", "R0_final", "SOH_final"]].round(3))

    # R1 + R2 + R4: multi-seed, ablation, gradual ageing, current offset
    print(f"[R1/R2/R4] {N_SEEDS} seeds ...")
    rows = []
    for seed in range(N_SEEDS):
        for m in bp.MODES:
            for ageing, plant in [("step", bp.make_aged), ("ramp", make_aged_ramp)]:
                rng = np.random.default_rng(1000 + seed)
                fn = noisy(fresh[m], rng)
                an = noisy(plant(fresh[m], ocv), rng)
                for off in OFFSETS:
                    if ageing == "ramp" and off != 0.0:
                        continue
                    if off == 0.0:
                        rows.append(dict(Seed=seed, Mode=m, Ageing=ageing, Offset_A=off, Case="Fresh",
                                         **metrics(run(fn, ocv, "Fresh", offset=off))))
                        rows.append(dict(Seed=seed, Mode=m, Ageing=ageing, Offset_A=off, Case="Fresh-Ungated",
                                         **metrics(run(fn, ocv, "Ungated", offset=off))))
                    for case in ["Static", "Ungated", "Gated"]:
                        rows.append(dict(Seed=seed, Mode=m, Ageing=ageing, Offset_A=off, Case=case,
                                         **metrics(run(an, ocv, case, offset=off))))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "multiseed_raw.csv"), index=False)
    cols = ["SOC_RMSE", "SOC_MaxErr", "Flags_pre", "Flags_post", "R0_final", "SOH_final"]
    mstd(R[R.Offset_A == 0], cols + ["Delay"], ["Ageing", "Mode", "Case"]).to_csv(
        os.path.join(OUT, "multiseed_summary.csv"), index=False)
    mstd(R[R.Ageing == "step"], ["SOC_RMSE"], ["Offset_A", "Mode", "Case"]).to_csv(
        os.path.join(OUT, "offset_summary.csv"), index=False)
    print(mstd(R[R.Offset_A == 0], ["SOC_RMSE", "R0_final"], ["Ageing", "Mode", "Case"]).to_string(index=False))

    # R3: threshold sweep (step ageing, gated)
    print("[R3] threshold sweep ...")
    rows = []
    for seed in range(N_SEEDS):
        for m in bp.MODES:
            rng = np.random.default_rng(1000 + seed)
            fn, an = noisy(fresh[m], rng), noisy(bp.make_aged(fresh[m], ocv), rng)
            for tau in TAUS:
                f, a = metrics(run(fn, ocv, "Gated", tau=tau)), metrics(run(an, ocv, "Gated", tau=tau))
                rows.append(dict(Seed=seed, Mode=m, Tau=tau, FalseAlarm=int(f["Flags_pre"] + f["Flags_post"] > 0),
                                 EarlyFlag=int(a["Flags_pre"] > 0), Detected=a["Detected"],
                                 Delay=a["Delay"], SOC_RMSE=a["SOC_RMSE"]))
    TS = pd.DataFrame(rows)
    TS.to_csv(os.path.join(OUT, "tau_sweep_raw.csv"), index=False)
    ts = TS.groupby(["Mode", "Tau"]).agg(FalseAlarmRate=("FalseAlarm", "mean"), DetectionRate=("Detected", "mean"),
                                          MedianDelay_s=("Delay", "median"), SOC_RMSE=("SOC_RMSE", "mean")).round(3).reset_index()
    ts.to_csv(os.path.join(OUT, "tau_sweep_summary.csv"), index=False)
    print(ts.to_string(index=False))

    # R5: independent table plant, chained 4x, deep SOC window
    print("[R5] table plant, chained cycles ...")
    tab = load_tables()
    tp = TablePlant(tab)
    pocv = PWLOCV(tp)
    val = []
    for m in bp.MODES:                        # validation against the Simulink export
        d = pd.read_csv(os.path.join(bp.BASE, f"{m}_timeseries.csv"))
        sim = tp.simulate(d.Time_s.values, (d.Voltage_V * d.Current_A).values / (bp.N_S * bp.N_P))
        val.append(dict(Mode=m, V_RMSE_V=float(np.sqrt(np.mean((sim.V * bp.N_S - d.Voltage_V) ** 2))),
                        V_MaxErr_V=float(np.abs(sim.V * bp.N_S - d.Voltage_V).max()),
                        FinalSOC_plant=float(sim.soc.iloc[-1] * 100), FinalSOC_simulink=float(d.SOC_pct.iloc[-1]),
                        T_end_degC=float(sim["T"].iloc[-1])))
    pd.DataFrame(val).round(4).to_csv(os.path.join(OUT, "table_plant_validation.csv"), index=False)
    print(pd.DataFrame(val).round(4).to_string(index=False))

    rows, example = [], {}
    for m in bp.MODES:
        t, p = chained_power(m)
        fresh_tp = tp.simulate(t, p)
        aged_tp = {k: tp.simulate(t, p, rf=ageing_profile(t, k)[0], qf=ageing_profile(t, k)[1])
                   for k in ["step", "ramp"]}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(5000 + seed)
            fn = noisy(fresh_tp.rename(columns={}), rng, SIGMA_T)
            for use_temp in [True, False]:
                det = "V+T" if use_temp else "V only"
                r = run(fn, pocv, "Gated", use_temp=use_temp)
                rows.append(dict(Seed=seed, Mode=m, Ageing="none", Detector=det, Case="Fresh", **metrics(r)))
            r = run(fn, pocv, "Ungated", use_temp=False)
            rows.append(dict(Seed=seed, Mode=m, Ageing="none", Detector="V only", Case="Fresh-Ungated", **metrics(r)))
            for k, ap in aged_tp.items():
                an = noisy(ap, rng, SIGMA_T)
                for case in ["Static", "Ungated", "Gated"]:
                    r = run(an, pocv, case, use_temp=False)
                    rows.append(dict(Seed=seed, Mode=m, Ageing=k, Detector="V only", Case=case, **metrics(r)))
                    if seed == 0 and m == "Sport" and k == "ramp":
                        example[case] = r
    TP = pd.DataFrame(rows)
    TP.to_csv(os.path.join(OUT, "table_plant_raw.csv"), index=False)
    s = mstd(TP, ["SOC_RMSE", "SOC_MaxErr", "Flags_pre", "Flags_post", "R0_final", "Q_final", "SOH_final"],
             ["Ageing", "Detector", "Mode", "Case"])
    s.to_csv(os.path.join(OUT, "table_plant_summary.csv"), index=False)
    print(s.to_string(index=False))
    fa = TP[TP.Case == "Fresh"].assign(Any=lambda x: (x.Flags_pre + x.Flags_post) > 0) \
        .groupby(["Mode", "Detector"]).agg(FalseAlarmRate=("Any", "mean"), MeanFlags=("Flags_post", "mean")).reset_index()
    fa.to_csv(os.path.join(OUT, "table_plant_false_alarms.csv"), index=False)
    print(fa.to_string(index=False))

    # true SOH and SOC window of the table-plant runs
    info = []
    for m in bp.MODES:
        t, p = chained_power(m)
        for k in ["step", "ramp"]:
            rf, qf = ageing_profile(t, k)
            a = tp.simulate(t, p, rf=rf, qf=qf)
            info.append(dict(Mode=m, Ageing=k, SOC_start=100.0, SOC_end=float(a.soc.iloc[-1] * 100),
                             T_end=float(a["T"].iloc[-1]), R0_true_end_mOhm=float(a.R0_true.iloc[-1] * 1000),
                             Q_true_end=float(a.Q_true.iloc[-1])))
    pd.DataFrame(info).round(3).to_csv(os.path.join(OUT, "table_plant_conditions.csv"), index=False)
    print(pd.DataFrame(info).round(3).to_string(index=False))

    # Figure: Sport, chained, gradual ageing, seed 0
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    e = example["Gated"]
    ax[0].plot(e.t / 60, e.SOC_true, "k-", lw=2, label="True SOC")
    ax[0].plot(example["Static"].t / 60, example["Static"].SOC_est, "r--", label="Static BMS")
    ax[0].plot(example["Ungated"].t / 60, example["Ungated"].SOC_est, ":", color="orange", label="Ungated FFRLS")
    ax[0].plot(e.t / 60, e.SOC_est, "b-", label="Gated (proposed)")
    ax[0].set_ylabel("SOC (%)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(e.t / 60, e.R0_true_mOhm, "k-", lw=2, label="True R0 (SOC- and T-dependent)")
    ax[1].plot(e.t / 60, e.R0_est_mOhm, "b-", label="Gated FFRLS estimate")
    ax[1].set_ylabel("R0 (mOhm)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(e.t / 60, e.Q_true_Ah, "k-", lw=2, label="True capacity")
    ax[2].plot(e.t / 60, e.Q_est_Ah, "b-", label="Rest-point estimate")
    ax[2].set_ylabel("Capacity (Ah)"); ax[2].set_xlabel("Time (min)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle("Sport x4 on the table plant, gradual ageing from 5 min (seed 0)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "table_plant_sport_ramp.png"), dpi=200); plt.close(fig)

    # R6: SOH weight sensitivity (table plant, gated, V-only detector)
    g = TP[(TP.Case == "Gated") & (TP.Ageing != "none")].groupby(["Ageing", "Mode"])[["Q_final", "R0_final"]].mean()
    cond = pd.DataFrame(info).set_index(["Mode", "Ageing"])
    rows = []
    for (k, m), v in g.iterrows():
        Rt, Qt = cond.loc[(m, k), "R0_true_end_mOhm"], cond.loc[(m, k), "Q_true_end"]
        R0n = tp.r0(cond.loc[(m, k), "SOC_end"] / 100, cond.loc[(m, k), "T_end"]) * 1000
        for wq, wr in SOH_WEIGHTS:
            est = wq * min(v.Q_final / bp.Q_CELL, 1) + wr * min(bp.R0_CELL * 1000 / v.R0_final, 1)
            tru = wq * Qt / bp.Q_CELL + wr * min(R0n / Rt, 1)
            rows.append(dict(Ageing=k, Mode=m, w_Q=wq, w_R=wr, SOH_est=100 * est, SOH_true=100 * tru))
    pd.DataFrame(rows).round(2).to_csv(os.path.join(OUT, "soh_weight_sensitivity.csv"), index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"\nDone. Results in {OUT}")
