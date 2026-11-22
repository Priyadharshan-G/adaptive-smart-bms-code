"""
bms_revision2.py - second-revision analyses (statistical tests, detector redesign,
capacity guard, SOH decomposition). Reads the runs made by bms_robustness.py and
runs new experiments on the table plant.
Outputs -> <ESS_BASE>/bms/results/revision2/
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bms_pipeline_v2 as bp
import bms_robustness as br

OUT = os.path.join(bp.BASE, "bms", "results", "revision2")
RES = os.path.join(bp.BASE, "bms", "results", "robustness")
N_SEEDS = br.N_SEEDS
TAUS = [3.0, 3.5, 4.0, 4.5, 5.0]
PERSIST = [(1, 1), (2, 3), (3, 5)]          # k of n
plt.rcParams.update({"font.size": 12, "legend.fontsize": 10})


def paired_tests():
    """R-N1: is gated recalibration really 'equally accurate' as continuous FFRLS?"""
    rows = []
    for plant, f, extra in [("Matched", "multiseed_raw.csv", lambda d: d[d.Offset_A == 0]),
                            ("Table", "table_plant_raw.csv", lambda d: d[d.Detector == "V only"])]:
        d0 = extra(pd.read_csv(os.path.join(RES, f)))
        for ag in ["step", "ramp"]:
            for m in bp.MODES:
                d = d0[(d0.Ageing == ag) & (d0.Mode == m)]
                g = d[d.Case == "Gated"].sort_values("Seed").SOC_RMSE.values
                u = d[d.Case == "Ungated"].sort_values("Seed").SOC_RMSE.values
                s = d[d.Case == "Static"].sort_values("Seed").SOC_RMSE.values
                _, p_gu = wilcoxon(g, u)
                _, p_gs = wilcoxon(g, s)
                rows.append(dict(Plant=plant, Ageing=ag, Mode=m,
                                 Static=s.mean(), Ungated=u.mean(), Gated=g.mean(),
                                 Gated_minus_Ungated=g.mean() - u.mean(), p_gated_vs_ungated=p_gu,
                                 Gate_worse=bool(p_gu < 0.05 and g.mean() > u.mean()),
                                 p_gated_vs_static=p_gs,
                                 Reduction_vs_static_pct=100 * (1 - g.mean() / s.mean())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)
    print(t.round(4).to_string(index=False))
    print(f"\ngate significantly worse in {t.Gate_worse.sum()}/{len(t)} comparisons")
    return t


def healthy_soh_distribution():
    """R-N2: the spread behind 'the gate protects SOH', not just the mean."""
    d = pd.read_csv(os.path.join(RES, "table_plant_raw.csv"))
    d = d[(d.Ageing == "none") & (d.Detector == "V only")]
    rows = []
    for m in bp.MODES:
        for case, lab in [("Fresh", "Gated"), ("Fresh-Ungated", "Ungated")]:
            v = d[(d.Mode == m) & (d.Case == case)].SOH_final.values
            rows.append(dict(Mode=m, Case=lab, Mean=v.mean(), SD=v.std(ddof=1),
                             Min=v.min(), P5=np.percentile(v, 5), Median=np.median(v),
                             Frac_below_95=float((v < 95).mean()), Frac_below_90=float((v < 90).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "healthy_soh_distribution.csv"), index=False)
    print(t.round(3).to_string(index=False))
    return t


def capacity_rail():
    """R-N3: on a healthy pack the capacity estimate rails against the 1.1*Qn clip."""
    d = pd.read_csv(os.path.join(RES, "table_plant_raw.csv"))
    d = d[(d.Ageing == "none") & (d.Detector == "V only")]
    rows = []
    for m in bp.MODES:
        for case, lab in [("Fresh", "Gated"), ("Fresh-Ungated", "Ungated")]:
            q = d[(d.Mode == m) & (d.Case == case)].Q_final.values
            rows.append(dict(Mode=m, Case=lab, Q_mean=q.mean(), Q_max=q.max(),
                             Over_nominal_pct=100 * (q.mean() / bp.Q_CELL - 1),
                             Frac_at_clip=float((q > 1.1 * bp.Q_CELL - 1e-3).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "capacity_rail.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t


def detector_redesign(tp, pocv, cycles):
    """R-N4: single-sample threshold vs a k-of-n persistence rule."""
    rows = []
    for m in bp.MODES:
        t, p = cycles[m]
        fresh = tp.simulate(t, p)
        rf, qf = br.ageing_profile(t, "step")
        aged = tp.simulate(t, p, rf=rf, qf=qf)
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(9000 + seed)
            fn = br.noisy(fresh, rng, br.SIGMA_T)
            an = br.noisy(aged, rng, br.SIGMA_T)
            for tau in TAUS:
                for pk, pn in PERSIST:
                    kw = dict(tau=tau, use_temp=False, persist=(pk, pn))
                    rf_ = br.run(fn, pocv, "Gated", **kw)
                    ra_ = br.run(an, pocv, "Gated", **kw)
                    fa = int(rf_.drift.sum() > 0)
                    post = ra_[ra_.t >= bp.T_AGE]
                    first = post.loc[post.drift == 1, "t"]
                    rows.append(dict(Seed=seed, Mode=m, Tau=tau, Rule=f"{pk}of{pn}",
                                     FalseAlarm=fa, Detected=int(len(first) > 0),
                                     Delay=float(first.iloc[0] - bp.T_AGE) if len(first) else np.nan,
                                     SOC_RMSE=float(np.sqrt(((ra_.SOC_est - ra_.SOC_true) ** 2).mean())),
                                     SOH_fresh=float(rf_.SOH.iloc[-1])))
        print(f"  {m} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OUT, "detector_raw.csv"), index=False)
    s = raw.groupby(["Rule", "Tau"]).agg(FalseAlarmRate=("FalseAlarm", "mean"),
                                         DetectionRate=("Detected", "mean"),
                                         MedianDelay_s=("Delay", "median"),
                                         SOC_RMSE=("SOC_RMSE", "mean"),
                                         SOH_fresh_min=("SOH_fresh", "min")).round(3).reset_index()
    s.to_csv(os.path.join(OUT, "detector_summary.csv"), index=False)
    print(s.to_string(index=False))
    per_mode = raw.groupby(["Rule", "Tau", "Mode"]).agg(
        FalseAlarmRate=("FalseAlarm", "mean"), DetectionRate=("Detected", "mean"),
        MedianDelay_s=("Delay", "median")).round(3).reset_index()
    per_mode.to_csv(os.path.join(OUT, "detector_per_mode.csv"), index=False)
    return raw, s, per_mode


def capacity_guard(tp, pocv, cycles):
    """R-N3 fix: require a wider SOC span and forget old rest points."""
    rows = []
    cfg = [("baseline", dict(q_span=0.03, q_lam=1.0)),
           ("span 10%", dict(q_span=0.10, q_lam=1.0)),
           ("span 10% + forget", dict(q_span=0.10, q_lam=0.97))]
    for m in bp.MODES:
        t, p = cycles[m]
        plants = {"none": tp.simulate(t, p)}
        for ag in ["step", "ramp"]:
            rf, qf = br.ageing_profile(t, ag)
            plants[ag] = tp.simulate(t, p, rf=rf, qf=qf)
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(11000 + seed)
            noisy = {k: br.noisy(v, rng, br.SIGMA_T) for k, v in plants.items()}
            for name, kw in cfg:
                for ag, d in noisy.items():
                    r = br.run(d, pocv, "Gated", use_temp=False, **kw)
                    q_true = bp.Q_CELL if ag == "none" else float(d["Q_true"].iloc[-1])
                    rows.append(dict(Seed=seed, Mode=m, Ageing=ag, Config=name,
                                     Q_est=float(r.Q_est_Ah.iloc[-1]), Q_true=q_true,
                                     Q_err_pct=100 * (float(r.Q_est_Ah.iloc[-1]) / q_true - 1),
                                     SOH=float(r.SOH.iloc[-1]),
                                     SOC_RMSE=float(np.sqrt(((r.SOC_est - r.SOC_true) ** 2).mean()))))
        print(f"  {m} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OUT, "capacity_guard_raw.csv"), index=False)
    s = raw.groupby(["Config", "Ageing", "Mode"]).agg(
        Q_err_pct=("Q_err_pct", "mean"), Q_err_sd=("Q_err_pct", "std"),
        SOH=("SOH", "mean"), SOC_RMSE=("SOC_RMSE", "mean")).round(3).reset_index()
    s.to_csv(os.path.join(OUT, "capacity_guard_summary.csv"), index=False)
    print(s.to_string(index=False))
    return raw, s


def soh_split(tp, cycles):
    """R-6: capacity-based SOH and resistance growth reported separately."""
    d = pd.read_csv(os.path.join(RES, "table_plant_raw.csv"))
    d = d[(d.Detector == "V only") & (d.Case == "Gated")]
    cond = pd.read_csv(os.path.join(RES, "table_plant_conditions.csv")).set_index(["Mode", "Ageing"])
    rows = []
    for ag in ["step", "ramp"]:
        for m in bp.MODES:
            q = d[(d.Ageing == ag) & (d.Mode == m)].Q_final
            r = d[(d.Ageing == ag) & (d.Mode == m)].R0_final
            qt = cond.loc[(m, ag), "Q_true_end"]
            rt = cond.loc[(m, ag), "R0_true_end_mOhm"]
            rows.append(dict(Ageing=ag, Mode=m,
                             SOHQ_est=100 * q.mean() / bp.Q_CELL, SOHQ_true=100 * qt / bp.Q_CELL,
                             Rgrowth_est=r.mean() / (bp.R0_CELL * 1000), Rgrowth_true=rt / (bp.R0_CELL * 1000)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "soh_split.csv"), index=False)
    print(t.round(2).to_string(index=False))
    return t


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("[1/6] Paired Wilcoxon tests (gated vs ungated vs static)")
    paired_tests()
    print("\n[2/6] Healthy-pack SOH distribution")
    healthy_soh_distribution()
    print("\n[3/6] Capacity rail on a healthy pack")
    capacity_rail()

    tp = br.TablePlant(br.load_tables())
    pocv = br.PWLOCV(tp)
    cycles = {m: br.chained_power(m) for m in bp.MODES}

    print("\n[4/6] Detector redesign: persistence rule")
    _, det, det_mode = detector_redesign(tp, pocv, cycles)
    print("\n[5/6] Capacity guard")
    capacity_guard(tp, pocv, cycles)
    print("\n[6/6] SOH decomposition")
    soh_split(tp, cycles)

    # Figure: false alarm vs detection for each rule
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    for rule, mk in zip(["1of1", "2of3", "3of5"], ["o-", "s-", "^-"]):
        q = det[det.Rule == rule]
        ax[0].plot(q.Tau, 100 * q.FalseAlarmRate, mk, label=rule)
        ax[1].plot(100 * q.FalseAlarmRate, 100 * q.DetectionRate, mk, label=rule)
    ax[0].set_xlabel(r"Threshold $\tau$ (–)"); ax[0].set_ylabel("False-alarm rate (%)")
    ax[1].set_xlabel("False-alarm rate (%)"); ax[1].set_ylabel("Detection rate (%)")
    for x in ax:
        x.grid(alpha=.3); x.legend(title="k of n")
    ax[0].set_ylim(-5, 105); ax[1].set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detector_rules.png"), dpi=250)
    plt.close(fig)
    print(f"\nDone. Results in {OUT}")
