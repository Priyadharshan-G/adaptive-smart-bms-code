"""Paper tables (II-XI) for an extra pack, with the P45B rerun alongside.
Usage: python tools/make_pack_tables.py LFP|NCR   (after tools/run_pack_analysis.sh)"""
import numpy as np, pandas as pd
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def pack_dir(p):
    """P45B lives at the repo root; extra packs under packs/<name>."""
    return ROOT if p == "P45B" else os.path.join(ROOT, "packs", {"LFP_poly": "LFP_polynomial_OCV"}.get(p, p))
MODES = ["Eco", "Normal", "Sport"]


def r(base, *p): return pd.read_csv(os.path.join(pack_dir(base), "bms", "results", *p))


def table3(base):
    R = r(base, "robustness", "multiseed_raw.csv"); R = R[R.Offset_A == 0]
    rows = []
    for ag, cases in [("none", ["Fresh"]), ("step", ["Static", "Ungated", "Gated"]), ("ramp", ["Static", "Ungated", "Gated"])]:
        for c in cases:
            row = {"Ageing": ag, "Case": c}
            for m in MODES:
                q = R[(R.Mode == m) & (R.Case == c) & (R.Ageing == ("step" if ag == "none" else ag))].SOC_RMSE
                row[m] = f"{q.mean():.3f} ± {q.std():.3f}"; row[m + "_mean"] = q.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def table4(base):
    t = r(base, "revision2", "paired_tests.csv")
    t["Gated_flag"] = np.where(t.Gate_worse, "†", "")
    return t[["Plant", "Ageing", "Mode", "Static", "Ungated", "Gated", "Gated_flag", "p_gated_vs_ungated", "Reduction_vs_static_pct"]]


def table5(base):
    d = r(base, "robustness", "table_plant_raw.csv"); d = d[(d.Ageing == "none") & (d.Detector == "V only")]
    det = r(base, "revision2", "detector_raw.csv")
    rows = []
    for m in MODES:
        for lab, v in [("Continuous", d[(d.Mode == m) & (d.Case == "Fresh-Ungated")].SOH_final.values),
                       ("Gated, 1 of 1", d[(d.Mode == m) & (d.Case == "Fresh")].SOH_final.values),
                       ("Gated, 2 of 3", det[(det.Mode == m) & (det.Rule == "2of3") & (det.Tau == 3.5)].SOH_fresh.values)]:
            rows.append(dict(Mode=m, Rule=lab, Mean=v.mean(), Median=np.median(v), P5=np.percentile(v, 5),
                             Min=v.min(), Below95_pct=100 * (v < 95).mean()))
    return pd.DataFrame(rows)


def table6(base):
    s = r(base, "revision2", "detector_summary.csv")
    keep = [("1of1", 3.5), ("1of1", 4.5), ("1of1", 5.0), ("2of3", 3.0), ("2of3", 3.5), ("2of3", 4.0), ("3of5", 3.5)]
    s = pd.concat([s[(s.Rule == a) & (s.Tau == b)] for a, b in keep])
    s[["FalseAlarmRate", "DetectionRate"]] *= 100
    return s.rename(columns={"FalseAlarmRate": "FalseAlarm_pct", "DetectionRate": "Detection_pct",
                             "SOH_fresh_min": "WorstHealthySOH_pct"})


def table7(base):
    d = r(base, "robustness", "table_plant_raw.csv"); d = d[d.Detector == "V only"]
    cond = r(base, "robustness", "table_plant_conditions.csv")
    rows = []
    for ag in ["step", "ramp"]:
        for m in MODES:
            q = d[(d.Ageing == ag) & (d.Mode == m)]
            g = q[q.Case == "Gated"]
            c = cond[(cond.Mode == m) & (cond.Ageing == ag)].iloc[0]
            rows.append(dict(Ageing=ag, Mode=m, Static=q[q.Case == "Static"].SOC_RMSE.mean(),
                             Ungated=q[q.Case == "Ungated"].SOC_RMSE.mean(),
                             Gated=f"{g.SOC_RMSE.mean():.3f} ± {g.SOC_RMSE.std():.3f}", Gated_mean=g.SOC_RMSE.mean(),
                             Q_est_Ah=g.Q_final.mean(), Q_true_Ah=c.Q_true_end, SOC_end_pct=c.SOC_end, T_end=c.T_end))
    fr = d[(d.Ageing == "none") & (d.Case == "Fresh")].groupby("Mode").SOC_RMSE.mean()
    return pd.DataFrame(rows), fr


def table8(base):
    s = r(base, "revision2", "capacity_guard_summary.csv")
    p = s.pivot_table(index=["Ageing", "Mode"], columns="Config", values="Q_err_pct").reset_index()
    return p[["Ageing", "Mode", "baseline", "span 10%", "span 10% + forget"]]


def table9(base):
    return r(base, "revision2", "soh_split.csv")


def offset(base):
    R = r(base, "robustness", "multiseed_raw.csv"); R = R[R.Ageing == "step"]
    g = R.groupby(["Offset_A", "Mode", "Case"]).SOC_RMSE.agg(["mean", "std"]).reset_index()
    return g[g.Case.isin(["Static", "Gated"])]


def tau_sweep(base):
    return r(base, "robustness", "tau_sweep_summary.csv")


def validation(base):
    return r(base, "robustness", "table_plant_validation.csv")


def imputation(base):
    return pd.read_csv(os.path.join(pack_dir(base), "bms", "results", "imputation_can", "imputation_summary.csv"))


if __name__ == "__main__":
    import sys
    PACK = sys.argv[1] if len(sys.argv) > 1 else "LFP"          # LFP or NCR
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    rep = os.path.join(pack_dir(PACK), "report")
    out = os.path.join(rep, f"{PACK}_results_tables.xlsx")
    resp = pd.read_csv(os.path.join(rep, "table_pack_response.csv"))
    with pd.ExcelWriter(out) as xw:
        resp.to_excel(xw, sheet_name="II Pack response", index=False)
        for name, fn in [("III SOC RMSE matched", table3), ("IV Paired tests", table4), ("V Healthy SOH", table5),
                         ("VI Detector rules", table6), ("VIII Capacity guard", table8), ("IX SOH split", table9),
                         ("Offset", offset), ("Tau sweep", tau_sweep), ("Table plant valid", validation),
                         ("XI Imputation", imputation)]:
            for base, tag in [(PACK, PACK), ("P45B", "P45B rerun")]:
                t = fn(base); t.insert(0, "Pack", tag)
                t.to_excel(xw, sheet_name=f"{name[:24]} {tag[:4]}", index=False)
        for base in [PACK, "P45B"]:
            t7, fr = table7(base); t7.insert(0, "Pack", base)
            t7.to_excel(xw, sheet_name=f"VII Table plant {base}", index=False)
        if PACK == "LFP" and os.path.isdir(pack_dir("LFP_poly")):
            t = table3("LFP_poly"); t.insert(0, "Pack", "LFP, eq.(5) polynomial OCV")
            t.to_excel(xw, sheet_name="III LFP polynomial OCV", index=False)
    print("saved", out)
