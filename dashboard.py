"""
dashboard.py - Adaptive Smart BMS dashboard
Run:  streamlit run dashboard.py
Reads results from bms_pipeline_v2.py, shap_lime.py, knn_imputation.py, soh_nasa.py, udp_receiver.py
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE = os.environ.get("ESS_BASE", r"C:\ESS")
RES = os.path.join(BASE, "bms", "results")
XAI = os.path.join(BASE, "bms")
IMP = os.path.join(RES, "imputation")
SOH = os.path.join(RES, "soh")
LIVE = os.path.join(RES, "live")
MODES = ["Eco", "Normal", "Sport"]
T_AGE, TAU_DM = 300, 3.5

st.set_page_config(page_title="Adaptive Smart BMS", page_icon="🔋", layout="wide")


@st.cache_data
def load_run(mode, case):
    return pd.read_csv(os.path.join(RES, f"{mode}_{case}.csv"))


@st.cache_data
def load_summary():
    return pd.read_csv(os.path.join(RES, "summary.csv"))


def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Not found: {path}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔋 Adaptive Smart BMS")
st.sidebar.caption("98S30P Molicel P45B pack · 23EEE332 ESS")
mode = st.sidebar.radio("Drive mode", MODES, index=2)
scenario = st.sidebar.radio("Pack condition", ["Fresh", "Aged (R0 ×2, capacity −20% at 300 s)"], index=1)
aged = scenario.startswith("Aged")

if not os.path.exists(os.path.join(RES, "summary.csv")):
    st.error(f"No results in {RES}. Run `python bms_pipeline_v2.py` first.")
    st.stop()

adaptive = load_run(mode, "Aged-adaptive" if aged else "Fresh")
static = load_run(mode, "Aged-static") if aged else None
t_max = int(adaptive["t"].max())
t_now = st.sidebar.slider("Playback time (s)", int(adaptive["t"].min()), t_max, t_max, step=10)
a = adaptive[adaptive["t"] <= t_now]
s = static[static["t"] <= t_now] if aged else None
now = a.iloc[-1]

tab1, tab5, tab2, tab6, tab7, tab3, tab4 = st.tabs(["📟 Live monitor", "📡 Live UDP", "📊 Results", "🩹 Imputation",
                                                    "🔋 SOH (NASA)", "🧠 Explainability (LG HG2)", "⚙️ Model"])

# ── Tab 1: live monitor ───────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{mode} mode · {'Aged pack' if aged else 'Fresh pack'} · t = {t_now} s")
    drift_now = bool(now["drift"])
    if aged and t_now >= T_AGE and (a.loc[a["t"] >= T_AGE, "drift"] == 1).any():
        first = a.loc[(a["t"] >= T_AGE) & (a["drift"] == 1), "t"].iloc[0]
        st.warning(f"⚠️ Parameter drift detected at t = {first:.0f} s · FFRLS recalibration active")
    elif drift_now:
        st.warning("⚠️ Drift flagged at this sample")
    else:
        st.success("✅ Pack operating within baseline behaviour")

    c = st.columns(5)
    c[0].metric("SOC estimate", f"{now['SOC_est']:.2f} %", f"{now['SOC_est'] - now['SOC_true']:+.2f} % vs true")
    c[1].metric("SOH", f"{now['SOH']:.1f} %")
    c[2].metric("R0 estimate", f"{now['R0_est_mOhm']:.2f} mΩ", f"true {now['R0_true_mOhm']:.0f} mΩ", delta_color="off")
    c[3].metric("Capacity estimate", f"{now['Q_est_Ah']:.3f} Ah", f"true {now['Q_true_Ah']:.3f} Ah", delta_color="off")
    c[4].metric("Mahalanobis DM", f"{now['DM']:.2f}", f"threshold {TAU_DM}", delta_color="off")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("State of charge", "Drift detection (Mahalanobis distance)",
                                        "Internal resistance R0 (FFRLS)", "State of health"))
    fig.add_trace(go.Scatter(x=a.t, y=a.SOC_true, name="True SOC", line=dict(color="black", width=3)), 1, 1)
    if aged:
        fig.add_trace(go.Scatter(x=s.t, y=s.SOC_est, name="Static BMS", line=dict(color="red", dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=a.t, y=a.SOC_est, name="Adaptive BMS", line=dict(color="royalblue")), 1, 1)
    fig.add_trace(go.Scatter(x=a.t, y=a.DM, name="DM", line=dict(color="royalblue"), showlegend=False), 2, 1)
    fig.add_hline(y=TAU_DM, line=dict(color="red", dash="dash"), row=2, col=1)
    d = a[a.drift == 1]
    fig.add_trace(go.Scatter(x=d.t, y=d.DM, mode="markers", name="Drift flagged",
                             marker=dict(color="red", size=8)), 2, 1)
    fig.add_trace(go.Scatter(x=a.t, y=a.R0_true_mOhm, name="True R0", line=dict(color="black", width=3)), 3, 1)
    fig.add_trace(go.Scatter(x=a.t, y=a.R0_est_mOhm, name="R0 estimate", line=dict(color="royalblue")), 3, 1)
    fig.add_trace(go.Scatter(x=a.t, y=a.SOH, name="SOH", line=dict(color="green"), showlegend=False), 4, 1)
    if aged:
        for r in range(1, 5):
            fig.add_vline(x=T_AGE, line=dict(color="grey", dash="dot"), row=r, col=1)
    fig.update_xaxes(range=[0, t_max], title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="DM", row=2, col=1)
    fig.update_yaxes(title_text="mΩ", row=3, col=1)
    fig.update_yaxes(title_text="%", row=4, col=1)
    fig.update_layout(height=900, legend=dict(orientation="h", y=1.04), margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw cell signals (V, I)"):
        f2 = make_subplots(rows=2, cols=1, shared_xaxes=True)
        f2.add_trace(go.Scatter(x=a.t, y=a.V_cell, name="Cell voltage (V)"), 1, 1)
        f2.add_trace(go.Scatter(x=a.t, y=a.I_cell, name="Cell current (A)"), 2, 1)
        f2.update_layout(height=400)
        st.plotly_chart(f2, use_container_width=True)

# ── Tab 2: results ────────────────────────────────────────────────────────────
with tab2:
    summ = load_summary()
    st.subheader("SOC error: static vs adaptive BMS on the aged pack")
    piv = summ.pivot(index="Mode", columns="Case", values="SOC_RMSE_pct").reindex(MODES)
    fb = go.Figure()
    for case, col in [("Fresh", "grey"), ("Aged-static", "red"), ("Aged-adaptive", "royalblue")]:
        fb.add_trace(go.Bar(x=piv.index, y=piv[case], name=case, marker_color=col,
                            text=piv[case].round(3), textposition="outside"))
    fb.update_layout(barmode="group", yaxis_title="SOC RMSE (%)", height=420)
    st.plotly_chart(fb, use_container_width=True)

    cols = st.columns(3)
    for i, m in enumerate(MODES):
        st_ = piv.loc[m, "Aged-static"]
        ad = piv.loc[m, "Aged-adaptive"]
        cols[i].metric(f"{m}: RMSE reduction", f"{(1 - ad / st_) * 100:.0f} %", f"{st_:.3f} → {ad:.3f} %")

    st.subheader("Full summary")
    st.dataframe(summ, use_container_width=True, hide_index=True)
    st.caption("Drifts_after_300s on Fresh runs are false alarms. Capacity is only re-estimated after "
               "≥3 % SOC change since drift detection (minimum-excitation condition).")

# ── Tab 3: explainability ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Which inputs drive SOC prediction? (Random Forest on LG HG2, normalised inputs)")
    fi_path = os.path.join(XAI, "feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        ff = go.Figure(go.Bar(x=fi["Mean_SHAP"], y=fi["Feature"], orientation="h", marker_color="royalblue"))
        ff.update_layout(xaxis_title="Mean |SHAP value|", yaxis=dict(autorange="reversed"), height=320)
        st.plotly_chart(ff, use_container_width=True)
        st.dataframe(fi, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        show_image(os.path.join(XAI, "shap_summary.png"), "SHAP summary (global)")
    with c2:
        show_image(os.path.join(XAI, "lime_explanation.png"), "LIME (single prediction)")

# ── Tab 5: live UDP stream ────────────────────────────────────────────────────
def read_live():
    p = os.path.join(LIVE, "live_stream.csv")
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p, on_bad_lines="skip")
    except Exception:
        return None


def live_view():
    df = read_live()
    if df is None or df.empty:
        st.info("Waiting for data. Run `python udp_receiver.py`, then `udp_stream.m` in MATLAB.")
        return
    now_ = df.iloc[-1]
    if (df["drift"] == 1).any():
        st.warning(f"⚠️ Drift detected at t = {df.loc[df.drift == 1, 't'].iloc[0]:.0f} s · FFRLS recalibration active")
    else:
        st.success("✅ Streaming · pack within baseline behaviour")
    c = st.columns(6)
    c[0].metric("Packets", f"{int(now_['seq'])}", f"lost {int(now_['lost_total'])}", delta_color="off")
    c[1].metric("Latency (p95)", f"{df['latency_ms'].quantile(.95):.2f} ms", f"max {df['latency_ms'].max():.2f}", delta_color="off")
    c[2].metric("BMS step", f"{df['proc_ms'].mean():.2f} ms", f"max {df['proc_ms'].max():.2f}", delta_color="off")
    c[3].metric("SOC", f"{now_['SOC_est']:.2f} %", f"{now_['SOC_est'] - now_['SOC_true']:+.2f} % vs true")
    c[4].metric("R0", f"{now_['R0_est_mOhm']:.2f} mΩ")
    c[5].metric("SOH", f"{now_['SOH']:.1f} %")
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                      subplot_titles=("SOC (live)", "Mahalanobis distance", "R0 estimate"))
    f.add_trace(go.Scatter(x=df.t, y=df.SOC_true, name="True SOC", line=dict(color="black", width=3)), 1, 1)
    f.add_trace(go.Scatter(x=df.t, y=df.SOC_est, name="BMS estimate", line=dict(color="royalblue")), 1, 1)
    f.add_trace(go.Scatter(x=df.t, y=df.DM, name="DM", line=dict(color="royalblue"), showlegend=False), 2, 1)
    f.add_hline(y=TAU_DM, line=dict(color="red", dash="dash"), row=2, col=1)
    d_ = df[df.drift == 1]
    f.add_trace(go.Scatter(x=d_.t, y=d_.DM, mode="markers", name="Drift", marker=dict(color="red", size=8)), 2, 1)
    f.add_trace(go.Scatter(x=df.t, y=df.R0_est_mOhm, name="R0", line=dict(color="green"), showlegend=False), 3, 1)
    f.update_xaxes(range=[0, 1800], title_text="Time (s)", row=3, col=1)
    f.update_layout(height=700, legend=dict(orientation="h", y=1.06))
    st.plotly_chart(f, use_container_width=True)


with tab5:
    st.subheader("MATLAB → Python UDP bridge (127.0.0.1:5005), BMS running live on each packet")
    if hasattr(st, "fragment"):
        st.fragment(run_every=1)(live_view)()
    else:
        st.button("🔄 Refresh")
        live_view()
    sp = os.path.join(LIVE, "udp_summary.csv")
    if os.path.exists(sp):
        st.caption("Last completed stream")
        st.dataframe(pd.read_csv(sp), use_container_width=True, hide_index=True)

# ── Tab 6: imputation ─────────────────────────────────────────────────────────
with tab6:
    st.subheader("Sensor dropout reconstruction (5–20 % random and burst dropouts)")
    sp = os.path.join(IMP, "imputation_summary.csv")
    if os.path.exists(sp):
        st.dataframe(pd.read_csv(sp), use_container_width=True, hide_index=True)
        st.caption("Hybrid = linear interpolation by default, KNN + temporal context for current gaps of ≥ 3 samples.")
    show_image(os.path.join(IMP, "imputation_example_sport_current.png"), "Sport mode: 20 % burst dropouts in pack current")
    show_image(os.path.join(IMP, "imputation_rmse.png"), "RMSE by method and channel")

# ── Tab 7: SOH on NASA ────────────────────────────────────────────────────────
with tab7:
    st.subheader("Data-driven SOH, leave-one-battery-out (NASA 18650 cells)")
    rp = os.path.join(SOH, "soh_results.csv")
    if os.path.exists(rp):
        r = pd.read_csv(rp)
        m = r[r.Test_cell == "MEAN"].set_index("Model")
        cc = st.columns(len(m))
        for i, (name, row) in enumerate(m.iterrows()):
            cc[i].metric(name, f"RMSE {row.RMSE:.2f} %", f"R² {row.R2:.3f}", delta_color="off")
        st.dataframe(r, use_container_width=True, hide_index=True)
    show_image(os.path.join(SOH, "soh_predictions.png"), "Predicted vs measured SOH on unseen cells")
    c1, c2 = st.columns(2)
    with c1:
        show_image(os.path.join(SOH, "soh_shap_summary.png"), "SHAP: which health indicators drive SOH")
    with c2:
        ip = os.path.join(SOH, "soh_shap_importance.csv")
        if os.path.exists(ip):
            st.dataframe(pd.read_csv(ip), use_container_width=True, hide_index=True)
        st.caption("EVDT = time for voltage to fall from 3.8 V to 3.5 V during discharge.")

# ── Tab 4: model ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Cell model and BMS settings")
    c1, c2 = st.columns([3, 2])
    with c1:
        show_image(os.path.join(RES, "ocv_fit.png"), "OCV-SOC curve identified from fresh simulation")
    with c2:
        st.table(pd.DataFrame({
            "Parameter": ["Pack", "Cell", "Cell capacity", "R0 (nominal)", "Ageing injection",
                          "Mahalanobis baseline", "Drift threshold", "FFRLS forgetting factor",
                          "Sensor noise"],
            "Value": ["98S30P", "Molicel INR-21700-P45B", "4.493 Ah", "10 mΩ",
                      "R0 ×2, capacity −20% at 300 s", "first 180 s", f"DM ≥ {TAU_DM}", "0.95",
                      "2 mV, 20 mA per cell"],
        }))