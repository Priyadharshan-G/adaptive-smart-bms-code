"""
dashboard_v2.py - Adaptive Smart BMS dashboard (three packs, live UDP)

Run from the repo root:
    pip install streamlit plotly pandas numpy scipy scikit-learn
    streamlit run dashboard_v2.py

Live UDP: open the "Live UDP" tab, press "Start listening", then either
  - MATLAB: run udp_stream_pack.m (set pack/mode inside), or
  - no MATLAB: press "Send demo stream" (or: python live_udp.py send --pack NCR --aged)
Packets: 7 doubles [seq, t, V_pack, I_pack, T, SOC_true, send_time] to 127.0.0.1:5005.
The old dashboard.py and udp_receiver.py are unchanged; do not run udp_receiver.py at the
same time (both need port 5005).
"""
import json
import os
import re
import sys
import threading
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import live_udp as lu  # noqa: E402

MODES = ["Eco", "Normal", "Sport"]
PACKS = ["P45B", "NCR", "LFP"]
T_AGE, TAU = 300, 3.5

# ── Look and feel ─────────────────────────────────────────────────────────────
INK, MUTED, GRID = "#1d2433", "#5b6475", "#e6e9ef"
PACK_COLOR = {"P45B": "#2a78d6", "NCR": "#eb6834", "LFP": "#1baf7a"}
MODE_COLOR = {"Eco": "#7a5af8", "Normal": "#d49a00", "Sport": "#0e8f9a"}
CASE = {"true": INK, "static": "#d64545", "gated": "#2a78d6", "ungated": "#e0a100"}
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

PACK_INFO = {
    "P45B": dict(name="Molicel P45B", cell="21700, Molicel P45B", config="98S30P", cells=2940, q_cell=4.493,
                 q_pack=134.8, energy=47.6, v0=408.5, r0=10, r0_range="8.3–13.0", ocv="Cubic", thermal="Yes"),
    "NCR": dict(name="Panasonic NCR18650BD", cell="18650, Panasonic NCR18650BD", config="98S45P", cells=4410,
                q_cell=3.153, q_pack=141.9, energy=50.1, v0=402.2, r0=47, r0_range="33.3–80.2", ocv="Cubic",
                thermal="No (25 °C)"),
    "LFP": dict(name="A123 ANR26650 (LFP)", cell="26650 LFP, A123 ANR26650", config="108S57P", cells=6156,
                q_cell=2.341, q_pack=133.4, energy=47.5, v0=415.4, r0=15, r0_range="12.4–20.4",
                ocv="PCHIP", thermal="No (25 °C)"),
}

st.set_page_config(page_title="Adaptive Smart BMS", page_icon="🔋", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{ font-family: {FONT}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
h1, h2, h3 {{ font-family: {FONT}; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; font-size: 2.3rem; margin-bottom: 0.1rem; }}
h2 {{ font-weight: 650; font-size: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.3rem; }}
p, li, label, .stMarkdown {{ font-size: 1.08rem; color: {INK}; }}
.subtitle {{ color: {MUTED}; font-size: 1.15rem; margin-bottom: 1.2rem; }}
[data-testid="stMetric"] {{ background: #f6f8fb; border: 1px solid {GRID}; border-radius: 12px; padding: 14px 18px; }}
[data-testid="stMetricLabel"] p {{ font-size: 1.0rem !important; color: {MUTED} !important; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-size: 2.0rem !important; font-weight: 650; color: {INK}; }}
[data-testid="stMetricDelta"] {{ font-size: 0.95rem !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 1.08rem; font-weight: 550; padding: 10px 16px; }}
[role="tab"] p, [data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p {{ font-size: 1.1rem !important; font-weight: 600; }}
[role="tab"] {{ padding: 10px 12px !important; }}
.stButton button p {{ font-size: 1.08rem; font-weight: 600; }}
.card.stat {{ background: #f6f8fb; padding: 14px 18px; min-height: 128px; margin-bottom: 12px; }}
.card.stat .big {{ font-size: 2.0rem; font-weight: 650; line-height: 1.25; }}
section[data-testid="stSidebar"] {{ background: #f6f8fb; }}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {{ font-size: 1.02rem; }}
.chip {{ display:inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 0.95rem; }}
.card {{ border: 1px solid {GRID}; border-radius: 12px; padding: 16px 20px; background: #ffffff; height: 100%; }}
.card h4 {{ margin: 0 0 6px 0; font-size: 1.0rem; color: {MUTED}; font-weight: 500; }}
.card .big {{ font-size: 1.7rem; font-weight: 700; color: {INK}; }}
.card .small {{ font-size: 0.98rem; color: {MUTED}; }}
.stDataFrame {{ font-size: 1.0rem; }}
</style>
""", unsafe_allow_html=True)

pio.templates["bms"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=16, color=INK), paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#b9c0cc", ticks="outside", tickcolor="#b9c0cc"),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#b9c0cc", ticks="outside", tickcolor="#b9c0cc"),
    legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, font=dict(size=16)),
    hoverlabel=dict(font=dict(family=FONT, size=15)), margin=dict(l=80, r=30, t=110, b=70),
    colorway=[PACK_COLOR["P45B"], PACK_COLOR["NCR"], PACK_COLOR["LFP"], "#e87ba4", "#8a63d2"]))
pio.templates.default = "plotly_white+bms"


# ── Small helpers (work on old and new Streamlit) ─────────────────────────────
def chart(fig, key=None):
    for tr in fig.data:                      # plain lines unless markers were asked for
        if tr.type == "scatter" and tr.mode is None:
            tr.mode = "lines"
    try:
        st.plotly_chart(fig, width="stretch", key=key, theme=None, config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, theme=None, config={"displaylogo": False})


def table(df, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def image(path, caption=None):
    if not os.path.exists(path):
        st.info(f"Not found: {os.path.relpath(path, ROOT)}")
        return
    try:
        st.image(path, caption=caption, width="stretch")
    except TypeError:
        st.image(path, caption=caption, use_container_width=True)


def fragment(run_every=None):
    f = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    return (lambda fn: f(run_every=run_every)(fn)) if f else (lambda fn: fn)


def chip(text, color):
    st.markdown(f'<span class="chip" style="background:{color}1f;color:{color};border:1px solid {color}55">{text}</span>',
                unsafe_allow_html=True)


def stat(col, label, value, sub=""):
    col.markdown(f'<div class="card stat"><h4>{label}</h4><div class="big">{value}</div>'
                 f'<div class="small">{sub}</div></div>', unsafe_allow_html=True)


def card(col, title, big, small=""):
    col.markdown(f'<div class="card"><h4>{title}</h4><div class="big">{big}</div><div class="small">{small}</div></div>',
                 unsafe_allow_html=True)


def res(pack, *p):
    return os.path.join(lu.pack_dir(pack), "bms", "results", *p)


@st.cache_data
def csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def cross():
    p = os.path.join(ROOT, "packs", "cross_pack.json")
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_data
def cell_tables(pack):
    """OCV and R0 against SOC (25 °C column) straight from the pack's Simulink model."""
    slx = os.path.join(ROOT, {"P45B": "ESSSimulation.slx", "NCR": "ESSSimulation_NCR.slx",
                              "LFP": "ESSSimulation_LFP.slx"}[pack])
    with zipfile.ZipFile(slx) as z:
        for n in z.namelist():
            if not re.match(r"simulink/systems/system_\d+\.xml", n):
                continue
            txt = z.read(n).decode("utf-8", "ignore")
            par = lambda k: np.array([[float(v) for v in r.replace(",", " ").split()]  # noqa: E731
                                      for r in re.search(r'Name="%s">([^<]*)<' % k, txt).group(1).strip("[]").split(";")])
            if 'Name="V0_matCell"' in txt:
                T = par("T_vecCell").ravel(); j = int(np.argmin(abs(T - 25)))
                return par("SOC_vecCell").ravel(), par("V0_matCell")[:, j], par("R0_matCell")[:, j]
            if 'Name="V0_vecCell"' in txt:
                return par("SOC_vecCell").ravel(), par("V0_vecCell").ravel(), par("R0_vecCell").ravel()
    return None


def pct(a, b):
    return 100 * (1 - b / a)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔋 Adaptive Smart BMS")
    st.caption("Drift-gated FFRLS recalibration · 23EEE332 ESS")
    pack = st.radio("Battery pack", PACKS, index=0,
                    format_func=lambda p: f"{PACK_INFO[p]['name']} · {PACK_INFO[p]['config']}")
    mode = st.radio("Drive mode", MODES, index=2, horizontal=True)
    aged = st.radio("Pack condition", ["Fresh", "Aged"], index=1, horizontal=True) == "Aged"
    st.caption("Aged = R0 × 2 and capacity −20 % from t = 300 s.")
    st.divider()
    info = PACK_INFO[pack]
    st.markdown(f"**{info['cell']}**  \n{info['config']} · {info['cells']} cells  \n"
                f"{info['q_pack']} Ah · {info['energy']} kWh · {info['v0']} V")

PC = PACK_COLOR[pack]
st.markdown("# Adaptive Smart BMS")
st.markdown(f'<div class="subtitle">Drift detection and online recalibration · '
            f'<b style="color:{PC}">{info["name"]} {info["config"]}</b> · {mode} mode · '
            f'{"aged pack" if aged else "fresh pack"}</div>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Playback", "Live UDP", "Ageing", "Detector", "Compare packs", "Imputation",
                "SOC & SOH models"])

J = cross()

# ── 1. Overview ───────────────────────────────────────────────────────────────
with tabs[0]:
    c = st.columns(5)
    stat(c[0], "Configuration", info["config"], f"{info['cells']} cells")
    stat(c[1], "Pack capacity", f"{info['q_pack']} Ah", f"{info['q_cell']} Ah per cell")
    stat(c[2], "Nominal energy", f"{info['energy']} kWh", f"start {info['v0']} V")
    stat(c[3], "BMS nominal R0", f"{info['r0']} mΩ", f"plant {info['r0_range']} mΩ")
    stat(c[4], "OCV model in the BMS", info["ocv"], f"thermal model: {info['thermal']}")

    if J:
        st.markdown("### Key results (30 noise seeds)")
        t3 = J[f"{pack}_t3"]
        tp = {r["Mode"]: r for r in J[f"{pack}_t7"] if r["Ageing"] == "step"}
        det = [d for d in J[f"{pack}_t6"] if d["Rule"] == "2of3" and d["Tau"] == 3.5][0]
        k = st.columns(4)
        card(k[0], "Sport SOC error after ageing (matched)",
             f"{t3['step_Static']['Sport']:.2f} → {t3['step_Gated']['Sport']:.2f} %",
             f"static → gated · {pct(t3['step_Static']['Sport'], t3['step_Gated']['Sport']):.0f} % lower")
        red = [pct(tp[m]["Static"], tp[m]["Gated_mean"]) for m in MODES]
        card(k[1], "Deep discharge, error reduction vs static",
             " / ".join(f"{r:.0f}" for r in red) + " %", "Eco / Normal / Sport, 4 chained cycles")
        qe = 100 * (tp["Sport"]["Q_est_Ah"] / tp["Sport"]["Q_true_Ah"] - 1)
        card(k[2], "Aged capacity estimate, Sport",
             f"{tp['Sport']['Q_est_Ah']:.3f} Ah", f"true {tp['Sport']['Q_true_Ah']:.3f} Ah · error {qe:+.1f} %")
        card(k[3], "Drift detector (2 of 3, τ = 3.5)",
             f"{det['FalseAlarm_pct']:.1f} % / {det['Detection_pct']:.1f} %", "false alarms / detection")

        st.markdown("### Three packs at a glance")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Matched plant, step ageing, Sport",
                                                            "Table plant, step ageing (4 cycles)"))
        for p in PACKS:
            t = J[f"{p}_t3"]
            fig.add_trace(go.Bar(name=p, x=["Static", "Gated"], y=[t["step_Static"]["Sport"], t["step_Gated"]["Sport"]],
                                 marker_color=PACK_COLOR[p], text=[f"{t['step_Static']['Sport']:.2f}",
                                 f"{t['step_Gated']['Sport']:.2f}"], textposition="outside", legendgroup=p), 1, 1)
            tps = {r["Mode"]: r for r in J[f"{p}_t7"] if r["Ageing"] == "step"}
            fig.add_trace(go.Bar(name=p, x=MODES, y=[pct(tps[m]["Static"], tps[m]["Gated_mean"]) for m in MODES],
                                 marker_color=PACK_COLOR[p], showlegend=False, legendgroup=p,
                                 text=[f"{pct(tps[m]['Static'], tps[m]['Gated_mean']):.0f} %" for m in MODES],
                                 textposition="outside"), 1, 2)
        fig.update_yaxes(type="log", title_text="SOC RMSE (%)", row=1, col=1)
        fig.update_yaxes(title_text="Error reduction vs static (%)", range=[0, 100], row=1, col=2)
        fig.update_layout(barmode="group", height=520, bargap=0.25, legend=dict(y=1.12))
        chart(fig, "ov_bars")
        st.markdown("The method carries over to the **NCR** pack (large resistance, steady OCV slope). "
                    "On the **LFP** pack the flat voltage plateau gives the observer and the drift test "
                    "almost no information, so Sport mode breaks down.")

# ── 2. BMS playback ───────────────────────────────────────────────────────────
with tabs[1]:
    a_full = csv(res(pack, f"{mode}_{'Aged-adaptive' if aged else 'Fresh'}.csv"))
    s_full = csv(res(pack, f"{mode}_Aged-static.csv")) if aged else None
    if a_full is None:
        st.error("Seed-42 results not found for this pack. Run tools/run_pack_analysis.sh first.")
    else:
        t_end = int(a_full.t.max())
        key = f"t_{pack}_{mode}_{aged}"
        st.session_state.setdefault(key, t_end)
        st.session_state.setdefault("playing", False)
        def _toggle():
            if not st.session_state.playing and st.session_state[key] >= t_end:
                st.session_state[key] = 10
            st.session_state.playing = not st.session_state.playing

        def _reset():
            st.session_state[key], st.session_state.playing = 10, False

        b1, b2, b3 = st.columns([1, 1, 6])
        b1.button("⏸  Pause" if st.session_state.playing else "▶  Play", key="play", type="primary", on_click=_toggle)
        b2.button("⟲  Reset", key="reset", on_click=_reset)

        @fragment(run_every=0.35 if st.session_state.playing else None)
        def playback():
            if st.session_state.playing:
                st.session_state[key] = min(st.session_state[key] + 20, t_end)
                if st.session_state[key] >= t_end:
                    st.session_state.playing = False
                    st.rerun()
            t_now = st.session_state[key]
            a = a_full[a_full.t <= t_now]
            s = s_full[s_full.t <= t_now] if aged else None
            now = a.iloc[-1]
            post = a[(a.t >= T_AGE) & (a.drift == 1)]
            if aged and len(post):
                chip(f"⚠  Drift detected at t = {post.t.iloc[0]:.0f} s · FFRLS recalibration active", "#d64545")
            elif now.drift:
                chip("⚠  Drift flagged at this sample", "#e0a100")
            else:
                chip(f"●  Within baseline behaviour · t = {t_now} s", "#1a9e5c")
            m = st.columns(5)
            stat(m[0], "SOC estimate", f"{now.SOC_est:.2f} %", f"true {now.SOC_true:.2f} % · error {now.SOC_est - now.SOC_true:+.2f} %")
            stat(m[1], "State of health", f"{now.SOH:.1f} %", "never allowed to rise")
            stat(m[2], "R0 estimate", f"{now.R0_est_mOhm:.1f} mΩ", f"true {now.R0_true_mOhm:.1f} mΩ")
            stat(m[3], "Capacity estimate", f"{now.Q_est_Ah:.3f} Ah", f"true {now.Q_true_Ah:.3f} Ah")
            stat(m[4], "Mahalanobis D_M", f"{now.DM:.2f}", f"threshold {TAU}")
            f = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              subplot_titles=("State of charge (%)", "Drift test: Mahalanobis distance",
                                              "Series resistance R0 (mΩ)", "State of health (%)"))
            f.add_trace(go.Scatter(x=a.t, y=a.SOC_true, name="True", line=dict(color=CASE["true"], width=3.5)), 1, 1)
            if aged:
                f.add_trace(go.Scatter(x=s.t, y=s.SOC_est, name="Static BMS",
                                       line=dict(color=CASE["static"], width=2.5, dash="dash")), 1, 1)
            f.add_trace(go.Scatter(x=a.t, y=a.SOC_est, name="Adaptive BMS", line=dict(color=PC, width=3)), 1, 1)
            f.add_trace(go.Scatter(x=a.t, y=a.DM, showlegend=False, line=dict(color=PC, width=2)), 2, 1)
            f.add_hline(y=TAU, line=dict(color=CASE["static"], dash="dash", width=2), row=2, col=1)
            d = a[a.drift == 1]
            f.add_trace(go.Scatter(x=d.t, y=d.DM, mode="markers", name="Drift flagged",
                                   marker=dict(color=CASE["static"], size=11, line=dict(color="white", width=2))), 2, 1)
            f.add_trace(go.Scatter(x=a.t, y=a.R0_true_mOhm, showlegend=False, line=dict(color=CASE["true"], width=3.5)), 3, 1)
            f.add_trace(go.Scatter(x=a.t, y=a.R0_est_mOhm, showlegend=False, line=dict(color=PC, width=3)), 3, 1)
            f.add_trace(go.Scatter(x=a.t, y=a.SOH, showlegend=False, line=dict(color="#1a9e5c", width=3)), 4, 1)
            if aged:
                for r in range(1, 5):
                    f.add_vline(x=T_AGE, line=dict(color="#9aa3b2", dash="dot", width=2), row=r, col=1)
            f.update_xaxes(range=[0, t_end + 10])
            f.update_xaxes(title_text="Time (s)", row=4, col=1)
            lo = min(a_full.SOC_true.min(), a_full.SOC_est.min(), s_full.SOC_est.min() if aged else 100)
            f.update_yaxes(range=[max(lo - 1, -1), 100.5], row=1, col=1)
            f.update_yaxes(range=[0, max(5, a_full.DM.max() * 1.1)], row=2, col=1)
            f.update_yaxes(range=[min(a_full.R0_est_mOhm.min(), a_full.R0_true_mOhm.min()) * 0.9,
                                  max(a_full.R0_est_mOhm.max(), a_full.R0_true_mOhm.max()) * 1.08], row=3, col=1)
            f.update_yaxes(range=[min(40, a_full.SOH.min() - 3), 101], row=4, col=1)
            for ann in f.layout.annotations:
                ann.font = dict(size=17, family=FONT, color=INK)
            f.update_layout(height=1000, margin=dict(t=120), legend=dict(y=1.05))
            chart(f, "playback_fig")

        playback()
        st.caption("Single run (seed 42). The adaptive BMS adopts FFRLS resistance updates only after the "
                   "drift test fires; capacity is re-estimated from rest points.")

# ── 3. Live UDP ───────────────────────────────────────────────────────────────
@st.cache_resource
def live_state():
    return {"listener": None, "sender": None, "stop": threading.Event(), "sent": [0, 0]}


with tabs[2]:
    S = live_state()
    L = S["listener"]
    st.markdown("### MATLAB → Python live link")
    st.markdown("The dashboard listens for UDP packets and runs the adaptive BMS on every one as it arrives. "
                f"The BMS uses the settings of the pack selected in the sidebar (**{pack}**).")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 3])
    port = c3.number_input("Port", 1024, 65535, 5005, step=1)
    if c1.button("●  Start listening", type="primary", key="lstart"):
        if L is not None and L.running and (L.pack != pack or L.port != port):
            L.stop()
        if L is None or not L.running:
            S["listener"] = L = lu.UdpBmsListener(pack, port=int(port))
            L.start()
    if c2.button("■  Stop", key="lstop") and L is not None:
        L.stop()
    with c4.expander("Send a demo stream (no MATLAB needed)", expanded=False):
        d1, d2, d3 = st.columns(3)
        dm = d1.selectbox("Mode", MODES, index=2, key="dmode")
        da = d2.selectbox("Condition", ["Aged", "Fresh"], key="dage") == "Aged"
        sp = d3.select_slider("Speed-up", [5, 10, 20, 50], value=20, key="dspeed")
        sender = S["sender"]
        if st.button("➤  Send demo stream", key="dsend", disabled=sender is not None and sender.is_alive()):
            S["stop"].clear()
            stream = lu.make_stream(pack, dm, da)
            S["sent"] = [0, len(stream)]
            S["sender"] = threading.Thread(target=lu.replay, daemon=True,
                                           args=(stream, "127.0.0.1", int(port), float(sp), S["stop"],
                                                 lambda k, n: S["sent"].__setitem__(0, k)))
            S["sender"].start()
        st.caption("MATLAB instead: run `udp_stream_pack.m` with the same pack, mode and port.")

    @fragment(run_every=1.0)
    def live_view():
        L = live_state()["listener"]
        sent = live_state()["sent"]
        if L is None:
            chip("○  Not listening · press Start listening", "#5b6475")
            return
        if L.status == "error":
            st.error(L.error)
            return
        colour = {"listening": "#5b6475", "streaming": "#1a9e5c", "finished": "#2a78d6", "stopped": "#5b6475"}
        label = {"listening": f"●  Listening on 127.0.0.1:{L.port} · waiting for packets",
                 "streaming": f"●  Streaming · {L.pack} BMS running live",
                 "finished": "✓  Stream finished · saved to bms/results/live_dashboard/",
                 "stopped": "○  Stopped"}
        chip(label.get(L.status, L.status), colour.get(L.status, "#5b6475"))
        if L.pack != pack:
            st.caption(f"Listener is running with the {L.pack} BMS. Stop and start again to switch to {pack}.")
        if sent[1] and sent[0] < sent[1]:
            st.progress(sent[0] / sent[1], text=f"Demo sender: {sent[0]} / {sent[1]} packets")
        df = L.frame()
        if df.empty:
            return
        now = df.iloc[-1]
        drift_t = df.loc[(df.drift == 1) & (df.t >= T_AGE), "t"]
        if len(drift_t):
            st.warning(f"Drift detected at t = {drift_t.iloc[0]:.0f} s · FFRLS recalibration active")
        m = st.columns(4)
        stat(m[0], "SOC estimate", f"{now.SOC_est:.2f} %", f"true {now.SOC_true:.2f} % · error {now.SOC_est - now.SOC_true:+.2f} %")
        stat(m[1], "Series resistance R0", f"{now.R0_est_mOhm:.1f} mΩ", "FFRLS, adopted after drift")
        stat(m[2], "State of health", f"{now.SOH:.1f} %", f"capacity {now.Q_est_Ah:.3f} Ah")
        stat(m[3], "Pack voltage · current", f"{now.V_pack:.1f} V", f"{now.I_pack:.1f} A at t = {now.t:.0f} s")
        n = st.columns(4)
        stat(n[0], "Packets received", f"{L.packets}", f"lost {L.lost}")
        stat(n[1], "Network latency (p95)", f"{df.latency_ms.quantile(.95):.2f} ms", f"mean {df.latency_ms.mean():.2f} ms")
        stat(n[2], "BMS step time", f"{df.proc_ms.mean():.2f} ms", f"max {df.proc_ms.max():.2f} ms")
        stat(n[3], "Drift test D_M", f"{now.DM:.2f}", f"threshold {TAU}")
        f = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.08,
                          subplot_titles=("State of charge (%)", "Mahalanobis distance",
                                          "Series resistance R0 (mΩ)", "Pack current (A)"))
        f.add_trace(go.Scatter(x=df.t, y=df.SOC_true, name="True", line=dict(color=INK, width=3.5)), 1, 1)
        f.add_trace(go.Scatter(x=df.t, y=df.SOC_est, name="Live BMS estimate",
                               line=dict(color=PACK_COLOR[L.pack], width=3)), 1, 1)
        f.add_trace(go.Scatter(x=df.t, y=df.DM, showlegend=False, line=dict(color=PACK_COLOR[L.pack], width=2)), 1, 2)
        f.add_hline(y=TAU, line=dict(color=CASE["static"], dash="dash", width=2), row=1, col=2)
        d = df[df.drift == 1]
        f.add_trace(go.Scatter(x=d.t, y=d.DM, mode="markers", name="Drift flagged",
                               marker=dict(color=CASE["static"], size=11, line=dict(color="white", width=2))), 1, 2)
        f.add_trace(go.Scatter(x=df.t, y=df.R0_est_mOhm, showlegend=False,
                               line=dict(color=PACK_COLOR[L.pack], width=3)), 2, 1)
        f.add_trace(go.Scatter(x=df.t, y=df.I_pack, showlegend=False, line=dict(color="#8a63d2", width=2)), 2, 2)
        for r in (1, 2):
            for cc in (1, 2):
                f.update_xaxes(range=[0, 1810], title_text="Time (s)" if r == 2 else None, row=r, col=cc)
        for ann in f.layout.annotations:
            ann.font = dict(size=17, family=FONT, color=INK)
        f.update_layout(height=760, margin=dict(t=130), legend=dict(y=1.1))
        chart(f, "live_fig")

    live_view()
    sp_ = csv(os.path.join(lu.pack_dir(pack), "bms", "results", "live_dashboard", "udp_summary.csv"))
    if sp_ is not None:
        st.caption(f"Last completed stream for {pack}")
        table(sp_)

# ── 4. Ageing results ─────────────────────────────────────────────────────────
with tabs[3]:
    if J is None:
        st.info("Run tools/cross_pack.py to build packs/cross_pack.json.")
    else:
        st.markdown(f"### SOC error on the matched plant · {info['name']} (30 seeds)")
        t3 = J[f"{pack}_t3"]
        rows = [("Fresh", "none_Fresh"), ("Step · static", "step_Static"), ("Step · ungated", "step_Ungated"),
                ("Step · gated", "step_Gated"), ("Ramp · static", "ramp_Static"), ("Ramp · ungated", "ramp_Ungated"),
                ("Ramp · gated", "ramp_Gated")]
        sd = J[f"{pack}_t3sd"]
        table(pd.DataFrame([{"Case": n, **{m: sd[k][m] for m in MODES}} for n, k in rows]))
        f = go.Figure()
        for lab, k, col, pat in [("Static", "step_Static", CASE["static"], "/"), ("Ungated", "step_Ungated", CASE["ungated"], ""),
                                 ("Gated", "step_Gated", PC, "")]:
            f.add_trace(go.Bar(name=lab, x=MODES, y=[t3[k][m] for m in MODES], marker_color=col,
                               marker_pattern_shape=pat, text=[f"{t3[k][m]:.3f}" for m in MODES], textposition="outside"))
        f.update_layout(barmode="group", height=430, yaxis_title="SOC RMSE (%), step ageing",
                        yaxis_type="log" if max(t3["step_Static"].values()) > 10 else None)
        chart(f, "age_bars")

        st.markdown("### Deep discharge on the independent table plant (4 chained cycles)")
        t7 = pd.DataFrame(J[f"{pack}_t7"])
        t7["Reduction vs static (%)"] = (100 * (1 - t7.Gated_mean / t7.Static)).round(0)
        t7["Q error (%)"] = (100 * (t7.Q_est_Ah / t7.Q_true_Ah - 1)).round(1)
        table(t7.rename(columns={"Gated_mean": "Gated", "Q_est_Ah": "Q est. (Ah)", "Q_true_Ah": "Q true (Ah)",
                                 "SOC_end_pct": "Lowest SOC (%)", "T_end": "End temp. (°C)"}).round(3))
        c1, c2, c3 = st.columns(3)
        for col, m in zip((c1, c2, c3), MODES):
            with col:
                image(res(pack, "paper_figures", f"table_plant_{m.lower()}_ramp.png"), f"{m}, gradual ageing (seed 0)")

        st.markdown("### Capacity fade and resistance growth, reported separately")
        t9 = pd.DataFrame(J[f"{pack}_t9"]).round(2)
        table(t9.rename(columns={"SOHQ_est": "SOH_Q est. (%)", "SOHQ_true": "SOH_Q true (%)",
                                 "Rgrowth_est": "R0/R0n est.", "Rgrowth_true": "R0/R0n true"}))

# ── 5. Drift detector ─────────────────────────────────────────────────────────
with tabs[4]:
    ts = csv(res(pack, "robustness", "tau_sweep_summary.csv"))
    ds = csv(res(pack, "revision2", "detector_summary.csv"))
    hs = csv(res(pack, "revision2", "healthy_soh_distribution.csv"))
    st.markdown(f"### Single-sample threshold sweep · matched plant · {info['name']}")
    if ts is not None:
        f = make_subplots(rows=1, cols=2, subplot_titles=("False alarms on a healthy pack (%)", "Detection of step ageing (%)"))
        for m in MODES:
            col = MODE_COLOR[m]
            q = ts[ts.Mode == m]
            f.add_trace(go.Scatter(x=q.Tau, y=100 * q.FalseAlarmRate, name=m, mode="lines+markers",
                                   line=dict(color=col, width=3), marker=dict(size=10)), 1, 1)
            f.add_trace(go.Scatter(x=q.Tau, y=100 * q.DetectionRate, name=m, mode="lines+markers", showlegend=False,
                                   line=dict(color=col, width=3), marker=dict(size=10)), 1, 2)
        f.update_xaxes(title_text="Threshold τ")
        f.update_yaxes(range=[-5, 105])
        for ann in f.layout.annotations:
            ann.font = dict(size=17, family=FONT, color=INK)
        f.update_layout(height=470, legend=dict(y=1.12))
        chart(f, "tau_fig")
    st.markdown("### Persistence rules · table plant (90 healthy + 90 aged runs)")
    if ds is not None:
        f = make_subplots(rows=1, cols=2, subplot_titles=("False alarms vs threshold (%)", "Detection vs false alarms (%)"))
        for rule, col, sym in [("1of1", "#d64545", "circle"), ("2of3", PC, "square"), ("3of5", "#8a63d2", "triangle-up")]:
            q = ds[ds.Rule == rule]
            lab = {"1of1": "1 of 1", "2of3": "2 of 3", "3of5": "3 of 5"}[rule]
            f.add_trace(go.Scatter(x=q.Tau, y=100 * q.FalseAlarmRate, name=lab, mode="lines+markers",
                                   line=dict(color=col, width=3), marker=dict(size=11, symbol=sym)), 1, 1)
            f.add_trace(go.Scatter(x=100 * q.FalseAlarmRate, y=100 * q.DetectionRate, name=lab, showlegend=False,
                                   mode="lines+markers", line=dict(color=col, width=3), marker=dict(size=11, symbol=sym)), 1, 2)
        f.update_xaxes(title_text="Threshold τ", row=1, col=1)
        f.update_xaxes(title_text="False-alarm rate (%)", row=1, col=2)
        f.update_yaxes(range=[-5, 105])
        for ann in f.layout.annotations:
            ann.font = dict(size=17, family=FONT, color=INK)
        f.update_layout(height=470, legend=dict(y=1.12))
        chart(f, "rules_fig")
        sel = ds[(ds.Rule == "2of3") & (ds.Tau == 3.5)].iloc[0]
        k = st.columns(4)
        k[0].metric("False alarms (2 of 3)", f"{100 * sel.FalseAlarmRate:.1f} %")
        k[1].metric("Detection", f"{100 * sel.DetectionRate:.1f} %")
        k[2].metric("Median delay", f"{sel.MedianDelay_s:.0f} s")
        k[3].metric("Worst healthy-pack SOH", f"{sel.SOH_fresh_min:.1f} %")
    if hs is not None:
        st.markdown("### Reported SOH on a healthy pack (true value 100 %)")
        table(hs.round(2))

# ── 6. Pack comparison ────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### The three packs")
    table(pd.DataFrame({"": ["Cell", "Configuration", "Cells", "Cell capacity (Ah)", "Pack capacity (Ah)",
                             "Nominal energy (kWh)", "Initial voltage (V)", "Plant R0, 10–100 % (mΩ)",
                             "BMS nominal R0 (mΩ)", "OCV model in BMS", "Thermal model"],
                        **{p: [PACK_INFO[p][k] for k in ("cell", "config", "cells", "q_cell", "q_pack", "energy",
                                                          "v0", "r0_range", "r0", "ocv", "thermal")]
                           for p in PACKS}}).astype(str))
    c1, c2 = st.columns(2)
    with c1:
        f = make_subplots(specs=[[{"secondary_y": False}]])
        for p in PACKS:
            tb = cell_tables(p)
            if tb:
                f.add_trace(go.Scatter(x=100 * tb[0], y=tb[1], name=p, line=dict(color=PACK_COLOR[p], width=3)))
        f.update_layout(title="Open-circuit voltage of the cell", xaxis_title="SOC (%)", yaxis_title="OCV (V)", height=440)
        chart(f, "ocv_cmp")
    with c2:
        f = go.Figure()
        for p in PACKS:
            tb = cell_tables(p)
            if tb:
                f.add_trace(go.Scatter(x=100 * tb[0], y=1000 * tb[2], name=p, line=dict(color=PACK_COLOR[p], width=3)))
        f.update_layout(title="Series resistance of the cell", xaxis_title="SOC (%)", yaxis_title="R0 (mΩ)",
                        yaxis_range=[0, 100], height=440)
        chart(f, "r0_cmp")
    st.markdown(f"### Pack response · {mode} mode (Simulink)")
    f = make_subplots(rows=1, cols=3, subplot_titles=("Pack voltage (V)", "SOC (%)", "Pack current (A)"),
                      horizontal_spacing=0.07)
    for p in PACKS:
        d = csv(os.path.join(lu.pack_dir(p), f"{mode}_timeseries.csv"))
        for i, col in enumerate(["Voltage_V", "SOC_pct", "Current_A"], 1):
            f.add_trace(go.Scatter(x=d.Time_s, y=d[col], name=p, showlegend=i == 1, legendgroup=p,
                                   line=dict(color=PACK_COLOR[p], width=2.5)), 1, i)
    f.update_xaxes(title_text="Time (s)")
    for ann in f.layout.annotations:
        ann.font = dict(size=17, family=FONT, color=INK)
    f.update_layout(height=470, legend=dict(y=1.12))
    chart(f, "resp_cmp")
    if J:
        st.markdown("### Adaptive BMS across packs (30 seeds)")
        f = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                          subplot_titles=("Matched plant, step ageing: gated SOC RMSE (%)",
                                          "Table plant, step ageing: gated SOC RMSE (%)",
                                          "Detector 2 of 3, τ = 3.5 (%)"))
        for p in PACKS:
            t = J[f"{p}_t3"]
            f.add_trace(go.Bar(name=p, x=MODES, y=[t["step_Gated"][m] for m in MODES], marker_color=PACK_COLOR[p],
                               legendgroup=p, text=[f"{t['step_Gated'][m]:.2f}" for m in MODES], textposition="outside"), 1, 1)
            tps = {r["Mode"]: r for r in J[f"{p}_t7"] if r["Ageing"] == "step"}
            f.add_trace(go.Bar(name=p, x=MODES, y=[tps[m]["Gated_mean"] for m in MODES], marker_color=PACK_COLOR[p],
                               legendgroup=p, showlegend=False,
                               text=[f"{tps[m]['Gated_mean']:.2f}" for m in MODES], textposition="outside"), 1, 2)
            dd = [d for d in J[f"{p}_t6"] if d["Rule"] == "2of3" and d["Tau"] == 3.5][0]
            f.add_trace(go.Bar(name=p, x=["False alarm", "Detection"], y=[dd["FalseAlarm_pct"], dd["Detection_pct"]],
                               marker_color=PACK_COLOR[p], legendgroup=p, showlegend=False,
                               text=[f"{dd['FalseAlarm_pct']:.1f}", f"{dd['Detection_pct']:.1f}"], textposition="outside"), 1, 3)
        f.update_yaxes(type="log", row=1, col=1)
        f.update_yaxes(type="log", row=1, col=2)
        f.update_yaxes(range=[0, 115], row=1, col=3)
        for ann in f.layout.annotations:
            ann.font = dict(size=16, family=FONT, color=INK)
        f.update_layout(barmode="group", height=540, legend=dict(y=1.12))
        chart(f, "bms_cmp")
        rows = []
        for p in PACKS:
            t = J[f"{p}_t3"]
            tps = {r["Mode"]: r for r in J[f"{p}_t7"] if r["Ageing"] == "step"}
            dd = [d for d in J[f"{p}_t6"] if d["Rule"] == "2of3" and d["Tau"] == 3.5][0]
            rows.append({"Pack": p,
                         "Sport, matched: static → gated (%)": f"{t['step_Static']['Sport']:.3f} → {t['step_Gated']['Sport']:.3f}",
                         "Table plant reduction E/N/S (%)": " / ".join(f"{pct(tps[m]['Static'], tps[m]['Gated_mean']):.0f}" for m in MODES),
                         "Sport capacity error (%)": f"{100 * (tps['Sport']['Q_est_Ah'] / tps['Sport']['Q_true_Ah'] - 1):+.1f}",
                         "False alarm / detection (%)": f"{dd['FalseAlarm_pct']:.1f} / {dd['Detection_pct']:.1f}",
                         "Worst healthy SOH (%)": f"{dd['WorstHealthySOH_pct']:.1f}"})
        table(pd.DataFrame(rows))

# ── 7. Imputation ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown(f"### Sensor-dropout reconstruction · {info['name']}")
    im = csv(res(pack, "imputation_can", "imputation_summary.csv"))
    if im is not None:
        b = im[(im.Pattern == "burst") & (im.Channel == "Current_A")].iloc[0]
        k = st.columns(3)
        k[0].metric("Burst current, linear", f"{b.Linear:.2f} A")
        k[1].metric("Burst current, hybrid KNN", f"{b.Hybrid:.2f} A")
        k[2].metric("Improvement", f"{b['Hybrid_vs_Linear_%']:.1f} %")
        table(im.round(3))
    image(res(pack, "paper_figures", "imputation_all_modes.png"), "Pack current with 20 % burst dropouts")
    st.caption("Inputs: pack voltage, pack current and vehicle speed (CAN bus). Hybrid = linear interpolation, "
               "with KNN + temporal context for current gaps of 3 or more samples.")

# ── 8. SOC & SOH models ───────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### Explainable SOC model (random forest, LG 18650HG2)")
    fi = csv(os.path.join(ROOT, "bms", "feature_importance.csv"))
    c1, c2 = st.columns([2, 3])
    with c1:
        if fi is not None:
            f = go.Figure(go.Bar(x=fi["Mean_SHAP"], y=fi["Feature"], orientation="h", marker_color="#2a78d6",
                                 text=fi["Mean_SHAP"].round(3), textposition="outside"))
            f.update_layout(xaxis_title="Mean |SHAP value|", yaxis=dict(autorange="reversed"), height=380,
                            title="Which inputs drive the SOC estimate")
            chart(f, "shap_bar")
    with c2:
        image(os.path.join(ROOT, "bms", "shap_summary.png"), "SHAP summary")
    st.markdown("### Data-driven SOH (NASA cells, leave one cell out)")
    sr = csv(os.path.join(ROOT, "bms", "results", "soh", "soh_results.csv"))
    if sr is not None:
        mm = sr[sr.Test_cell == "MEAN"].set_index("Model")
        k = st.columns(len(mm))
        for i, (name, row) in enumerate(mm.iterrows()):
            stat(k[i], name, f"RMSE {row.RMSE:.2f} %", f"R² {row.R2:.3f}")
    image(os.path.join(ROOT, "bms", "results", "soh", "soh_predictions.png"), "Predicted vs measured SOH on unseen cells")
    st.caption("These two models are trained on public cell datasets and do not depend on the pack selected.")
