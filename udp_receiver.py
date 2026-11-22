"""
udp_receiver.py - Objective 3: live MATLAB -> Python UDP bridge
Listens on 127.0.0.1:5005 for packets from udp_stream.m:
  7 doubles (little-endian): [seq, t, V_pack, I_pack, T, SOC_true, send_time]
Runs the adaptive BMS live on every packet, logs packet loss, network latency
and BMS processing time, and writes C:/ESS/bms/results/live/live_stream.csv
for the dashboard's Live UDP tab.  Start this BEFORE running udp_stream.m.
"""
import os
import socket
import struct
import time
import numpy as np
import pandas as pd
import bms_pipeline_v2 as bp

HOST, PORT = "127.0.0.1", 5005
FMT = "<7d"
IDLE_TIMEOUT = 30          # s without packets (after the first) -> stop
LIVE = os.path.join(bp.BASE, "bms", "results", "live")
LOG = os.path.join(LIVE, "live_stream.csv")
COLS = ["seq", "t", "V_cell", "I_cell", "T", "SOC_true", "SOC_est", "R0_est_mOhm",
        "Q_est_Ah", "DM", "drift", "SOH", "latency_ms", "proc_ms", "lost_total"]

if __name__ == "__main__":
    os.makedirs(LIVE, exist_ok=True)
    print("Identifying OCV curve from fresh simulation data...")
    ocv = bp.identify_ocv({m: bp.load_mode(m) for m in bp.MODES})
    with open(LOG, "w") as f:
        f.write(",".join(COLS) + "\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(1.0)
    print(f"Listening on {HOST}:{PORT} ... start udp_stream.m in MATLAB (Ctrl+C to stop)")

    bms, last_seq, lost, prev = None, 0, 0, None
    rows, last_rx = [], None
    try:
        while True:
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                if last_rx and time.time() - last_rx > IDLE_TIMEOUT:
                    print("No packets for 30 s - stopping.")
                    break
                continue
            rx = time.time()
            last_rx = rx
            if len(data) != struct.calcsize(FMT):
                continue
            seq, t, Vp, Ip, T, soc, tx = struct.unpack(FMT, data)
            if seq < 0:
                print("End-of-stream packet received.")
                break
            seq = int(seq)
            if bms is None or seq <= last_seq:            # first packet or a new stream
                if bms is not None:
                    print("New stream detected - resetting BMS.")
                    with open(LOG, "w") as f:
                        f.write(",".join(COLS) + "\n")
                    rows, lost = [], 0
                bms = bp.AdaptiveBMS(ocv, adaptive=True)
                bms.init_soc(Vp / bp.N_S + Ip / bp.N_P * bp.R0_CELL)
                last_seq, prev = seq, (t, Ip / bp.N_P)
                print(f"Stream started (seq {seq}, t = {t:.0f} s)")
                continue
            lost += max(0, seq - last_seq - 1)
            last_seq = seq
            V, I = Vp / bp.N_S, Ip / bp.N_P
            t0 = time.perf_counter()
            bms.step(t, V, I, prev[1], T, t - prev[0])
            proc = (time.perf_counter() - t0) * 1000
            prev = (t, I)
            s = bms.log[-1]
            row = [seq, t, V, I, T, soc, s["SOC_est"], s["R0_est_mOhm"], s["Q_est_Ah"], s["DM"],
                   s["drift"], s["SOH"], (rx - tx) * 1000, proc, lost]
            rows.append(row)
            with open(LOG, "a") as f:
                f.write(",".join(f"{x:.6g}" for x in row) + "\n")
            flag = "  DRIFT" if s["drift"] else ""
            print(f"seq {seq:4d}  t={t:6.0f}s  SOC {s['SOC_est']:6.2f}% (true {soc:6.2f})  "
                  f"R0 {s['R0_est_mOhm']:5.2f} mOhm  SOH {s['SOH']:5.1f}%  lat {row[12]:5.2f} ms{flag}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()

    if rows:
        df = pd.DataFrame(rows, columns=COLS)
        n = len(df) + 1 + lost                         # +1 = initialising packet
        err = df.SOC_est - df.SOC_true
        drift_t = df.loc[df.drift == 1, "t"]
        summary = pd.DataFrame([dict(
            Packets_received=len(df) + 1, Packets_lost=lost, Loss_pct=round(100 * lost / n, 2),
            Latency_mean_ms=round(df.latency_ms.mean(), 3), Latency_p95_ms=round(df.latency_ms.quantile(.95), 3),
            Latency_max_ms=round(df.latency_ms.max(), 3), BMS_proc_mean_ms=round(df.proc_ms.mean(), 3),
            BMS_proc_max_ms=round(df.proc_ms.max(), 3), SOC_RMSE_pct=round(float(np.sqrt((err ** 2).mean())), 3),
            First_drift_s=(float(drift_t.iloc[0]) if len(drift_t) else None),
            Final_R0_mOhm=round(df.R0_est_mOhm.iloc[-1], 2), Final_SOH_pct=round(df.SOH.iloc[-1], 1))])
        summary.to_csv(os.path.join(LIVE, "udp_summary.csv"), index=False)
        print("\n=== UDP bridge summary ===")
        print(summary.T.to_string(header=False))