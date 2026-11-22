"""
live_udp.py - live MATLAB -> Python UDP link used by dashboard_v2.py

Packet format (unchanged from udp_stream.m / udp_receiver.py):
    7 little-endian doubles  [seq, t, V_pack, I_pack, T, SOC_true, send_time]
    seq < 0 marks the end of a stream.

Pieces
  load_bms_module(pack)   bms_pipeline_v2 configured for P45B / NCR / LFP (own copy per pack)
  make_stream(pack, ...)  noisy pack-level stream (fresh or aged), same as export_streams.py
  UdpBmsListener          background thread: receives packets, runs the adaptive BMS on each one
  replay(...)             Python sender with the same packet format, for demos without MATLAB

Command line
  python live_udp.py export                         # write packs/NCR|LFP/<Mode>_<cond>_stream.csv for udp_stream_pack.m
  python live_udp.py send --pack NCR --mode Sport --aged --speedup 20
"""
import argparse
import importlib.util
import os
import socket
import struct
import threading
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FMT = "<7d"
PKT_SIZE = struct.calcsize(FMT)
PACKS = ["P45B", "NCR", "LFP"]
MODES = ["Eco", "Normal", "Sport"]
OCV_FIT = {"P45B": "poly", "NCR": "poly", "LFP": "pchip"}   # as used in the paper
COLS = ["seq", "t", "V_pack", "I_pack", "V_cell", "I_cell", "T", "SOC_true", "SOC_est", "R0_est_mOhm",
        "Q_est_Ah", "DM", "drift", "SOH", "latency_ms", "proc_ms", "lost_total"]


def pack_dir(pack):
    return ROOT if pack == "P45B" else os.path.join(ROOT, "packs", pack)


_modules = {}
_lock = threading.Lock()


def load_bms_module(pack):
    """A separate copy of bms_pipeline_v2 configured for one pack (its settings are module globals)."""
    with _lock:
        if pack not in _modules:
            saved = {k: os.environ.get(k) for k in ("ESS_PACK", "ESS_BASE", "ESS_OCV")}
            os.environ.update(ESS_PACK=pack, ESS_BASE=pack_dir(pack), ESS_OCV=OCV_FIT[pack])
            try:
                spec = importlib.util.spec_from_file_location(f"bp_{pack}", os.path.join(ROOT, "bms_pipeline_v2.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.OCV = mod.identify_ocv({m: mod.load_mode(m) for m in mod.MODES})
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            _modules[pack] = mod
        return _modules[pack]


def make_stream(pack, mode, aged, seed=42):
    """Pack-level noisy stream, identical in construction to export_streams.py."""
    bp = load_bms_module(pack)
    d = bp.load_mode(mode)
    if aged:
        d = bp.make_aged(d, bp.OCV)
    dn = bp.add_noise(d, np.random.default_rng(seed))
    return pd.DataFrame(dict(Time_s=dn["t"], Voltage_V=dn["V_meas"] * bp.N_S, Current_A=dn["I_meas"] * bp.N_P,
                             Temp_degC=dn["T"], SOC_pct=dn["soc"] * 100))


class UdpBmsListener:
    """Receives packets on host:port in a background thread and runs the adaptive BMS on each one."""

    def __init__(self, pack="P45B", host="127.0.0.1", port=5005, log_dir=None):
        self.pack, self.host, self.port = pack, host, port
        self.log_dir = log_dir or os.path.join(pack_dir(pack), "bms", "results", "live_dashboard")
        self.rows, self.status, self.error = [], "stopped", None
        self.lost, self.packets = 0, 0
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    # ---- control -------------------------------------------------------------------------
    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        for _ in range(50):                      # wait until bound (or failed)
            if self.status != "starting" and self.status != "stopped":
                break
            time.sleep(0.02)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.status = "stopped"

    def clear(self):
        with self._lock:
            self.rows, self.lost, self.packets = [], 0, 0

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def frame(self):
        with self._lock:
            return pd.DataFrame(self.rows, columns=COLS)

    # ---- worker --------------------------------------------------------------------------
    def _run(self):
        self.status = "starting"
        try:
            bp = load_bms_module(self.pack)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.host, self.port))
            sock.settimeout(0.5)
        except OSError as e:
            self.status, self.error = "error", f"Cannot listen on {self.host}:{self.port} ({e}). " \
                                              "Close udp_receiver.py or any other program using the port."
            return
        self.status = "listening"
        bms, last_seq, prev = None, 0, None
        try:
            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(1024)
                except socket.timeout:
                    continue
                rx = time.time()
                if len(data) != PKT_SIZE:
                    continue
                seq, t, Vp, Ip, T, soc, tx = struct.unpack(FMT, data)
                if seq < 0:
                    self.status = "finished"
                    self._save()
                    bms = None
                    continue
                seq = int(seq)
                if bms is None or seq <= last_seq:          # first packet, or a new stream started
                    self.clear()
                    bms = bp.AdaptiveBMS(bp.OCV, adaptive=True)
                    bms.init_soc(Vp / bp.N_S + Ip / bp.N_P * bp.R0_CELL)
                    last_seq, prev = seq, (t, Ip / bp.N_P)
                    self.status = "streaming"
                    with self._lock:
                        self.packets = 1
                    continue
                V, I = Vp / bp.N_S, Ip / bp.N_P
                t0 = time.perf_counter()
                bms.step(t, V, I, prev[1], T, t - prev[0])
                proc = (time.perf_counter() - t0) * 1000
                prev = (t, I)
                s = bms.log[-1]
                with self._lock:
                    self.lost += max(0, seq - last_seq - 1)
                    self.packets += 1
                    self.rows.append([seq, t, Vp, Ip, V, I, T, soc, s["SOC_est"], s["R0_est_mOhm"], s["Q_est_Ah"],
                                      s["DM"], s["drift"], s["SOH"], (rx - tx) * 1000, proc, self.lost])
                last_seq = seq
                self.status = "streaming"
        finally:
            sock.close()
            if self.status != "finished":
                self.status = "stopped"

    def _save(self):
        df = self.frame()
        if df.empty:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        df.to_csv(os.path.join(self.log_dir, "live_stream.csv"), index=False)
        err = df.SOC_est - df.SOC_true
        n = self.packets + self.lost
        drift_t = df.loc[df.drift == 1, "t"]
        pd.DataFrame([dict(Pack=self.pack, Packets_received=self.packets, Packets_lost=self.lost,
                           Loss_pct=round(100 * self.lost / max(n, 1), 2),
                           Latency_mean_ms=round(df.latency_ms.mean(), 3),
                           Latency_p95_ms=round(df.latency_ms.quantile(.95), 3),
                           BMS_proc_mean_ms=round(df.proc_ms.mean(), 3),
                           SOC_RMSE_pct=round(float(np.sqrt((err ** 2).mean())), 3),
                           First_drift_s=(float(drift_t.iloc[0]) if len(drift_t) else None),
                           Final_R0_mOhm=round(df.R0_est_mOhm.iloc[-1], 2),
                           Final_SOH_pct=round(df.SOH.iloc[-1], 1))]
                     ).to_csv(os.path.join(self.log_dir, "udp_summary.csv"), index=False)


def replay(stream, host="127.0.0.1", port=5005, speedup=20.0, stop_event=None, progress=None):
    """Send a stream DataFrame with the udp_stream.m packet format."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t = stream["Time_s"].values
    dt = np.r_[0.0, np.diff(t)]
    try:
        for k in range(len(stream)):
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(dt[k] / speedup)
            r = stream.iloc[k]
            pkt = struct.pack(FMT, k + 1, r.Time_s, r.Voltage_V, r.Current_A, r.Temp_degC, r.SOC_pct, time.time())
            sock.sendto(pkt, (host, port))
            if progress:
                progress(k + 1, len(stream))
        sock.sendto(struct.pack(FMT, -1, 0, 0, 0, 0, 0, time.time()), (host, port))
    finally:
        sock.close()


def export_all():
    # P45B keeps the original files from export_streams.py at the repo root (used by udp_stream.m)
    for pack in ("NCR", "LFP"):
        for mode in MODES:
            for aged in (False, True):
                s = make_stream(pack, mode, aged)
                p = os.path.join(pack_dir(pack), f"{mode}_{'aged' if aged else 'fresh'}_stream.csv")
                s.to_csv(p, index=False)
                print("saved", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export", help="write stream CSVs for udp_stream_pack.m")
    s = sub.add_parser("send", help="replay a stream over UDP (no MATLAB needed)")
    s.add_argument("--pack", choices=PACKS, default="P45B")
    s.add_argument("--mode", choices=MODES, default="Sport")
    s.add_argument("--aged", action="store_true")
    s.add_argument("--speedup", type=float, default=20)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5005)
    a = ap.parse_args()
    if a.cmd == "export":
        export_all()
    else:
        st = make_stream(a.pack, a.mode, a.aged)
        print(f"Streaming {a.pack} {a.mode} {'aged' if a.aged else 'fresh'} ({len(st)} packets) to {a.host}:{a.port}")
        replay(st, a.host, a.port, a.speedup, progress=lambda k, n: print(f"  sent {k}/{n}") if k % 30 == 0 else None)
        print("done")
