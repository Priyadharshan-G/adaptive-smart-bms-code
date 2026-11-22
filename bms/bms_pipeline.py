"""
bms_pipeline.py — Adaptive Smart BMS
1-RC observer → Mahalanobis drift detection → FFRLS recalibration → monotonic SOH
Runs on the three Simulink pack simulation CSVs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── OCV-SOC polynomial (from paper Table II) ──────────────────────────────────
A = [3.14, 2.15, -5.82, 11.20, -10.90, 4.43, 0.08, -0.03]

def ocv(soc):
    s = np.clip(soc, 0.001, 0.999)
    return (A[0] + A[1]*s + A[2]*s**2 + A[3]*s**3 + A[4]*s**4 +
            A[5]*s**5 + A[6]*np.log(s) + A[7]*np.log(1-s))

# ── BMS class ─────────────────────────────────────────────────────────────────
class AdaptiveBMS:
    def __init__(self, R0=0.045, R1=0.025, C1=1500, Q=4.493, lam=0.97, tau=0.60):
        # 1-RC parameters (nominal)
        self.R0 = R0;  self.R1 = R1;  self.C1 = C1
        self.tau1 = R1 * C1
        self.Q  = Q          # cell capacity Ah
        self.lam = lam       # FFRLS forgetting factor
        self.tau_drift = tau # Mahalanobis threshold

        # State
        self.soc  = 1.0
        self.Vp   = 0.0      # polarisation voltage
        self.R0_eff = R0

        # FFRLS
        self.theta = np.array([0.99, -R0, 0.0])
        self.P     = np.eye(3) * 100
        self.y_prev = 0.0

        # Baseline window for Mahalanobis (first 60 s)
        self.baseline = []
        self.mu0  = None
        self.S0inv = None

        # SOH
        self.soh = 1.0
        self.Q_nom  = Q
        self.R0_nom = R0

        # Outputs
        self.log = []

    def step(self, t, V_meas, I, T, dt=1.0):
        eta = 0.995
        # SOC update (Coulomb counting + Luenberger)
        soc_prior = self.soc - eta * I * dt / (3600 * self.Q)
        V_pred = ocv(soc_prior) - I * self.R0_eff - self.Vp
        ev = V_meas - V_pred
        L  = 0.01
        self.soc = float(np.clip(soc_prior + L * ev, 0, 1))

        # Polarisation dynamics
        self.Vp = (np.exp(-dt/self.tau1) * self.Vp +
                   self.R1 * (1 - np.exp(-dt/self.tau1)) * I)

        # ── Mahalanobis drift detection ──────────────────────────────────────
        dT   = T - 25.0
        x_k  = np.array([ev, dT])
        DM   = 0.0
        drift = False
        if t < 60:
            self.baseline.append(x_k)
        else:
            if self.mu0 is None and len(self.baseline) > 5:
                B = np.array(self.baseline)
                self.mu0   = B.mean(axis=0)
                cov        = np.cov(B.T) + np.eye(2)*1e-6
                self.S0inv = np.linalg.inv(cov)
            if self.mu0 is not None:
                diff = x_k - self.mu0
                DM   = float(np.sqrt(diff @ self.S0inv @ diff))
                if DM >= self.tau_drift:
                    drift = True
                    self._ffrls(V_meas, I, dt)

        # ── Monotonic SOH ────────────────────────────────────────────────────
        soh_raw = (0.70 * (self.Q / self.Q_nom) +
                   0.30 * min(self.R0_nom / self.R0_eff, 1.0))
        self.soh = min(self.soh, soh_raw)

        self.log.append(dict(t=t, V=V_meas, I=I, T=T,
                             SOC_true=None, SOC_est=self.soc*100,
                             R0=self.R0_eff, DM=DM, drift=drift,
                             SOH=self.soh*100))

    def _ffrls(self, V_meas, I, dt):
        y_k  = V_meas - ocv(self.soc)
        phi  = np.array([-self.y_prev, I, 0.0])
        denom = self.lam + phi @ self.P @ phi
        K    = self.P @ phi / denom
        err  = y_k - phi @ self.theta
        self.theta += K * err
        self.P      = (1/self.lam) * (np.eye(3) - np.outer(K, phi)) @ self.P
        self.y_prev = y_k

        # Recover R0
        R0_raw       = -self.theta[1]
        if R0_raw > 0:
            self.R0_eff  = 0.75*self.R0_eff + 0.25*R0_raw

# ── Run on all three modes ─────────────────────────────────────────────────────
def run_mode(csv_path, mode_name):
    df  = pd.read_csv(csv_path)
    bms = AdaptiveBMS()
    times = df['Time_s'].values
    for k in range(1, len(times)):
        dt = times[k] - times[k-1]
        if dt <= 0: dt = 1.0
        bms.step(times[k],
                 df['Voltage_V'].iloc[k],
                 df['Current_A'].iloc[k],
                 df['Temp_degC'].iloc[k],
                 dt)
    result = pd.DataFrame(bms.log)
    result['SOC_true'] = df['SOC_pct'].values[:len(result)]
    out = os.path.join(os.path.dirname(csv_path), f'{mode_name}_bms.csv')
    result.to_csv(out, index=False)
    print(f'{mode_name}: SOC RMSE = '
          f'{np.sqrt(((result.SOC_est - result.SOC_true)**2).mean()):.3f} %  '
          f'Drifts = {result.drift.sum()}  '
          f'Final SOH = {result.SOH.iloc[-1]:.1f} %')
    return result

if __name__ == '__main__':
    base = r'C:\ESS'
    results = {}
    for mode in ['Eco', 'Normal', 'Sport']:
        csv = os.path.join(base, f'{mode}_timeseries.csv')
        results[mode] = run_mode(csv, mode)
    print('\nDone. BMS CSV files saved to C:\\ESS\\')