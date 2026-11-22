# shap_lime.py
# SHAP + LIME explainability on LG HG2 dataset

import numpy as np
import pandas as pd
import scipy.io as sio
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── PATHS ─────────────────────────────────────────────────────────────────────
TRAIN_MAT = r"C:\ESS\datasets\lg_hg2_data\Train\TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat"
OUT_DIR   = r"C:\ESS\bms"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. LOAD .mat ──────────────────────────────────────────────────────────────
print("[1/6] Loading LG HG2 .mat file...")
mat = sio.loadmat(TRAIN_MAT)

# Print keys so we can see structure
keys = [k for k in mat.keys() if not k.startswith("__")]
print(f"  Keys in mat: {keys}")

# Standard McMaster format: X = inputs, Y = SOC
X_raw = None
Y_raw = None
for k in keys:
    arr = mat[k]
    if hasattr(arr, 'shape'):
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    if 'X' in k.upper() or 'INPUT' in k.upper():
        X_raw = arr
    if 'Y' in k.upper() or 'SOC' in k.upper() or 'OUTPUT' in k.upper():
        Y_raw = arr

if X_raw is None or Y_raw is None:
    # Fallback: take first two numeric arrays by size
    numeric = [(k, mat[k]) for k in keys if hasattr(mat[k], 'shape') and mat[k].ndim >= 2]
    numeric.sort(key=lambda x: x[1].size, reverse=True)
    X_raw = numeric[0][1]
    Y_raw = numeric[1][1]
    print(f"  Fallback: X from '{numeric[0][0]}', Y from '{numeric[1][0]}'")

# Flatten to 2D
if X_raw.ndim == 3:
    X_raw = X_raw.reshape(-1, X_raw.shape[-1])
if Y_raw.ndim > 1:
    Y_raw = Y_raw.flatten()

print(f"  X shape: {X_raw.shape}, Y shape: {Y_raw.shape}")

# ── 2. FEATURE NAMES ─────────────────────────────────────────────────────────
# McMaster 5-input: Voltage, Current, Temperature, Avg_V, Avg_I
n_features = X_raw.shape[1]
if n_features == 5:
    feature_names = ["Voltage(V)", "Current(A)", "Temperature(°C)", "Avg_Voltage", "Avg_Current"]
else:
    feature_names = [f"Feature_{i}" for i in range(n_features)]

print(f"  Features: {feature_names}")

# ── 3. SUBSAMPLE (keep manageable) ───────────────────────────────────────────
print("[2/6] Fixing orientation and subsampling...")
# McMaster mat files are often stored transposed: (features x samples)
# If X has fewer rows than columns, transpose it
if X_raw.shape[0] < X_raw.shape[1]:
    X_raw = X_raw.T
    print(f"  Transposed X to: {X_raw.shape}")
if Y_raw.ndim == 1 and len(Y_raw) != X_raw.shape[0]:
    Y_raw = Y_raw.reshape(-1, 1).flatten()

# Re-flatten Y
Y_raw = Y_raw.flatten()

# Trim to same length
min_len = min(X_raw.shape[0], len(Y_raw))
X_raw = X_raw[:min_len]
Y_raw = Y_raw[:min_len]
print(f"  X shape: {X_raw.shape}, Y shape: {Y_raw.shape}")

MAX_SAMPLES = 5000
if len(Y_raw) > MAX_SAMPLES:
    idx = np.random.choice(len(Y_raw), MAX_SAMPLES, replace=False)
    idx.sort()
    X = X_raw[idx]
    Y = Y_raw[idx]
else:
    X = X_raw
    Y = Y_raw

print(f"  Using {len(Y)} samples")
# Rebuild feature names after transpose
n_features = X.shape[1]
if n_features == 5:
    feature_names = ["Voltage(V)", "Current(A)", "Temperature(°C)", "Avg_Voltage", "Avg_Current"]
else:
    feature_names = [f"Feature_{i}" for i in range(n_features)]
print(f"  Features: {feature_names}")

# ── 4. TRAIN RANDOM FOREST ───────────────────────────────────────────────────
print("[3/6] Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"  RF RMSE on LG HG2: {rmse*100:.3f}%")

# ── 5. SHAP ───────────────────────────────────────────────────────────────────
print("[4/6] Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test[:200])  # limit for speed

# SHAP summary plot
shap.summary_plot(shap_values, X_test[:200],
                  feature_names=feature_names,
                  show=False)
shap_path = os.path.join(OUT_DIR, "shap_summary.png")
plt.savefig(shap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  SHAP summary saved → {shap_path}")

# SHAP bar plot
shap.summary_plot(shap_values, X_test[:200],
                  feature_names=feature_names,
                  plot_type="bar", show=False)
shap_bar_path = os.path.join(OUT_DIR, "shap_bar.png")
plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  SHAP bar saved → {shap_bar_path}")

# ── 6. LIME ───────────────────────────────────────────────────────────────────
print("[5/6] Computing LIME explanation...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    mode='regression'
)

# Explain one instance from test set
instance_idx = 0
exp = lime_explainer.explain_instance(
    X_test[instance_idx],
    rf.predict,
    num_features=n_features
)

# Save LIME plot
fig = exp.as_pyplot_figure()
lime_path = os.path.join(OUT_DIR, "lime_explanation.png")
fig.savefig(lime_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  LIME explanation saved → {lime_path}")

# ── 7. FEATURE IMPORTANCE TABLE ──────────────────────────────────────────────
print("[6/6] Saving feature importance...")
importances = rf.feature_importances_
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': importances,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

fi_path = os.path.join(OUT_DIR, "feature_importance.csv")
fi_df.to_csv(fi_path, index=False)
print(f"  Feature importance saved → {fi_path}")
print(fi_df.to_string(index=False))

print("\n\nDone! Outputs in C:\\ESS\\bms\\")
print("  shap_summary.png")
print("  shap_bar.png")
print("  lime_explanation.png")
print("  feature_importance.csv")