# ================================================================
# BOOTSTRAP ANALYSIS NOTEBOOK (Single-cell version for GitHub)
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set(style="whitegrid", context="notebook")

# --------------------------- PATHS ------------------------------


DATA_PATH = "data/chf_database.csv"
IMPUTER_PATH = "preprocessing/imputer.pkl"
MODEL_PATH   = "trained_models/xgboost_final.pkl"


FEATURE_COLUMNS = ["ks", "P", "Tsat", "Ra", "kf"]
RANDOM_STATE = 42
N_BOOT = 1000

# --------------------------- LOAD DATA --------------------------
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLUMNS].copy()
y = df["CHF"].astype(float).copy()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

# --------------------- LOAD IMPUTER & MODEL ---------------------
imputer = load(IMPUTER_PATH)
model   = load(MODEL_PATH)

expected_cols = list(getattr(imputer, "feature_names_in_", FEATURE_COLUMNS))
X_test_aligned = X_test[expected_cols]

# ---------------------- BASELINE METRICS -------------------------
X_test_imp   = imputer.transform(X_test_aligned)
y_pred_final = model.predict(X_test_imp)

rmse_base = float(np.sqrt(mean_squared_error(y_test, y_pred_final)))
mae_base  = mean_absolute_error(y_test, y_pred_final)
r2_base   = r2_score(y_test, y_pred_final)

print("=== Baseline Test Metrics ===")
print(f"RMSE: {rmse_base:.3f}")
print(f"MAE : {mae_base:.3f}")
print(f"R²  : {r2_base:.4f}")

# ----------------------- BOOTSTRAP LOOP -------------------------
rng = np.random.default_rng(RANDOM_STATE)
y_test_arr = np.asarray(y_test)
y_pred_arr = np.asarray(y_pred_final)
n_test = len(y_test_arr)

rmse_boot, mae_boot, r2_boot = [], [], []

for _ in range(N_BOOT):
    idx = rng.integers(0, n_test, size=n_test)
    yt, yp = y_test_arr[idx], y_pred_arr[idx]

    rmse_boot.append(float(np.sqrt(mean_squared_error(yt, yp))))
    mae_boot.append(mean_absolute_error(yt, yp))
    r2_boot.append(r2_score(yt, yp))

rmse_boot, mae_boot, r2_boot = map(np.array, [rmse_boot, mae_boot, r2_boot])

rmse_ci = np.percentile(rmse_boot, [2.5, 97.5])
mae_ci  = np.percentile(mae_boot,  [2.5, 97.5])
r2_ci   = np.percentile(r2_boot,   [2.5, 97.5])

print("\n=== Bootstrap 95% Confidence Intervals ===")
print(f"RMSE 95% CI: [{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]")
print(f"MAE  95% CI: [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")
print(f"R²   95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")

# ---------------------- SAVE FIGURES -----------------------------
def plot_boot(values, title, xlabel, filename):
    plt.figure(figsize=(6,4))
    sns.histplot(values, bins=25, kde=True)
    low, high = np.percentile(values, [2.5, 97.5])
    plt.axvline(low,  color="red", linestyle="--", label=f"2.5%={low:.3f}")
    plt.axvline(high, color="red", linestyle="--", label=f"97.5%={high:.3f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

# plot_boot(rmse_boot, "Bootstrap RMSE Distribution", "RMSE", "bootstrap_rmse_xgb.jpg")
# plot_boot(mae_boot,  "Bootstrap MAE Distribution",  "MAE",  "bootstrap_mae_xgb.jpg")
# plot_boot(r2_boot,   "Bootstrap R² Distribution",   "R²",   "bootstrap_r2_xgb.jpg")

print("\nBootstrap analysis completed successfully.")
