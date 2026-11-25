"""
Demo: Full paper generation pipeline for TPU Wind Pressure + ML study.
Target journal: JWEIA
Topic: ML-based prediction of peak wind pressure coefficients on low-rise building
       roofs from short-duration pressure records (synthetic TPU-style data).

Prediction task (leakage-free by construction): every record is split into an
observation window (the first N_OBS samples) and a disjoint prediction window
(the remaining samples).  All signal features are computed from the observation
window only; the target is the peak suction (minimum Cp) over the prediction
window, i.e. over the part of the record the models never see.

This script executes Steps 2-5 of the PaperFactory pipeline:
  Step 2: Research Design
  Step 3: Code Execution (synthetic TPU-style data + ML training)
  Step 4: Result Analysis
  Step 5: Paper Assembly + Quality Check
"""

import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.figure_utils import get_colors, get_figsize, save_figure, setup_style
from utils.quality_checker import check_paper
from utils.word_generator import generate_word

# ── Setup ────────────────────────────────────────────────────────────────────
setup_style()
colors = get_colors()
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("PaperFactory Demo: TPU Wind Pressure + ML Prediction")
print("=" * 60)

# ── Record layout and prediction-task definition ─────────────────────────────
N_TAPS = 144
N_TIME = 4096  # samples per record
DT = 0.001  # s (1,000 Hz)
FS = 1 / DT
N_OBS = 512  # observation window: the first 12.5% of each record (features)
N_PRED = N_TIME - N_OBS  # prediction window: the remaining samples (target)
GUARD = 256  # samples skipped before the target window in the guard-gap control
BURN_IN = 256  # filter start-up samples discarded before each record
WELCH_NPERSEG = 128
N_FOLDS = 10
WIND_ANGLES = np.arange(0, 360, 10)
N_DIR = len(WIND_ANGLES)
N_SAMPLES = N_TAPS * N_DIR
OBS_PCT = 100 * N_OBS / N_TIME
PRED_PCT = 100 * N_PRED / N_TIME
T_OBS = N_OBS * DT
T_PRED = N_PRED * DT

# Stochastic model parameters (quoted in the manuscript text)
RUN_FACTOR = (0.8, 1.2)  # uniform range of the run-level turbulence factor
TAP_SIGMA = 0.1  # log-std of the tap-level RMS factor
EVENT_RATE = 10.0  # mean number of suction events per record at separation intensity 1
EVENT_SIGMA = 0.5  # log-std of the record-level intermittency factor
PULSE_WIDTH = (8, 40)  # samples (std of the Gaussian pulse shape)
PULSE_AMP = (1.0, 3.0)  # pulse amplitude in units of the local RMS
CUTOFF = (0.40, 0.22)  # Butterworth cutoff = CUTOFF[0] - CUTOFF[1] * min(s, 1) (x Nyquist)

tap_x = np.linspace(0, 1, 12)
tap_y = np.linspace(0, 1, 12)
tap_xx, tap_yy = np.meshgrid(tap_x, tap_y)
tap_xx, tap_yy = tap_xx.flatten(), tap_yy.flatten()
TAP_CORNER, TAP_CENTER = 0, 66  # (x/B, y/D) = (0, 0) and (0.55, 0.45)

FEATURE_GROUPS = {
    "statistical": [
        "mean_cp",
        "rms_cp",
        "min_cp",
        "max_cp",
        "skewness",
        "kurtosis",
        "peak_factor",
    ],
    "spectral": ["dominant_freq", "spectral_bandwidth", "spectral_centroid"],
    "spatial-directional": ["tap_x", "tap_y", "wind_angle_sin", "wind_angle_cos"],
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: RESEARCH DESIGN (embedded)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Step 2] Research Design")
print("-" * 40)

PAPER_TITLE = (
    "Machine Learning Prediction of Peak Wind Pressure Coefficients "
    "on Low-Rise Building Roofs from Short-Duration TPU-Style Pressure Records"
)

RESEARCH_DESIGN = {
    "title": PAPER_TITLE,
    "hypothesis": (
        "ML regressors trained on statistical, spectral, and spatial-directional features of a "
        f"short observation window ({OBS_PCT:.1f}% of a record) predict the peak suction Cp over "
        "the unobserved remainder of the record more accurately than training-free estimates "
        "(the observed short-record peak and a Gaussian peak-factor extrapolation)."
    ),
    "methodology": "Supervised regression, 10-fold cross-validation grouped by wind direction",
    "models": ["Random Forest", "Gradient Boosting (GBR)", "Deep Neural Network"],
    "target": f"Peak suction Cp (minimum Cp) over the last {N_PRED} samples of each record",
    "features": FEATURE_GROUPS,
    "planned_figures": 8,
    "planned_tables": 4,
}

print(f"  Title: {PAPER_TITLE}")
print(f"  Models: {', '.join(RESEARCH_DESIGN['models'])}")
print(
    f"  Planned: {RESEARCH_DESIGN['planned_figures']} figures, {RESEARCH_DESIGN['planned_tables']} tables"
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Step 3] Code Execution")
print("-" * 40)


# ── Synthetic TPU-style data ─────────────────────────────────────────────────
def flapping_profile(d):
    """RMS profile behind a windward edge; peaks at d = 0.1 (shear-layer flapping)."""
    u = (d + 0.03) / 0.13
    return u * np.exp(1 - u)


def aero_fields(angle_deg):
    """Mean Cp and separation intensity of every tap for one wind direction.

    The approach flow travels in the direction theta measured from the x-axis, so
    the windward edge is x/B = 0 for theta = 0 and y/D = 0 for theta = 90 degrees.
    """
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    dx = tap_xx if c >= 0 else 1 - tap_xx  # distance from the windward edge across x
    dy = tap_yy if s >= 0 else 1 - tap_yy
    lead = abs(c) * np.exp(-dx / 0.12) + abs(s) * np.exp(-dy / 0.12)  # separation bubble
    cone = abs(np.sin(2 * th)) * np.exp(-(dx + dy) / 0.2)  # corner (conical) vortices
    side = s**2 * np.exp(-np.minimum(tap_xx, 1 - tap_xx) / 0.08)
    side = side + c**2 * np.exp(-np.minimum(tap_yy, 1 - tap_yy) / 0.08)  # side edges
    mean_cp = -0.3 - 0.9 * lead - 1.1 * cone - 0.25 * side
    flap = abs(c) * flapping_profile(dx) + abs(s) * flapping_profile(dy)
    sep = np.clip(0.7 * flap + 0.8 * cone + 0.2 * side, 0, 1.5)
    return mean_cp, sep


def generate_cp_timeseries(angle_deg):
    """One synthetic wind tunnel run: Cp records of all taps for one wind direction."""
    mean_cp, sep = aero_fields(angle_deg)
    run_factor = np.random.uniform(*RUN_FACTOR)  # run-to-run variation of inflow turbulence
    n = np.arange(N_TIME)
    impulse = np.zeros(2048)
    impulse[0] = 1.0
    Cp = np.zeros((N_TAPS, N_TIME))
    for i in range(N_TAPS):
        rms = (0.07 + 0.28 * sep[i]) * run_factor * np.random.lognormal(0.0, TAP_SIGMA)
        b, a = signal.butter(4, CUTOFF[0] - CUTOFF[1] * min(sep[i], 1.0))  # narrower if separated
        gain = np.sqrt(np.sum(signal.lfilter(b, a, impulse) ** 2))  # theoretical std
        gauss = signal.lfilter(b, a, np.random.randn(N_TIME + BURN_IN))[BURN_IN:] / gain
        fluct = gauss * rms
        rate = EVENT_RATE * sep[i] * np.random.lognormal(0.0, EVENT_SIGMA)  # suction events
        for _ in range(np.random.poisson(rate)):
            loc = np.random.randint(0, N_TIME)
            width = np.random.uniform(*PULSE_WIDTH)
            amp = rms * np.random.uniform(*PULSE_AMP)
            fluct -= amp * np.exp(-0.5 * ((n - loc) / width) ** 2)
        Cp[i, :] = mean_cp[i] + fluct
    return Cp


def extract_features(cp_window):
    """Statistical and spectral features of each tap's observation window."""
    rows = []
    for cp in cp_window:
        mean, rms, cp_min = np.mean(cp), np.std(cp), np.min(cp)
        f, psd = signal.welch(cp, fs=FS, nperseg=WELCH_NPERSEG)
        power = max(np.sum(psd), 1e-12)
        centroid = np.sum(f * psd) / power
        rows.append(
            {
                "mean_cp": mean,
                "rms_cp": rms,
                "min_cp": cp_min,
                "max_cp": np.max(cp),
                "skewness": stats.skew(cp),
                "kurtosis": stats.kurtosis(cp),
                "peak_factor": (mean - cp_min) / max(rms, 1e-8),
                "dominant_freq": f[np.argmax(psd)],
                "spectral_bandwidth": np.sqrt(np.sum((f - centroid) ** 2 * psd) / power),
                "spectral_centroid": centroid,
            }
        )
    return pd.DataFrame(rows)


def gaussian_peak_estimate(cp_window, duration):
    """Expected minimum of a stationary Gaussian process over `duration` seconds."""
    mean = cp_window.mean(axis=1)
    rms = cp_window.std(axis=1)
    fluct = cp_window - mean[:, None]
    upcrossings = np.sum((fluct[:, :-1] < 0) & (fluct[:, 1:] >= 0), axis=1)
    nu = upcrossings / (cp_window.shape[1] * DT)
    root = np.sqrt(2 * np.log(np.maximum(nu * duration, np.e)))
    g = root + 0.5772 / root
    return mean - g * rms, g


print(f"  Generating synthetic TPU-style records ({N_DIR} wind directions)...")
frames, targets, targets_guard, groups = [], [], [], []
frames_full, targets_full = [], []  # leaky reference formulation (control only)
obs_peak, gauss_estimate, gauss_g = [], [], []
mean_cp_full, cp_example = {}, None
for k, angle in enumerate(WIND_ANGLES):
    Cp_angle = generate_cp_timeseries(angle)
    window = Cp_angle[:, :N_OBS]  # the only samples the features may use
    assert window.shape[1] == N_OBS
    feats = extract_features(window)
    feats["tap_x"] = tap_xx
    feats["tap_y"] = tap_yy
    feats["wind_angle_sin"] = np.sin(np.radians(angle))
    feats["wind_angle_cos"] = np.cos(np.radians(angle))
    frames.append(feats)
    targets.append(Cp_angle[:, N_OBS:].min(axis=1))  # peak over the disjoint prediction window
    targets_guard.append(Cp_angle[:, N_OBS + GUARD :].min(axis=1))
    frames_full.append(extract_features(Cp_angle))
    targets_full.append(Cp_angle.min(axis=1))
    groups.append(np.full(N_TAPS, k))
    obs_peak.append(window.min(axis=1))
    est, g = gaussian_peak_estimate(window, T_PRED)
    gauss_estimate.append(est)
    gauss_g.append(g)
    mean_cp_full[int(angle)] = Cp_angle.mean(axis=1)
    if angle == 0:
        cp_example = Cp_angle[[TAP_CORNER, TAP_CENTER]].copy()

X = pd.concat(frames, ignore_index=True)
FEATURES = list(X.columns)
assert sorted(FEATURES) == sorted(sum(FEATURE_GROUPS.values(), []))
X_arr = X.to_numpy()
y = np.concatenate(targets)
y_guard = np.concatenate(targets_guard)
y_full = np.concatenate(targets_full)
X_full = pd.concat(frames_full, ignore_index=True)
X_full = pd.concat([X_full, X[FEATURE_GROUPS["spatial-directional"]]], axis=1)[FEATURES]
groups = np.concatenate(groups)
obs_peak = np.concatenate(obs_peak)
gauss_estimate = np.concatenate(gauss_estimate)
gauss_g = np.concatenate(gauss_g)
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(
    f"  Observation window: {N_OBS} samples ({OBS_PCT:.1f}%), "
    f"prediction window: {N_PRED} samples ({PRED_PCT:.1f}%)"
)
print(f"  Target (peak Cp): min {y.min():.3f}, median {np.median(y):.3f}, max {y.max():.3f}")

# 10-fold CV grouped by wind direction: fold k holds out every tenth direction
# (theta = 10k, 10k + 100, ... degrees), so all taps of a wind tunnel run share a fold.
fold_of = groups % N_FOLDS
cv_splits = [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(N_FOLDS)]
fold_angles = [WIND_ANGLES[np.arange(N_DIR) % N_FOLDS == k].tolist() for k in range(N_FOLDS)]


def make_model(estimator):
    """Standardization (fitted on the training folds only) followed by the regressor."""
    return make_pipeline(StandardScaler(), clone(estimator))


def evaluate(target, pred):
    return {
        "R2": r2_score(target, pred),
        "RMSE": float(np.sqrt(mean_squared_error(target, pred))),
        "MAE": mean_absolute_error(target, pred),
    }


def cv_predict(estimator, X_in, target):
    """Out-of-fold predictions under the grouped 10-fold scheme."""
    pred = np.zeros_like(target)
    for train_idx, test_idx in cv_splits:
        model = make_model(estimator).fit(X_in[train_idx], target[train_idx])
        pred[test_idx] = model.predict(X_in[test_idx])
    return pred


# ── ML Training ──────────────────────────────────────────────────────────────
print("  Training ML models with 10-fold CV (grouped by wind direction)...")

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, max_features=1 / 3, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
    ),
    "Deep Neural Network": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True
    ),
}
ABBR = {"Random Forest": "RF", "Gradient Boosting": "GBR", "Deep Neural Network": "DNN"}
rf_est = models["Random Forest"]
dnn_est = models["Deep Neural Network"]
rf_split_features = max(1, int(rf_est.max_features * len(FEATURES)))
dnn_layers = [len(FEATURES), *dnn_est.hidden_layer_sizes, 1]
dnn_params = sum(a * b + b for a, b in zip(dnn_layers[:-1], dnn_layers[1:], strict=True))

cv_results = {name: {"R2": [], "RMSE": [], "MAE": []} for name in models}
y_pred_all = {name: np.zeros_like(y) for name in models}
perm_folds, dnn_epochs = [], []
for train_idx, test_idx in cv_splits:
    for name, estimator in models.items():
        model = make_model(estimator).fit(X_arr[train_idx], y[train_idx])
        pred = model.predict(X_arr[test_idx])
        y_pred_all[name][test_idx] = pred
        for metric, value in evaluate(y[test_idx], pred).items():
            cv_results[name][metric].append(value)
        if name == "Random Forest":
            perm = permutation_importance(
                model, X_arr[test_idx], y[test_idx], n_repeats=5, random_state=42, scoring="r2"
            )
            perm_folds.append(perm.importances_mean)
        elif name == "Deep Neural Network":
            dnn_epochs.append(model[-1].n_iter_)
perm_imp = np.mean(perm_folds, axis=0)

# Overall (pooled out-of-fold) metrics
overall = {name: evaluate(y, y_pred_all[name]) for name in models}
for name in models:
    print(
        f"  {name}: R²={overall[name]['R2']:.4f}, RMSE={overall[name]['RMSE']:.4f}, "
        f"MAE={overall[name]['MAE']:.4f}"
    )

# Training-free reference estimators (same samples, no fitting)
BASELINES = {
    "Observed short-record peak": obs_peak,
    "Gaussian peak-factor estimate": gauss_estimate,
}
baseline_metrics = {name: evaluate(y, pred) for name, pred in BASELINES.items()}
for name, m in baseline_metrics.items():
    print(f"  {name}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

# Per-direction errors, bootstrap confidence intervals and paired tests
all_preds = {**y_pred_all, **BASELINES}
mse_dir = {
    name: np.array([np.mean((y[groups == k] - pred[groups == k]) ** 2) for k in range(N_DIR)])
    for name, pred in all_preds.items()
}
rmse_dir = {name: np.sqrt(v) for name, v in mse_dir.items()}
boot = np.random.default_rng(42).integers(0, N_DIR, size=(2000, N_DIR))
rmse_boot = {name: np.sqrt(v[boot].mean(axis=1)) for name, v in mse_dir.items()}
rmse_ci = {name: np.percentile(v, [2.5, 97.5]) for name, v in rmse_boot.items()}

# ── Leakage controls ─────────────────────────────────────────────────────────
print("  Running leakage controls...")
# (a) how often the observation-window peak is the more extreme one: with an
#     overlapping (full-record) target, min_cp would equal the target in these records
share_window_extreme = float(np.mean(obs_peak <= y))
r_min_target = float(stats.pearsonr(X["min_cp"], y)[0])
# (b) guard gap: target measured only after a further GUARD samples
pred_guard = cv_predict(rf_est, X_arr, y_guard)
guard_metrics = evaluate(y_guard, pred_guard)
# (c) negative control: permuted targets must not be predictable
y_perm = np.random.default_rng(42).permutation(y)
perm_control_r2 = r2_score(y_perm, cv_predict(rf_est, X_arr, y_perm))
# (d) leaky reference formulation: features and target from the same full record
leaky_metrics = evaluate(y_full, cv_predict(rf_est, X_full.to_numpy(), y_full))
# (e) ablation: spatial-directional inputs only vs. short-record signal features only
spatial_cols = [FEATURES.index(f) for f in FEATURE_GROUPS["spatial-directional"]]
signal_cols = [i for i in range(len(FEATURES)) if i not in spatial_cols]
ablation = {
    "spatial": evaluate(y, cv_predict(rf_est, X_arr[:, spatial_cols], y)),
    "signal": evaluate(y, cv_predict(rf_est, X_arr[:, signal_cols], y)),
    "all": overall["Random Forest"],
}
print(
    f"    window peak more extreme than target: {share_window_extreme:.1%}; "
    f"r(min_cp, target) = {r_min_target:.3f}"
)
print(
    f"    RF guard-gap R² = {guard_metrics['R2']:.4f} (vs {overall['Random Forest']['R2']:.4f}); "
    f"permuted-target R² = {perm_control_r2:.4f}; leaky formulation R² = {leaky_metrics['R2']:.4f}"
)
print(
    f"    ablation R²: spatial-only {ablation['spatial']['R2']:.4f}, "
    f"signal-only {ablation['signal']['R2']:.4f}, all {ablation['all']['R2']:.4f}"
)

# Feature importance (RF trained on all samples) and prediction time
rf_final = make_model(rf_est).fit(X_arr, y)
t_start = time.perf_counter()
rf_final.predict(X_arr)
t_predict = time.perf_counter() - t_start
importances = rf_final[-1].feature_importances_

# ── Data-driven findings (computed before the figures that depend on them) ────
ranking = sorted(models, key=lambda n: overall[n]["RMSE"])
best, second, third = ranking
best_ab = ABBR[best]
residuals = y - y_pred_all[best]

# ── Generate Figures ─────────────────────────────────────────────────────────
print("\n  Generating figures...")
figure_paths = []

# Fig 1: Mean Cp distribution (full-record mean, three wind directions)
FIG1_ANGLES = (0, 40, 90)
fig1_fields = {a: mean_cp_full[a].reshape(12, 12) for a in FIG1_ANGLES}
levels = np.linspace(
    min(f.min() for f in fig1_fields.values()), max(f.max() for f in fig1_fields.values()), 21
)
fig, axes = plt.subplots(1, 3, figsize=get_figsize("double"))
for ax, angle in zip(axes, FIG1_ANGLES, strict=True):
    im = ax.contourf(tap_x, tap_y, fig1_fields[angle], levels=levels, cmap="RdBu_r")
    ax.annotate(
        "",
        xy=(0.5 + 0.18 * np.cos(np.radians(angle)), 0.5 + 0.18 * np.sin(np.radians(angle))),
        xytext=(0.5 - 0.18 * np.cos(np.radians(angle)), 0.5 - 0.18 * np.sin(np.radians(angle))),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
    )
    ax.set_xlabel("x/B")
    ax.set_ylabel("y/D")
    ax.set_title(f"$\\theta$ = {angle}°")
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04).set_label("Mean $C_p$")
p = save_figure(fig, "fig_1_mean_cp_distribution", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 1: {os.path.basename(p)}")

# Fig 2: Time history (observation / prediction windows) + PSD of the observation window
t = np.arange(N_TIME) * DT
fig, axes = plt.subplots(2, 2, figsize=get_figsize("double"))
for i, tap_name in enumerate(["Corner", "Center"]):
    cp = cp_example[i]
    ax = axes[0, i]
    ax.plot(t, cp, color=colors[i], linewidth=0.4)
    ax.axvspan(0, T_OBS, color="0.85", zorder=0, label="Observation window")
    ax.axhline(np.mean(cp[:N_OBS]), color="black", linestyle="--", linewidth=0.8)
    k_peak = N_OBS + int(np.argmin(cp[N_OBS:]))
    ax.plot(t[k_peak], cp[k_peak], "v", color="black", markersize=5, label="Target peak")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"{tap_name} tap, $\\theta$ = 0°")
    lo, hi = cp.min(), cp.max()
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))  # room for the legend
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    f_obs, psd_obs = signal.welch(cp[:N_OBS], fs=FS, nperseg=WELCH_NPERSEG)
    f_full, psd_full = signal.welch(cp, fs=FS, nperseg=512)
    axes[1, i].loglog(f_full[1:], psd_full[1:], color="0.6", linewidth=0.8, label="Full record")
    axes[1, i].loglog(
        f_obs[1:], psd_obs[1:], color=colors[i], linewidth=1.2, label="Observation window"
    )
    axes[1, i].set_xlabel("Frequency (Hz)")
    axes[1, i].set_ylabel("PSD ($C_p^2$/Hz)")
    axes[1, i].set_title(f"PSD — {tap_name}")
    axes[1, i].legend(fontsize=7)
p = save_figure(fig, "fig_2_time_history_psd", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 2: {os.path.basename(p)}")

# Fig 3: Feature importance (impurity-based and permutation-based)
fig, axes = plt.subplots(1, 2, figsize=get_figsize("double"))
for ax, values, title in [
    (axes[0], importances, "(a) Impurity-based (RF)"),
    (axes[1], perm_imp, "(b) Permutation, held-out folds (RF)"),
]:
    order = np.argsort(values)
    y_pos = np.arange(len(FEATURES))
    bar_colors = [colors[1] if r >= len(FEATURES) - 5 else colors[0] for r in range(len(FEATURES))]
    ax.barh(y_pos, values[order], color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURES[i] for i in order], fontsize=8)
    ax.set_xlabel("Importance" if ax is axes[0] else "Mean decrease in $R^2$")
    ax.set_title(title, fontsize=10)
p = save_figure(fig, "fig_3_feature_importance", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 3: {os.path.basename(p)}")

# Fig 4: Model comparison scatter (out-of-fold predictions)
fig, axes = plt.subplots(1, 3, figsize=get_figsize("double"))
for idx, name in enumerate(models):
    ax = axes[idx]
    ax.scatter(y, y_pred_all[name], s=3, alpha=0.3, color=colors[idx], rasterized=True)
    lims = [min(y.min(), y_pred_all[name].min()), max(y.max(), y_pred_all[name].max())]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Measured Peak $C_p$")
    ax.set_ylabel("Predicted Peak $C_p$")
    ax.set_title(name, fontsize=10)
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {overall[name]['R2']:.4f}\nRMSE = {overall[name]['RMSE']:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax.set_aspect("equal")
p = save_figure(fig, "fig_4_model_comparison", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 4: {os.path.basename(p)}")

# Fig 5: CV box plot
fig, axes = plt.subplots(1, 3, figsize=get_figsize("double"))
metric_labels = ["$R^2$", "RMSE", "MAE"]
for ax_idx, metric in enumerate(["R2", "RMSE", "MAE"]):
    data = [cv_results[name][metric] for name in models]
    bp = axes[ax_idx].boxplot(data, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors[:3], strict=False):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    axes[ax_idx].set_xticklabels([ABBR[n] for n in models], fontsize=9)
    axes[ax_idx].set_ylabel(metric_labels[ax_idx])
    axes[ax_idx].set_title(f"10-Fold CV: {metric_labels[ax_idx]}")
p = save_figure(fig, "fig_5_cv_boxplot", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 5: {os.path.basename(p)}")

# Fig 6: Wind direction polar error (every direction is predicted while held out)
fig, ax = plt.subplots(figsize=get_figsize("single"), subplot_kw={"projection": "polar"})
theta = np.radians(WIND_ANGLES)
theta_c = np.append(theta, theta[0])
for idx, name in enumerate(models):
    rmse_c = np.append(rmse_dir[name], rmse_dir[name][0])
    lw = 1.6 if name == best else 0.9
    ax.plot(
        theta_c, rmse_c, "o-", color=colors[idx], markersize=2.5, linewidth=lw, label=ABBR[name]
    )
ax.set_title("Out-of-fold RMSE vs Wind Direction", pad=20, fontsize=11)
ax.set_rlabel_position(112.5)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.05, 1.0))
p = save_figure(fig, "fig_6_wind_direction_error", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 6: {os.path.basename(p)}")

# Fig 7: Residual analysis (best model)
fig, axes = plt.subplots(1, 3, figsize=get_figsize("double"))
axes[0].scatter(y_pred_all[best], residuals, s=2, alpha=0.2, color=colors[0], rasterized=True)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("Predicted Peak $C_p$")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")
axes[1].hist(residuals, bins=50, color=colors[0], edgecolor="white", density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[1].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()), "r-", linewidth=1.5)
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Density")
axes[1].set_title("Residual Distribution")
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[2].scatter(osm, osr, s=3, alpha=0.3, color=colors[0], rasterized=True)
axes[2].plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.2)
axes[2].set_xlabel("Theoretical Quantiles")
axes[2].set_ylabel("Sample Quantiles")
axes[2].set_title("Q-Q Plot")
p = save_figure(fig, "fig_7_residual_analysis", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 7: {os.path.basename(p)}")

# Fig 8: Peak Cp distribution comparison (best model)
fig, ax = plt.subplots(figsize=get_figsize("single"))
ax.hist(y, bins=50, alpha=0.6, color=colors[0], label="Measured", density=True, edgecolor="white")
ax.hist(
    y_pred_all[best],
    bins=50,
    alpha=0.6,
    color=colors[1],
    label=f"{best_ab} Predicted",
    density=True,
    edgecolor="white",
)
ax.set_xlabel("Peak $C_p$")
ax.set_ylabel("Density")
ax.set_title("Distribution: Measured vs Predicted Peak $C_p$")
ax.legend()
p = save_figure(fig, "fig_8_peak_cp_distribution", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 8: {os.path.basename(p)}")

print(f"\n  Total figures generated: {len(figure_paths)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: RESULT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Step 4] Result Analysis")
print("-" * 40)

# ── Data-driven findings ─────────────────────────────────────────────────────
# Every comparative or quantitative statement in the manuscript is derived from
# these values, so the narrative stays consistent with the computed results
# (which can shift slightly across scikit-learn / NumPy versions).
TREE_MODELS = {"Random Forest", "Gradient Boosting"}
FEATURE_LABELS = {
    "mean_cp": "short-record mean Cp",
    "rms_cp": "short-record RMS Cp",
    "min_cp": "short-record minimum Cp",
    "max_cp": "short-record maximum Cp",
    "skewness": "short-record skewness",
    "kurtosis": "short-record kurtosis",
    "peak_factor": "short-record peak factor",
    "dominant_freq": "dominant frequency",
    "spectral_bandwidth": "spectral bandwidth",
    "spectral_centroid": "spectral centroid",
    "tap_x": "tap coordinate x/B",
    "tap_y": "tap coordinate y/D",
    "wind_angle_sin": "direction encoding sin(theta)",
    "wind_angle_cos": "direction encoding cos(theta)",
}
PEAK_MODEL_FEATURES = {"mean_cp", "rms_cp", "min_cp", "peak_factor"}


def fmt_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + ", and " + items[-1]


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def label(feature, value=None):
    """Readable feature name with its code name (and optionally its importance)."""
    extra = f", {value:.3f}" if value is not None else ""
    return f"{FEATURE_LABELS[feature]} ({feature}{extra})"


GAUSS = "Gaussian peak-factor estimate"
OBSPK = "Observed short-record peak"

# Model ranking (pooled out-of-fold RMSE; pooled R-squared gives the same order) and
# paired Wilcoxon signed-rank tests on the 36 per-direction mean squared errors
r2_lo = min(o["R2"] for o in overall.values())
r2_hi = max(o["R2"] for o in overall.values())
pair_p = {
    (a, b): float(stats.wilcoxon(mse_dir[a], mse_dir[b]).pvalue)
    for a, b in ((best, second), (best, third), (second, third))
}
p12, p13, p23 = pair_p[(best, second)], pair_p[(best, third)], pair_p[(second, third)]
rmse_gap = {n: 100 * (overall[n]["RMSE"] / overall[best]["RMSE"] - 1) for n in (second, third)}
mae_best = min(models, key=lambda n: overall[n]["MAE"])
best_tree = min(TREE_MODELS, key=lambda n: overall[n]["RMSE"])
pair_text = [f"{ABBR[a]} versus {ABBR[b]}, {fmt_p(p)}" for (a, b), p in pair_p.items()]

# Training-free reference estimators
best_baseline = min(BASELINES, key=lambda n: baseline_metrics[n]["RMSE"])
gain_vs = {b: 100 * (1 - overall[best]["RMSE"] / baseline_metrics[b]["RMSE"]) for b in BASELINES}
worst_ml = max(models, key=lambda n: overall[n]["RMSE"])
all_beat_baselines = overall[worst_ml]["RMSE"] < baseline_metrics[best_baseline]["RMSE"]
bias = {b: float(np.mean(y - pred)) for b, pred in BASELINES.items()}
ml_spread = overall[worst_ml]["RMSE"] - overall[best]["RMSE"]  # range of the ML models' RMSE
ml_margin = baseline_metrics[best_baseline]["RMSE"] - overall[worst_ml]["RMSE"]
gauss_g_median = float(np.median(gauss_g))

# Fold-to-fold stability
cv_std = {n: float(np.std(cv_results[n]["RMSE"])) for n in models}
by_stability = sorted(models, key=lambda n: cv_std[n])
fold_r2_min = {n: float(np.min(cv_results[n]["R2"])) for n in models}

# Most severe 5% of measured peaks
severe = y <= np.percentile(y, 5)
severe_cut = float(np.percentile(y, 5))
severe_rmse = {n: float(np.sqrt(np.mean((y[severe] - y_pred_all[n][severe]) ** 2))) for n in models}
severe_bias = {n: float(np.mean(y[severe] - y_pred_all[n][severe])) for n in models}
rest_rmse_best = float(np.sqrt(np.mean(residuals[~severe] ** 2)))
all_underestimate_severe = all(v < 0 for v in severe_bias.values())

# Leakage controls and information content
guard_delta = overall["Random Forest"]["R2"] - guard_metrics["R2"]
signal_beats_spatial = ablation["signal"]["R2"] > ablation["spatial"]["R2"]
spatial_gain_pct = 100 * (1 - ablation["all"]["RMSE"] / ablation["signal"]["RMSE"])
spatial_adds_little = abs(spatial_gain_pct) < 2.0
duration_ratio = N_TIME / N_OBS
n_welch_segments = (N_OBS - WELCH_NPERSEG) // (WELCH_NPERSEG // 2) + 1

# Feature importance
imp_order = [int(i) for i in np.argsort(-importances)]
perm_order = [int(i) for i in np.argsort(-perm_imp)]
top3_imp = [FEATURES[i] for i in imp_order[:3]]
top3_perm = [FEATURES[i] for i in perm_order[:3]]
rank_of = {FEATURES[i]: r + 1 for r, i in enumerate(imp_order)}
perm_pos = np.clip(perm_imp, 0, None)
group_share = {
    g: float(sum(importances[FEATURES.index(f)] for f in fs) / importances.sum())
    for g, fs in FEATURE_GROUPS.items()
}
group_share_perm = {
    g: float(sum(perm_pos[FEATURES.index(f)] for f in fs) / perm_pos.sum())
    for g, fs in FEATURE_GROUPS.items()
}
dominant_group = max(group_share, key=group_share.get)
rank_rho = float(stats.spearmanr(importances, perm_imp)[0])
same_top = imp_order[0] == perm_order[0]
top_spectral = max(FEATURE_GROUPS["spectral"], key=lambda f: importances[FEATURES.index(f)])
spectral_beats_spatial = group_share["spectral"] > group_share["spatial-directional"]
top3_peak_model = set(top3_imp) <= PEAK_MODEL_FEATURES
higher_moments = ["skewness", "kurtosis", "peak_factor"]
higher_moment_share = float(
    sum(importances[FEATURES.index(f)] for f in higher_moments) / importances.sum()
)

# Wind direction dependence (each direction is predicted while held out)
rb = rmse_dir[best]
k_min, k_max = int(np.argmin(rb)), int(np.argmax(rb))
dir_cov = float(np.std(rb) / np.mean(rb))
obliqueness = np.abs(np.sin(2 * np.radians(WIND_ANGLES)))
rho_obl, p_obl = (float(v) for v in stats.spearmanr(rb, obliqueness))
sd_dir = np.array([np.std(y[groups == k]) for k in range(N_DIR)])
rho_sd, p_sd = (float(v) for v in stats.spearmanr(rb, sd_dir))
rho_nobl, p_nobl = (float(v) for v in stats.spearmanr(rb / sd_dir, obliqueness))
rho_sd_obl, p_sd_obl = (float(v) for v in stats.spearmanr(sd_dir, obliqueness))
oblique = obliqueness >= np.sin(np.radians(60))  # within 15 degrees of a diagonal
near_normal = obliqueness <= np.sin(np.radians(20))  # within 10 degrees of a roof axis
rmse_oblique, rmse_normal = float(np.mean(rb[oblique])), float(np.mean(rb[near_normal]))

# Residuals of the best model
res_mean, res_std = float(np.mean(residuals)), float(np.std(residuals))
res_skew, res_kurt = float(stats.skew(residuals)), float(stats.kurtosis(residuals))
normal_p = float(stats.normaltest(residuals).pvalue)
residuals_normal = normal_p >= 0.05
het_rho, het_p = (float(v) for v in stats.spearmanr(np.abs(residuals), y_pred_all[best]))
heteroscedastic = het_p < 0.05
pi_lo, pi_hi = (float(v) for v in np.percentile(residuals, [2.5, 97.5]))
cover_gauss = float(np.mean(np.abs(residuals - res_mean) <= 1.96 * res_std))
ks = stats.ks_2samp(y, y_pred_all[best])
ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
std_ratio = float(np.std(y_pred_all[best]) / np.std(y))
q01_meas, q01_pred = float(np.percentile(y, 1)), float(np.percentile(y_pred_all[best], 1))

# Figures 1-2: where the mean suction is strongest, corner vs. centre statistics
fig1_min = {}
for a in FIG1_ANGLES:
    i_min = int(np.argmin(mean_cp_full[a]))
    fig1_min[a] = (float(mean_cp_full[a][i_min]), float(tap_xx[i_min]), float(tap_yy[i_min]))
corner_cp, center_cp = cp_example
ex_stats = {
    lab: {
        "rms": float(np.std(cp)),
        "skew": float(stats.skew(cp)),
        "kurt": float(stats.kurtosis(cp)),
        "centroid": float(extract_features(cp[None, :N_OBS])["spectral_centroid"].iloc[0]),
    }
    for lab, cp in (("corner", corner_cp), ("center", center_cp))
}

print("  Data-driven findings:")
print(f"    ranking: {[ABBR[n] for n in ranking]}  paired tests: {pair_text}")
print(f"    gain vs baselines: { {b: round(v, 1) for b, v in gain_vs.items()} }")
print(f"    top-3 impurity: {top3_imp}  top-3 permutation: {top3_perm}  rho = {rank_rho:.2f}")
print(f"    group share (impurity): { {g: round(v, 3) for g, v in group_share.items()} }")
print(f"    direction RMSE CoV {dir_cov:.2f}; rho(obliqueness) = {rho_obl:.2f} ({fmt_p(p_obl)})")
print(f"    residual skew {res_skew:.2f}, normality {fmt_p(normal_p)}, het rho {het_rho:.2f}")

# ── Conditional narrative blocks ──────────────────────────────────────────────
best_stats = (
    f"R-squared = {overall[best]['R2']:.4f}, RMSE = {overall[best]['RMSE']:.4f}, "
    f"MAE = {overall[best]['MAE']:.4f}"
)
model_claim = (
    f"{best_ab} achieved the lowest prediction error ({best_stats}), followed by "
    f"{ABBR[second]} (RMSE {rmse_gap[second]:.1f}% higher) and {ABBR[third]} "
    f"(RMSE {rmse_gap[third]:.1f}% higher)"
)
if p12 < 0.05 and p13 < 0.05:
    model_verdict = f"{best_ab} is significantly more accurate than both other models"
elif p13 < 0.05:
    model_verdict = (
        f"{best_ab} is significantly more accurate than {ABBR[third]}, but its difference from "
        f"{ABBR[second]} is not statistically significant"
        + (
            f", and neither is the difference between {ABBR[second]} and {ABBR[third]}"
            if p23 >= 0.05
            else f", whereas {ABBR[second]} is also significantly more accurate than {ABBR[third]}"
        )
    )
elif p12 < 0.05:
    model_verdict = (
        f"{best_ab} is significantly more accurate than {ABBR[second]}; the remaining "
        "differences are not statistically significant"
    )
else:
    model_verdict = "none of the differences between the three models is statistically significant"

significance_sentence = (
    "Pairwise differences were tested with the Wilcoxon signed-rank test applied to the "
    f"{N_DIR} paired per-direction mean squared errors ({'; '.join(pair_text)}). At the 0.05 "
    f"level, {model_verdict}. The 95% bootstrap confidence intervals of the RMSE ({best_ab}: "
    f"{rmse_ci[best][0]:.4f}-{rmse_ci[best][1]:.4f}; {ABBR[second]}: "
    f"{rmse_ci[second][0]:.4f}-{rmse_ci[second][1]:.4f}; {ABBR[third]}: "
    f"{rmse_ci[third][0]:.4f}-{rmse_ci[third][1]:.4f}) "
    + (
        "overlap, because they also contain the direction-to-direction variability that is "
        "common to all models; the paired test removes this shared component and is therefore "
        "the more sensitive comparison. "
        if rmse_ci[best][1] > rmse_ci[third][0]
        else "do not overlap. "
    )
)
if mae_best != best:
    significance_sentence += (
        f"Note that {ABBR[mae_best]} attains a slightly lower MAE ({overall[mae_best]['MAE']:.4f} "
        f"versus {overall[best]['MAE']:.4f}), so the ranking depends on whether large errors are "
        "penalized quadratically (RMSE) or linearly (MAE). "
    )
else:
    significance_sentence += f"{best_ab} also attains the lowest MAE. "

if all_beat_baselines:
    baseline_claim = (
        f"All three ML models outperform both training-free estimators; {best_ab} reduces the RMSE "
        f"by {gain_vs[GAUSS]:.1f}% relative to the Gaussian peak-factor estimate and by "
        f"{gain_vs[OBSPK]:.1f}% relative to the observed short-record peak"
    )
else:
    baseline_claim = (
        f"Not all ML models outperform the {best_baseline.lower()} (RMSE "
        f"{baseline_metrics[best_baseline]['RMSE']:.4f}); {best_ab} changes the RMSE by "
        f"{-gain_vs[best_baseline]:+.1f}% relative to this training-free estimator"
    )

dnn_note = (
    f"The DNN stopped early after {min(dnn_epochs)}-{max(dnn_epochs)} epochs in the ten folds; with "
    f"{dnn_params:,} trainable weights and about {len(y) * 9 // 10:,} training samples per fold, "
    "more data, regularization (e.g., dropout), or tuning could change its standing, which was not "
    "explored here. "
)
if best in TREE_MODELS:
    other_tree = next(n for n in TREE_MODELS if n != best)
    best_mechanism = (
        f"Random feature subsampling ({rf_split_features} of {len(FEATURES)} features per split) "
        "decorrelates the trees of the Random Forest and averages over many partially redundant "
        "inputs, which suits the present feature set, in which the short-record minimum, mean, and "
        "RMS carry overlapping information. "
        if best == "Random Forest"
        else "Sequential boosting corrects the residual structure of earlier trees, which helps "
        "where the peak depends on interactions between the short-record statistics. "
    )
    model_discussion = (
        f"Among the three models, {best_ab} gives the lowest error. {best_mechanism}"
        + (
            f"The ranking does not, however, indicate a general advantage of tree ensembles: "
            f"{ABBR[other_tree]}, the other tree ensemble, is the least accurate of the three models"
            + (
                ", although its difference from the DNN is not statistically significant. "
                if p23 >= 0.05
                else ", significantly less accurate than the DNN. "
            )
            + "Possible reasons, which were not investigated, include the untuned depth and "
            f"learning rate of {ABBR[other_tree]}. "
            if ranking[-1] == other_tree
            else "Both tree ensembles rank ahead of the DNN, consistent with the common "
            "observation that tree-based ensembles are strong learners for tabular data of "
            "moderate size. "
        )
        + dnn_note
    )
else:
    model_discussion = (
        f"The DNN gives the lowest error; the RMSE of the best tree ensemble ({ABBR[best_tree]}) is "
        f"{rmse_gap[best_tree]:.1f}% higher. A plausible explanation is that the peak depends "
        "smoothly on the short-record mean, RMS, and minimum, which favors the continuous response "
        "surface of a neural network over the piecewise-constant predictions of trees. Tree "
        "ensembles nevertheless remain attractive because they require little tuning and provide "
        f"direct importance estimates. The DNN stopped early after {min(dnn_epochs)}-"
        f"{max(dnn_epochs)} epochs with {dnn_params:,} trainable weights. "
    )

if all_underestimate_severe:
    severe_sentence = (
        f"For the most severe 5% of measured peaks (Cp below {severe_cut:.2f}), all three models "
        "underestimate the suction magnitude on average (mean residual "
        + ", ".join(f"{ABBR[n]} {severe_bias[n]:.3f}" for n in models)
        + f"), and the RMSE of {best_ab} rises to {severe_rmse[best]:.3f}, compared with "
        f"{rest_rmse_best:.3f} for the remaining 95% of samples. This regression toward the mean is "
        "unconservative for cladding design and is examined further in the residual analysis. "
    )
else:
    severe_sentence = (
        f"For the most severe 5% of measured peaks (Cp below {severe_cut:.2f}) the mean residuals "
        "are " + ", ".join(f"{ABBR[n]} {severe_bias[n]:.3f}" for n in models) + f", and the RMSE "
        f"of {best_ab} is {severe_rmse[best]:.3f}, compared with {rest_rmse_best:.3f} for the "
        "remaining samples. "
    )

stability_sentence = (
    f"Across the ten folds, the standard deviation of the fold RMSE is smallest for "
    f"{ABBR[by_stability[0]]} ({cv_std[by_stability[0]]:.4f}) and largest for "
    f"{ABBR[by_stability[-1]]} ({cv_std[by_stability[-1]]:.4f}), and the lowest fold R-squared is "
    + ", ".join(f"{fold_r2_min[n]:.4f} ({ABBR[n]})" for n in models)
    + ". "
    + (
        "The fold-to-fold spread is therefore similar for all three models; "
        if cv_std[by_stability[-1]] < 1.5 * cv_std[by_stability[0]]
        else f"{ABBR[by_stability[-1]]} is thus noticeably more sensitive to the fold composition; "
    )
    + (
        "part of it reflects which wind directions are held out, since, as shown in the wind "
        "direction analysis below, the errors are largest at oblique directions. "
        if rho_obl > 0 and p_obl < 0.05
        else "part of it reflects which wind directions are held out in each fold. "
    )
)

leakage_paragraph = (
    "The disjoint window design ensures by construction that no input is computed from samples "
    "of the prediction window. The observed short-record minimum is more extreme than the target "
    f"in only {share_window_extreme:.1%} of the records and is otherwise a less severe, partial "
    f"sample of the extreme (correlation with the target r = {r_min_target:.3f}). If the target "
    "were instead defined as the minimum of the full record, the input min_cp would coincide "
    f"exactly with the target in these {share_window_extreme:.1%} of the records; this is the "
    "leakage that the disjoint windows avoid. Three further controls confirm that the reported "
    "accuracy does not stem from leakage. First, moving the start of the target window by a guard "
    f"gap of {GUARD} samples ({GUARD * DT:.3f} s, {GUARD / PULSE_WIDTH[1]:.1f} times the largest "
    f"pulse width) changes the RF out-of-fold R-squared from {overall['Random Forest']['R2']:.4f} "
    f"to {guard_metrics['R2']:.4f} (difference {-guard_delta:+.4f}), so the continuity of the "
    "signal across the window boundary contributes "
    + ("negligibly " if abs(guard_delta) < 0.005 else "measurably ")
    + "to the skill. Second, when the RF is trained and evaluated with randomly permuted targets "
    f"under the same grouped cross-validation, the out-of-fold R-squared is {perm_control_r2:.4f}, "
    + (
        "confirming that the pipeline (including the standardization fitted within the training "
        "folds) provides no route by which test targets can influence predictions. "
        if perm_control_r2 < 0.02
        else "which indicates that the pipeline is not free of leakage and requires inspection. "
    )
    + "Third, when the same 14 features are computed from the full record and the target is the "
    "minimum of the full record, so that the input min_cp equals the target whenever the "
    "full-record minimum lies in the observation window and bounds it otherwise, the RF reaches "
    f"R-squared = {leaky_metrics['R2']:.4f} (RMSE = {leaky_metrics['RMSE']:.4f}) under the same "
    "protocol. "
    + (
        "Such near-perfect scores are an artifact of the leaky formulation and say nothing about "
        "the ability to predict an unobserved peak. "
        if leaky_metrics["R2"] > 0.99
        else "The higher score of this formulation is an artifact of leakage and says nothing "
        "about the ability to predict an unobserved peak. "
    )
)

if signal_beats_spatial:
    ablation_paragraph = (
        "The ablation study quantifies the value of the short-duration measurement. An RF that "
        "receives only the four spatial-directional inputs (tap coordinates and encoded wind "
        "direction), and therefore requires no measurement of the record at all, reaches "
        f"R-squared = {ablation['spatial']['R2']:.4f} (RMSE = {ablation['spatial']['RMSE']:.4f}); "
        "an RF that receives only the ten statistical and spectral features of the observation "
        f"window reaches R-squared = {ablation['signal']['R2']:.4f} "
        f"(RMSE = {ablation['signal']['RMSE']:.4f}). The full feature set gives R-squared = "
        f"{ablation['all']['R2']:.4f} (RMSE = {ablation['all']['RMSE']:.4f}), so adding the "
        f"spatial-directional inputs to the short-record features changes the RMSE by "
        f"{-spatial_gain_pct:+.1f}%. "
        + (
            "The short record therefore carries essentially all of the predictive information. "
            if spatial_adds_little
            else "Both sources of information therefore contribute to the predictions. "
        )
        + "A model based on tap position and wind direction can only interpolate the pressure "
        "field between neighbouring wind directions of the same building and cannot account for "
        "the run-to-run and record-to-record variations of turbulence and intermittency, which "
        "the short-record statistics measure directly. "
    )
else:
    ablation_paragraph = (
        "The ablation study shows that tap position and wind direction alone (R-squared = "
        f"{ablation['spatial']['R2']:.4f}) are at least as informative as the ten short-record "
        f"features alone (R-squared = {ablation['signal']['R2']:.4f}); the full set gives "
        f"R-squared = {ablation['all']['R2']:.4f}. In this case the benefit of the short "
        "measurement is limited, and the model mainly interpolates the pressure field between "
        "neighbouring wind directions. "
    )

if top3_peak_model:
    importance_interpretation = (
        f"The prominence of {fmt_list('the ' + FEATURE_LABELS[f] for f in top3_imp)} is "
        "physically consistent with the peak-factor representation of an extreme value, "
        "Cp_peak = Cp_mean - g Cp_rms: the short-record mean and RMS estimate the location and "
        "scale of the pressure distribution, and the short-record minimum is a direct, although "
        "incomplete, sample of its lower tail. The models effectively learn a data-driven peak "
        "factor g that adapts to the local flow regime. "
    )
else:
    importance_interpretation = (
        f"Only part of this ranking ({fmt_list(FEATURE_LABELS[f] for f in top3_imp)}) follows the "
        "peak-factor representation Cp_peak = Cp_mean - g Cp_rms, in which the mean and RMS set "
        "the location and scale of the extreme. "
    )

importance_consistency = (
    "Permutation importance on the held-out folds, which measures the loss of out-of-fold "
    f"R-squared when one feature is shuffled, ranks {fmt_list('the ' + label(f) for f in top3_perm)} "
    f"highest. The Spearman rank correlation between the two importance measures is {rank_rho:.2f}"
    + (
        ", and both identify the same most important feature. "
        if same_top
        else ", but they disagree on the most important feature. "
    )
    + "Permutation importance is systematically smaller for features that share information "
    "with others (e.g., the short-record mean, RMS, and spectral bandwidth all reflect the "
    "separation intensity), because the remaining correlated features compensate when one of "
    "them is shuffled; the impurity-based measure distributes credit among such features "
    "instead. Neither measure should therefore be read as the causal contribution of an "
    "individual feature. "
)

if spectral_beats_spatial:
    spectral_paragraph = (
        f"The spectral features carry {group_share['spectral']:.1%} of the impurity-based "
        f"importance ({group_share_perm['spectral']:.1%} of the positive permutation importance), "
        f"led by the {label(top_spectral)}, which ranks {ordinal(rank_of[top_spectral])} of "
        f"{len(FEATURES)}. In the synthetic records the bandwidth of the Gaussian fluctuations "
        "narrows as the separation intensity increases, so the spectral descriptors act as "
        "indirect indicators of separated flow, and their contribution overlaps with that of the "
        f"RMS. The dominant frequency ranks {ordinal(rank_of['dominant_freq'])}"
        + (
            ": the spectrum of the synthetic fluctuations is nearly flat below the filter cutoff, so "
            "the frequency of the spectral maximum is poorly defined in a short-window estimate with "
            f"{n_welch_segments} averaged segments. "
            if rank_of["dominant_freq"] > len(FEATURES) // 2
            else ". "
        )
    )
else:
    spectral_paragraph = (
        f"The spectral features carry only {group_share['spectral']:.1%} of the impurity-based "
        f"importance, less than the spatial-directional inputs "
        f"({group_share['spatial-directional']:.1%}). With {n_welch_segments} averaged segments "
        f"and a {FS / WELCH_NPERSEG:.1f} Hz resolution, the short-window spectrum adds little "
        "information beyond the statistical moments. "
    )

spatial_paragraph = (
    f"The spatial-directional features carry {group_share['spatial-directional']:.1%} of the "
    f"impurity-based importance and {group_share_perm['spatial-directional']:.1%} of the positive "
    "permutation importance. "
    + (
        "This low importance is consistent with the ablation result: the effect of tap position "
        "and wind direction on the peak acts through the local mean, RMS, and intermittency of "
        "the pressure signal, which the short-record statistics already measure for the specific "
        "record, so the explicit position and direction inputs are largely redundant once the "
        "short record is available. "
        if group_share["spatial-directional"] < 0.05
        else "Tap position and wind direction thus retain information that the short-record "
        "statistics do not capture, presumably through the spatial pattern of separated flow. "
    )
    + "The higher-order descriptors (skewness, kurtosis, and peak factor) together account for "
    f"{higher_moment_share:.1%} of the impurity-based importance"
    + (
        f". In a window of only {N_OBS} samples these statistics are strongly affected by sampling "
        "variability, because the window contains few intermittent suction events at most taps. "
        if higher_moment_share < 0.05
        else ", reflecting the non-Gaussian character of the separated-flow pressures. "
    )
)

if p_obl < 0.05 and rho_obl > 0:
    direction_pattern = (
        "The error is larger at oblique wind directions: the Spearman rank correlation between "
        f"the per-direction RMSE and the obliqueness |sin(2 theta)| is {rho_obl:.2f} "
        f"({fmt_p(p_obl)}), and the mean RMSE of the directions within 15 degrees of a roof "
        f"diagonal is {rmse_oblique:.4f}, compared with {rmse_normal:.4f} for the directions within "
        "10 degrees of a roof axis. "
    )
elif p_obl < 0.05:
    direction_pattern = (
        "The error is larger near the normal wind directions: the Spearman rank correlation "
        f"between the per-direction RMSE and the obliqueness |sin(2 theta)| is {rho_obl:.2f} "
        f"({fmt_p(p_obl)}); the mean RMSE is {rmse_oblique:.4f} for near-diagonal and "
        f"{rmse_normal:.4f} for near-axis directions. "
    )
else:
    direction_pattern = (
        "The per-direction RMSE shows no statistically significant association with the "
        f"obliqueness |sin(2 theta)| (Spearman rho = {rho_obl:.2f}, {fmt_p(p_obl)}); the mean RMSE "
        f"is {rmse_oblique:.4f} for near-diagonal and {rmse_normal:.4f} for near-axis directions. "
    )
spread_tracks_obliqueness = p_sd_obl < 0.05 and rho_sd_obl > 0
error_tracks_spread = p_sd < 0.05 and rho_sd > 0 and p_nobl >= 0.05
direction_cause = (
    "The per-direction RMSE is "
    + ("strongly " if rho_sd >= 0.5 else "")
    + f"associated with the spread (standard deviation) of the measured peaks in that direction "
    f"(Spearman rho = {rho_sd:.2f}, {fmt_p(p_sd)}), and this spread itself "
    + (
        f"increases with obliqueness (rho = {rho_sd_obl:.2f}, {fmt_p(p_sd_obl)}). "
        if spread_tracks_obliqueness
        else f"has a rank correlation of {rho_sd_obl:.2f} with obliqueness ({fmt_p(p_sd_obl)}). "
    )
    + "After the RMSE is normalized by the spread, the association with obliqueness is "
    f"rho = {rho_nobl:.2f} ({fmt_p(p_nobl)}). "
    + (
        "The directional variation of the error therefore mainly reflects the larger and more "
        "variable peaks generated by the corner vortices at oblique incidence, rather than a loss "
        "of relative predictive skill. "
        if error_tracks_spread and spread_tracks_obliqueness
        else "The relative skill itself thus depends on the wind direction. "
        if p_nobl < 0.05
        else "The relative skill shows no significant dependence on obliqueness. "
    )
)

residual_shape = (
    f"The residuals of {best_ab} (measured minus predicted peak Cp) have mean {res_mean:.4f} and "
    f"standard deviation {res_std:.4f}, so the predictions are unbiased on average. "
    + (
        f"Their distribution is, however, clearly non-Gaussian (skewness {res_skew:.2f}, excess "
        f"kurtosis {res_kurt:.2f}; D'Agostino-Pearson test {fmt_p(normal_p)}), with a heavier "
        "lower tail, as the Q-Q plot in Fig. 7 shows. "
        if not residuals_normal and res_skew < 0
        else f"Their distribution deviates from a Gaussian (skewness {res_skew:.2f}, excess "
        f"kurtosis {res_kurt:.2f}; D'Agostino-Pearson test {fmt_p(normal_p)}). "
        if not residuals_normal
        else f"The normality test does not reject a Gaussian distribution ({fmt_p(normal_p)}; "
        f"skewness {res_skew:.2f}, excess kurtosis {res_kurt:.2f}). "
    )
    + (
        "The absolute residuals grow as the predicted peak becomes more severe (Spearman rho "
        f"between |residual| and predicted Cp = {het_rho:.2f}, {fmt_p(het_p)}), i.e., the errors "
        "are heteroscedastic: the taps with the most severe predicted peaks, which lie near the "
        "windward edges and corners, are predicted less precisely than interior taps. "
        if heteroscedastic and het_rho < 0
        else f"The Spearman correlation between |residual| and the predicted Cp is {het_rho:.2f} "
        f"({fmt_p(het_p)}). "
    )
)
interval_sentence = (
    f"A symmetric interval of +/- 1.96 standard deviations (+/- {1.96 * res_std:.4f}) covers "
    f"{cover_gauss:.1%} of the out-of-fold residuals overall, but the empirical 2.5% and 97.5% "
    f"residual quantiles ({pi_lo:.4f} and {pi_hi:.4f}) are asymmetric, reflecting the heavier "
    "lower tail. "
    if abs(pi_lo) > 1.1 * abs(pi_hi)
    else f"A symmetric interval of +/- 1.96 standard deviations (+/- {1.96 * res_std:.4f}) covers "
    f"{cover_gauss:.1%} of the out-of-fold residuals; the empirical 2.5% and 97.5% residual "
    f"quantiles are {pi_lo:.4f} and {pi_hi:.4f}. "
)
q01_close = abs(q01_pred - q01_meas) <= 0.05 * abs(q01_meas)
distribution_sentence = (
    f"The two-sample Kolmogorov-Smirnov statistic between the measured and predicted "
    f"distributions is {ks_stat:.4f} ({fmt_p(ks_p)}), "
    + (
        "so the difference between the distributions is not statistically significant at the 5% "
        "level. "
        if ks_p >= 0.05
        else "so the distributions differ significantly. "
    )
    + f"The standard deviation of the predicted peaks is {std_ratio:.3f} times that of the "
    f"measured peaks, and the 1st percentiles are {q01_pred:.3f} (predicted) and {q01_meas:.3f} "
    "(measured). "
    + (
        "The lower tail of the marginal distribution is thus reproduced closely, even though the "
        "most severe individual peaks are underestimated: the marginal distributions can agree "
        "while the predictions for individual records regress toward the mean. "
        if q01_close
        else "The predicted lower tail is thus truncated relative to the measurements. "
        if q01_pred > q01_meas
        else "The predicted lower tail thus extends further than the measured one. "
    )
)
dispersion_sentence = (
    "The reduced dispersion of the predictions is the expected behaviour of a conditional-mean "
    "predictor: the models estimate the expected peak given the features, whereas each measured "
    "peak contains an additional random component. "
    if std_ratio < 1
    else "Although a conditional-mean predictor would be expected to produce less dispersed "
    "values than the measurements, the predictions here are not less dispersed. "
)

# ── Highlights (computed, <= 85 characters each) ─────────────────────────────
if p12 < 0.05 and p13 < 0.05:
    hl_model = (
        f"{best_ab} is the most accurate model (out-of-fold R-squared = {overall[best]['R2']:.3f})"
    )
elif p13 < 0.05:
    hl_model = (
        f"{best_ab} is most accurate (R-squared = {overall[best]['R2']:.3f}); significantly so "
        f"only versus {ABBR[third]}"
    )
else:
    hl_model = f"RF, GBR and DNN perform alike (out-of-fold R-squared {r2_lo:.3f}-{r2_hi:.3f})"
highlights = [
    f"Peak roof Cp over unseen {PRED_PCT:.1f}% of a record predicted from first {OBS_PCT:.1f}%",
    hl_model,
    (
        f"{best_ab} cuts RMSE by {gain_vs[GAUSS]:.0f}% versus Gaussian peak-factor extrapolation"
        if gain_vs[GAUSS] > 0
        else "Gaussian peak-factor extrapolation is as accurate as the ML models"
    ),
    (
        f"Short-record features: R-squared {ablation['signal']['R2']:.3f}; "
        f"position/direction alone: {ablation['spatial']['R2']:.3f}"
    ),
    (
        "Larger errors at oblique wind directions track the spread of peak suction"
        if p_obl < 0.05 and rho_obl > 0 and error_tracks_spread and spread_tracks_obliqueness
        else f"Per-direction RMSE varies from {rb.min():.3f} to {rb.max():.3f} over 36 directions"
    ),
]
assert 3 <= len(highlights) <= 5, len(highlights)
assert all(len(h) <= 85 for h in highlights), [len(h) for h in highlights]


def location_phrase(angle, x, y):
    """Describe a tap position relative to the windward edges of a wind direction."""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    on_x = (c > 1e-9 and x == 0) or (c < -1e-9 and x == 1)
    on_y = (s > 1e-9 and y == 0) or (s < -1e-9 and y == 1)
    if on_x and on_y:
        return "at the windward corner"
    if on_x or on_y:
        return "on the windward edge"
    return "away from the windward edges"


fig1_sentence = "; ".join(
    f"for theta = {a} degrees it occurs at (x/B, y/D) = ({fig1_min[a][1]:.2f}, {fig1_min[a][2]:.2f}), "
    f"{location_phrase(a, fig1_min[a][1], fig1_min[a][2])}, with mean Cp = {fig1_min[a][0]:.2f}"
    for a in FIG1_ANGLES
)
full_scale_obs_s = 600 * N_OBS / N_TIME
full_scale_pred_min = 10 * N_PRED / N_TIME
fold_sizes = sorted({len(a) for a in fold_angles})
all_metrics = {**overall, **baseline_metrics}  # Table 1: ML models, then baselines

abstract_r2 = f"R-squared = {overall[best]['R2']:.3f}, RMSE = {overall[best]['RMSE']:.3f}"
if p12 < 0.05 and p13 < 0.05:
    abstract_model = (
        f"{best_ab} achieved the lowest error ({abstract_r2}), significantly lower than "
        f"{ABBR[second]} and {ABBR[third]}"
    )
elif p13 < 0.05:
    abstract_model = (
        f"{best_ab} achieved the lowest error ({abstract_r2}), significantly lower than "
        f"{ABBR[third]} (R-squared = {overall[third]['R2']:.3f}) but not than {ABBR[second]} "
        f"(R-squared = {overall[second]['R2']:.3f})"
    )
else:
    abstract_model = (
        f"{best_ab} achieved the lowest error ({abstract_r2}), but the differences from "
        + (
            f"{ABBR[second]} and {ABBR[third]} were not statistically significant"
            if p12 >= 0.05
            else f"{ABBR[third]} were not statistically significant"
        )
    )
abstract_baseline = (
    f"{best_ab} reduced the RMSE by {gain_vs[GAUSS]:.0f}% relative to a Gaussian peak-factor "
    f"extrapolation and by {gain_vs[OBSPK]:.0f}% relative to the observed short-record peak"
    if gain_vs[GAUSS] > 0
    else f"{best_ab} did not improve on a Gaussian peak-factor extrapolation "
    f"(RMSE change {-gain_vs[GAUSS]:+.0f}%)"
)
abstract_features = (
    f"Short-record features alone gave R-squared = {ablation['signal']['R2']:.3f}, versus "
    f"{ablation['spatial']['R2']:.3f} for tap position and wind direction alone. "
)
abstract_direction = (
    "Errors were larger at oblique wind directions, where the peaks are more variable. "
    if p_obl < 0.05 and rho_obl > 0 and error_tracks_spread and spread_tracks_obliqueness
    else f"The per-direction RMSE ranged from {rb.min():.3f} to {rb.max():.3f}. "
)
abstract_residual = (
    f"The most severe 5% of peaks were underestimated by {abs(severe_bias[best]):.2f} on average. "
    if severe_bias[best] < 0
    else ""
)

if all_underestimate_severe and not residuals_normal:
    conclusion_residual = (
        f"6. The residuals of {best_ab} are unbiased on average (mean {res_mean:.4f}) but "
        f"non-Gaussian (skewness {res_skew:.2f}, excess kurtosis {res_kurt:.2f}) and "
        + ("heteroscedastic, " if heteroscedastic else "")
        + f"and the most severe 5% of peaks are underestimated by {abs(severe_bias[best]):.3f} on "
        "average. Point predictions are therefore unconservative for the most exposed corner and "
        "edge taps, and design use requires prediction intervals that are conditioned on the "
        "predicted value or flow regime rather than a single symmetric error band. "
    )
else:
    conclusion_residual = (
        f"6. The residuals of {best_ab} have mean {res_mean:.4f} and standard deviation "
        f"{res_std:.4f} (skewness {res_skew:.2f}, normality test {fmt_p(normal_p)}); the mean "
        f"residual for the most severe 5% of peaks is {severe_bias[best]:.3f}. "
    )

# Build paper_content dict
paper_content = {
    "title": PAPER_TITLE,
    "authors": "S.M. Lee",
    "abstract": (
        "Peak suction governs roof cladding design on low-rise buildings, but peaks of short "
        "pressure records underestimate design-length peaks. This study tests whether machine "
        "learning (ML) can predict the peak pressure coefficient (Cp) over the unobserved remainder "
        "of a record from a short observation window. Synthetic records of a Tokyo Polytechnic "
        f"University (TPU)-style flat-roof wind tunnel model ({N_TAPS} taps, {N_DIR} wind "
        f"directions, {N_SAMPLES:,} samples) were used. Fourteen statistical, spectral, and "
        f"spatial-directional features were computed from the first {OBS_PCT:.1f}% of each "
        f"record, and the target was the minimum Cp over the remaining {PRED_PCT:.1f}%, so that "
        "no input contains target samples. Random Forest (RF), Gradient Boosting (GBR), and a "
        "Deep Neural Network (DNN) were compared using 10-fold cross-validation grouped by wind "
        f"direction. {abstract_model}. {abstract_baseline}. A guard gap between the windows "
        f"changed the RF R-squared by {-guard_delta:+.4f}, and permuted targets gave R-squared = "
        f"{perm_control_r2:.3f}, confirming the absence of leakage. {abstract_features}"
        f"{abstract_direction}{abstract_residual}The approach could shorten pressure measurements "
        "but requires validation with measured TPU records."
    ),
    "keywords": "wind pressure coefficient; machine learning; TPU aerodynamic database; low-rise building; peak pressure prediction; random forest; wind tunnel",
    "highlights": highlights,
    "sections": [
        {
            "heading": "INTRODUCTION",
            "content": (
                "Wind loads on low-rise buildings are a primary concern in structural engineering, particularly "
                "for roof systems that experience severe suction pressures during windstorms. Low-rise buildings, "
                "defined as structures with heights less than approximately 18 m, constitute the vast majority of "
                "the built environment and are disproportionately affected by wind damage. Roof failures during "
                "hurricanes, typhoons, and severe thunderstorms account for a significant portion of insured losses "
                "in the building sector. Accurate prediction of peak wind pressure coefficients is therefore "
                "essential for safe and economical building design, enabling engineers to optimize structural "
                "members while maintaining adequate safety margins against extreme wind events. "
                "\n\n"
                "Traditionally, wind pressure data has been obtained through boundary layer wind tunnel (BLWT) "
                "testing, which remains the gold standard for determining wind loads on structures. However, "
                "wind tunnel testing is time-consuming, expensive, and requires specialized facilities that are "
                "not always accessible to practicing engineers. To address this limitation, several aerodynamic "
                "databases have been developed to provide standardized wind pressure data for common building "
                "configurations. Among these, the Tokyo Polytechnic University (TPU) aerodynamic database is one "
                "of the most comprehensive, offering wind tunnel measurement data for various low-rise and high-rise "
                "building models across multiple wind directions and terrain exposures. The TPU database has been "
                "widely used in research and practice, and its data quality has been validated through comparison "
                "with full-scale measurements by Wang et al. (2020) and cross-database comparisons with the NIST "
                "aerodynamic database by Shelley et al. (2023). "
                "\n\n"
                "Recent advances in machine learning (ML) have opened new possibilities for wind engineering "
                "applications, as comprehensively reviewed by Mostafa et al. (2022) and Wu and Snaiki (2022). "
                "Several studies have demonstrated the potential of ML techniques for predicting "
                "wind loads on buildings. Bre et al. (2018) pioneered the use of artificial neural networks (ANNs) "
                "for wind pressure coefficient prediction on building surfaces, establishing a baseline for "
                "subsequent studies. Tian et al. (2020) applied deep neural networks (DNNs) to predict wind "
                "pressures on low-rise gable roof buildings, achieving significant accuracy improvements over "
                "empirical methods. Hu et al. (2020) extended deep learning approaches to investigate wind "
                "pressures on tall buildings under interference effects. More recently, Weng and Paal (2022) "
                "developed a machine learning-based framework for wind pressure prediction on low-rise non-isolated "
                "buildings, demonstrating the effectiveness of ensemble methods. Ding et al. (2022) combined neural "
                "networks with genetic algorithm and Bayesian optimization for hyperparameter tuning. Meddage et al. "
                "(2022) introduced explainable ML (XML) techniques, using SHAP values to interpret wind pressure "
                "predictions on low-rise buildings in urban settings. Huang et al. (2023) further advanced the "
                "field by applying deep neural networks for mean and RMS wind pressure coefficient prediction, "
                "while Huang et al. (2024) proposed CNN-based approaches that leverage spatial correlations in "
                "pressure tap measurements. Recent work by An and Jung (2024) addressed the challenging problem "
                "of wind pressure prediction in complex heterogeneous terrains using data-driven approaches. "
                "\n\n"
                "Advanced deep learning architectures have also been explored. Wu et al. (2023) developed "
                "super-resolution CNNs for evaluating peak wind pressures from sparse measurements, while "
                "Tong et al. (2024) proposed deep learning-based methods for extending wind pressure time series. "
                "Kim et al. (2021) applied generative adversarial networks (GANs) for wind pressure imputation, "
                "and Chen et al. (2022) combined proper orthogonal decomposition (POD) with neural networks for "
                "predicting roof-surface wind pressures induced by conical vortices. Physics-informed ML approaches "
                "have also emerged, with Zhu et al. (2023) developing surrogate models that incorporate physical "
                "constraints for wind pressure prediction and sensor placement optimization. Most recently, "
                "Nav et al. (2025) proposed a hybrid ML framework for buildings with constrained sensor networks, "
                "and Yan et al. (2024) used explainable machine learning to predict the aerodynamic interference "
                "of a low-rise building on a high-rise building. "
                "\n\n"
                "A further practical issue concerns the duration of the pressure records from which peak values "
                "are obtained. Peak pressure coefficients are extreme-value statistics of a random process, so the "
                "peak observed in a record of finite length is itself a random variable whose expected magnitude "
                "increases with the record duration. Design peaks refer to a specified observation period, "
                "typically 10 min at full scale, whereas measurements are sometimes available only for shorter "
                "periods, for example when wind tunnel time is limited, when many configurations must be screened, "
                "or when a full-scale monitoring record is interrupted. The classical way to extrapolate from a "
                "short record is a peak-factor model, in which the expected extreme is expressed as the mean minus "
                "a peak factor times the RMS, with the peak factor derived from the extreme-value statistics of a "
                "stationary Gaussian process. This assumption is questionable in separated-flow regions, where "
                "intermittent suction events make the pressure strongly non-Gaussian. Data-driven alternatives "
                "address related problems, such as the extension of wind pressure time series by deep learning "
                "(Tong et al., 2024) and the evaluation of peak pressures from spatially sparse measurements "
                "(Wu et al., 2023). "
                "\n\n"
                "Despite these advances, several research gaps remain. First, systematic comparison of multiple "
                "ML paradigms (bagging and boosting ensembles versus deep learning) on the same dataset and under "
                "the same validation protocol is limited, making it difficult to assess the relative merits of "
                "different approaches objectively. Second, most existing studies rely primarily on geometric and "
                "directional parameters as inputs, and the extent to which statistical and spectral features of "
                "a short measured record improve the prediction of peak pressures beyond such parameters has not "
                "been quantified. Third, the prediction of the peak over a design-length record from a "
                "short-duration record, and the comparison of such ML predictions with training-free peak-factor "
                "estimates, have received little attention. Fourth, wind direction-dependent prediction accuracy "
                "has rarely been analyzed in a systematic manner, despite the well-known sensitivity of peak "
                "pressures to the wind angle of attack. Finally, when input features and target are computed from "
                "the same pressure record, the features can determine the target trivially: the minimum Cp of a "
                "record is then both an input statistic and the target, so that the apparent accuracy reflects "
                "target leakage rather than predictive skill. Validation designs that exclude such leakage, and "
                "controls that demonstrate its absence, are therefore essential for a credible assessment. "
                "\n\n"
                "This study addresses these gaps by: (1) formulating peak pressure prediction as a leakage-free "
                "extrapolation task, in which 14 statistical, spectral, and spatial-directional features are "
                f"computed from the first {OBS_PCT:.1f}% of each pressure record (the observation window) and the "
                f"target is the peak suction over the disjoint remaining {PRED_PCT:.1f}% (the prediction window); "
                "(2) comparing three ML paradigms, Random Forest (bagging ensemble), Gradient Boosting (boosting "
                "ensemble), and a Deep Neural Network (connectionist approach), against two training-free "
                "estimators using 10-fold cross-validation in which all taps of a wind direction are held out "
                "together; (3) verifying the absence of leakage with a guard-gap test, a permuted-target negative "
                "control, and a comparison with a deliberately leaky formulation, and isolating the information "
                "contributed by the short record with a feature ablation; and (4) analyzing feature importance, "
                "wind direction dependence, and residuals to identify where the models succeed and where they "
                "fail. Because full TPU time series were not used in this demonstration, the framework is developed "
                "and evaluated on synthetic records generated by a stochastic model that mimics the configuration "
                f"of a TPU-style flat-roof low-rise model ({N_TAPS} roof taps, {N_DIR} wind directions); the "
                "pipeline can be applied unchanged to measured TPU records. The paper is organized as follows: "
                "Section 2 describes the methodology; Section 3 presents and discusses the results; and Section 4 "
                "summarizes the conclusions, the limitations, and directions for future research. "
            ),
        },
        {
            "heading": "METHODOLOGY",
            "content": (
                "The proposed framework consists of five sequential stages: (1) generation of synthetic "
                "TPU-style pressure coefficient records for a flat-roof low-rise building; (2) division of every "
                "record into an observation window and a disjoint prediction window; (3) feature engineering on "
                "the observation window; (4) training and validation of three ML models with 10-fold "
                "cross-validation grouped by wind direction, together with two training-free reference "
                "estimators; and (5) leakage controls and error analysis, including feature importance, wind "
                "direction dependence, and residual diagnostics. The framework is modular, so that individual "
                "components can be replaced independently: the synthetic generator can be replaced by measured "
                "TPU records, and the feature extraction module could be extended with additional descriptors "
                "(e.g., wavelet coefficients or POD modal amplitudes) without modifying the downstream training "
                "and validation components. "
                "\n\n"
                f"The data flow is as follows. Each of the {N_SAMPLES:,} records (one per tap and wind direction) "
                f"is reduced to a feature vector with 14 entries, computed from its first {N_OBS} samples together "
                "with the tap coordinates and the wind direction, and to a single target value, the minimum Cp "
                f"over its last {N_PRED:,} samples. The feature matrix and the target vector are then split into "
                "training and test sets according to the grouped 10-fold protocol. Each model is trained on nine "
                "folds and evaluated on the held-out fold, cycling through all ten combinations, so that every "
                "sample receives exactly one out-of-fold prediction from a model that has never seen any record "
                "of its wind direction. All performance measures reported in Section 3 are computed from these "
                "out-of-fold predictions. The following subsections describe the data, the window definition, the "
                "features, the models, and the evaluation protocol in detail. "
            ),
            "subsections": [
                {
                    "heading": "Synthetic TPU-style dataset",
                    "content": (
                        "The dataset consists of synthetic pressure coefficient records generated to mimic the "
                        "configuration of a TPU-style wind tunnel test of an isolated flat-roof low-rise building "
                        f"with nominal full-scale dimensions H = 16 m and B = D = 24 m. The roof carries {N_TAPS} "
                        "pressure taps on a uniform 12 x 12 grid with normalized coordinates x/B and y/D between 0 "
                        "and 1, which provides dense spatial coverage including taps on the roof edges and corners. "
                        f"Records were generated for {N_DIR} wind directions from 0 to 350 degrees at 10-degree "
                        "intervals. The wind direction theta is measured from the x-axis such that the approach "
                        "flow travels in the direction (cos theta, sin theta); the windward edge is therefore "
                        "x/B = 0 for theta = 0 degrees and y/D = 0 for theta = 90 degrees, and the windward corner "
                        f"is (0, 0) for directions between 0 and 90 degrees. Each record consists of {N_TIME:,} time "
                        f"steps at a sampling frequency of {FS:,.0f} Hz ({N_TIME * DT:.3f} s at model scale), and "
                        "it is treated as representing a nominal 10-min period at full scale. The dataset comprises "
                        f"{N_SAMPLES:,} records ({N_TAPS} taps x {N_DIR} directions). The stochastic model is not a "
                        "substitute for measured data, but it reproduces the main mechanisms that control roof "
                        "peak pressures and therefore provides a controlled test bed in which the true "
                        "data-generating process is known. "
                        "\n\n"
                        "The mean pressure field combines three contributions. Leading-edge separation produces a "
                        "suction band behind each windward edge, modeled as L = |cos theta| exp(-d_x/0.12) + "
                        "|sin theta| exp(-d_y/0.12), where d_x and d_y are the normalized distances from the "
                        "windward edges perpendicular to the x- and y-axes. Conical vortices at oblique incidence "
                        "produce concentrated suction near the windward corner, modeled as C = |sin 2 theta| "
                        "exp(-(d_x + d_y)/0.2), which vanishes for flow normal to an edge and is largest at 45 "
                        "degrees. Separation at the side edges parallel to the flow adds a weaker term "
                        "S = sin^2(theta) exp(-min(x, 1 - x)/0.08) + cos^2(theta) exp(-min(y, 1 - y)/0.08), with "
                        "x and y the normalized tap coordinates. The mean pressure coefficient of a tap is "
                        "Cp_mean = -0.3 - 0.9 L - 1.1 C - 0.25 S, so that the reattached interior of the roof has "
                        "a mean Cp of about -0.3 and the strongest mean suction occurs near the windward corner at "
                        "oblique incidence. "
                        "\n\n"
                        "The fluctuating pressure is the sum of a Gaussian component and intermittent suction "
                        "events. A separation intensity s = min(1.5, 0.7 F + 0.8 C + 0.2 S) is defined with "
                        "F = |cos theta| phi(d_x) + |sin theta| phi(d_y), where phi(d) = u exp(1 - u) with "
                        "u = (d + 0.03)/0.13 peaks at d = 0.1 and represents the RMS maximum slightly downstream "
                        "of the separation line caused by shear-layer flapping. The RMS of the Gaussian component "
                        "is sigma = (0.07 + 0.28 s) r_run e_tap, where r_run is drawn from a uniform distribution "
                        f"on [{RUN_FACTOR[0]}, {RUN_FACTOR[1]}] once per wind direction to represent run-to-run "
                        "variation of the inflow turbulence, and e_tap is a lognormal factor with log-standard "
                        f"deviation {TAP_SIGMA} that represents tap-to-tap variability. The Gaussian component is "
                        "white noise passed through a causal fourth-order Butterworth low-pass filter whose cutoff "
                        f"decreases from {CUTOFF[0] * FS / 2:.0f} Hz in attached flow to "
                        f"{(CUTOFF[0] - CUTOFF[1]) * FS / 2:.0f} Hz in fully separated flow, scaled to the "
                        "required RMS with the theoretical variance of the filter output; the first "
                        f"{BURN_IN} filter output samples are discarded. Causal filtering ensures that the signal "
                        "in the observation window does not depend on random inputs that occur later in the "
                        "record. Intermittent suction events, which represent the bursts of strong suction "
                        "associated with separation bubble and corner vortex dynamics, are superimposed as "
                        f"Gaussian-shaped negative pulses with widths drawn uniformly between {PULSE_WIDTH[0]} and "
                        f"{PULSE_WIDTH[1]} samples and amplitudes between {PULSE_AMP[0]:.0f} and "
                        f"{PULSE_AMP[1]:.0f} times the local RMS, at uniformly distributed random times. The number "
                        f"of events per record follows a Poisson distribution with mean {EVENT_RATE:.0f} s lambda, "
                        "where lambda is a lognormal record-level intermittency factor with log-standard "
                        f"deviation {EVENT_SIGMA}. Interior taps (s close to zero) are thus nearly Gaussian, "
                        "whereas taps near the windward edges and corners, where s is large, have negatively "
                        "skewed, intermittent pressure signals, and both the turbulence level and the "
                        "intermittency vary from record to record in ways that tap position and wind direction "
                        "alone cannot predict. All random numbers are drawn from a generator seeded "
                        "with 42, so the dataset is exactly reproducible. "
                        "\n\n"
                        "Fig. 1 shows the full-record mean Cp on the roof for three wind directions. The strongest "
                        f"mean suction varies with the wind direction as follows: {fig1_sentence}. The suction "
                        "decays rapidly downstream of the windward edges, and the interior of the roof shows the "
                        "nearly uniform mean pressure of the reattached flow. Fig. 2 presents the time histories "
                        "and power spectral densities of a corner tap, (x/B, y/D) = (0, 0), and an interior tap, "
                        f"({tap_xx[TAP_CENTER]:.2f}, {tap_yy[TAP_CENTER]:.2f}), for theta = 0 degrees. Over the full "
                        f"record, the corner tap has an RMS of {ex_stats['corner']['rms']:.3f}, a skewness of "
                        f"{ex_stats['corner']['skew']:.2f}, and an excess kurtosis of "
                        f"{ex_stats['corner']['kurt']:.2f}, compared with {ex_stats['center']['rms']:.3f}, "
                        f"{ex_stats['center']['skew']:.2f}, and {ex_stats['center']['kurt']:.2f} for the interior "
                        "tap, which illustrates the contrast between separation-dominated and reattached flow. The "
                        "spectral centroid of the corner tap in the observation window "
                        f"({ex_stats['corner']['centroid']:.1f} Hz) is "
                        f"{'lower' if ex_stats['corner']['centroid'] < ex_stats['center']['centroid'] else 'higher'} "
                        f"than that of the interior tap ({ex_stats['center']['centroid']:.1f} Hz). The shaded part of "
                        "each time history is the observation window used for feature extraction, and the marker "
                        "indicates the target peak in the prediction window. "
                        "\n\n"
                        "The quality of the TPU database, which is the intended source of measured records for this "
                        "framework, has been examined in the literature. Wang et al. (2020) compared TPU database "
                        "values with full-scale measurements during actual typhoon events, finding good agreement "
                        "for mean pressure coefficients and acceptable agreement for peak values, with differences "
                        "attributed to the inherent variability of full-scale wind conditions. Shelley et al. "
                        "(2023) conducted a systematic comparison between the TPU and NIST aerodynamic databases, "
                        "demonstrating consistent results when accounting for differences in terrain simulation "
                        "and measurement methodology. These studies support the use of measured TPU records for "
                        "the validation step that the present synthetic demonstration cannot replace. "
                        "\n\n"
                        "Table 4 summarizes the key parameters of the dataset. The flat roof configuration was "
                        "selected because it represents one of the most common roof types for low-rise commercial "
                        "and industrial buildings, and because the aerodynamics of flat roofs (leading-edge "
                        "separation, reattachment, and conical vortex formation at oblique wind angles) produce a "
                        "wide range of peak pressure values that provide a challenging test case for ML "
                        f"prediction. In the dataset, the target peak Cp ranges from {y.min():.2f} to "
                        f"{y.max():.2f}, with a median of {np.median(y):.2f}. "
                    ),
                },
                {
                    "heading": "Observation and prediction windows",
                    "content": (
                        f"Every record is divided into two disjoint parts: an observation window formed by the first "
                        f"{N_OBS} samples ({OBS_PCT:.1f}% of the record, {T_OBS:.3f} s at model scale or about "
                        f"{full_scale_obs_s:.0f} s at full scale under the nominal 10-min interpretation) and a "
                        f"prediction window formed by the remaining {N_PRED:,} samples ({PRED_PCT:.1f}%, about "
                        f"{full_scale_pred_min:.2f} min at full scale). All signal features are computed "
                        "exclusively from the observation window, and the target is the peak suction coefficient "
                        "over the prediction window, i.e., the minimum of Cp(t) over its samples. The task "
                        "therefore emulates a short-duration measurement from which the peak over a much longer, "
                        "unobserved period must be inferred; the ratio of the total record duration to the observed "
                        f"duration is {duration_ratio:.0f}. "
                        "\n\n"
                        "The target is deliberately defined over the prediction window rather than over the full "
                        "record. If the target were the minimum of the full record, the observation window would be "
                        "part of the target interval, and the short-record minimum, which is an input feature, "
                        "would coincide with the target whenever the full-record minimum happens to fall in the "
                        "observation window. With disjoint windows, no input is computed from any sample that "
                        "enters the target. The peak over the full record, if required, follows directly as the "
                        "more severe of the observed short-record peak and the predicted peak of the prediction "
                        "window. "
                        "\n\n"
                        "The window lengths balance two requirements. The observation window must be long enough "
                        f"to provide usable estimates of the mean and RMS and a spectrum with {n_welch_segments} "
                        "averaged segments (see below), and short enough that its own peak is a poor estimate of "
                        "the peak over the remainder, so that the prediction task is not trivial. For a stationary "
                        "record, the overall extreme falls in the observation window with a probability close to "
                        f"the observed fraction of the record ({OBS_PCT:.1f}%); in the dataset this occurs for "
                        f"{share_window_extreme:.1%} of the records. The sensitivity of the results to the window "
                        "length was not examined in this study. "
                    ),
                },
                {
                    "heading": "Feature engineering",
                    "content": (
                        "A total of 14 input features were computed for each record, organized into three "
                        "categories as summarized in Table 3. The statistical and spectral features are computed "
                        f"from the {N_OBS}-sample observation window only, while the spatial-directional features "
                        "describe the tap location and the wind direction. The feature set was designed to capture "
                        "complementary aspects of the pressure signal: statistical features characterize the "
                        "probability distribution, spectral features capture the frequency content, and "
                        "spatial-directional features encode the geometric context. "
                        "\n\n"
                        "Statistical features (7): The first category comprises seven statistical descriptors "
                        "of the observation window. The mean Cp represents the time-averaged pressure "
                        "coefficient, which is related to the mean flow pattern around the building. The RMS "
                        "(standard deviation) of Cp quantifies the overall intensity of pressure fluctuations, "
                        "which is a key indicator of turbulence-induced loads. The minimum Cp (the observed "
                        "short-record peak suction) and the maximum Cp capture the extreme values of the window. "
                        "The skewness measures the asymmetry of the Cp probability distribution, with negative "
                        "skewness indicating a tendency toward extreme suction events. The excess kurtosis "
                        "quantifies the heaviness of the distribution tails relative to a normal distribution, "
                        "with high kurtosis indicating frequent extreme pressure events. Finally, the observed "
                        "peak factor, defined as (Cp_mean - Cp_min) / Cp_rms, normalizes the extreme suction of the "
                        "window relative to the local fluctuation intensity, providing a dimensionless measure of "
                        "peak severity that is independent of the local mean pressure level. "
                        "\n\n"
                        "Spectral features (3): The second category comprises three features extracted from the "
                        "power spectral density (PSD) of the observation window, computed using Welch's method "
                        f"with a Hann window, a segment length of {WELCH_NPERSEG} samples, and 50% overlap, which "
                        f"gives {n_welch_segments} averaged segments and a frequency resolution of "
                        f"{FS / WELCH_NPERSEG:.1f} Hz; the mean of each segment is removed. The dominant frequency "
                        "is defined as the frequency at which the PSD reaches its maximum value, representing the "
                        "characteristic time scale of the dominant pressure fluctuation mechanism. The spectral "
                        "centroid (first spectral moment) represents the frequency-weighted center of gravity of "
                        "the PSD, providing a robust measure of the average frequency content. The spectral "
                        "bandwidth, computed as the square root of the second central spectral moment about the "
                        "centroid, quantifies the spread of the PSD around the centroid, with broader bandwidth "
                        "indicating a wider range of active frequency scales. Together, these three spectral "
                        "features provide a compact characterization of the frequency domain behavior that "
                        "complements the time-domain statistical features. "
                        "\n\n"
                        "Spatial and directional features (4): The third category encodes the geometric context "
                        "of each pressure measurement. Normalized tap coordinates (x/B, y/D) locate each tap on "
                        "the roof surface, enabling the model to learn the spatial dependence of peak pressures. "
                        "The wind direction is encoded as sin(theta) and cos(theta) rather than the raw angle "
                        "to preserve angular continuity (i.e., 0 degrees and 350 degrees are represented as "
                        "nearby points in the feature space, avoiding the discontinuity inherent in angular "
                        "representations). "
                        "\n\n"
                        "All features were standardized (zero mean, unit variance) before model training to ensure "
                        "numerical stability and comparable scales. The standardization parameters are estimated "
                        "from the training folds only, as the first step of a scikit-learn pipeline, so that no "
                        "information from the test fold enters the preprocessing. The target variable is the peak "
                        "suction coefficient over the prediction window, which is the critical design parameter "
                        "for roof cladding and structural member sizing. "
                        "\n\n"
                        "The selection of the minimum Cp (peak suction) as the prediction target, rather than "
                        "other peak pressure statistics such as the 78th percentile fractile or the Cook-Mayne "
                        "expected peak, was motivated by its direct relevance to structural design codes. Building "
                        "codes such as ASCE 7, Eurocode 1, and the AIJ Recommendations specify design wind "
                        "pressures in terms of peak pressure coefficients that account for both the mean pressure "
                        "and the peak fluctuation component. The minimum Cp over the prediction window is the "
                        f"observed extreme of a single realization over about {full_scale_pred_min:.2f} min at full "
                        "scale under the nominal time scaling; it is closely related to, but more variable than, "
                        "the expected peak that forms the basis of component and cladding design pressures, and "
                        "part of its variability is therefore unpredictable from any input. Future extensions of "
                        "this work could consider alternative peak definitions that incorporate return period "
                        "adjustment through extreme value analysis techniques such as the Lieblein BLUE method or "
                        "the peak-over-threshold approach, but the minimum Cp provides a simple, unambiguous, and "
                        "design-relevant target for the framework development and validation presented here. "
                    ),
                },
                {
                    "heading": "Machine learning models",
                    "content": (
                        "Three ML models were implemented with scikit-learn and compared, representing three "
                        "distinct algorithmic paradigms: bagging ensemble (RF), boosting ensemble (GBR), and deep "
                        "learning (DNN). The hyperparameters were fixed a priori at commonly used values and were "
                        "not tuned on the data, so that no information from the test folds entered model "
                        "selection; the same settings were used in every fold. Each model is preceded by the "
                        "standardization step described above. "
                        "\n\n"
                        "Random Forest (RF): An ensemble of 200 regression trees with a maximum depth of 12, each "
                        "grown on a bootstrap sample of the training data with the squared-error splitting "
                        "criterion and a minimum of one sample per leaf. At each split, a random subset of one "
                        f"third of the features ({rf_split_features} of {len(FEATURES)}) is considered, which "
                        "decorrelates the trees and reduces the variance of the ensemble average. The maximum "
                        "depth of 12 allows sufficient tree complexity to capture the nonlinear relationships "
                        "between the input features and the peak Cp, while limiting the computational cost of "
                        "fully grown trees. RF provides feature importance estimates through the mean decrease in "
                        "impurity, which quantifies the contribution of each feature to reducing the squared "
                        "error over all splits of all trees. "
                        "\n\n"
                        "Gradient Boosting Regressor (GBR): An ensemble of 200 sequentially trained regression "
                        "trees with a maximum depth of 6 and a learning rate of 0.1, using the squared-error loss, "
                        "the Friedman mean squared error splitting criterion, and no row subsampling. GBR builds "
                        "trees additively using the gradient descent framework, with each new tree fitted to the "
                        "negative gradient (pseudo-residuals) of the loss function evaluated at the current "
                        "ensemble prediction. The shallow individual trees serve as weak learners, and the "
                        "learning rate scales the contribution of each tree to limit overfitting. GBR is "
                        "particularly effective for capturing complex nonlinear relationships and feature "
                        "interactions through its sequential boosting mechanism. "
                        "\n\n"
                        "Deep Neural Network (DNN): A fully connected feedforward network (multilayer perceptron) "
                        "with three hidden layers of 128, 64, and 32 neurons, Rectified Linear Unit (ReLU) "
                        f"activation functions, and a linear output neuron, giving {dnn_params:,} trainable weights "
                        "and biases. The network is trained with the Adam optimizer (learning rate 0.001, "
                        "mini-batches of 200 samples, L2 penalty 0.0001) on the squared-error loss for at most 500 "
                        "epochs. Early stopping holds out 10% of the training fold as a validation set and stops "
                        "training when the validation R-squared has not improved by at least 0.0001 for 10 "
                        "consecutive epochs; the weights with the best validation score are retained. In the ten "
                        f"folds, training stopped after {min(dnn_epochs)} to {max(dnn_epochs)} epochs. "
                        "\n\n"
                        "The choice of these three specific model architectures was motivated by the desire to "
                        "compare fundamentally different algorithmic paradigms applied to the same prediction "
                        "task. Random Forest represents the bagging paradigm, where multiple models are trained "
                        "independently on bootstrap samples and their predictions are aggregated through "
                        "averaging. Gradient Boosting represents the boosting paradigm, where models are trained "
                        "sequentially with each new model focusing on correcting the errors of the previous "
                        "ensemble. The Deep Neural Network represents the connectionist paradigm, where a "
                        "parametric function composed of multiple nonlinear transformation layers is optimized "
                        "end-to-end using gradient descent. By comparing these three paradigms on identical data "
                        "with identical features and evaluation protocol, the study provides a fair comparison "
                        "of the algorithmic approaches for the peak pressure prediction task. "
                        "\n\n"
                        "For RF, the number of trees (200) is large enough for the ensemble prediction to be "
                        "stable, and the maximum depth (12) balances model complexity against overfitting. For "
                        "GBR, the lower maximum depth (6) and the learning rate (0.1) follow the established "
                        "principle that boosting performs best with shallow individual trees and conservative "
                        "step sizes. For the DNN, the three-layer architecture (128-64-32) provides a gradually "
                        "narrowing representation that encourages hierarchical feature abstractions while keeping "
                        "the number of trainable parameters moderate. Because the hyperparameters were not "
                        "optimized, the comparison reflects the performance of reasonable default configurations "
                        "rather than of the best achievable configuration of each model class. "
                    ),
                },
                {
                    "heading": "Reference estimators and validation protocol",
                    "content": (
                        "Two training-free estimators provide reference points that use the same observation "
                        "window but no learning. The first is the observed short-record peak, i.e., the minimum Cp "
                        "of the observation window used directly as the prediction (the input feature min_cp). The "
                        "second is a Gaussian peak-factor estimate, Cp_peak = Cp_mean - g Cp_rms, with the mean and "
                        "RMS of the observation window and the peak factor g = sqrt(2 ln(nu T)) + 0.5772 / "
                        "sqrt(2 ln(nu T)), which is the classical expected-extreme formula for a stationary "
                        "Gaussian process; nu is the mean zero-upcrossing rate of the fluctuating pressure in the "
                        f"observation window, T = {T_PRED:.3f} s is the duration of the prediction window, and nu T "
                        "is bounded below by e. The median peak factor obtained in this way is "
                        f"{gauss_g_median:.2f}. Neither estimator uses the spatial-directional features or any "
                        f"training data, and both are evaluated on all {N_SAMPLES:,} samples. "
                        "\n\n"
                        "Model performance was evaluated with 10-fold cross-validation grouped by wind direction. "
                        "Fold k (k = 0, ..., 9) contains every record of the wind directions theta = 10k, "
                        "10k + 100, 10k + 200, and (for k < 6) 10k + 300 degrees, so that each fold holds out "
                        f"{fold_sizes[0]} or {fold_sizes[-1]} complete wind directions and every held-out direction "
                        "has both neighbouring directions (theta +/- 10 degrees) in the training set. Grouping by "
                        "wind direction ensures that records from the same simulated wind tunnel run, which share "
                        "the run-level turbulence factor, never appear in both the training and the test set; a "
                        "random split of individual records would allow the models to infer the run-level "
                        "conditions of a test record from other taps of the same run. The protocol is not "
                        "stratified by the target value. Random seeds are fixed (42) for the data generator, the "
                        "models, and all resampling procedures. "
                        "\n\n"
                        "Three performance metrics were computed: the coefficient of determination (R-squared), "
                        "which measures the proportion of variance explained by the model; the root mean square "
                        "error (RMSE), which penalizes large errors more heavily due to the squaring operation "
                        "and is therefore sensitive to outlier predictions; and the mean absolute error (MAE), "
                        "which provides a robust measure of average prediction deviation that is less influenced "
                        "by occasional large errors. The metrics are reported pooled over all out-of-fold "
                        "predictions (Table 1) and as the mean and standard deviation over the ten folds (Table 2 "
                        "and Fig. 5). Differences between models were tested with the two-sided Wilcoxon "
                        f"signed-rank test applied to the paired per-direction mean squared errors ({N_DIR} pairs), "
                        "and 95% confidence intervals of the RMSE were obtained by bootstrap resampling of the "
                        f"{N_DIR} wind directions (2,000 resamples). A significance level of 0.05 is used "
                        "throughout. "
                        "\n\n"
                        "Leakage was examined with three controls in addition to the disjoint window design: (a) a "
                        f"guard-gap test, in which the target window starts {GUARD} samples after the end of the "
                        "observation window, so that any information carried across the window boundary by the "
                        "continuity of the signal is removed; (b) a negative control, in which the RF is trained "
                        "and evaluated on randomly permuted targets under the same protocol and should show no "
                        "skill; and (c) a deliberately leaky reference formulation, in which the same features and "
                        "the target are both computed from the full record, to illustrate how strongly leakage "
                        "inflates the scores. The information contributed by the short record was isolated with a "
                        "feature ablation, in which RF models were trained on the four spatial-directional features "
                        "only and on the ten statistical and spectral features only. Feature importance was "
                        "assessed for RF with two methods: the mean decrease in impurity of an RF trained on all "
                        "samples, and the permutation importance on each held-out fold (five repetitions per "
                        "feature, measured as the decrease in R-squared), averaged over the ten folds. The error "
                        "analysis comprises the per-direction RMSE, the D'Agostino-Pearson normality test and "
                        "Spearman rank correlations for the residuals, and the two-sample Kolmogorov-Smirnov test "
                        "for the distributions of measured and predicted peaks. "
                    ),
                },
            ],
        },
        {
            "heading": "RESULTS AND DISCUSSION",
            "content": "",
            "subsections": [
                {
                    "heading": "Overall model performance",
                    "content": (
                        "Table 1 summarizes the out-of-fold prediction performance of the three ML models and "
                        f"the two training-free estimators. {model_claim[0].upper() + model_claim[1:]}. "
                        f"{significance_sentence}"
                        "\n\n"
                        f"{baseline_claim}. The mean residual (measured minus estimated peak) is "
                        f"{bias[GAUSS]:.3f} for the Gaussian peak-factor estimate and {bias[OBSPK]:.3f} for the "
                        "observed short-record peak"
                        + (
                            ", i.e., both training-free estimators underestimate the suction magnitude on "
                            "average. The observed short-record peak is biased because a window that is "
                            f"{duration_ratio:.0f} times shorter than the record samples fewer extreme "
                            "excursions, and the Gaussian estimate is biased because the intermittent suction "
                            "events at edge and corner taps make the lower tail of the pressure distribution "
                            "heavier than Gaussian. The ML models learn corrections for both effects from the "
                            "training data. "
                            if bias[GAUSS] < 0 and bias[OBSPK] < 0
                            else ". "
                        )
                        + "\n\n"
                        "Fig. 4 presents the measured versus predicted peak Cp values for the three models, "
                        "with the identity line shown for reference. The out-of-fold predictions of all models "
                        "cluster around the identity line over most of the range, with R-squared between "
                        f"{r2_lo:.4f} and {r2_hi:.4f}"
                        + (
                            ", and the scatter widens toward the most severe peaks. "
                            if severe_rmse[best] > rest_rmse_best
                            else ". "
                        )
                        + f"{severe_sentence}"
                        "\n\n"
                        "Table 2 presents the mean and standard deviation of the performance metrics over the "
                        f"ten folds, and the box plots in Fig. 5 show their distributions. {stability_sentence}"
                        "\n\n"
                        f"{model_discussion}"
                        "\n\n"
                        "A direct numerical comparison with previous ML studies of roof pressures is not "
                        "meaningful, because those studies predict different quantities from different inputs; "
                        "for example, Huang et al. (2023) predicted mean and RMS wind pressure coefficients, "
                        "which are time-averaged statistics, whereas the present target is the extreme of a single "
                        "realization over an unobserved interval and contains an irreducible random component. "
                        "The reported accuracies of the studies reviewed in the Introduction should therefore not "
                        "be compared with the values in Table 1. "
                        + (
                            "The advantage of all three ML models over the training-free estimators parallels "
                            "the finding of Weng and Paal (2022) that ML models significantly outperformed "
                            "traditional empirical approaches for wind pressure estimation on non-isolated "
                            "low-rise buildings, although the tasks differ. "
                            if all_beat_baselines
                            else ""
                        )
                        + "The closest related problem in the cited literature is the deep learning-based "
                        "extension of wind pressure time series by Tong et al. (2024); the present approach "
                        "differs in that it predicts the peak value directly from 14 scalar features instead of "
                        "synthesizing the missing part of the record. "
                        "\n\n"
                        "An important distinction between the present study and many prior works is the use of "
                        "features derived from a measured signal. While most existing studies rely primarily on "
                        "geometric and directional parameters as inputs (building dimensions, roof slope, wind "
                        "direction), the present models also receive statistical and spectral features of a short "
                        "record. The inclusion of spectral features is motivated by the relationship between the "
                        "frequency content of pressure signals and the underlying flow mechanisms, which the "
                        "synthetic model represents by a lower fluctuation bandwidth in separated flow. The value "
                        "of the short-record features, relative to position and direction alone, is quantified by "
                        "the ablation study in the next subsection. "
                        "\n\n"
                        "The computational cost of the trained models is small: the RF pipeline predicts the peak "
                        f"Cp of all {N_SAMPLES:,} records in {t_predict:.2f} s in the present implementation. The "
                        "more relevant saving lies in the measurement, because the predictions require only the "
                        f"observation window, i.e., 1/{duration_ratio:.0f} of the record duration, once a training "
                        "set of full-length records is available. A single wind tunnel test for a low-rise "
                        "building model typically requires several hours of facility time across multiple wind "
                        "directions, and computational fluid dynamics (CFD) simulations using large-eddy simulation "
                        "(LES) can require 24-72 hours of computation on high-performance computing clusters for a "
                        "single wind direction, so a reduction of the required record duration translates "
                        "directly into savings when many configurations or wind directions must be screened. "
                        "\n\n"
                        "It should be noted, however, that the ML model is a surrogate trained on a specific set "
                        "of records, here synthetic ones, and cannot replace wind tunnel testing for final design "
                        "verification. The model's validity is inherently limited to the range of building "
                        "configurations, terrain conditions, and wind characteristics present in the training "
                        "data. Extrapolation beyond these conditions should be approached with caution, and the "
                        "uncertainty of predictions should be quantified and communicated to the design engineer. "
                        "Physics-informed ML approaches, as explored by Zhu et al. (2023), offer a promising "
                        "pathway to improve extrapolation reliability by incorporating known physical constraints "
                        "into the learning process. "
                    ),
                },
                {
                    "heading": "Leakage controls and information content of the short record",
                    "content": (
                        f"{leakage_paragraph}"
                        "\n\n"
                        f"{ablation_paragraph}"
                        "\n\n"
                        "Together, these results indicate that the accuracy reported in Table 1 reflects genuine "
                        f"predictive skill: the best model explains {overall[best]['R2']:.1%} of the variance of "
                        f"the peak over the unobserved {PRED_PCT:.1f}% of each record. In the synthetic records, "
                        "the suction events in the prediction window are statistically independent of those in "
                        "the observation window, given the record-level turbulence and intermittency parameters, "
                        "so their timing and amplitude cannot be anticipated from any feature of the observation "
                        "window; this component sets a lower bound on the achievable error. What the short record "
                        "provides is information about the record-level parameters, namely the local mean, the "
                        "fluctuation intensity, and the rate of intermittent events, which determine the "
                        "distribution from which the unobserved peak is drawn. "
                    ),
                },
                {
                    "heading": "Feature importance analysis",
                    "content": (
                        "Fig. 3 presents the RF feature importance obtained with two methods: (a) the mean "
                        "decrease in impurity of the RF trained on all samples and (b) the permutation importance "
                        "on the held-out folds. The impurity-based ranking places "
                        + fmt_list(
                            "the " + label(f, importances[FEATURES.index(f)]) for f in top3_imp
                        )
                        + f" at the top, and the {dominant_group} features together account for "
                        f"{group_share[dominant_group]:.1%} of the total importance. "
                        f"{importance_interpretation}"
                        "\n\n"
                        f"{importance_consistency}"
                        "\n\n"
                        f"{spectral_paragraph}"
                        "\n\n"
                        f"{spatial_paragraph}"
                        "\n\n"
                        "These findings have practical implications for sensor deployment and feature selection. "
                        f"In the present setting, the {fmt_list(label(f) for f in top3_imp)} provide most of the "
                        "information, so a reduced feature set built around these statistics would probably retain "
                        "most of the accuracy; this was not tested here, and retaining the full set is inexpensive "
                        "because all features are computed from the same short window. The results also suggest "
                        "avenues for feature engineering: features that capture the temporal clustering of extreme "
                        "pressure events, such as the number or the mean duration of threshold exceedances in the "
                        "observation window, could provide additional information about the intermittency that "
                        "controls the unobserved peak. "
                    ),
                },
                {
                    "heading": "Wind direction dependency",
                    "content": (
                        "Fig. 6 presents the out-of-fold RMSE of the three models as a function of wind direction "
                        "in a polar plot. Because the folds are grouped by wind direction, each point is the error "
                        "for a direction that was entirely absent from the training data of the corresponding "
                        f"model. For {best_ab}, the per-direction RMSE ranges from {rb.min():.4f} (theta = "
                        f"{WIND_ANGLES[k_min]} degrees) to {rb.max():.4f} (theta = {WIND_ANGLES[k_max]} degrees), "
                        f"with a coefficient of variation of {dir_cov:.2f} across the {N_DIR} directions. "
                        f"{direction_pattern}{direction_cause}"
                        "\n\n"
                        "For wind normal to a roof edge, the flow separates at the leading edge and reattaches "
                        "downstream, producing a well-defined separation bubble. Oblique wind directions generate "
                        "conical (delta-wing) vortices at the windward roof corner, producing highly localized "
                        "peak suction with large spatial and temporal variability. In the synthetic model this "
                        "mechanism enters through the corner term C, which increases the mean suction and the "
                        "separation intensity, and hence the RMS and the rate of intermittent suction events, near "
                        "the windward corner at oblique incidence. "
                        + (
                            "The elevated absolute errors at oblique angles therefore reflect the greater severity "
                            "and intermittency of the corner peaks, which are inherently more difficult to predict "
                            "from a short record. "
                            if p_obl < 0.05 and rho_obl > 0
                            else "Despite this mechanism, the per-direction errors show no significant "
                            "systematic increase at oblique angles. "
                        )
                        + "\n\n"
                        "A single model was trained for all wind directions. Under the grouped protocol, a "
                        "held-out direction is predicted from training data of other directions, including its "
                        "neighbours at +/- 10 degrees, so the results measure the generalization across wind "
                        "directions for a single building; a single model avoids the data fragmentation of "
                        "direction-specific models. Whether the accuracy is sufficient for design depends on the "
                        "required tolerance, which should be judged against the residual distribution in the next "
                        "subsection rather than against the average error alone. "
                        "\n\n"
                        "The wind direction is encoded by two continuous variables, sin(theta) and cos(theta), "
                        "rather than by a discrete categorical variable or the raw angle. This representation "
                        "preserves the angular continuity and symmetry of the wind direction, so that nearby wind "
                        "directions (e.g., 0 degrees and 350 degrees) are represented as nearby points in the "
                        "feature space, which is a prerequisite for smooth interpolation between measured wind "
                        "directions. Alternative encoding schemes, such as one-hot encoding of discretized wind "
                        "sectors, would lose this continuity property. "
                        + (
                            "In the present results, however, the explicit direction features carry little "
                            "importance once the short-record statistics are available, so the encoding has "
                            "little influence on the accuracy reported here. "
                            if group_share["spatial-directional"] < 0.05
                            else ""
                        )
                        + "\n\n"
                        "The 10-degree angular resolution of the dataset determines how far a held-out direction "
                        "is from the nearest training direction. A finer resolution (e.g., 5-degree intervals) "
                        "would reduce this distance and increase the dataset size; its effect on the accuracy was "
                        "not examined in this study. "
                    ),
                },
                {
                    "heading": "Residual analysis",
                    "content": (
                        f"Fig. 7 presents the residual analysis for the {best} model, which has the lowest "
                        f"out-of-fold RMSE. {residual_shape}{interval_sentence}"
                        "\n\n"
                        + (
                            "The heavier lower tail corresponds to the most severe peaks: "
                            if res_skew < 0 and severe_bias[best] < 0
                            else ""
                        )
                        + f"for the 5% most severe measured peaks, the mean residual of {best_ab} is "
                        f"{severe_bias[best]:.3f}, i.e., the suction magnitude is "
                        f"{'underestimated' if severe_bias[best] < 0 else 'overestimated'} on "
                        "average. These extreme values occur at roof corner and edge taps, where several "
                        "intermittent suction events in the prediction window can coincide with a negative "
                        "excursion of the Gaussian component. Their timing and amplitude are random and cannot be "
                        "inferred from the observation window, so a model that predicts the conditional mean of "
                        "the peak necessarily regresses toward the mean for the most extreme realizations. "
                        "\n\n"
                        "Fig. 8 compares the probability density distributions of the measured and "
                        f"{best_ab}-predicted peak Cp values. {distribution_sentence}{dispersion_sentence}"
                        "The predicted values "
                        "should therefore be interpreted as estimates of the expected peak, and an allowance for "
                        "the random component is required for design. "
                        "\n\n"
                        "The practical significance of these results is twofold. First, the small mean residual "
                        "shows that the model is unbiased on average, which is a prerequisite for applications in "
                        "which both overestimation and underestimation of peak pressures should be avoided. "
                        + (
                            "Second, the heavy lower tail and the growth of the errors with the severity of the "
                            "predicted peak mean that a single symmetric prediction interval based on the "
                            "residual standard deviation is inadequate for the most exposed taps. Prediction "
                            "intervals should be conditioned on the predicted value or on the flow regime, for "
                            "example with quantile regression forests or conformal prediction, before the "
                            "predictions are used in reliability-based design. "
                            if (not residuals_normal) and heteroscedastic
                            else "Second, the residual distribution allows prediction intervals to be constructed "
                            "from the residual standard deviation, which supports reliability-based design. "
                        )
                    ),
                },
            ],
        },
        {
            "heading": "CONCLUSIONS",
            "content": (
                "This study developed and compared three machine learning models for predicting the peak wind "
                "pressure coefficient on a low-rise flat roof over the unobserved part of a pressure record from a "
                f"short observation window (the first {OBS_PCT:.1f}% of the record). Fourteen statistical, "
                "spectral, and spatial-directional features were computed from synthetic TPU-style records, and "
                f"the models were evaluated with 10-fold cross-validation grouped by wind direction on {N_SAMPLES:,} "
                f"samples spanning {N_DIR} wind directions. The following conclusions are drawn: "
                "\n\n"
                f"1. {model_claim}; {model_verdict} (paired Wilcoxon tests over the {N_DIR} wind "
                "directions). "
                + (
                    "The differences between the models are small compared with their common advantage over "
                    "the training-free estimators, so the choice between them can be guided by "
                    "interpretability and computational cost as well as accuracy. "
                    if all_beat_baselines and ml_spread < ml_margin
                    else "The model ranking should be interpreted together with the training-free reference "
                    "estimators, which are competitive in this setting. "
                )
                + "\n\n"
                f"2. {baseline_claim}. "
                + (
                    "The Gaussian peak-factor extrapolation underestimates the suction magnitude on average "
                    f"(mean residual {bias[GAUSS]:.3f}), consistent with the non-Gaussian, intermittent "
                    "character of the pressures at edge and corner taps. "
                    if bias[GAUSS] < 0
                    else f"The mean residual of the Gaussian peak-factor extrapolation is {bias[GAUSS]:.3f}. "
                )
                + "\n\n"
                "3. The prediction task is leakage-free: no input is computed from the prediction window, a "
                f"guard gap of {GUARD} samples changes the RF R-squared by {-guard_delta:+.4f}, and permuted "
                f"targets yield R-squared = {perm_control_r2:.3f}. A leaky formulation in which features and "
                f"target come from the same full record yields R-squared = {leaky_metrics['R2']:.4f}, which "
                "illustrates how strongly leakage inflates apparent accuracy. "
                "\n\n"
                f"4. The {fmt_list(label(f) for f in top3_imp)} are the most important features "
                f"(impurity-based), and the {dominant_group} features carry {group_share[dominant_group]:.1%} of "
                "the total importance. "
                + (
                    f"Short-record features alone reach R-squared = {ablation['signal']['R2']:.4f}, compared "
                    f"with {ablation['spatial']['R2']:.4f} for tap position and wind direction alone, so the short "
                    "measurement carries most of the predictive information. "
                    if signal_beats_spatial
                    else f"Tap position and wind direction alone reach R-squared = {ablation['spatial']['R2']:.4f}, "
                    f"compared with {ablation['signal']['R2']:.4f} for the short-record features alone. "
                )
                + "\n\n"
                f"5. For {best_ab}, the per-direction RMSE ranges from {rb.min():.4f} to {rb.max():.4f} "
                f"(coefficient of variation {dir_cov:.2f}). "
                + (
                    "The errors are significantly larger at oblique wind directions, where the measured peaks "
                    "are more variable, but the dependence is no longer significant when the errors are "
                    "normalized by the spread of the peaks in each direction, so the relative accuracy of the "
                    "single all-direction model does not deteriorate significantly at oblique incidence. "
                    if p_obl < 0.05
                    and rho_obl > 0
                    and error_tracks_spread
                    and spread_tracks_obliqueness
                    else f"The association with wind obliqueness is rho = {rho_obl:.2f} ({fmt_p(p_obl)}). "
                )
                + "\n\n"
                f"{conclusion_residual}"
                "\n\n"
                "Limitations of this study include: (a) synthetic records from a stochastic model were used to "
                "demonstrate the framework, and the model omits features of measured records such as "
                "non-stationary inflow, spatial correlation of suction events between neighbouring taps, and "
                "Reynolds number effects, so validation with measured TPU records is needed to establish "
                "realistic accuracy levels; (b) only a single flat-roof geometry (nominal B/H = D/H = 1.5) was "
                "considered, and the framework should be extended to gable and hip roofs and other aspect ratios, "
                "for which the spatial-directional features would need to be generalized; (c) interference "
                "effects from neighbouring buildings were not included, although they can significantly modify "
                "the pressure distribution in urban settings; (d) the target is the peak of a single realization, "
                "which contains an irreducible random component, and expected peaks obtained by extreme value "
                "analysis would be more stable targets; (e) the observation window was always the first "
                f"{OBS_PCT:.1f}% of the record and stationary conditions were assumed, so the sensitivity to the "
                "window length and position and the behaviour under non-stationary events such as thunderstorm "
                "downbursts were not examined; and (f) the hyperparameters were not tuned, and a single "
                "terrain and inflow condition was represented. "
                "\n\n"
                "Future research should address these limitations through several avenues. First, the framework "
                "should be validated using measured TPU database records to establish realistic accuracy "
                "benchmarks. Second, the model should be extended to multiple building geometries using transfer "
                "learning techniques that leverage shared aerodynamic knowledge across configurations. Third, "
                "building interference effects should be incorporated, potentially using multi-task learning to "
                "simultaneously predict peak pressures for the target building and characterize the interference "
                "pattern. Fourth, physics-informed ML approaches should be explored to incorporate known physical "
                "constraints, such as the monotonic relationship between distance from the roof edge and mean "
                "suction intensity. Fifth, the framework should be integrated with building information modeling "
                "(BIM) systems to enable automated wind load assessment during the design process. Finally, "
                "uncertainty quantification through conformal prediction or quantile regression should be "
                "incorporated to provide prediction intervals on peak Cp predictions that reflect the "
                "heteroscedastic, heavy-tailed errors observed here, supporting risk-informed design decisions. "
            ),
        },
    ],
    "tables": [
        {
            "caption": (
                "Table 1. Out-of-fold prediction performance (10-fold CV grouped by wind direction) of the "
                "three ML models and two training-free estimators; RMSE 95% CI from bootstrap over wind "
                "directions."
            ),
            "headers": ["Estimator", "R-squared", "RMSE", "RMSE 95% CI", "MAE"],
            "rows": [
                [
                    name,
                    f"{all_metrics[name]['R2']:.4f}",
                    f"{all_metrics[name]['RMSE']:.4f}",
                    f"{rmse_ci[name][0]:.4f}-{rmse_ci[name][1]:.4f}",
                    f"{all_metrics[name]['MAE']:.4f}",
                ]
                for name in all_metrics
            ],
        },
        {
            "caption": "Table 2. Cross-validation statistics (mean +/- std over the 10 folds).",
            "headers": ["Model", "R-squared", "RMSE", "MAE"],
            "rows": [
                [
                    name,
                    f"{np.mean(cv_results[name]['R2']):.4f} +/- {np.std(cv_results[name]['R2']):.4f}",
                    f"{np.mean(cv_results[name]['RMSE']):.4f} +/- {np.std(cv_results[name]['RMSE']):.4f}",
                    f"{np.mean(cv_results[name]['MAE']):.4f} +/- {np.std(cv_results[name]['MAE']):.4f}",
                ]
                for name in models
            ],
        },
        {
            "caption": (
                f"Table 3. Feature engineering summary (14 input features; signal features from the first "
                f"{N_OBS} samples of each record)."
            ),
            "headers": ["Category", "Features", "Count", "Computed from"],
            "rows": [
                [
                    "Statistical",
                    ", ".join(FEATURE_GROUPS["statistical"]),
                    str(len(FEATURE_GROUPS["statistical"])),
                    f"Observation window ({N_OBS} samples)",
                ],
                [
                    "Spectral",
                    ", ".join(FEATURE_GROUPS["spectral"]),
                    str(len(FEATURE_GROUPS["spectral"])),
                    f"Welch PSD of the observation window ({WELCH_NPERSEG}-sample segments)",
                ],
                [
                    "Spatial/Directional",
                    ", ".join(FEATURE_GROUPS["spatial-directional"]),
                    str(len(FEATURE_GROUPS["spatial-directional"])),
                    "Tap layout and wind direction",
                ],
            ],
        },
        {
            "caption": "Table 4. Dataset parameters.",
            "headers": ["Parameter", "Value"],
            "rows": [
                ["Data source", "Synthetic TPU-style records (stochastic model, Section 2)"],
                ["Building model (nominal)", "Flat roof, H=16m, B=D=24m"],
                ["Pressure taps", f"{N_TAPS} (12x12 grid)"],
                ["Wind directions", f"{N_DIR} (0-350 deg, 10-deg interval)"],
                ["Sampling frequency", f"{FS:.0f} Hz"],
                ["Time steps per record", f"{N_TIME}"],
                ["Observation window (features)", f"First {N_OBS} samples ({OBS_PCT:.1f}%)"],
                ["Prediction window (target)", f"Last {N_PRED} samples ({PRED_PCT:.1f}%)"],
                ["Total samples", f"{N_SAMPLES}"],
                [
                    "Cross-validation",
                    f"10 folds grouped by wind direction ({fold_sizes[0]}-{fold_sizes[-1]} directions per fold)",
                ],
            ],
        },
    ],
    "references": [
        "[1] F. Bre, J.M. Gimenez, V.D. Fachinotti, Prediction of wind pressure coefficients on building surfaces using artificial neural networks, Energy Build. 158 (2018) 1429-1441. https://doi.org/10.1016/j.enbuild.2017.11.045",
        "[2] J. Tian, K.R. Gurley, M.T. Diaz, P.L. Fernandez-Caban, F.J. Masters, R. Fang, Low-rise gable roof buildings pressure prediction using deep neural networks, J. Wind Eng. Ind. Aerodyn. 196 (2020) 104026. https://doi.org/10.1016/j.jweia.2019.104026",
        "[3] G. Hu, L. Liu, D. Tao, J. Song, K.T. Tse, K.C.S. Kwok, Deep learning-based investigation of wind pressures on tall building under interference effects, J. Wind Eng. Ind. Aerodyn. 201 (2020) 104138. https://doi.org/10.1016/j.jweia.2020.104138",
        "[4] X.J. Wang, Q.S. Li, B.W. Yan, Full-scale measurements of wind pressures on a low-rise building during typhoons and comparison with wind tunnel test results and aerodynamic database, J. Struct. Eng. 146 (10) (2020) 04020196. https://doi.org/10.1061/(ASCE)ST.1943-541X.0002769",
        "[5] B. Kim, N. Yuvaraj, K.R. Sri Preethaa, G. Hu, D.-E. Lee, Wind-induced pressure prediction on tall buildings using generative adversarial imputation network, Sensors 21 (7) (2021) 2515. https://doi.org/10.3390/s21072515",
        "[6] Y. Weng, S.G. Paal, Machine learning-based wind pressure prediction of low-rise non-isolated buildings, Eng. Struct. 258 (2022) 114148. https://doi.org/10.1016/j.engstruct.2022.114148",
        "[7] Z. Ding, W. Zhang, D. Zhu, Neural-network based wind pressure prediction for low-rise buildings with genetic algorithm and Bayesian optimization, Eng. Struct. 260 (2022) 114203. https://doi.org/10.1016/j.engstruct.2022.114203",
        "[8] D.P.P. Meddage, I.U. Ekanayake, A.U. Weerasuriya, C.S. Lewangamage, K.T. Tse, T.P. Miyanawala, C.D.E. Ramanayaka, Explainable Machine Learning (XML) to predict external wind pressure of a low-rise building in urban-like settings, J. Wind Eng. Ind. Aerodyn. 226 (2022) 105027. https://doi.org/10.1016/j.jweia.2022.105027",
        "[9] K. Mostafa, I. Zisis, M.A. Moustafa, Machine learning techniques in structural wind engineering: A state-of-the-art review, Appl. Sci. 12 (10) (2022) 5232. https://doi.org/10.3390/app12105232",
        "[10] T. Wu, R. Snaiki, Applications of machine learning to wind engineering, Front. Built Environ. 8 (2022) 811460. https://doi.org/10.3389/fbuil.2022.811460",
        "[11] F. Chen, W. Kang, Z. Shu, Q. Li, Y. Li, Y.F. Chen, K. Zhou, Predicting roof-surface wind pressure induced by conical vortex using a BP neural network combined with POD, Build. Simul. 15 (8) (2022) 1475-1490. https://doi.org/10.1007/s12273-021-0867-7",
        "[12] Y. Huang, G. Ou, J. Fu, H. Zhang, Prediction of mean and RMS wind pressure coefficients for low-rise buildings using deep neural networks, Eng. Struct. 274 (2023) 115149. https://doi.org/10.1016/j.engstruct.2022.115149",
        "[13] E. Shelley, E. Hubbard, W. Zhang, Comparison and uncertainty quantification of roof pressure measurements using the NIST and TPU aerodynamic databases, J. Wind Eng. Ind. Aerodyn. 232 (2023) 105246. https://doi.org/10.1016/j.jweia.2022.105246",
        "[14] H. Wu, Y. Chen, P. Xie, D. Zhou, T. Tamura, Y. Cao, Sparse-measurement-based peak wind pressure evaluation by super-resolution convolutional neural networks, J. Wind Eng. Ind. Aerodyn. 242 (2023) 105574. https://doi.org/10.1016/j.jweia.2023.105574",
        "[15] Q. Zhu, Z. Zhao, J. Yan, Physics-informed machine learning for surrogate modeling of wind pressure and optimization of pressure sensor placement, Comput. Mech. 71 (2023) 481-491. https://doi.org/10.1007/s00466-022-02251-1",
        "[16] Y. Huang, H. Wu, J. Fu, H. Zhang, H. Li, Convolutional neural network-based wind pressure prediction on low-rise buildings, Eng. Struct. 309 (2024) 118078. https://doi.org/10.1016/j.engstruct.2024.118078",
        "[17] L.-S. An, S. Jung, Data-driven prediction of wind pressure on low-rise buildings in complex heterogeneous terrains, Build. Environ. 265 (2024) 112022. https://doi.org/10.1016/j.buildenv.2024.112022",
        "[18] B. Yan, W. Ding, Z. Jin, L. Zhang, L. Wang, M. Du, Q. Yang, Y. He, Explainable machine learning-based prediction for aerodynamic interference of a low-rise building on a high-rise building, J. Build. Eng. 82 (2024) 108285. https://doi.org/10.1016/j.jobe.2023.108285",
        "[19] B. Tong, Y. Liang, J. Song, G. Hu, A. Kareem, Deep learning-based extension of wind pressure time series, J. Wind Eng. Ind. Aerodyn. 254 (2024) 105909. https://doi.org/10.1016/j.jweia.2024.105909",
        "[20] F.M. Nav, S.F. Mirfakhar, R. Snaiki, A hybrid machine learning framework for wind pressure prediction on buildings with constrained sensor networks, Comput.-Aided Civ. Infrastruct. Eng. (2025). https://doi.org/10.1111/mice.13488",
    ],
    "figure_captions": [
        "Fig. 1. Full-record mean Cp on the roof surface for three wind directions (arrows: flow direction).",
        (
            f"Fig. 2. Cp time histories (shaded: {N_OBS}-sample observation window; marker: target peak) "
            "and power spectral densities of the corner and center taps at 0 deg."
        ),
        (
            "Fig. 3. Feature importance for peak Cp prediction: (a) RF impurity-based importance; "
            "(b) RF permutation importance on the held-out folds."
        ),
        "Fig. 4. Measured vs. out-of-fold predicted peak Cp over the prediction window for three ML models.",
        "Fig. 5. Box plots of the 10-fold (wind-direction-grouped) cross-validation metrics for three ML models.",
        "Fig. 6. Out-of-fold RMSE of the three ML models for each held-out wind direction.",
        f"Fig. 7. Residual analysis for the {best} model (lowest out-of-fold RMSE).",
        f"Fig. 8. Distribution comparison of measured and {best_ab}-predicted peak Cp values.",
    ],
    "conflict_of_interest": (
        "The authors declare that they have no known competing financial interests or personal "
        "relationships that could have appeared to influence the work reported in this paper."
    ),
    "data_availability": (
        "The wind pressure data used in this study were generated synthetically with the stochastic "
        "model described in Section 2 to demonstrate the proposed framework; the generating script "
        "reproduces them exactly (fixed random seed). The TPU aerodynamic database is publicly available "
        "at https://db.wind.arch.t-kougei.ac.jp/."
    ),
    "acknowledgments": (
        "The authors acknowledge the Tokyo Polytechnic University for making the aerodynamic "
        "database publicly available."
    ),
}

# ── Generate Word document ───────────────────────────────────────────────────
print("\n[Step 5] Generating Word document...")
papers_dir = os.path.join(OUTPUT_DIR, "paper")
os.makedirs(papers_dir, exist_ok=True)
word_path = generate_word(paper_content, "jweia", figures=figure_paths, output_dir=papers_dir)
print(f"  Word: {word_path}")

# ── Quality Check ────────────────────────────────────────────────────────────
print("\n[Quality Check]")
print("-" * 40)
result = check_paper(paper_content, "jweia", figures=figure_paths)
print(result["summary"])

# Save quality report
report_path = os.path.join(OUTPUT_DIR, "quality_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Report saved: {report_path}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
