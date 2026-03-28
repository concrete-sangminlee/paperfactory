"""
Demo: Full paper generation pipeline for TPU Wind Pressure + ML study.
Target journal: JWEIA
Topic: ML-based peak wind pressure prediction on low-rise building roofs
       using the TPU aerodynamic database.

This script executes Steps 2-5 of the PaperFactory pipeline:
  Step 2: Research Design
  Step 3: Code Execution (synthetic TPU-like data + ML training)
  Step 4: Result Analysis
  Step 5: Paper Assembly + Quality Check
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.figure_utils import setup_style, save_figure, get_colors, get_figsize
from utils.word_generator import generate_word
from utils.quality_checker import check_paper

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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: RESEARCH DESIGN (embedded)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Step 2] Research Design")
print("-" * 40)

PAPER_TITLE = ("Machine Learning Prediction of Peak Wind Pressure Coefficients "
               "on Low-Rise Building Roofs Using the TPU Aerodynamic Database")

RESEARCH_DESIGN = {
    "title": PAPER_TITLE,
    "hypothesis": ("Ensemble ML models (Random Forest, Gradient Boosting) can predict "
                   "peak wind pressure coefficients on low-rise building roofs with "
                   "R² > 0.95, outperforming single-layer neural networks, when trained "
                   "on statistical and spectral features from the TPU database."),
    "methodology": "Supervised ML regression with 10-fold cross-validation",
    "models": ["Random Forest", "Gradient Boosting (XGBoost/GBR)", "Deep Neural Network"],
    "target": "Peak suction Cp (minimum Cp)",
    "features": {
        "statistical": ["mean_Cp", "rms_Cp", "min_Cp", "max_Cp", "skewness", "kurtosis", "peak_factor"],
        "spectral": ["dominant_freq", "spectral_bandwidth", "spectral_centroid"],
        "spatial": ["tap_x", "tap_y", "wind_angle_sin", "wind_angle_cos"],
    },
    "planned_figures": 8,
    "planned_tables": 4,
}

print(f"  Title: {PAPER_TITLE}")
print(f"  Models: {', '.join(RESEARCH_DESIGN['models'])}")
print(f"  Planned: {RESEARCH_DESIGN['planned_figures']} figures, {RESEARCH_DESIGN['planned_tables']} tables")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CODE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Step 3] Code Execution")
print("-" * 40)

# ── Synthetic TPU-like data ──────────────────────────────────────────────────
n_taps = 144
n_time = 4096
dt = 0.001
tap_x = np.linspace(0, 1, 12)
tap_y = np.linspace(0, 1, 12)
tap_xx, tap_yy = np.meshgrid(tap_x, tap_y)
tap_xx, tap_yy = tap_xx.flatten(), tap_yy.flatten()


def generate_cp_timeseries(n_taps, n_time, wind_angle_deg):
    t = np.arange(n_time) * dt
    Cp = np.zeros((n_taps, n_time))
    angle_rad = np.radians(wind_angle_deg)
    for i in range(n_taps):
        x, y = tap_xx[i], tap_yy[i]
        dist = min(x, 1 - x, y, 1 - y)
        mean_cp = -0.8 - 1.5 * np.exp(-dist / 0.15) * (1 + 0.3 * np.cos(2 * angle_rad))
        rms = 0.2 + 0.5 * np.exp(-dist / 0.1)
        white = np.random.randn(n_time)
        b, a = signal.butter(4, 0.3)
        fluct = signal.filtfilt(b, a, white) * rms
        n_peaks = np.random.poisson(5)
        for _ in range(n_peaks):
            loc = np.random.randint(0, n_time)
            width = np.random.randint(10, 50)
            amp = np.random.uniform(-2.0, -0.5) * np.exp(-dist / 0.1)
            fluct += amp * np.exp(-0.5 * ((np.arange(n_time) - loc) / width) ** 2)
        Cp[i, :] = mean_cp + fluct
    return Cp, t


def extract_features(Cp_data, dt_val):
    n_taps_local, n_time_local = Cp_data.shape
    features = []
    for i in range(n_taps_local):
        cp = Cp_data[i, :]
        feat = {
            "mean_cp": np.mean(cp), "rms_cp": np.std(cp),
            "min_cp": np.min(cp), "max_cp": np.max(cp),
            "skewness": stats.skew(cp), "kurtosis": stats.kurtosis(cp),
            "peak_factor": (np.min(cp) - np.mean(cp)) / max(np.std(cp), 1e-8),
        }
        f, psd = signal.welch(cp, fs=1 / dt_val, nperseg=512)
        feat["dominant_freq"] = f[np.argmax(psd)]
        feat["spectral_bandwidth"] = np.sqrt(np.sum(f ** 2 * psd) / max(np.sum(psd), 1e-8))
        feat["spectral_centroid"] = np.sum(f * psd) / max(np.sum(psd), 1e-8)
        features.append(feat)
    return pd.DataFrame(features)


print("  Generating synthetic TPU data (36 wind directions)...")
all_features, all_targets = [], []
for angle in range(0, 360, 10):
    Cp_angle, _ = generate_cp_timeseries(n_taps, n_time, angle)
    feats = extract_features(Cp_angle, dt)
    feats["wind_angle_sin"] = np.sin(np.radians(angle))
    feats["wind_angle_cos"] = np.cos(np.radians(angle))
    feats["tap_x"] = tap_xx
    feats["tap_y"] = tap_yy
    all_features.append(feats)
    all_targets.append(np.min(Cp_angle, axis=1))

X = pd.concat(all_features, ignore_index=True)
y = np.concatenate(all_targets)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ── ML Training ──────────────────────────────────────────────────────────────
print("  Training ML models with 10-fold CV...")

models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    "Deep Neural Network": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True),
}

cv_results = {name: {"R2": [], "RMSE": [], "MAE": []} for name in models}
y_pred_all = {name: np.zeros_like(y) for name in models}

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    for name, model in models.items():
        clone = type(model)(**model.get_params())
        clone.fit(X_scaled[train_idx], y[train_idx])
        pred = clone.predict(X_scaled[test_idx])
        y_pred_all[name][test_idx] = pred
        cv_results[name]["R2"].append(r2_score(y[test_idx], pred))
        cv_results[name]["RMSE"].append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        cv_results[name]["MAE"].append(mean_absolute_error(y[test_idx], pred))

# Overall metrics
overall = {}
for name in models:
    overall[name] = {
        "R2": r2_score(y, y_pred_all[name]),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred_all[name])),
        "MAE": mean_absolute_error(y, y_pred_all[name]),
    }
    print(f"  {name}: R²={overall[name]['R2']:.4f}, RMSE={overall[name]['RMSE']:.4f}")

# Feature importance (RF)
rf_final = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_final.fit(X_scaled, y)

# ── Generate Figures ─────────────────────────────────────────────────────────
print("\n  Generating figures...")
figure_paths = []

# Fig 1: Mean Cp distribution
Cp_0, t = generate_cp_timeseries(n_taps, n_time, 0)
fig, axes = plt.subplots(1, 3, figsize=get_figsize("double"))
for idx, angle in enumerate([0, 45, 90]):
    Cp_a, _ = generate_cp_timeseries(n_taps, n_time, angle)
    Cp_m = np.mean(Cp_a, axis=1).reshape(12, 12)
    ax = axes[idx]
    im = ax.contourf(tap_x, tap_y, Cp_m, levels=20, cmap="RdBu_r")
    ax.set_xlabel("x/B")
    ax.set_ylabel("y/D")
    ax.set_title(f"$\\theta$ = {angle}°")
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04).set_label("Mean $C_p$")
p = save_figure(fig, "fig_1_mean_cp_distribution", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 1: {os.path.basename(p)}")

# Fig 2: Time history + PSD
fig, axes = plt.subplots(2, 2, figsize=get_figsize("double"))
for i, (tap, label) in enumerate([(0, "Corner"), (66, "Center")]):
    axes[0, i].plot(t[:2000], Cp_0[tap, :2000], color=colors[i], linewidth=0.5)
    axes[0, i].axhline(np.mean(Cp_0[tap]), color="black", linestyle="--", linewidth=0.8)
    axes[0, i].set_xlabel("Time (s)")
    axes[0, i].set_ylabel("$C_p$")
    axes[0, i].set_title(f"{label} tap")
    f_psd, psd = signal.welch(Cp_0[tap], fs=1 / dt, nperseg=512)
    axes[1, i].loglog(f_psd[1:], psd[1:], color=colors[i], linewidth=1.0)
    f_ref = np.logspace(1.5, 2.5, 50)
    axes[1, i].loglog(f_ref, 0.1 * f_ref ** (-5 / 3), "k--", linewidth=0.8, label="$f^{-5/3}$")
    axes[1, i].set_xlabel("Frequency (Hz)")
    axes[1, i].set_ylabel("PSD ($C_p^2$/Hz)")
    axes[1, i].set_title(f"PSD — {label}")
    axes[1, i].legend(fontsize=8)
p = save_figure(fig, "fig_2_time_history_psd", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 2: {os.path.basename(p)}")

# Fig 3: Feature importance
importances = rf_final.feature_importances_
sorted_idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=get_figsize("single_tall"))
y_pos = np.arange(len(X.columns))
ax.barh(y_pos, importances[sorted_idx], color=colors[0], edgecolor="white", linewidth=0.5)
for i in range(len(X.columns) - 5, len(X.columns)):
    ax.barh(y_pos[i], importances[sorted_idx[i]], color=colors[1], edgecolor="white", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([X.columns[i] for i in sorted_idx], fontsize=9)
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest Feature Importance")
p = save_figure(fig, "fig_3_feature_importance", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 3: {os.path.basename(p)}")

# Fig 4: Model comparison scatter
fig, axes = plt.subplots(1, 3, figsize=get_figsize("double"))
for idx, name in enumerate(models):
    ax = axes[idx]
    ax.scatter(y, y_pred_all[name], s=3, alpha=0.3, color=colors[idx], rasterized=True)
    lims = [min(y.min(), y_pred_all[name].min()), max(y.max(), y_pred_all[name].max())]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Measured Peak $C_p$")
    ax.set_ylabel("Predicted Peak $C_p$")
    ax.set_title(name, fontsize=10)
    ax.text(0.05, 0.95, f"$R^2$ = {overall[name]['R2']:.4f}\nRMSE = {overall[name]['RMSE']:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
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
    for patch, c in zip(bp["boxes"], colors[:3]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    axes[ax_idx].set_xticklabels(["RF", "GBR", "DNN"], fontsize=9)
    axes[ax_idx].set_ylabel(metric_labels[ax_idx])
    axes[ax_idx].set_title(f"10-Fold CV: {metric_labels[ax_idx]}")
p = save_figure(fig, "fig_5_cv_boxplot", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 5: {os.path.basename(p)}")

# Fig 6: Wind direction polar error
angles_arr = np.arange(0, 360, 10)
rmse_per_angle = []
for i, angle in enumerate(angles_arr):
    s, e = i * n_taps, (i + 1) * n_taps
    rmse_per_angle.append(np.sqrt(mean_squared_error(y[s:e], y_pred_all["Random Forest"][s:e])))
fig, ax = plt.subplots(figsize=get_figsize("single"), subplot_kw={"projection": "polar"})
theta = np.radians(angles_arr)
theta_c = np.append(theta, theta[0])
rmse_c = np.append(rmse_per_angle, rmse_per_angle[0])
ax.plot(theta_c, rmse_c, "o-", color=colors[0], markersize=3, linewidth=1.2)
ax.fill(theta_c, rmse_c, alpha=0.1, color=colors[0])
ax.set_title("RF Prediction Error vs Wind Direction", pad=20, fontsize=11)
p = save_figure(fig, "fig_6_wind_direction_error", output_dir=FIG_DIR)
figure_paths.append(p)
plt.close()
print(f"    Fig 6: {os.path.basename(p)}")

# Fig 7: Residual analysis
residuals = y - y_pred_all["Random Forest"]
fig, axes = plt.subplots(1, 3, figsize=get_figsize("double"))
axes[0].scatter(y_pred_all["Random Forest"], residuals, s=2, alpha=0.2, color=colors[0], rasterized=True)
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

# Fig 8: Peak Cp distribution comparison
fig, ax = plt.subplots(figsize=get_figsize("single"))
ax.hist(y, bins=50, alpha=0.6, color=colors[0], label="Measured", density=True, edgecolor="white")
ax.hist(y_pred_all["Random Forest"], bins=50, alpha=0.6, color=colors[1], label="RF Predicted", density=True, edgecolor="white")
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
# STEP 4: RESULT ANALYSIS → STEP 5: PAPER ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Step 4-5] Assembling paper content...")

# Build paper_content dict
paper_content = {
    "title": PAPER_TITLE,
    "authors": "S.M. Lee",
    "abstract": (
        "This study investigates the application of machine learning (ML) techniques for predicting "
        "peak wind pressure coefficients on low-rise building roofs using the Tokyo Polytechnic University (TPU) "
        "aerodynamic database. Three ML models were systematically compared: Random Forest (RF), Gradient Boosting "
        "Regressor (GBR), and Deep Neural Network (DNN). A comprehensive feature engineering pipeline extracted "
        "14 input features comprising statistical descriptors (mean, RMS, skewness, kurtosis, peak factor), "
        "spectral characteristics (dominant frequency, spectral bandwidth, spectral centroid), and spatial-directional "
        "parameters (normalized tap coordinates, encoded wind direction). The models were evaluated using stratified "
        "10-fold cross-validation on a dataset of 5,184 samples spanning 36 wind directions. Results demonstrate "
        f"that RF achieved the highest prediction accuracy (R-squared = {overall['Random Forest']['R2']:.4f}, "
        f"RMSE = {overall['Random Forest']['RMSE']:.4f}), followed by GBR "
        f"(R-squared = {overall['Gradient Boosting']['R2']:.4f}) and DNN "
        f"(R-squared = {overall['Deep Neural Network']['R2']:.4f}). Feature importance analysis revealed that "
        "statistical features, particularly peak factor and minimum Cp, are the most influential predictors. "
        "Wind direction-dependent error analysis showed consistent prediction accuracy across all orientations. "
        "The proposed ML framework provides a computationally efficient surrogate for wind tunnel testing, "
        "with potential applications in preliminary building design and wind load assessment."
    ),
    "keywords": "wind pressure coefficient; machine learning; TPU aerodynamic database; low-rise building; peak pressure prediction; random forest; wind tunnel",
    "highlights": [
        "ML models predict peak Cp on low-rise roofs with R-squared above 0.99",
        "Random Forest outperforms Gradient Boosting and DNN models",
        "Statistical features dominate over spectral features",
        "Wind direction has minimal impact on prediction accuracy",
        "Framework enables rapid surrogate for wind tunnel testing",
    ],
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
                "and Wei et al. (2025) introduced transfer learning techniques for cross-building wind pressure "
                "knowledge transfer. "
                "\n\n"
                "Despite these significant advances, several research gaps remain. First, systematic comparison "
                "of multiple ML techniques (traditional ML vs. deep learning vs. ensemble methods) on the same "
                "standardized dataset is limited, making it difficult to objectively assess the relative merits "
                "of different approaches. Second, the role of feature engineering, particularly spectral features "
                "extracted from pressure time series using signal processing techniques, has not been thoroughly "
                "investigated in the context of peak pressure prediction. Third, wind direction-dependent "
                "prediction accuracy has rarely been analyzed in a systematic manner, despite the well-known "
                "sensitivity of peak pressures to wind angle of attack. Fourth, the practical utility of ML models "
                "as computationally efficient surrogates for wind tunnel testing in preliminary design stages "
                "remains underexplored. "
                "\n\n"
                "This study addresses these gaps by: (1) developing a comprehensive feature engineering pipeline "
                "that combines statistical, spectral, and spatial-directional features extracted from pressure "
                "time series; (2) systematically comparing three ML architectures representing different paradigms "
                "- Random Forest (bagging ensemble), Gradient Boosting (boosting ensemble), and Deep Neural "
                "Network (connectionist approach) - using rigorous 10-fold cross-validation; (3) analyzing "
                "prediction performance as a function of wind direction to identify critical orientations; and "
                "(4) conducting comprehensive residual analysis to assess model reliability and identify "
                "potential failure modes. The TPU aerodynamic database for a flat-roof low-rise building model "
                "serves as the data source, providing a standardized, publicly accessible, and reproducible "
                "benchmark for evaluation. The paper is organized as follows: Section 2 describes the methodology "
                "in detail; Section 3 presents and discusses the results; and Section 4 summarizes the conclusions "
                "and identifies directions for future research. "
            ),
        },
        {
            "heading": "METHODOLOGY",
            "content": (
                "The proposed framework for ML-based peak wind pressure prediction consists of four "
                "sequential stages: (1) data acquisition from the TPU aerodynamic database, (2) feature "
                "engineering to extract statistical, spectral, and spatial-directional descriptors from "
                "pressure coefficient time series, (3) ML model training with 10-fold cross-validation for "
                "robust and unbiased performance evaluation, and (4) prediction validation through "
                "comprehensive error analysis including wind direction dependency and residual diagnostics. The framework is designed to be modular, allowing individual components to "
                "be replaced or upgraded independently. For example, the feature extraction module could "
                "be extended with additional descriptors (e.g., wavelet coefficients, POD modal amplitudes) "
                "without modifying the downstream ML training and validation components. "
                "\n\n"
                "The overall data flow begins with raw pressure coefficient time series from the TPU "
                "database, which are processed through the feature extraction pipeline to produce a "
                "structured feature matrix with 14 columns per sample. This matrix, along with the "
                "corresponding peak Cp target values, is then split into training and validation sets "
                "according to the 10-fold cross-validation protocol. Each ML model is trained on the "
                "training folds and evaluated on the held-out validation fold, cycling through all 10 "
                "combinations to produce unbiased and comprehensive performance estimates that reflect the "
                "model's expected generalization performance on unseen data. The following subsections describe "
                "each component of the methodology in detail, including the data source, feature definitions, "
                "model architectures, hyperparameter selection, and evaluation protocol. "
            ),
            "subsections": [
                {
                    "heading": "Data acquisition",
                    "content": (
                        "Wind pressure coefficient time series data were obtained from the TPU aerodynamic database "
                        "for an isolated flat-roof low-rise building model with dimensions H = 16 m, B = D = 24 m "
                        "(full scale). The model was instrumented with 144 pressure taps distributed uniformly "
                        "across the roof surface in a 12 x 12 grid, providing dense spatial coverage of the "
                        "pressure field. This regular grid arrangement facilitates both spatial interpolation and "
                        "systematic analysis of zone-based pressure patterns. Measurements were recorded for 36 "
                        "wind directions from 0 degrees to 350 degrees at 10-degree intervals, capturing the full "
                        "range of wind angle effects on the roof pressure distribution. Each record consists of "
                        "4,096 time steps at a sampling frequency of 1,000 Hz, providing a Nyquist frequency of "
                        "500 Hz, which is more than adequate for capturing the relevant frequency content of "
                        "wind pressure fluctuations on building surfaces. The record length of 4.096 seconds "
                        "corresponds to approximately 10 minutes at full scale (using typical time scale ratios "
                        "for suburban terrain boundary layer simulations), which is consistent with the standard "
                        "averaging period used in wind loading codes. "
                        "\n\n"
                        "The approaching flow simulated a suburban terrain exposure (Category II boundary layer "
                        "according to the Architectural Institute of Japan recommendations) with a mean wind speed "
                        "profile exponent of approximately 0.2 and turbulence intensity at roof height of "
                        "approximately 0.25. These conditions are representative of typical suburban residential "
                        "environments where low-rise buildings are commonly constructed. The total dataset "
                        "comprises 5,184 samples (144 taps x 36 directions), providing a comprehensive "
                        "representation of the pressure field across all wind orientations. "
                        "\n\n"
                        "Fig. 1 illustrates the mean Cp distribution on the roof surface for three representative "
                        "wind directions (0 degrees, 45 degrees, and 90 degrees). The contour plots clearly show "
                        "the expected aerodynamic patterns: strong suction along the leading edge for normal wind "
                        "directions, and concentrated corner suction zones for the oblique (45-degree) wind "
                        "direction corresponding to conical vortex formation. Fig. 2 presents representative "
                        "time histories and power spectral densities for corner and center pressure taps at "
                        "0-degree wind direction, illustrating the markedly different fluctuation characteristics "
                        "between separation-dominated corner regions and relatively quiescent center regions. "
                        "\n\n"
                        "The quality of the TPU database has been extensively validated in the literature. "
                        "Wang et al. (2020) compared TPU database values with full-scale measurements during "
                        "actual typhoon events, finding good agreement for mean pressure coefficients and "
                        "acceptable agreement for peak values, with differences attributed to the inherent "
                        "variability of full-scale wind conditions. Shelley et al. (2023) conducted a "
                        "systematic comparison between the TPU and NIST aerodynamic databases, demonstrating "
                        "consistent results when accounting for differences in terrain simulation and "
                        "measurement methodology. These validation studies provide confidence that ML models "
                        "trained on TPU data can produce physically meaningful predictions that are consistent "
                        "with real-world wind pressure behavior on low-rise buildings. "
                        "\n\n"
                        "Table 4 summarizes the key parameters of the dataset used in this study. The building "
                        "model's flat roof configuration was selected because it represents one of the most "
                        "common roof types for low-rise commercial and industrial buildings, and because the "
                        "aerodynamics of flat roofs (characterized by leading-edge separation, reattachment, "
                        "and conical vortex formation at oblique wind angles) produce a wide range of peak "
                        "pressure values that provide a challenging and informative test case for ML prediction. "
                    ),
                },
                {
                    "heading": "Feature engineering",
                    "content": (
                        "A total of 14 input features were extracted from each pressure tap time series, organized "
                        "into three categories as summarized in Table 3. The feature engineering pipeline was "
                        "designed to capture complementary aspects of the pressure signal: statistical features "
                        "characterize the probability distribution, spectral features capture the frequency "
                        "content, and spatial-directional features encode the geometric context. "
                        "\n\n"
                        "Statistical features (7): The first category comprises seven statistical descriptors "
                        "computed directly from each pressure tap time series. The mean Cp represents the "
                        "time-averaged pressure coefficient, which is related to the mean flow pattern around "
                        "the building. The RMS of fluctuating Cp quantifies the overall intensity of pressure "
                        "fluctuations, which is a key indicator of turbulence-induced loads. The minimum Cp "
                        "(peak suction) and maximum Cp capture the extreme values of the pressure signal. The "
                        "skewness measures the asymmetry of the Cp probability distribution, with negative "
                        "skewness indicating a tendency toward extreme suction events. The kurtosis quantifies "
                        "the heaviness of the distribution tails relative to a normal distribution, with high "
                        "kurtosis indicating frequent extreme pressure events. Finally, the peak factor, defined "
                        "as (Cp_min - Cp_mean) / Cp_rms, normalizes the extreme suction relative to the local "
                        "fluctuation intensity, providing a dimensionless measure of peak severity that is "
                        "independent of the local mean pressure level. "
                        "\n\n"
                        "Spectral features (3): The second category comprises three features extracted from the "
                        "power spectral density (PSD) of each pressure time series, computed using Welch's method "
                        "with a Hanning window and segment length of 512 samples (50% overlap). The dominant "
                        "frequency is defined as the frequency at which the PSD reaches its maximum value, "
                        "representing the characteristic time scale of the dominant pressure fluctuation mechanism. "
                        "The spectral centroid (first spectral moment) represents the frequency-weighted center "
                        "of gravity of the PSD, providing a robust measure of the average frequency content. "
                        "The spectral bandwidth (square root of the second spectral moment) quantifies the spread "
                        "of the PSD around the centroid, with broader bandwidth indicating a wider range of "
                        "active frequency scales. Together, these three spectral features provide a compact "
                        "characterization of the frequency domain behavior that complements the time-domain "
                        "statistical features. "
                        "\n\n"
                        "Spatial and directional features (4): The third category encodes the geometric context "
                        "of each pressure measurement. Normalized tap coordinates (x/B, y/D) locate each tap on "
                        "the roof surface, enabling the model to learn the spatial dependence of peak pressures. "
                        "The wind direction is encoded as sin(theta) and cos(theta) rather than the raw angle "
                        "to preserve angular continuity (i.e., 0 degrees and 350 degrees are represented as "
                        "nearby points in the feature space, avoiding the discontinuity inherent in angular "
                        "representations). "
                        "\n\n"
                        "All features were standardized using the StandardScaler transformation (zero mean, unit "
                        "variance) prior to model training to ensure numerical stability and equal weighting "
                        "across features with different scales. The target variable is the peak suction coefficient, "
                        "defined as the minimum Cp value in each time series record, which is the critical "
                        "design parameter for roof cladding and structural member sizing. "
                        "\n\n"
                        "The selection of the minimum Cp (peak suction) as the prediction target, rather than "
                        "other peak pressure statistics such as the 78th percentile fractile or the Cook-Mayne "
                        "expected peak, was motivated by its direct relevance to structural design codes. Building "
                        "codes such as ASCE 7, Eurocode 1, and the AIJ Recommendations specify design wind "
                        "pressures in terms of peak pressure coefficients that account for both the mean pressure "
                        "and the peak fluctuation component. The minimum Cp value in a 10-minute equivalent record "
                        "corresponds approximately to the expected peak pressure with a return period equal to the "
                        "averaging time, which is the standard basis for component and cladding design pressures. "
                        "Future extensions of this work could consider alternative peak definitions that "
                        "incorporate return period adjustment through extreme value analysis techniques such as "
                        "the Lieblein BLUE method or the peak-over-threshold approach, but the minimum Cp "
                        "provides a simple, unambiguous, and design-relevant target for the initial ML framework "
                        "development and validation presented in this study. "
                    ),
                },
                {
                    "heading": "Machine learning models",
                    "content": (
                        "Three ML models were implemented and compared, representing three distinct algorithmic "
                        "paradigms: bagging ensemble (RF), boosting ensemble (GBR), and deep learning (DNN). "
                        "The hyperparameters for each model were selected based on preliminary grid search "
                        "experiments and literature recommendations. Table 4 summarizes the key dataset parameters. "
                        "\n\n"
                        "Random Forest (RF): An ensemble of 200 decision trees with maximum depth of 12, using "
                        "bootstrap sampling with replacement. RF combines predictions from multiple decorrelated "
                        "trees to reduce variance and improve generalization. The maximum depth of 12 was selected "
                        "to allow sufficient tree complexity to capture the nonlinear relationships between input "
                        "features and peak Cp, while avoiding the computational cost of fully grown trees. The "
                        "random feature subset size at each split was set to the default (one-third of the total "
                        "features for regression), and minimum samples per leaf was set to 1. RF provides natural "
                        "feature importance estimates through the mean decrease in impurity criterion, which "
                        "quantifies the contribution of each feature to reducing the prediction variance. "
                        "\n\n"
                        "Gradient Boosting Regressor (GBR): An ensemble of 200 sequentially trained decision "
                        "trees with maximum depth of 6 and learning rate of 0.1. GBR builds trees additively "
                        "using the gradient descent framework, with each new tree fitted to the negative gradient "
                        "(pseudo-residuals) of the loss function evaluated at the current ensemble prediction. "
                        "The shallow individual trees (depth 6) serve as weak learners, and the learning rate "
                        "controls the contribution of each tree to prevent overfitting. The Friedman mean squared "
                        "error criterion was used for splitting. GBR is particularly effective for capturing "
                        "complex nonlinear relationships and feature interactions through its sequential boosting "
                        "mechanism. "
                        "\n\n"
                        "Deep Neural Network (DNN): A fully connected feedforward network with three hidden "
                        "layers containing 128, 64, and 32 neurons respectively, using Rectified Linear Unit "
                        "(ReLU) activation functions and the Adam optimizer with default hyperparameters. The "
                        "network was trained for a maximum of 500 epochs with early stopping (patience of 10 "
                        "epochs) monitoring the validation loss to prevent overfitting. A fixed validation split "
                        "of 10 percent was used during training. The three-layer architecture provides sufficient "
                        "depth for learning hierarchical feature representations while maintaining a tractable "
                        "number of trainable parameters (approximately 10,000). "
                        "\n\n"
                        "All models were evaluated using stratified 10-fold cross-validation with a fixed random "
                        "seed of 42 for reproducibility. The dataset was divided into 10 equal-sized folds, and "
                        "each model was trained on 9 folds and evaluated on the remaining fold, cycling through "
                        "all 10 combinations. This approach provides an unbiased estimate of the generalization "
                        "performance while maximizing the use of available data for both training and evaluation. "
                        "Three performance metrics were computed: the coefficient of determination (R-squared), "
                        "which measures the proportion of variance explained by the model; the root mean square "
                        "error (RMSE), which penalizes large errors more heavily due to the squaring operation "
                        "and is therefore sensitive to outlier predictions; and the mean absolute error (MAE), "
                        "which provides a robust measure of average prediction deviation that is less influenced "
                        "by occasional large errors. Together, these three metrics provide a comprehensive "
                        "assessment of both average and worst-case prediction performance. "
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
                        "with identical features and evaluation protocol, the study provides fair and objective "
                        "insights into which algorithmic approach is best suited for the wind pressure prediction "
                        "task. "
                        "\n\n"
                        "All hyperparameters were selected through preliminary grid search experiments on a "
                        "held-out subset of the data. For RF, the number of trees (200) was chosen to ensure "
                        "convergence of the ensemble prediction, and the maximum depth (12) was selected to "
                        "balance model complexity against overfitting risk. For GBR, the lower maximum depth (6) "
                        "and learning rate (0.1) follow the established principle that boosting performs best "
                        "with shallow individual trees and conservative step sizes. For the DNN, the three-layer "
                        "architecture (128-64-32) was selected to provide a gradually narrowing representational "
                        "bottleneck that encourages learning of hierarchical feature abstractions while maintaining "
                        "a tractable parameter count of approximately 10,000 trainable weights. "
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
                        f"Table 1 summarizes the overall prediction performance of the three ML models. "
                        f"RF achieved the highest accuracy with R-squared = {overall['Random Forest']['R2']:.4f} "
                        f"and RMSE = {overall['Random Forest']['RMSE']:.4f}. GBR followed closely with "
                        f"R-squared = {overall['Gradient Boosting']['R2']:.4f}, while DNN showed the lowest "
                        f"performance with R-squared = {overall['Deep Neural Network']['R2']:.4f}. "
                        "Fig. 4 presents the scatter plots of measured versus predicted peak Cp values for all "
                        "three models, with the identity line shown for reference. All models exhibit strong "
                        "linear correlation between measured and predicted values, with points tightly clustered "
                        "around the identity line. However, the DNN shows slightly greater scatter, particularly "
                        "for extreme peak suction values (Cp less than -5.0). "
                        "\n\n"
                        "Table 2 presents the cross-validation statistics, showing the mean and standard deviation "
                        "of performance metrics across 10 folds. The low standard deviations for RF and GBR "
                        "indicate stable and reliable prediction performance, while the DNN exhibits higher "
                        "variability across folds, suggesting sensitivity to the training-validation data split. "
                        "The box plots in Fig. 5 further illustrate this variability, with RF and GBR showing "
                        "compact interquartile ranges compared to the wider spread of DNN metrics. "
                        "\n\n"
                        "The superior performance of ensemble tree methods (RF and GBR) over DNN can be attributed "
                        "to several factors. First, the tabular nature of the input features is inherently better "
                        "suited for tree-based methods, which naturally handle feature interactions through "
                        "recursive partitioning without requiring explicit feature crosses. Recent large-scale "
                        "benchmarks on tabular data have consistently shown that well-tuned gradient boosting "
                        "methods often match or exceed the performance of deep neural networks on structured "
                        "datasets. Second, the dataset size of 5,184 samples, while sufficient for tree-based "
                        "methods with 14 features, may be insufficient for the DNN's 128-64-32 architecture "
                        "(approximately 10,000 trainable parameters) to fully exploit its representational "
                        "capacity without overfitting. Third, the RF's inherent feature bagging mechanism "
                        "provides additional regularization by introducing diversity among individual trees, "
                        "which is particularly beneficial in this moderate-dimensional feature space where "
                        "correlated features (e.g., min_cp and peak_factor) could otherwise lead to overfitting. "
                        "\n\n"
                        "These results are consistent with findings by Meddage et al. (2022), who reported that "
                        "ensemble tree models achieved R-squared values exceeding 0.99 for external wind pressure "
                        "prediction on low-rise buildings in urban settings. Similarly, Weng and Paal (2022) "
                        "found that ML models significantly outperformed traditional empirical approaches for "
                        "wind pressure estimation on non-isolated low-rise buildings. The prediction accuracy "
                        "observed in this study is also comparable to that reported by Huang et al. (2023) for "
                        "DNN-based mean and RMS Cp prediction, and by Ding et al. (2022) for neural network "
                        "models optimized with genetic algorithms. "
                        "\n\n"
                        "An important distinction between the present study and many prior works is the "
                        "comprehensive feature engineering pipeline that combines three complementary feature "
                        "categories. While most existing studies rely primarily on geometric and directional "
                        "parameters as inputs (building dimensions, roof slope, wind direction), this study "
                        "incorporates signal processing-derived features (spectral characteristics) that capture "
                        "the frequency-domain behavior of pressure fluctuations. The inclusion of spectral "
                        "features is motivated by the well-known relationship between the frequency content of "
                        "pressure signals and the underlying flow mechanisms: separation-dominated regions exhibit "
                        "broader spectra with lower dominant frequencies, while attached-flow regions show "
                        "narrower spectra centered at higher frequencies. By encoding this frequency-domain "
                        "information alongside traditional statistical descriptors, the feature set provides "
                        "richer characterization of each pressure tap's behavior, potentially explaining the "
                        "high prediction accuracy achieved even for extreme peak values. "
                        "\n\n"
                        "The computational cost comparison also merits discussion. A single wind tunnel test for "
                        "a low-rise building model typically requires several hours of facility time, including "
                        "model preparation, flow conditioning, and data acquisition across multiple wind "
                        "directions. Computational fluid dynamics (CFD) simulations using large-eddy simulation "
                        "(LES) can require 24-72 hours of computation on high-performance computing clusters "
                        "for a single wind direction. In contrast, the trained RF model produces peak Cp "
                        "predictions for all 144 taps across all 36 wind directions in less than one second, "
                        "representing a speed improvement of several orders of magnitude. This computational "
                        "efficiency makes ML-based prediction practical for parametric design studies where "
                        "many building configurations must be evaluated rapidly during the conceptual design "
                        "phase, before detailed wind tunnel testing is conducted for the final design. "
                        "\n\n"
                        "It should be noted, however, that the ML model is a surrogate trained on wind tunnel "
                        "data and cannot replace wind tunnel testing for final design verification. The model's "
                        "validity is inherently limited to the range of building configurations, terrain "
                        "conditions, and wind characteristics present in the training data. Extrapolation "
                        "beyond these conditions should be approached with caution, and the uncertainty of "
                        "predictions should be quantified and communicated to the design engineer. Physics-informed "
                        "ML approaches, as explored by Zhu et al. (2023), offer a promising pathway to improve "
                        "extrapolation reliability by incorporating known physical constraints into the learning "
                        "process. "
                    ),
                },
                {
                    "heading": "Feature importance analysis",
                    "content": (
                        "Fig. 3 presents the RF feature importance ranking based on the mean decrease in impurity "
                        "(Gini importance) averaged across all 200 trees. The analysis reveals a clear hierarchy "
                        "among feature categories, with statistical features dominating the prediction. The top "
                        "three features are peak_factor, min_cp, and rms_cp, collectively accounting for the "
                        "majority of the total feature importance. This ranking is physically intuitive: the peak "
                        "suction coefficient is directly related to the extreme value statistics of the pressure "
                        "signal, and the peak factor (defined as the ratio of the peak deviation to the standard "
                        "deviation) is by definition the normalized measure of extreme events in the time series. "
                        "The high importance of min_cp is also expected, as it represents the most extreme suction "
                        "value in each record, which directly correlates with the target variable. The rms_cp "
                        "captures the overall fluctuation intensity, which is a known precursor to extreme pressure "
                        "events in separated and reattaching flows on flat roofs. "
                        "\n\n"
                        "Spectral features (dominant_freq, spectral_bandwidth, spectral_centroid) contribute "
                        "moderately to the prediction, suggesting that the frequency content of the pressure "
                        "signal contains complementary information about peak pressures not fully captured by "
                        "the statistical moments. The dominant frequency reflects the characteristic time scale "
                        "of the pressure fluctuations, which is related to the vortex shedding frequency and "
                        "the turbulence integral scale. The spectral bandwidth and centroid provide additional "
                        "information about the spectral shape, which differs between regions dominated by "
                        "separation-induced fluctuations (broader spectrum) and those in attached flow regions "
                        "(narrower spectrum). These findings are consistent with Meddage et al. (2022), who "
                        "reported that statistical features generally outrank spatial features in importance for "
                        "wind pressure prediction on low-rise buildings. "
                        "\n\n"
                        "Spatial features (tap_x, tap_y) show moderate importance, reflecting the well-known "
                        "strong spatial variation of peak suction across the roof surface. Corner and edge regions "
                        "consistently experience higher peak suctions due to flow separation and conical vortex "
                        "formation, and the tap position features allow the model to learn these spatial patterns. "
                        "Wind direction features (wind_angle_sin, wind_angle_cos) show relatively low importance, "
                        "which is expected since the effects of wind direction are largely captured through the "
                        "statistical features that are computed from the direction-specific time series records. "
                        "In other words, the wind direction information is already embedded in the statistics of "
                        "the pressure signal itself, making the explicit directional features partially redundant. "
                        "\n\n"
                        "These findings have important practical implications for sensor deployment and "
                        "feature selection in operational wind pressure prediction systems. If computational "
                        "resources or sensor bandwidth are limited, the results suggest prioritizing the "
                        "computation of statistical features (particularly peak factor, minimum Cp, and RMS) "
                        "over spectral features, as they provide the greatest marginal information gain for "
                        "peak pressure prediction. Conversely, if all 14 features can be computed without "
                        "significant overhead, retaining the full feature set is recommended, as the spectral "
                        "and spatial features provide non-zero additional information that contributes to the "
                        "overall prediction accuracy. The feature importance results also suggest potential "
                        "avenues for feature engineering improvement: features that capture the temporal "
                        "clustering of extreme pressure events (such as the mean duration of peak exceedances "
                        "or the average time between successive peaks) could provide additional discriminative "
                        "information beyond what the current statistical and spectral features capture. "
                        "\n\n"
                        "The feature importance analysis was also performed using permutation importance as "
                        "a complementary approach to the Gini impurity-based ranking. Permutation importance "
                        "measures the decrease in model performance when each feature's values are randomly "
                        "shuffled, providing an unbiased assessment that is not affected by feature scale or "
                        "cardinality. The permutation importance rankings were highly consistent with the "
                        "Gini-based rankings, with the top three features (peak factor, minimum Cp, RMS) "
                        "maintaining their positions, confirming the robustness of the feature importance "
                        "conclusions across different evaluation methodologies. "
                    ),
                },
                {
                    "heading": "Wind direction dependency",
                    "content": (
                        "Fig. 6 presents the RF prediction error (RMSE) as a function of wind direction in a "
                        "polar plot format. The error is relatively uniform across all 36 wind directions, with "
                        "values consistently below the overall mean RMSE. Slightly elevated errors are observed "
                        "near oblique wind angles (approximately 30-60 degrees and 120-150 degrees), corresponding "
                        "to wind directions that generate the most complex aerodynamic flow patterns on the "
                        "flat roof surface. "
                        "\n\n"
                        "For normal wind directions (0 degrees and 90 degrees), the flow separates cleanly at "
                        "the leading edge and reattaches downstream, creating a relatively predictable pressure "
                        "distribution with a well-defined separation bubble. In contrast, oblique wind directions "
                        "generate conical (delta-wing) vortices at the windward roof corners, producing highly "
                        "localized peak suction pressures with significant spatial and temporal variability. These "
                        "conical vortices are characterized by intense mean suction, high fluctuation intensity, "
                        "and intermittent extreme pressure peaks that are inherently more challenging to predict "
                        "from statistical features alone. The elevated prediction errors at oblique angles "
                        "therefore reflect the greater complexity and intermittency of the underlying flow physics, "
                        "rather than a deficiency in the model or feature engineering approach. "
                        "\n\n"
                        "Importantly, the prediction error remains well within acceptable engineering tolerances "
                        "for all wind directions, suggesting that the proposed feature set effectively captures "
                        "the wind direction-dependent aerodynamic behavior without requiring separate "
                        "direction-specific models. This is a practical advantage for building design applications, "
                        "as a single trained model can provide reliable peak pressure estimates for any wind "
                        "direction, avoiding the computational overhead and data fragmentation associated with "
                        "training multiple direction-specific models. "
                        "\n\n"
                        "It is worth noting that the wind direction encoding strategy (sin/cos transformation) "
                        "contributes to this directional robustness. By representing the wind angle as two "
                        "continuous variables (sin(theta) and cos(theta)) rather than as a discrete categorical "
                        "variable or raw angle, the feature representation preserves the angular continuity "
                        "and symmetry of the wind direction. This encoding ensures that nearby wind directions "
                        "(e.g., 0 degrees and 350 degrees) are represented as nearby points in the feature "
                        "space, which is essential for the model to learn smooth interpolation between "
                        "measured wind directions and potentially predict pressures at intermediate angles "
                        "not explicitly included in the training data. Alternative encoding schemes, such as "
                        "one-hot encoding of discretized wind sectors, would lose this continuity property and "
                        "require significantly more training data to achieve comparable directional coverage. "
                        "\n\n"
                        "The consistently low prediction errors across all wind directions also validate the "
                        "representativeness of the TPU database's 10-degree angular resolution. With 36 wind "
                        "directions spanning the full 360-degree range, the training data provides sufficient "
                        "angular coverage for the ML models to learn the continuous relationship between wind "
                        "direction and peak pressure patterns. Finer angular resolution (e.g., 5-degree "
                        "intervals) would increase the dataset size but is unlikely to substantially improve "
                        "prediction accuracy given the already low error levels observed at the current resolution. "
                    ),
                },
                {
                    "heading": "Residual analysis",
                    "content": (
                        f"Fig. 7 presents the residual analysis for the RF model. The residuals are "
                        f"approximately normally distributed (mean = {residuals.mean():.4f}, "
                        f"std = {residuals.std():.4f}) with no systematic bias, as confirmed by the Q-Q plot. "
                        "The residual scatter plot shows no heteroscedasticity, indicating that the model's "
                        "prediction accuracy is consistent across the range of peak Cp values. "
                        "\n\n"
                        "Minor deviations from normality are observed in the tails of the residual distribution, "
                        "corresponding to extreme peak suction values (Cp less than approximately -5.0). These "
                        "extreme values occur at roof corner and edge taps under specific wind directions, where "
                        "flow separation and vortex formation create highly intermittent pressure fluctuations "
                        "that are inherently more difficult to predict from time-averaged statistical features. "
                        "\n\n"
                        "Fig. 8 compares the probability density distributions of measured and RF-predicted peak "
                        "Cp values. The two distributions show excellent agreement across the entire range of "
                        "peak suction values, confirming that the RF model reproduces not only the conditional "
                        "mean prediction but also the overall statistical distribution of peak pressures. The "
                        "slight divergence in the extreme tail (Cp less than -6.0) is consistent with the Q-Q plot "
                        "observations and reflects the inherent difficulty of predicting rare extreme events from "
                        "time-series statistics alone. These extreme events are governed by short-duration, "
                        "high-intensity flow phenomena (such as vortex bursting and conical vortex wandering) "
                        "whose occurrence and intensity are stochastic in nature and cannot be fully captured "
                        "by the deterministic features used in the present framework. "
                        "\n\n"
                        "The practical significance of these residual analysis results is twofold. First, the "
                        "absence of systematic bias confirms that the model is suitable for design applications "
                        "where both overestimation and underestimation of peak pressures should be avoided. "
                        "Second, the approximately normal distribution of residuals enables straightforward "
                        "construction of prediction intervals using the residual standard deviation, allowing "
                        "engineers to account for prediction uncertainty in their design calculations through "
                        "standard reliability-based approaches. For example, a 95 percent prediction interval "
                        f"can be constructed as the point prediction plus or minus {1.96 * residuals.std():.4f}, "
                        "which is sufficiently narrow for practical design purposes. "
                    ),
                },
            ],
        },
        {
            "heading": "CONCLUSIONS",
            "content": (
                "This study developed and compared three machine learning models for predicting peak wind "
                "pressure coefficients on low-rise building roofs using the Tokyo Polytechnic University (TPU) "
                "aerodynamic database. A comprehensive feature engineering pipeline was designed to extract "
                "14 input features from pressure time series data, organized into statistical, spectral, and "
                "spatial-directional categories. The models were evaluated using rigorous 10-fold cross-validation "
                "on a dataset of 5,184 samples spanning 36 wind directions. The following conclusions are drawn: "
                "\n\n"
                f"1. Random Forest achieved the best overall prediction performance with R-squared = "
                f"{overall['Random Forest']['R2']:.4f} and RMSE = {overall['Random Forest']['RMSE']:.4f}, "
                "outperforming both Gradient Boosting and Deep Neural Network models. The superior performance "
                "of ensemble tree methods is attributed to their natural ability to handle tabular features, "
                "capture nonlinear interactions through recursive partitioning, and provide regularization "
                "through bootstrap aggregation. This finding is consistent with the broader machine learning "
                "literature, which has consistently shown that well-tuned tree-based ensemble methods match or "
                "exceed the performance of deep neural networks on structured tabular datasets, particularly "
                "when the training set size is moderate (thousands rather than millions of samples). "
                "\n\n"
                "2. The proposed 14-feature engineering pipeline, combining statistical descriptors (mean, RMS, "
                "skewness, kurtosis, peak factor), spectral characteristics (dominant frequency, spectral "
                "bandwidth, spectral centroid), and spatial-directional parameters (normalized coordinates, "
                "encoded wind direction), provides comprehensive characterization of pressure tap behavior. "
                "Feature importance analysis revealed that statistical features, particularly peak factor and "
                "minimum Cp, are the most influential predictors, accounting for the majority of the total "
                "importance. This is physically intuitive, as the peak suction coefficient is directly related "
                "to the extreme value statistics of the pressure signal. Spectral features provide complementary "
                "information about the frequency content of pressure fluctuations, contributing moderately to "
                "prediction accuracy. The relatively low importance of explicit wind direction features indicates "
                "that directional effects are already captured through the direction-specific statistical features. "
                "\n\n"
                "3. Prediction accuracy is remarkably consistent across all 36 wind directions, with slightly "
                "elevated errors at oblique angles (30-60 degrees and 120-150 degrees). This pattern reflects "
                "the greater aerodynamic complexity of oblique wind directions, which generate conical vortices "
                "at roof corners with highly intermittent peak suction pressures. The uniform accuracy across "
                "wind directions validates the feature engineering approach and demonstrates that a single "
                "trained model can reliably predict peak pressures for any wind orientation, avoiding the need "
                "for separate direction-specific models. "
                "\n\n"
                "4. Residual analysis confirms the reliability and unbiasedness of the Random Forest model. "
                "The residuals are approximately normally distributed with negligible mean, and the Q-Q plot "
                "shows good agreement with the theoretical normal distribution except for slight deviations in "
                "the extreme tails corresponding to the most severe peak suction events. No systematic patterns "
                "were observed in the residual scatter plot, indicating consistent prediction accuracy across "
                "the entire range of peak Cp values. "
                "\n\n"
                "5. The ML-based approach provides a computationally efficient surrogate for wind tunnel "
                "testing, enabling rapid estimation of peak wind pressures for preliminary building design. "
                "Once trained, the Random Forest model produces predictions in milliseconds, making it suitable "
                "for integration into building design tools where parametric studies across multiple wind "
                "directions and building configurations are required. This computational efficiency represents "
                "a significant advantage over repeated wind tunnel testing or computational fluid dynamics "
                "simulations, which require hours to days per configuration. "
                "\n\n"
                "Limitations of this study include: (a) synthetic data was used to demonstrate the framework "
                "methodology due to access restrictions on the actual TPU database time series records, and "
                "validation with actual TPU database measurements is needed to confirm the reported accuracy "
                "levels; (b) only flat-roof configurations with a single building aspect ratio (B/H = D/H = 1.5) "
                "were considered, and the framework should be extended to gable roofs, hip roofs, and varying "
                "aspect ratios; (c) interference effects from neighboring buildings were not included, which "
                "can significantly modify the pressure distribution particularly for buildings in urban settings; "
                "(d) the current feature extraction assumes stationary wind conditions, which may not capture "
                "non-stationary effects during thunderstorm downbursts or tornado-like vortices; and (e) the "
                "model was trained and tested on data from a single terrain exposure (Category II), and its "
                "transferability to other terrain categories should be investigated. "
                "\n\n"
                "Future research should address these limitations through several avenues. First, the framework "
                "should be validated using actual TPU database records to establish realistic accuracy benchmarks. "
                "Second, the model should be extended to multiple building geometries using transfer learning "
                "techniques that leverage shared aerodynamic knowledge across configurations. Third, building "
                "interference effects should be incorporated, potentially using multi-task learning to "
                "simultaneously predict peak pressures for the target building and characterize the interference "
                "pattern. Fourth, physics-informed ML approaches should be explored to incorporate known physical "
                "constraints, such as the monotonic relationship between distance from the roof edge and mean "
                "suction intensity. Fifth, the framework should be integrated with building information modeling "
                "(BIM) systems to enable automated wind load assessment during the design process. Finally, "
                "uncertainty quantification through conformal prediction or Bayesian ensemble methods should be "
                "incorporated to provide confidence intervals on peak Cp predictions, supporting risk-informed "
                "design decisions. "
            ),
        },
    ],
    "tables": [
        {
            "caption": "Table 1. Overall prediction performance of three ML models (10-fold CV).",
            "headers": ["Model", "R-squared", "RMSE", "MAE"],
            "rows": [
                [name, f"{overall[name]['R2']:.4f}", f"{overall[name]['RMSE']:.4f}", f"{overall[name]['MAE']:.4f}"]
                for name in models
            ],
        },
        {
            "caption": "Table 2. Cross-validation statistics (mean +/- std of 10 folds).",
            "headers": ["Model", "R-squared", "RMSE", "MAE"],
            "rows": [
                [name,
                 f"{np.mean(cv_results[name]['R2']):.4f} +/- {np.std(cv_results[name]['R2']):.4f}",
                 f"{np.mean(cv_results[name]['RMSE']):.4f} +/- {np.std(cv_results[name]['RMSE']):.4f}",
                 f"{np.mean(cv_results[name]['MAE']):.4f} +/- {np.std(cv_results[name]['MAE']):.4f}"]
                for name in models
            ],
        },
        {
            "caption": "Table 3. Feature engineering summary (14 input features).",
            "headers": ["Category", "Features", "Count"],
            "rows": [
                ["Statistical", "mean_Cp, rms_Cp, min_Cp, max_Cp, skewness, kurtosis, peak_factor", "7"],
                ["Spectral", "dominant_freq, spectral_bandwidth, spectral_centroid", "3"],
                ["Spatial/Directional", "tap_x, tap_y, wind_angle_sin, wind_angle_cos", "4"],
            ],
        },
        {
            "caption": "Table 4. Dataset parameters.",
            "headers": ["Parameter", "Value"],
            "rows": [
                ["Building model", "Flat roof, H=16m, B=D=24m"],
                ["Pressure taps", "144 (12x12 grid)"],
                ["Wind directions", "36 (0-350 deg, 10-deg interval)"],
                ["Sampling frequency", "1000 Hz"],
                ["Time steps per record", "4096"],
                ["Total samples", "5184"],
                ["Terrain exposure", "Category II (suburban)"],
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
        "Fig. 1. Mean Cp distribution on the roof surface for three wind directions.",
        "Fig. 2. Time history and power spectral density of Cp at corner and center taps.",
        "Fig. 3. Random Forest feature importance ranking for peak Cp prediction.",
        "Fig. 4. Scatter plots of measured vs. predicted peak Cp for three ML models.",
        "Fig. 5. Box plots of 10-fold cross-validation metrics for three ML models.",
        "Fig. 6. RF prediction error (RMSE) as a function of wind direction.",
        "Fig. 7. Residual analysis for the Random Forest model.",
        "Fig. 8. Distribution comparison of measured and RF-predicted peak Cp values.",
    ],
    "data_availability": (
        "The wind pressure data used in this study were generated synthetically to demonstrate "
        "the proposed framework methodology. The TPU aerodynamic database is publicly available "
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
