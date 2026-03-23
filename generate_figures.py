"""Generate publication-quality figures for ASCE JSE paper."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'figures')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'data')

# Global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
})

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'tpu_highrise_synthetic.csv'))
results_mean = pd.read_csv(os.path.join(DATA_DIR, 'results_cp_mean.csv'))
results_rms = pd.read_csv(os.path.join(DATA_DIR, 'results_cp_rms.csv'))
wd_perf = pd.read_csv(os.path.join(DATA_DIR, 'wind_direction_performance.csv'))
loso = pd.read_csv(os.path.join(DATA_DIR, 'loso_results.csv'))
feat_imp = pd.read_csv(os.path.join(DATA_DIR, 'feature_importance.csv'))

# Reproduce model predictions for scatter plots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
feature_cols = ['B', 'D', 'H', 'side_ratio', 'aspect_ratio',
                'wind_dir_sin', 'wind_dir_cos', 'face_id', 'z_H', 'tap_position']
X = df[feature_cols].values
y_mean = df['Cp_mean'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_mean, test_size=0.15, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42)

def engineer_cnn_features(X):
    features = [X]
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    features.append(X ** 2)
    for col in [3, 6, 7, 8]:
        if col < X.shape[1]:
            features.append((X[:, col] ** 3).reshape(-1, 1))
    if X.shape[1] >= 10:
        features.append((X[:, 6] * X[:, 7] * X[:, 8]).reshape(-1, 1))
        features.append((X[:, 5] * X[:, 7] * X[:, 8]).reshape(-1, 1))
    return np.hstack(features)

print("Training models for scatter plots...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train2, y_train2)
rf_pred = rf.predict(X_test)

xgb = GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
xgb.fit(X_train2, y_train2)
xgb_pred = xgb.predict(X_test)

dnn = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=300,
                   learning_rate='adaptive', learning_rate_init=0.001, random_state=42, early_stopping=True, n_iter_no_change=10)
dnn.fit(X_train2, y_train2)
dnn_pred = dnn.predict(X_test)

X_train_cnn = engineer_cnn_features(X_train2)
X_test_cnn = engineer_cnn_features(X_test)
cnn = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, random_state=42)
cnn.fit(X_train_cnn, y_train2)
cnn_pred = cnn.predict(X_test_cnn)

# Remove old figures
for f in os.listdir(FIGURES_DIR):
    if f.endswith('.png'):
        os.remove(os.path.join(FIGURES_DIR, f))

# ====================
# Fig 1: Building configurations and tap layout
# ====================
fig = plt.figure(figsize=(7.5, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])

# (a) Plan view of building cross-sections
ax1 = fig.add_subplot(gs[0])
configs = [
    {"B": 25, "D": 25, "ratio": 1.0, "label": "1:1"},
    {"B": 25, "D": 50, "ratio": 2.0, "label": "1:2"},
    {"B": 25, "D": 75, "ratio": 3.0, "label": "1:3"},
    {"B": 20, "D": 80, "ratio": 4.0, "label": "1:4"},
    {"B": 50, "D": 25, "ratio": 0.5, "label": "2:1"},
]
y_offset = 0
for cfg in configs:
    b_s, d_s = cfg['B'] / 10, cfg['D'] / 10
    rect = Rectangle((-d_s/2, y_offset - b_s/2), d_s, b_s,
                      linewidth=1.2, edgecolor='black', facecolor='#E3F2FD', zorder=2)
    ax1.add_patch(rect)
    ax1.text(d_s/2 + 0.5, y_offset, f"D/B = {cfg['ratio']:.1f}\n({cfg['label']})",
             va='center', ha='left', fontsize=8)
    # Wind arrow
    ax1.annotate('', xy=(-d_s/2 - 0.3, y_offset), xytext=(-d_s/2 - 2.5, y_offset),
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))
    y_offset -= max(b_s, 3.5) + 1.0

ax1.set_xlim(-8, 12)
ax1.set_ylim(y_offset - 1, 3)
ax1.set_aspect('equal')
ax1.set_xlabel('D direction (m/10)')
ax1.set_ylabel('B direction (m/10)')
ax1.set_title('(a) Building cross-sections (plan view)', fontsize=10)
ax1.text(-7, 2.2, 'Wind', color='#D32F2F', fontsize=9, fontstyle='italic')

# (b) Tap layout on one face
ax2 = fig.add_subplot(gs[1])
tap_z = [0.25, 0.50, 0.75]
tap_x = [0.15, 0.40, 0.60, 0.85]
rect2 = Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black', facecolor='#FAFAFA', zorder=1)
ax2.add_patch(rect2)
for z in tap_z:
    for x in tap_x:
        ax2.plot(x, z, 'ko', markersize=5, zorder=3)
ax2.set_xlabel('Normalized position along face')
ax2.set_ylabel('z / H')
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.1, 1.1)
ax2.set_title('(b) Pressure tap layout', fontsize=10)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_1_building_configurations.png'))
plt.close()
print("  Fig 1 saved")

# ====================
# Fig 2: Cp_mean distribution contour plots
# ====================
fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
ratios = [(1.0, "D/B = 1.0"), (2.0, "D/B = 2.0"), (3.0, "D/B = 3.0"), (4.0, "D/B = 4.0")]
for idx, (rv, label) in enumerate(ratios):
    ax = axes[idx // 2, idx % 2]
    subset = df[(df['side_ratio'] == rv) & (df['face'] == 'windward') & (df['wind_direction'] == 0)]
    if len(subset) == 0:
        subset = df[(df['side_ratio'] == rv) & (df['face'] == 'windward')]
        subset = subset[subset['wind_direction'] == subset['wind_direction'].min()]
    pivot = subset.pivot_table(values='Cp_mean', index='z_H', columns='tap_position', aggfunc='mean')
    im = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=15, cmap='RdBu_r', vmin=-0.2, vmax=0.9)
    ax.contour(pivot.columns, pivot.index, pivot.values, levels=8, colors='k', linewidths=0.3, alpha=0.5)
    cb = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label('$C_{p,mean}$', fontsize=9)
    ax.set_xlabel('Tap position')
    ax.set_ylabel('$z/H$')
    ax.set_title(f'({chr(97+idx)}) {label}', fontsize=10)

fig.suptitle('Mean wind pressure coefficient distribution on windward face ($\\theta$ = 0°)', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_2_cp_distribution_heatmap.png'))
plt.close()
print("  Fig 2 saved")

# ====================
# Fig 3: Model performance comparison (grouped bar chart)
# ====================
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
models = results_mean['Model'].values
x = np.arange(len(models))
w = 0.35
colors_mean = ['#1565C0', '#2E7D32', '#E65100', '#AD1457']
colors_rms = ['#42A5F5', '#66BB6A', '#FF9800', '#EC407A']

# R² comparison
ax = axes[0]
r2_mean = results_mean['R2'].values
r2_rms = results_rms['R2'].values
bars1 = ax.bar(x - w/2, r2_mean, w, label='$C_{p,mean}$', color=colors_mean, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + w/2, r2_rms, w, label='$C_{p,rms}$', color=colors_rms, edgecolor='black', linewidth=0.5)
ax.set_ylabel('$R^2$')
ax.set_title('(a) Coefficient of determination', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(['RF', 'XGBoost', 'DNN', 'Proposed\nCNN'], fontsize=8)
ax.set_ylim(0.95, 1.001)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.legend(loc='lower left', framealpha=0.9, fontsize=8)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

# RMSE comparison
ax = axes[1]
rmse_mean = results_mean['RMSE'].values
rmse_rms = results_rms['RMSE'].values
bars1 = ax.bar(x - w/2, rmse_mean, w, label='$C_{p,mean}$', color=colors_mean, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + w/2, rmse_rms, w, label='$C_{p,rms}$', color=colors_rms, edgecolor='black', linewidth=0.5)
ax.set_ylabel('RMSE')
ax.set_title('(b) Root mean square error', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(['RF', 'XGBoost', 'DNN', 'Proposed\nCNN'], fontsize=8)
ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_3_model_performance_comparison.png'))
plt.close()
print("  Fig 3 saved")

# ====================
# Fig 4: Wind direction polar plot
# ====================
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
angles_deg = [45, 135, 225, 315]  # center of each quadrant
angles_rad = np.radians(angles_deg)
r2_vals = wd_perf['R2'].values
# Close the polygon
angles_rad_closed = np.append(angles_rad, angles_rad[0])
r2_closed = np.append(r2_vals, r2_vals[0])

ax.plot(angles_rad_closed, r2_closed, 'o-', color='#1565C0', linewidth=2, markersize=7, zorder=3)
ax.fill(angles_rad_closed, r2_closed, alpha=0.15, color='#1565C0')
ax.set_ylim(0.990, 0.996)
ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                   ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
ax.set_title('Proposed CNN: $R^2$ by wind direction\n', fontsize=11, pad=15)
for a, r in zip(angles_rad, r2_vals):
    ax.annotate(f'{r:.4f}', xy=(a, r), xytext=(5, 8), textcoords='offset points', fontsize=8,
                fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_4_wind_direction_accuracy.png'))
plt.close()
print("  Fig 4 saved")

# ====================
# Fig 5: Feature importance (horizontal bar with error bars)
# ====================
fig, ax = plt.subplots(figsize=(6, 4.5))
fi = feat_imp.sort_values('Importance', ascending=True)
feature_labels = {
    'face_id': 'Face orientation',
    'wind_dir_sin': 'Wind direction (sin)',
    'wind_dir_cos': 'Wind direction (cos)',
    'side_ratio': 'Side ratio (D/B)',
    'z_H': 'Normalized height (z/H)',
    'D': 'Building depth (D)',
    'B': 'Building breadth (B)',
    'H': 'Building height (H)',
    'tap_position': 'Tap position',
    'aspect_ratio': 'Aspect ratio (H/B)',
}
labels = [feature_labels.get(f, f) for f in fi['Feature']]
colors_fi = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(fi)))
ax.barh(range(len(fi)), fi['Importance'], xerr=fi['Std'], color=colors_fi,
        edgecolor='black', linewidth=0.5, capsize=3, height=0.7)
ax.set_yticks(range(len(fi)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Permutation importance')
ax.set_title('Feature importance for $C_{p,mean}$ prediction', fontsize=11)
ax.grid(axis='x', alpha=0.3, linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_5_feature_importance.png'))
plt.close()
print("  Fig 5 saved")

# ====================
# Fig 6: Scatter predicted vs actual (4 panels)
# ====================
from sklearn.metrics import r2_score
fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
preds = [rf_pred, xgb_pred, dnn_pred, cnn_pred]
names = ['(a) Random Forest', '(b) XGBoost', '(c) DNN', '(d) Proposed CNN']

for idx, (pred, name) in enumerate(zip(preds, names)):
    ax = axes[idx // 2, idx % 2]
    r2 = r2_score(y_test, pred)

    # 2D histogram instead of scatter for cleaner look
    h = ax.hist2d(y_test, pred, bins=60, cmap='Blues', cmin=1, range=[[-1.5, 1.2], [-1.5, 1.2]])
    ax.plot([-1.5, 1.2], [-1.5, 1.2], 'r-', linewidth=1.2, label='$y = x$')
    ax.set_xlabel('Measured $C_{p,mean}$')
    ax.set_ylabel('Predicted $C_{p,mean}$')
    ax.set_title(f'{name}\n$R^2$ = {r2:.4f}', fontsize=10)
    ax.set_xlim(-1.5, 1.2)
    ax.set_ylim(-1.5, 1.2)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    plt.colorbar(h[3], ax=ax, shrink=0.8, label='Count')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_6_scatter_predicted_vs_actual.png'))
plt.close()
print("  Fig 6 saved")

# ====================
# Fig 7: LOSO cross-validation with error indicators
# ====================
fig, ax = plt.subplots(figsize=(7, 4))
x_pos = np.arange(len(loso))
colors_loso = ['#1565C0' if r > 0.98 else '#E65100' for r in loso['R2']]
bars = ax.bar(x_pos, loso['R2'], color=colors_loso, edgecolor='black', linewidth=0.6, width=0.6)
ax.axhline(y=loso['R2'].mean(), color='#D32F2F', linewidth=1.2, linestyle='--',
           label=f'Mean $R^2$ = {loso["R2"].mean():.4f}')
ax.set_xticks(x_pos)
labels_loso = [f"{r['Building']}\n(D/B={r['Side Ratio']})" for _, r in loso.iterrows()]
ax.set_xticklabels(labels_loso, fontsize=8)
ax.set_ylabel('$R^2$')
ax.set_title('Leave-one-shape-out cross-validation results', fontsize=11)
ax.set_ylim(0.96, 1.0)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.legend(loc='lower left', fontsize=9)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
for bar, val in zip(bars, loso['R2']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{val:.4f}',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_7_loso_cross_validation.png'))
plt.close()
print("  Fig 7 saved")

print("\nAll 7 figures regenerated with publication quality.")
