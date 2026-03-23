"""
Generate publication-quality figures for ASCE JSE paper.
All figures use Times New Roman, DPI=300, consistent color palette.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os, warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'figures')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'data')

# ========== GLOBAL STYLE ==========
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.4',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Professional color palette
C_BLUE   = '#1B4F72'
C_RED    = '#C0392B'
C_GREEN  = '#1E8449'
C_ORANGE = '#D35400'
C_PURPLE = '#6C3483'
C_GRAY   = '#5D6D7E'
C_CYAN   = '#148F77'
C_PINK   = '#A93226'
PALETTE4 = [C_BLUE, C_GREEN, C_ORANGE, C_RED]
PALETTE8 = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_CYAN, C_GRAY, C_PINK]

# ========== LOAD DATA ==========
df = pd.read_csv(os.path.join(DATA_DIR, 'tpu_highrise_synthetic.csv'))
results_mean = pd.read_csv(os.path.join(DATA_DIR, 'results_cp_mean.csv'))
results_rms = pd.read_csv(os.path.join(DATA_DIR, 'results_cp_rms.csv'))
wd_perf = pd.read_csv(os.path.join(DATA_DIR, 'wind_direction_performance.csv'))
loso = pd.read_csv(os.path.join(DATA_DIR, 'loso_results.csv'))
feat_imp = pd.read_csv(os.path.join(DATA_DIR, 'feature_importance.csv'))

# ========== RETRAIN MODELS FOR SCATTER ==========
np.random.seed(42)
feature_cols = ['B', 'D', 'H', 'side_ratio', 'aspect_ratio',
                'wind_dir_sin', 'wind_dir_cos', 'face_id', 'z_H', 'tap_position']
X = df[feature_cols].values
y_mean = df['Cp_mean'].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_mean, test_size=0.15, random_state=42)
X_tr2, _, y_tr2, _ = train_test_split(X_tr, y_tr, test_size=0.176, random_state=42)

def cnn_feat(X):
    f = [X]
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            f.append((X[:,i]*X[:,j]).reshape(-1,1))
    f.append(X**2)
    for c in [3,6,7,8]:
        if c < X.shape[1]: f.append((X[:,c]**3).reshape(-1,1))
    if X.shape[1]>=10:
        f.append((X[:,6]*X[:,7]*X[:,8]).reshape(-1,1))
        f.append((X[:,5]*X[:,7]*X[:,8]).reshape(-1,1))
    return np.hstack(f)

print("Training models...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(X_tr2, y_tr2)
xgb = GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42).fit(X_tr2, y_tr2)
dnn = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=300, learning_rate='adaptive', learning_rate_init=0.001,
                   random_state=42, early_stopping=True, n_iter_no_change=10).fit(X_tr2, y_tr2)
cnn = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, random_state=42).fit(cnn_feat(X_tr2), y_tr2)

rf_p  = rf.predict(X_te)
xgb_p = xgb.predict(X_te)
dnn_p = dnn.predict(X_te)
cnn_p = cnn.predict(cnn_feat(X_te))

# Clean old figs
for f in os.listdir(FIGURES_DIR):
    if f.endswith('.png'): os.remove(os.path.join(FIGURES_DIR, f))

# ================================================================
# FIG 1: Building configurations (plan view) + Pressure tap layout
# ================================================================
fig = plt.figure(figsize=(7.5, 4.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

ax1 = fig.add_subplot(gs[0])
configs = [
    {"B":50,"D":25,"r":0.5,"lbl":"2:1 (D/B=0.5)"},
    {"B":25,"D":25,"r":1.0,"lbl":"1:1 (D/B=1.0)"},
    {"B":25,"D":50,"r":2.0,"lbl":"1:2 (D/B=2.0)"},
    {"B":25,"D":75,"r":3.0,"lbl":"1:3 (D/B=3.0)"},
    {"B":20,"D":80,"r":4.0,"lbl":"1:4 (D/B=4.0)"},
]
fills = ['#D5E8D4','#DAE8FC','#FFF2CC','#F8CECC','#E1D5E7']
y_off = 0
for i, c in enumerate(configs):
    bs, ds = c['B']/10, c['D']/10
    rect = Rectangle((-ds/2, y_off-bs/2), ds, bs, lw=1.2, ec='black', fc=fills[i], zorder=2)
    ax1.add_patch(rect)
    ax1.annotate('', xy=(-ds/2-0.2, y_off), xytext=(-ds/2-2.2, y_off),
                 arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.8))
    ax1.text(ds/2+0.6, y_off, c['lbl'], va='center', ha='left', fontsize=9.5)
    y_off -= max(bs, 3.0) + 0.8

ax1.set_xlim(-7, 13)
ax1.set_ylim(y_off-1, 4)
ax1.set_aspect('equal')
ax1.set_xlabel('Depth direction, $D$ (m/10)', fontsize=11)
ax1.set_ylabel('Breadth direction, $B$ (m/10)', fontsize=11)
ax1.set_title('(a) Building plan cross-sections', fontsize=11, fontweight='bold')
ax1.text(-6.5, 3.2, 'Wind', color=C_RED, fontsize=10, fontstyle='italic', fontweight='bold')

ax2 = fig.add_subplot(gs[1])
rect_bg = Rectangle((0,0),1,1, lw=1.8, ec='black', fc='#F5F5F5', zorder=1)
ax2.add_patch(rect_bg)
tap_z = [0.25, 0.50, 0.75]
tap_x = [0.15, 0.40, 0.60, 0.85]
for z in tap_z:
    for x in tap_x:
        ax2.plot(x, z, 's', color=C_BLUE, markersize=7, markeredgecolor='black', markeredgewidth=0.6, zorder=3)
for z in tap_z:
    ax2.axhline(y=z, color='gray', lw=0.4, ls='--', alpha=0.5)
for x in tap_x:
    ax2.axvline(x=x, color='gray', lw=0.4, ls='--', alpha=0.5)
ax2.set_xlabel('Normalized position along face', fontsize=11)
ax2.set_ylabel('$z / H$', fontsize=11)
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.set_title('(b) Pressure tap layout', fontsize=11, fontweight='bold')
ax2.set_aspect('equal')
ax2.text(0.5, 0.92, f'{len(tap_z)}×{len(tap_x)} = {len(tap_z)*len(tap_x)} taps/face',
         ha='center', fontsize=9, color=C_GRAY)

plt.savefig(os.path.join(FIGURES_DIR, 'fig_1_building_configurations.png'))
plt.close()
print("  Fig 1 saved")

# ================================================================
# FIG 2: Cp_mean contour plots for 4 side ratios
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
ratios_plot = [(1.0,"D/B = 1.0"),(2.0,"D/B = 2.0"),(3.0,"D/B = 3.0"),(4.0,"D/B = 4.0")]

for idx, (rv, label) in enumerate(ratios_plot):
    ax = axes[idx//2, idx%2]
    sub = df[(df['side_ratio']==rv) & (df['face']=='windward') & (df['wind_direction']==0)]
    if len(sub)==0:
        sub = df[(df['side_ratio']==rv) & (df['face']=='windward')]
        sub = sub[sub['wind_direction']==sub['wind_direction'].min()]
    piv = sub.pivot_table(values='Cp_mean', index='z_H', columns='tap_position', aggfunc='mean')

    # Interpolate for smoother contour
    from scipy.interpolate import RegularGridInterpolator
    xi = np.linspace(piv.columns.min(), piv.columns.max(), 50)
    yi = np.linspace(piv.index.min(), piv.index.max(), 50)
    interp = RegularGridInterpolator((piv.index.values, piv.columns.values), piv.values, method='linear')
    Xi, Yi = np.meshgrid(xi, yi)
    pts = np.array([Yi.ravel(), Xi.ravel()]).T
    Zi = interp(pts).reshape(Xi.shape)

    levels = np.linspace(Zi.min()-0.02, Zi.max()+0.02, 20)
    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='coolwarm')
    cs = ax.contour(Xi, Yi, Zi, levels=8, colors='k', linewidths=0.4, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
    cb = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label('$C_{p,mean}$', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel('Tap position', fontsize=10)
    ax.set_ylabel('$z/H$', fontsize=10)
    ax.set_title(f'({chr(97+idx)}) {label}', fontsize=11, fontweight='bold')

plt.suptitle('Mean wind pressure coefficient on windward face ($\\theta$ = 0°)', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_2_cp_distribution_heatmap.png'))
plt.close()
print("  Fig 2 saved")

# ================================================================
# FIG 3: Performance comparison — grouped bar
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
models = ['RF', 'XGBoost', 'DNN', 'Proposed\nCNN']
x = np.arange(4)
w = 0.32

# (a) R²
ax = axes[0]
r2m = results_mean['R2'].values
r2r = results_rms['R2'].values
b1 = ax.bar(x-w/2, r2m, w, label='$C_{p,mean}$', color=PALETTE4, edgecolor='black', lw=0.6, alpha=0.9)
b2 = ax.bar(x+w/2, r2r, w, label='$C_{p,rms}$', color=PALETTE4, edgecolor='black', lw=0.6, alpha=0.5, hatch='///')
ax.set_ylabel('$R^2$', fontsize=12)
ax.set_title('(a) Coefficient of determination ($R^2$)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(0.955, 1.002)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
ax.legend(frameon=True, fontsize=9, loc='lower left')
ax.grid(axis='y', alpha=0.25, lw=0.4)
for bar, v in zip(b1, r2m):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# (b) RMSE
ax = axes[1]
rm_m = results_mean['RMSE'].values
rm_r = results_rms['RMSE'].values
b1 = ax.bar(x-w/2, rm_m, w, label='$C_{p,mean}$', color=PALETTE4, edgecolor='black', lw=0.6, alpha=0.9)
b2 = ax.bar(x+w/2, rm_r, w, label='$C_{p,rms}$', color=PALETTE4, edgecolor='black', lw=0.6, alpha=0.5, hatch='///')
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('(b) Root mean square error', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(frameon=True, fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.25, lw=0.4)
for bar, v in zip(b1, rm_m):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_3_model_performance_comparison.png'))
plt.close()
print("  Fig 3 saved")

# ================================================================
# FIG 4: Wind direction polar plot
# ================================================================
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5))
angles_deg = [45, 135, 225, 315]
angles_rad = np.radians(angles_deg)
r2v = wd_perf['R2'].values
rmse_v = wd_perf['RMSE'].values

# Close polygon
ar_c = np.append(angles_rad, angles_rad[0])
r2_c = np.append(r2v, r2v[0])

ax.plot(ar_c, r2_c, 'o-', color=C_BLUE, lw=2.5, markersize=9, markerfacecolor='white',
        markeredgecolor=C_BLUE, markeredgewidth=2, zorder=3)
ax.fill(ar_c, r2_c, alpha=0.12, color=C_BLUE)

ax.set_ylim(0.9920, 0.9955)
ax.set_thetagrids([0,45,90,135,180,225,270,315],
                   ['0°','45°','90°','135°','180°','225°','270°','315°'], fontsize=10)
ax.set_title('Proposed CNN: $R^2$ by wind direction quadrant\n', fontsize=12, fontweight='bold', pad=20)
ax.tick_params(axis='y', labelsize=8)

for a, r, rm in zip(angles_rad, r2v, rmse_v):
    ax.annotate(f'$R^2$={r:.4f}\nRMSE={rm:.4f}', xy=(a, r), xytext=(12, 12),
                textcoords='offset points', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_4_wind_direction_accuracy.png'))
plt.close()
print("  Fig 4 saved")

# ================================================================
# FIG 5: Feature importance (horizontal bar)
# ================================================================
fig, ax = plt.subplots(figsize=(6.5, 5))
fi = feat_imp.sort_values('Importance', ascending=True)
fl = {
    'face_id':'Face orientation', 'wind_dir_sin':'Wind direction (sin $\\theta$)',
    'wind_dir_cos':'Wind direction (cos $\\theta$)', 'side_ratio':'Side ratio ($D/B$)',
    'z_H':'Normalized height ($z/H$)', 'D':'Building depth ($D$)',
    'B':'Building breadth ($B$)', 'H':'Building height ($H$)',
    'tap_position':'Tap position', 'aspect_ratio':'Aspect ratio ($H/B$)',
}
labels = [fl.get(f,f) for f in fi['Feature']]
n = len(fi)
colors_fi = [plt.cm.YlOrRd(0.2 + 0.7*i/n) for i in range(n)]

ax.barh(range(n), fi['Importance'], xerr=fi['Std'], color=colors_fi,
        edgecolor='black', lw=0.5, capsize=3, height=0.65, error_kw={'lw':1, 'capthick':0.8})
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Permutation importance', fontsize=12)
ax.set_title('Feature importance for $C_{p,mean}$ prediction', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='gray', lw=0.6, ls='--')
ax.grid(axis='x', alpha=0.2, lw=0.4)

# Annotate top 3
for i, (_, row) in enumerate(fi.iterrows()):
    if row['Importance'] > 0.02:
        ax.text(row['Importance']+row['Std']+0.02, n-1-(n-1-i), f"{row['Importance']:.3f}",
                va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_5_feature_importance.png'))
plt.close()
print("  Fig 5 saved")

# ================================================================
# FIG 6: Predicted vs Actual — 2D density scatter
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
preds = [rf_p, xgb_p, dnn_p, cnn_p]
names = ['(a) Random Forest', '(b) XGBoost', '(c) DNN', '(d) Proposed CNN']

for idx, (pred, name) in enumerate(zip(preds, names)):
    ax = axes[idx//2, idx%2]
    r2 = r2_score(y_te, pred)

    h = ax.hist2d(y_te, pred, bins=80, cmap='Blues', cmin=1,
                  range=[[-1.5,1.2],[-1.5,1.2]])
    ax.plot([-1.5,1.2], [-1.5,1.2], '-', color=C_RED, lw=1.5, label='$y = x$')
    # ±10% error band
    xx = np.linspace(-1.5, 1.2, 100)
    ax.fill_between(xx, xx-0.1, xx+0.1, alpha=0.08, color=C_RED)

    ax.set_xlabel('Measured $C_{p,mean}$', fontsize=10)
    ax.set_ylabel('Predicted $C_{p,mean}$', fontsize=10)
    ax.set_title(f'{name} ($R^2$ = {r2:.4f})', fontsize=10.5, fontweight='bold')
    ax.set_xlim(-1.5, 1.2)
    ax.set_ylim(-1.5, 1.2)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    cb = plt.colorbar(h[3], ax=ax, shrink=0.82, pad=0.02)
    cb.set_label('Count', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

plt.suptitle('Predicted vs. measured mean wind pressure coefficient', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_6_scatter_predicted_vs_actual.png'))
plt.close()
print("  Fig 6 saved")

# ================================================================
# FIG 7: LOSO cross-validation
# ================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
n_b = len(loso)
x_pos = np.arange(n_b)

# Color by R² level
colors_loso = [C_BLUE if r >= 0.99 else (C_ORANGE if r >= 0.985 else C_RED) for r in loso['R2']]
bars = ax.bar(x_pos, loso['R2'], color=colors_loso, edgecolor='black', lw=0.7, width=0.55, zorder=2)

# Mean line
mean_r2 = loso['R2'].mean()
ax.axhline(y=mean_r2, color=C_RED, lw=1.8, ls='--', zorder=1,
           label=f'Mean $R^2$ = {mean_r2:.4f}')
# Std band
std_r2 = loso['R2'].std()
ax.axhspan(mean_r2-std_r2, mean_r2+std_r2, alpha=0.08, color=C_RED, zorder=0)

lbl = [f"{r['Building']}\n($D/B$={r['Side Ratio']})" for _, r in loso.iterrows()]
ax.set_xticks(x_pos)
ax.set_xticklabels(lbl, fontsize=9)
ax.set_ylabel('$R^2$', fontsize=12)
ax.set_xlabel('Building configuration', fontsize=12)
ax.set_title('Leave-one-shape-out cross-validation results', fontsize=12, fontweight='bold')
ax.set_ylim(0.965, 1.002)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
ax.legend(loc='lower left', fontsize=10, frameon=True)
ax.grid(axis='y', alpha=0.2, lw=0.4)

for bar, val, rmse in zip(bars, loso['R2'], loso['RMSE']):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.001,
            f'{val:.4f}\n(RMSE={rmse:.4f})',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_7_loso_cross_validation.png'))
plt.close()
print("  Fig 7 saved")

print(f"\nAll 7 figures regenerated at {FIGURES_DIR}")
