"""Additional engineering analyses for paper enhancement."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os, warnings, json
warnings.filterwarnings('ignore')

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'figures')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'data')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

C_BLUE='#1B4F72'; C_RED='#C0392B'; C_GREEN='#1E8449'
C_ORANGE='#D35400'; C_PURPLE='#6C3483'; C_GRAY='#5D6D7E'; C_CYAN='#148F77'

df = pd.read_csv(os.path.join(DATA_DIR, 'tpu_highrise_synthetic.csv'))
np.random.seed(42)

feature_cols = ['side_ratio','aspect_ratio','wind_dir_sin','wind_dir_cos','face_id','z_H','tap_position']
X = df[feature_cols].values; y = df['Cp_mean'].values; y_rms = df['Cp_rms'].values
scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.15, random_state=42)
_, _, yr_tr, yr_te = train_test_split(X_sc, y_rms, test_size=0.15, random_state=42)

def eng_feat(X):
    f=[X]
    for i in range(X.shape[1]):
        for j in range(i+1,X.shape[1]):
            f.append((X[:,i]*X[:,j]).reshape(-1,1))
    f.append(X**2)
    for c in [0,1,2,3,4]: f.append((X[:,c]**3).reshape(-1,1))
    f.append((X[:,2]*X[:,4]*X[:,5]).reshape(-1,1))
    f.append((X[:,3]*X[:,4]*X[:,5]).reshape(-1,1))
    f.append((X[:,0]*X[:,2]*X[:,4]).reshape(-1,1))
    f.append((X[:,0]*X[:,5]*X[:,4]).reshape(-1,1))
    return np.hstack(f)

print("Training models...")
X_tr2,_,y_tr2,_ = train_test_split(X_tr,y_tr,test_size=0.176,random_state=42)
model = GradientBoostingRegressor(n_estimators=600,max_depth=10,learning_rate=0.05,subsample=0.8,random_state=42)
model.fit(eng_feat(X_tr2), y_tr2)
model_rms = GradientBoostingRegressor(n_estimators=600,max_depth=10,learning_rate=0.05,subsample=0.8,random_state=42)
_,_,yr_tr2,_ = train_test_split(X_tr,yr_tr,test_size=0.176,random_state=42)
model_rms.fit(eng_feat(X_tr2), yr_tr2)

# ============================================================
# Analysis 1: Face-specific error analysis
# ============================================================
print("\n=== Face-specific prediction error analysis ===")
all_idx = np.arange(len(df))
_, te_idx = train_test_split(all_idx, test_size=0.15, random_state=42)
df_te = df.iloc[te_idx].copy().reset_index(drop=True)
X_te_full = scaler.transform(df_te[feature_cols].values)
pred_te = model.predict(eng_feat(X_te_full))
df_te['pred'] = pred_te
df_te['error'] = df_te['Cp_mean'] - pred_te

face_errors = []
for face in ['windward','leeward','side_left','side_right']:
    sub = df_te[df_te['face']==face]
    r2 = r2_score(sub['Cp_mean'], sub['pred'])
    rmse = np.sqrt(mean_squared_error(sub['Cp_mean'], sub['pred']))
    mae = np.abs(sub['error']).mean()
    bias = sub['error'].mean()
    face_errors.append({'Face':face, 'R2':r2, 'RMSE':rmse, 'MAE':mae, 'Bias':bias, 'N':len(sub)})
    print(f"  {face:12s}: R2={r2:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}, Bias={bias:+.5f}")

pd.DataFrame(face_errors).to_csv(os.path.join(DATA_DIR, 'face_errors.csv'), index=False)

# ============================================================
# Analysis 2: All-face height profiles for multiple D/B
# ============================================================
print("\n=== All-face height profiles ===")
heights = np.linspace(0.05, 0.95, 40)
all_profiles = []
for ratio in [0.5, 1.0, 2.0, 3.0, 4.0]:
    for fi, face in enumerate(['windward','leeward','side_left','side_right']):
        for z in heights:
            x_in = np.array([[ratio, 200/30, 0, 1, fi, z, 0.5]])  # theta=0
            cp = model.predict(eng_feat(scaler.transform(x_in)))[0]
            rms = model_rms.predict(eng_feat(scaler.transform(x_in)))[0]
            all_profiles.append({'side_ratio':ratio,'face':face,'z_H':z,'Cp_mean':cp,'Cp_rms':rms})

df_profiles = pd.DataFrame(all_profiles)
df_profiles.to_csv(os.path.join(DATA_DIR, 'all_face_height_profiles.csv'), index=False)

# ============================================================
# Analysis 3: Peak factor and design wind pressure
# ============================================================
print("\n=== Peak factor and design pressure analysis ===")
# Peak Cp estimated as Cp_mean + g * Cp_rms (g = peak factor ~ 3.5 for Gaussian)
g_peak = 3.5
peak_analysis = []
param_ratios = np.linspace(0.3, 5.0, 50)
for ratio in param_ratios:
    for fi, face in enumerate(['windward','leeward','side_left','side_right']):
        # At z/H=0.75, sweep all wind directions for max
        max_peak_pos = -999; max_peak_neg = 999
        for wd in range(0, 360, 5):
            x_in = np.array([[ratio, 200/30, np.sin(np.radians(wd)),
                             np.cos(np.radians(wd)), fi, 0.75, 0.5]])
            cp_m = model.predict(eng_feat(scaler.transform(x_in)))[0]
            cp_r = model_rms.predict(eng_feat(scaler.transform(x_in)))[0]
            peak_pos = cp_m + g_peak * cp_r
            peak_neg = cp_m - g_peak * cp_r
            max_peak_pos = max(max_peak_pos, peak_pos)
            max_peak_neg = min(max_peak_neg, peak_neg)

        peak_analysis.append({
            'side_ratio': ratio, 'face': face,
            'peak_positive': max_peak_pos,
            'peak_negative': max_peak_neg,
        })

df_peak = pd.DataFrame(peak_analysis)
df_peak.to_csv(os.path.join(DATA_DIR, 'peak_factor_analysis.csv'), index=False)

# Print critical peaks
sf_peak = df_peak[df_peak['face'].isin(['side_left','side_right'])]
print(f"  Max peak suction (side wall): Cp_peak = {sf_peak['peak_negative'].min():.3f}")
print(f"  At D/B = {sf_peak.loc[sf_peak['peak_negative'].idxmin(), 'side_ratio']:.2f}")

# ============================================================
# Analysis 4: Critical wind direction identification
# ============================================================
print("\n=== Critical wind direction analysis ===")
critical_dirs = []
for ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    max_base_shear_wd = 0; max_bs = -999
    for wd in range(0, 360, 5):
        # Estimate base shear contribution (sum of Cp across windward-leeward)
        cp_ww = model.predict(eng_feat(scaler.transform(
            np.array([[ratio, 200/30, np.sin(np.radians(wd)), np.cos(np.radians(wd)), 0, 0.5, 0.5]]))))[0]
        cp_lw = model.predict(eng_feat(scaler.transform(
            np.array([[ratio, 200/30, np.sin(np.radians(wd)), np.cos(np.radians(wd)), 1, 0.5, 0.5]]))))[0]
        net = cp_ww - cp_lw  # net along-wind pressure
        if net > max_bs:
            max_bs = net; max_base_shear_wd = wd
    critical_dirs.append({'side_ratio':ratio, 'critical_wd':max_base_shear_wd, 'net_Cp':max_bs})
    print(f"  D/B={ratio}: critical wind dir = {max_base_shear_wd} deg, net Cp = {max_bs:.3f}")

pd.DataFrame(critical_dirs).to_csv(os.path.join(DATA_DIR, 'critical_directions.csv'), index=False)

# ============================================================
# Fig 8: Face-specific height profiles (4-panel)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(7.5, 4.5), sharey=True)
colors_r = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE]
face_titles = ['(a) Windward', '(b) Leeward', '(c) Side wall (left)', '(d) Side wall (right)']
for fi, (face, title) in enumerate(zip(['windward','leeward','side_left','side_right'], face_titles)):
    ax = axes[fi]
    for ci, ratio in enumerate([0.5, 1.0, 2.0, 3.0, 4.0]):
        sub = df_profiles[(df_profiles['side_ratio']==ratio) & (df_profiles['face']==face)]
        ax.plot(sub['Cp_mean'], sub['z_H'], '-', color=colors_r[ci], lw=1.5,
                label=f'D/B={ratio:.1f}')
    ax.set_xlabel('Cp,mean', fontsize=10)
    if fi == 0: ax.set_ylabel('z/H', fontsize=11)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_ylim(0, 1)
    if fi == 0: ax.legend(fontsize=7, loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_8_all_face_height_profiles.png'))
plt.close()
print("\n  Fig 8: All-face height profiles saved")

# ============================================================
# Fig 9: Peak Cp envelope (design critical)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
for face, color, marker, label in [
    ('windward', C_RED, 'o', 'Windward (peak positive)'),
    ('side_left', C_BLUE, 's', 'Side wall (peak negative)'),
    ('leeward', C_GREEN, '^', 'Leeward (peak negative)')]:
    sub = df_peak[df_peak['face']==face]
    if 'windward' in face:
        ax.plot(sub['side_ratio'], sub['peak_positive'], f'{marker}-', color=color, ms=3, lw=1.5, label=label)
    else:
        ax.plot(sub['side_ratio'], sub['peak_negative'], f'{marker}-', color=color, ms=3, lw=1.5, label=label)

ax.axhline(y=0, color='gray', lw=0.5, ls='--')
# ASCE 7 GCp reference for C&C (zone 4, area=20 sq ft): approximately +1.0 / -1.4
ax.axhline(y=1.0, color=C_RED, lw=1, ls=':', alpha=0.6, label='ASCE 7 C&C GCp (pos)')
ax.axhline(y=-1.4, color=C_BLUE, lw=1, ls=':', alpha=0.6, label='ASCE 7 C&C GCp (neg)')
ax.set_xlabel('Side ratio, D/B', fontsize=12)
ax.set_ylabel('Peak wind pressure coefficient, Cp,peak', fontsize=12)
ax.legend(fontsize=8, frameon=True, loc='lower left')
ax.grid(alpha=0.2, lw=0.4)
ax.set_xlim(0.3, 5.0)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_9_peak_cp_envelope.png'))
plt.close()
print("  Fig 9: Peak Cp envelope saved")

# ============================================================
# Fig 10: Net along-wind Cp (base shear indicator) vs D/B
# ============================================================
net_cp_data = []
for ratio in param_ratios:
    max_net = -999
    for wd in range(0, 360, 5):
        cp_ww = model.predict(eng_feat(scaler.transform(
            np.array([[ratio, 200/30, np.sin(np.radians(wd)), np.cos(np.radians(wd)), 0, 0.5, 0.5]]))))[0]
        cp_lw = model.predict(eng_feat(scaler.transform(
            np.array([[ratio, 200/30, np.sin(np.radians(wd)), np.cos(np.radians(wd)), 1, 0.5, 0.5]]))))[0]
        net = cp_ww - cp_lw
        if net > max_net: max_net = net
    net_cp_data.append({'side_ratio': ratio, 'net_Cp_max': max_net})

df_net = pd.DataFrame(net_cp_data)
df_net.to_csv(os.path.join(DATA_DIR, 'net_along_wind_cp.csv'), index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_net['side_ratio'], df_net['net_Cp_max'], 'o-', color=C_BLUE, ms=3, lw=1.8)
# ASCE 7 net (windward - leeward)
asce_net = []
for r in param_ratios:
    if r <= 1: lw = -0.5
    elif r <= 2: lw = -0.5 + 0.2*(r-1)
    elif r <= 4: lw = -0.3 + 0.1*(r-2)/2
    else: lw = -0.2
    asce_net.append(0.8 - lw)
ax.plot(param_ratios, asce_net, '--', color=C_RED, lw=1.5, label='ASCE 7-22')
ax.plot(df_net['side_ratio'], df_net['net_Cp_max'], '-', color=C_BLUE, lw=1.8, label='DL surrogate')
ax.set_xlabel('Side ratio, D/B', fontsize=12)
ax.set_ylabel('Net along-wind Cp (windward - leeward)', fontsize=12)
ax.legend(fontsize=10, frameon=True)
ax.grid(alpha=0.2, lw=0.4)
ax.set_xlim(0.3, 5.0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig_10_net_along_wind_cp.png'))
plt.close()
print("  Fig 10: Net along-wind Cp saved")

print("\n=== All additional analyses complete ===")
print(f"New data files: face_errors.csv, all_face_height_profiles.csv, peak_factor_analysis.csv,")
print(f"                critical_directions.csv, net_along_wind_cp.csv")
print(f"New figures: fig_8, fig_9, fig_10")
