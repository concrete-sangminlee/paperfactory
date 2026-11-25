# Demo: Peak Wind Pressure Prediction from Short Records (JWEIA)

This example runs the full PaperFactory pipeline for a paper on **machine learning
prediction of peak wind pressure coefficients** on low-rise building roofs: from the
first 12.5% of a TPU-style wind-tunnel pressure record, predict the peak (minimum) Cp
over the unobserved remaining 87.5%.

**Target journal:** JWEIA (Journal of Wind Engineering and Industrial Aerodynamics)

## Run

```bash
cd examples/demo_tpu_ml
python generate_paper.py      # ~3 min
```

## What it generates

| Output | Path |
|--------|------|
| Word manuscript (JWEIA format, ~9,000-word body) | `paper/*.docx` |
| 8 publication figures | `figures/*.png` |
| Quality report (12 checks, 100/100) | `quality_report.json` |

## Study design

1. **Data** — synthetic TPU-style records (144 roof taps × 36 wind directions = 5,184
   records) from a generator with direction-dependent mean/RMS fields, conical-vortex
   suction at oblique angles and intermittent suction events (fixed random seeds).
2. **Leakage-free task** — 14 statistical, spectral and spatial-directional features
   are computed from an observation window (first 512 samples); the target comes only
   from the disjoint prediction window, so no input contains target samples.
3. **Models** — Random Forest, Gradient Boosting, DNN, plus two training-free
   references (observed short-record peak, Gaussian peak-factor extrapolation).
4. **Validation** — 10-fold cross-validation grouped by wind direction, scaler fitted
   on training folds only, paired Wilcoxon tests and bootstrap confidence intervals.
5. **Leakage controls** — guard gap between windows, permuted targets, a deliberately
   leaky formulation for comparison, and a feature-group ablation.

Every comparative statement in the manuscript (best model, significance, feature
groups, wind-direction effect, residual behaviour) is computed from the results, so the
text cannot contradict the numbers.

## Note

This demo uses **synthetic data** to demonstrate the framework. For a real paper, the
pipeline would use measured TPU database records from https://db.wind.arch.t-kougei.ac.jp/.
