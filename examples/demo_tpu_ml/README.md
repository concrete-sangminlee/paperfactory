# Demo: TPU Wind Pressure + ML Prediction

This example demonstrates the full PaperFactory pipeline for a research paper on **machine learning prediction of peak wind pressure coefficients** on low-rise building roofs using the TPU aerodynamic database.

**Target journal:** JWEIA (Journal of Wind Engineering and Industrial Aerodynamics)

## Run

```bash
cd examples/demo_tpu_ml
python generate_paper.py
```

## What it generates

| Output | Path |
|--------|------|
| Word manuscript | `paper/*.docx` |
| 8 publication figures | `figures/*.png` |
| Quality report | `quality_report.json` |

## Pipeline steps executed

1. **Research Design** — hypothesis, methodology, feature engineering plan
2. **Code Execution** — synthetic TPU data generation, 3 ML models (RF, GBR, DNN), 10-fold CV
3. **Result Analysis** — model comparison, feature importance, wind direction analysis, residuals
4. **Paper Assembly** — JWEIA-formatted Word document with all sections, tables, figures
5. **Quality Check** — automated validation against journal requirements

## Quality report

The `quality_report.json` contains checks for: abstract word count, body word count, required sections, reference count, recent reference ratio, figure/table counts, keywords, highlights, and data availability.

## Note

This demo uses **synthetic data** to demonstrate the framework. For a real paper, the pipeline would use actual TPU database records obtained from https://db.wind.arch.t-kougei.ac.jp/.
