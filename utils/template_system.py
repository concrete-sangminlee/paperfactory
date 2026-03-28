"""Paper template system for common research types.

Provides skeleton paper_content dicts with pre-filled section headings,
suggested content outlines, and recommended figure/table plans.
"""

import os

TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
)

TEMPLATES = {
    "ml_comparison": {
        "name": "ML Model Comparison Study",
        "description": "Compare multiple ML/DL models on the same dataset with cross-validation",
        "sections": [
            {
                "heading": "INTRODUCTION",
                "outline": [
                    "Problem statement and engineering motivation",
                    "Literature review of existing ML approaches (15+ refs)",
                    "Research gaps: lack of systematic comparison, missing feature analysis",
                    "Objectives: (1) compare N models, (2) feature importance, (3) error analysis",
                ],
            },
            {
                "heading": "METHODOLOGY",
                "subsections": [
                    {
                        "heading": "Data description",
                        "outline": ["Source, size, features, target variable"],
                    },
                    {
                        "heading": "Feature engineering",
                        "outline": ["Feature categories, extraction methods, normalization"],
                    },
                    {
                        "heading": "ML models",
                        "outline": ["Model architectures, hyperparameters, training details"],
                    },
                    {
                        "heading": "Evaluation protocol",
                        "outline": ["K-fold CV, metrics (R², RMSE, MAE or Accuracy, F1)"],
                    },
                ],
            },
            {
                "heading": "RESULTS AND DISCUSSION",
                "subsections": [
                    {
                        "heading": "Overall performance comparison",
                        "outline": ["Table + scatter plots"],
                    },
                    {
                        "heading": "Feature importance analysis",
                        "outline": ["Bar chart, physical interpretation"],
                    },
                    {
                        "heading": "Comparison with existing studies",
                        "outline": ["Literature comparison (3+ studies)"],
                    },
                    {
                        "heading": "Error analysis",
                        "outline": ["Residuals, Q-Q plot, parameter-dependent errors"],
                    },
                ],
            },
            {
                "heading": "CONCLUSIONS",
                "outline": [
                    "Summary of key findings (numbered)",
                    "Limitations (3+)",
                    "Future work directions",
                ],
            },
        ],
        "figures_plan": [
            "Data distribution/visualization",
            "Feature importance ranking",
            "Model comparison scatter plots",
            "Cross-validation box plots",
            "Error analysis (residuals, Q-Q)",
            "Parameter-dependent error plot",
        ],
        "tables_plan": [
            "Overall performance comparison",
            "Cross-validation statistics (mean ± std)",
            "Feature summary table",
            "Dataset parameters",
        ],
    },
    "experimental": {
        "name": "Experimental Study",
        "description": "Wind tunnel, shake table, or material testing with data analysis",
        "sections": [
            {
                "heading": "INTRODUCTION",
                "outline": [
                    "Engineering context and motivation",
                    "Review of previous experimental studies",
                    "Limitations of existing data/knowledge",
                    "Scope and objectives of the present study",
                ],
            },
            {
                "heading": "EXPERIMENTAL PROGRAM",
                "subsections": [
                    {"heading": "Test specimens", "outline": ["Geometry, materials, fabrication"]},
                    {
                        "heading": "Test setup and instrumentation",
                        "outline": ["Equipment, sensors, data acquisition"],
                    },
                    {"heading": "Loading protocol", "outline": ["Loading type, rate, sequence"]},
                ],
            },
            {
                "heading": "RESULTS",
                "subsections": [
                    {"heading": "Observed behavior", "outline": ["Failure modes, crack patterns"]},
                    {
                        "heading": "Load-displacement response",
                        "outline": ["Hysteresis curves, envelope curves"],
                    },
                    {"heading": "Strain distribution", "outline": ["Strain gauge data analysis"]},
                ],
            },
            {
                "heading": "DISCUSSION",
                "subsections": [
                    {
                        "heading": "Comparison with code predictions",
                        "outline": ["ACI, ASCE, Eurocode comparison"],
                    },
                    {"heading": "Parametric effects", "outline": ["Effect of key variables"]},
                    {"heading": "Design implications", "outline": ["Practical recommendations"]},
                ],
            },
            {
                "heading": "CONCLUSIONS",
                "outline": [
                    "Key experimental findings",
                    "Comparison with analytical predictions",
                    "Practical design recommendations",
                    "Limitations and future work",
                ],
            },
        ],
        "figures_plan": [
            "Test setup photograph/schematic",
            "Specimen geometry and reinforcement details",
            "Failure mode photographs",
            "Load-displacement curves",
            "Strain distribution plots",
            "Parametric comparison charts",
        ],
        "tables_plan": [
            "Specimen details and material properties",
            "Test matrix",
            "Summary of test results",
            "Comparison with code predictions",
        ],
    },
    "numerical": {
        "name": "Numerical/Computational Study",
        "description": "FEM, CFD, or other computational analysis with validation",
        "sections": [
            {
                "heading": "INTRODUCTION",
                "outline": [
                    "Problem background",
                    "Review of numerical studies",
                    "Limitations of existing approaches",
                    "Objectives and scope",
                ],
            },
            {
                "heading": "NUMERICAL MODEL",
                "subsections": [
                    {"heading": "Model description", "outline": ["Software, element types, mesh"]},
                    {"heading": "Material models", "outline": ["Constitutive laws, parameters"]},
                    {
                        "heading": "Boundary conditions and loading",
                        "outline": ["Supports, loads, analysis type"],
                    },
                    {
                        "heading": "Model validation",
                        "outline": ["Comparison with experimental data"],
                    },
                ],
            },
            {
                "heading": "PARAMETRIC STUDY",
                "subsections": [
                    {
                        "heading": "Parameters investigated",
                        "outline": ["Variable ranges, analysis matrix"],
                    },
                    {"heading": "Results", "outline": ["Effect of each parameter"]},
                ],
            },
            {
                "heading": "DISCUSSION",
                "outline": [
                    "Physical interpretation of results",
                    "Comparison with analytical solutions",
                    "Practical implications",
                ],
            },
            {
                "heading": "CONCLUSIONS",
                "outline": [
                    "Model validation results",
                    "Key parametric findings",
                    "Design recommendations",
                    "Limitations",
                ],
            },
        ],
        "figures_plan": [
            "FEM mesh and model geometry",
            "Validation: numerical vs experimental",
            "Deformation/stress contour plots",
            "Parametric study results",
            "Sensitivity analysis charts",
            "Design chart/nomogram",
        ],
        "tables_plan": [
            "Material properties used",
            "Model validation comparison",
            "Parametric study matrix",
            "Summary of key results",
        ],
    },
    "review": {
        "name": "Review/State-of-the-Art Paper",
        "description": "Comprehensive literature review with classification and gap analysis",
        "sections": [
            {
                "heading": "INTRODUCTION",
                "outline": [
                    "Topic definition and scope",
                    "Motivation for the review",
                    "Review methodology (search strategy, databases, criteria)",
                    "Paper organization",
                ],
            },
            {
                "heading": "CLASSIFICATION FRAMEWORK",
                "outline": [
                    "Taxonomy of reviewed studies",
                    "Classification criteria",
                ],
            },
            {
                "heading": "REVIEW OF METHODS",
                "subsections": [
                    {
                        "heading": "Category 1",
                        "outline": ["Methods, key findings, strengths/limitations"],
                    },
                    {
                        "heading": "Category 2",
                        "outline": ["Methods, key findings, strengths/limitations"],
                    },
                    {
                        "heading": "Category 3",
                        "outline": ["Methods, key findings, strengths/limitations"],
                    },
                ],
            },
            {
                "heading": "DISCUSSION",
                "subsections": [
                    {
                        "heading": "Comparative analysis",
                        "outline": ["Performance comparison across categories"],
                    },
                    {
                        "heading": "Research gaps",
                        "outline": ["Identified gaps and unresolved challenges"],
                    },
                    {"heading": "Future directions", "outline": ["Promising research directions"]},
                ],
            },
            {
                "heading": "CONCLUSIONS",
                "outline": [
                    "Summary of the state of the art",
                    "Key research gaps",
                    "Recommended future research priorities",
                ],
            },
        ],
        "figures_plan": [
            "Publication trend chart (papers per year)",
            "Classification taxonomy diagram",
            "Performance comparison chart",
            "Research gap visualization",
        ],
        "tables_plan": [
            "Summary of reviewed studies",
            "Method comparison matrix",
            "Research gap classification",
        ],
    },
}


def list_templates() -> list:
    """List available paper templates."""
    return [
        {"key": k, "name": v["name"], "description": v["description"]} for k, v in TEMPLATES.items()
    ]


def get_template(template_key: str) -> dict:
    """Get a specific template."""
    if template_key not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(f"Unknown template '{template_key}'. Available: {available}")
    return TEMPLATES[template_key]


def generate_skeleton(template_key: str, topic: str = "", journal_key: str = "") -> dict:
    """Generate a paper_content skeleton from a template.

    Returns a paper_content dict with pre-filled section headings and
    outline comments that can be filled in during the pipeline.
    """
    template = get_template(template_key)

    sections = []
    for sec_template in template["sections"]:
        section = {"heading": sec_template["heading"]}

        outline = sec_template.get("outline", [])
        if outline:
            section["content"] = "[OUTLINE]\n" + "\n".join(f"- {item}" for item in outline)

        subsections = []
        for sub_template in sec_template.get("subsections", []):
            sub = {"heading": sub_template["heading"]}
            sub_outline = sub_template.get("outline", [])
            if sub_outline:
                sub["content"] = "[OUTLINE]\n" + "\n".join(f"- {item}" for item in sub_outline)
            subsections.append(sub)

        if subsections:
            section["subsections"] = subsections
        sections.append(section)

    return {
        "title": f"[FILL] {topic}" if topic else "[FILL TITLE]",
        "authors": "[FILL AUTHORS]",
        "abstract": "[FILL ABSTRACT — max 250 words]",
        "keywords": "[FILL KEYWORDS — 3-7 terms separated by semicolons]",
        "sections": sections,
        "tables": [
            {"caption": f"[PLAN] {t}", "headers": [], "rows": []}
            for t in template.get("tables_plan", [])
        ],
        "references": [],
        "figure_captions": [f"[PLAN] {f}" for f in template.get("figures_plan", [])],
        "data_availability": "[FILL]",
        "_template": template_key,
        "_journal": journal_key,
    }
