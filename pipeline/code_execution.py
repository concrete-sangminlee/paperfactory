import json
import logging
from utils.claude_cli import call_claude
from utils.code_runner import run_python_code, list_generated_figures, FIGURES_DIR, DATA_DIR

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Python programmer specializing in structural engineering research and AI/ML.
You write clean, efficient, well-documented research code.
All figures must be saved to the FIGURES_DIR path variable (already defined).
All data must be saved to the DATA_DIR path variable (already defined).
Use matplotlib for figures (set DPI=300, use tight_layout).
Always use publicly available datasets or generate synthetic data for demonstration.
Always respond in English.
IMPORTANT: Output ONLY the Python code. No markdown, no explanations, no code blocks."""

MAX_RETRIES = 3


def run(topic: str, design: dict, literature: dict, journal_guideline: dict, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback("Generating research code...")

    research_design = design.get("design", {})
    methodology = research_design.get("methodology", {})
    data_plan = research_design.get("data_plan", {})
    expected_figs = research_design.get("paper_outline", {}).get("expected_figures", [])
    expected_tables = research_design.get("paper_outline", {}).get("expected_tables", [])

    # Build context from design, handling both parsed and raw designs
    raw_design = research_design.get("raw_design", "")
    objective = research_design.get("research_objective", "")
    approach = methodology.get("approach", "")

    if raw_design and not objective:
        # Design was not parsed as JSON, use raw text
        design_context = f"Research Design:\n{raw_design[:3000]}"
    else:
        design_context = f"""Research Objective: {objective}

Methodology:
- Approach: {approach}
- Steps: {json.dumps(methodology.get('steps', []), indent=2)}
- ML Models: {json.dumps(methodology.get('ml_models', []), indent=2)}
- Evaluation Metrics: {json.dumps(methodology.get('evaluation_metrics', []))}
- Baselines: {json.dumps(methodology.get('baseline_comparisons', []))}

Data Plan:
- Primary Dataset: {json.dumps(data_plan.get('primary_dataset', {}))}
- Splits: {json.dumps(data_plan.get('data_splits', {}))}

Expected Figures: {json.dumps(expected_figs)}
Expected Tables: {json.dumps(expected_tables)}"""

    # Add literature context
    review = literature.get("review", {})
    lit_context = review.get("suggested_methodology", "")

    code_prompt = f"""Write a complete Python research script for the following study.

Topic: "{topic}"
{design_context}

Literature-suggested approach: {lit_context[:500]}

Requirements:
1. Use publicly available datasets. If unavailable, generate REALISTIC synthetic data that simulates real structural engineering data.
2. Save ALL figures to FIGURES_DIR with descriptive filenames (e.g., 'fig_1_model_performance.png')
3. Save ALL result data to DATA_DIR as CSV or JSON
4. Set matplotlib DPI=300, use tight_layout()
5. Print all numerical results, metrics, and comparison tables to stdout
6. Include proper error handling
7. Use these packages only: numpy, pandas, scikit-learn, matplotlib, scipy (NO tensorflow, NO pytorch — use scikit-learn for ML)
8. The script must be self-contained and runnable
9. Generate at least 4 figures: data distribution, model comparison, confusion matrix, performance metrics

Write ONLY the Python code. No markdown, no code blocks, no explanations."""

    code = call_claude(code_prompt, system=SYSTEM_PROMPT, timeout=600)
    code = _clean_code(code)

    results = {"code": code, "execution_attempts": [], "figures": [], "data_files": []}

    for attempt in range(1, MAX_RETRIES + 1):
        if progress_callback:
            progress_callback(f"Executing research code (attempt {attempt}/{MAX_RETRIES})...")

        exec_result = run_python_code(code, timeout=600)
        results["execution_attempts"].append(exec_result)

        if exec_result["success"]:
            results["stdout"] = exec_result["stdout"]
            results["figures"] = list_generated_figures()
            results["status"] = "completed"

            if progress_callback:
                progress_callback(f"Code executed successfully. Generated {len(results['figures'])} figures.")
            return results

        if attempt < MAX_RETRIES:
            if progress_callback:
                progress_callback(f"Attempt {attempt} failed. Fixing code...")

            fix_prompt = f"""The following Python research code produced an error. Fix it.

ORIGINAL CODE:
{code}

ERROR:
{exec_result['stderr']}

STDOUT (partial):
{exec_result['stdout'][:2000]}

Fix the code and return ONLY the corrected Python code. No markdown, no code blocks, no explanations.
Common fixes:
- If a dataset URL is broken, generate synthetic data instead
- If a package is missing, use an alternative
- Fix any import errors or typos"""

            code = call_claude(fix_prompt, system=SYSTEM_PROMPT, timeout=600)
            code = _clean_code(code)
            results["code"] = code

    results["status"] = "failed"
    results["error"] = exec_result["stderr"]
    if progress_callback:
        progress_callback("Code execution failed after all retries.")
    return results


def _clean_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()
