import json
import os
import logging
from utils.claude_cli import call_claude
from utils.code_runner import DATA_DIR

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior research scientist specializing in structural engineering and AI/ML.
Your task is to analyze experimental results and write scholarly interpretations.
Provide rigorous statistical analysis, comparison with existing literature, and identify limitations.
Always respond in English."""


def run(topic: str, design: dict, literature: dict, code_results: dict, journal_guideline: dict, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback("Analyzing research results...")

    research_design = design.get("design", {})
    review = literature.get("review", {})

    data_summaries = _load_data_summaries()

    analysis_prompt = f"""Analyze the following research results for a scholarly paper.

Topic: "{topic}"
Target Journal: {journal_guideline.get('journal_name', '')}
Research Objective: {research_design.get('research_objective', '')}
Hypotheses: {json.dumps(research_design.get('hypotheses', []))}

Methodology Summary: {research_design.get('methodology', {}).get('approach', '')}
Evaluation Metrics: {json.dumps(research_design.get('methodology', {}).get('evaluation_metrics', []))}

Code Execution Output:
{code_results.get('stdout', 'No output available')[:5000]}

Generated Figures: {json.dumps([os.path.basename(f) for f in code_results.get('figures', [])])}

Data Summaries: {json.dumps(data_summaries)}

State of the Art (from literature): {review.get('state_of_the_art', '')}
Key Papers for Comparison: {json.dumps([p.get('key_findings', '') for p in review.get('key_papers', [])[:5]])}

Provide a thorough scholarly analysis. Respond in JSON format:
{{
    "results_summary": "Concise summary of key results",
    "detailed_findings": [
        {{
            "finding": "Description of finding",
            "evidence": "Supporting data/metrics",
            "significance": "Statistical/practical significance",
            "comparison_to_literature": "How it compares to existing work"
        }}
    ],
    "hypothesis_evaluation": [
        {{
            "hypothesis": "The hypothesis",
            "supported": true/false,
            "evidence": "Supporting/contradicting evidence"
        }}
    ],
    "performance_comparison": {{
        "proposed_method": {{"metric1": "value1"}},
        "baselines": [{{"name": "baseline", "metric1": "value1"}}]
    }},
    "discussion_points": [
        {{
            "point": "Discussion topic",
            "analysis": "Detailed scholarly analysis"
        }}
    ],
    "limitations": ["limitation1", "limitation2"],
    "future_work": ["direction1", "direction2"],
    "figure_descriptions": [
        {{
            "filename": "figure filename",
            "caption": "Suggested figure caption",
            "description": "What the figure shows and its significance"
        }}
    ]
}}"""

    analysis_raw = call_claude(analysis_prompt, system=SYSTEM_PROMPT, timeout=300)
    try:
        cleaned = analysis_raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        analysis = json.loads(cleaned)
    except json.JSONDecodeError:
        analysis = {"raw_analysis": analysis_raw}

    if progress_callback:
        progress_callback("Result analysis completed.")

    return {
        "analysis": analysis,
        "status": "completed",
    }


def _load_data_summaries() -> list[dict]:
    summaries = []
    if not os.path.exists(DATA_DIR):
        return summaries
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith(".csv"):
            try:
                import pandas as pd
                df = pd.read_csv(fpath)
                summaries.append({
                    "file": fname,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "describe": df.describe().to_dict(),
                })
            except Exception:
                summaries.append({"file": fname, "error": "Could not parse"})
        elif fname.endswith(".json"):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    summaries.append({"file": fname, "keys": list(data.keys())[:20]})
                elif isinstance(data, list):
                    summaries.append({"file": fname, "length": len(data)})
            except Exception:
                summaries.append({"file": fname, "error": "Could not parse"})
    return summaries
