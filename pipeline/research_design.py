import json
import logging
from utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior research scientist specializing in structural engineering and AI/ML applications.
Your task is to design a rigorous research methodology based on the literature review findings.
The methodology should be novel, scientifically sound, and suitable for the target journal.
Always respond in English."""


def run(topic: str, literature: dict, journal_guideline: dict, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback("Designing research methodology...")

    review = literature.get("review", {})

    design_prompt = f"""Based on the following literature review, design a complete research methodology.

Topic: "{topic}"
Target journal: {journal_guideline.get('journal_name', '')}

Literature Summary: {review.get('summary', '')}

Research Gaps: {json.dumps(review.get('research_gaps', []))}

State of the Art: {review.get('state_of_the_art', '')}

Available Datasets: {json.dumps(review.get('available_datasets', []))}

Potential Contributions: {json.dumps(review.get('potential_contributions', []))}

Suggested Methodology: {review.get('suggested_methodology', '')}

Design a complete research plan. Respond in JSON format:
{{
    "research_title": "Proposed paper title",
    "research_objective": "Clear statement of the research objective",
    "hypotheses": ["hypothesis1", "hypothesis2", ...],
    "methodology": {{
        "approach": "Overall methodological approach",
        "steps": [
            {{
                "step_number": 1,
                "name": "Step name",
                "description": "Detailed description",
                "tools_required": ["tool1", "tool2"]
            }}
        ],
        "ml_models": [
            {{
                "name": "Model name",
                "purpose": "Why this model",
                "architecture": "Brief architecture description",
                "hyperparameters": {{"param1": "value1"}}
            }}
        ],
        "evaluation_metrics": ["metric1", "metric2", ...],
        "baseline_comparisons": ["baseline1", "baseline2", ...]
    }},
    "data_plan": {{
        "primary_dataset": {{
            "name": "Dataset name",
            "source": "Where to get it",
            "preprocessing": "Preprocessing steps"
        }},
        "data_splits": {{
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15
        }},
        "augmentation": "Data augmentation strategy if applicable"
    }},
    "expected_results": "What results are anticipated",
    "novelty_statement": "What makes this research novel",
    "paper_outline": {{
        "introduction_points": ["point1", "point2", ...],
        "methodology_sections": ["section1", "section2", ...],
        "expected_figures": ["fig1 description", "fig2 description", ...],
        "expected_tables": ["table1 description", "table2 description", ...]
    }}
}}"""

    design_raw = call_claude(design_prompt, system=SYSTEM_PROMPT, timeout=300)
    try:
        cleaned = design_raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        design = json.loads(cleaned)
    except json.JSONDecodeError:
        design = {"raw_design": design_raw}

    if progress_callback:
        progress_callback("Research design completed.")

    return {
        "design": design,
        "status": "completed",
    }
