import json
import re
import logging
from utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior research scientist specializing in structural engineering and AI/ML applications.
Your task is to design a rigorous research methodology based on the literature review findings.
The methodology should be novel, scientifically sound, and suitable for the target journal.
Always respond in English.
You MUST respond with valid JSON only. No markdown code blocks, no extra text."""


def _clean_json(raw: str) -> str:
    cleaned = raw.strip()
    match = re.search(r'```(?:json)?\s*\n(.*?)```', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    if not cleaned.startswith('{'):
        brace_start = cleaned.find('{')
        if brace_start != -1:
            cleaned = cleaned[brace_start:]
    depth = 0
    end = 0
    for i, c in enumerate(cleaned):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end > 0:
        cleaned = cleaned[:end]
    return cleaned


def run(topic: str, literature: dict, journal_guideline: dict, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback("Designing research methodology...")

    review = literature.get("review", {})

    # Summarize key papers for context
    papers_summary = ""
    for p in review.get("key_papers", [])[:8]:
        papers_summary += f"- {p.get('title', '')} ({p.get('year', '')}) - {p.get('key_findings', '')}\n"

    design_prompt = f"""Based on the following literature review, design a complete research methodology.

Topic: "{topic}"
Target journal: {journal_guideline.get('journal_name', '')}

Literature Summary: {review.get('summary', '')[:2000]}

Key Papers Found:
{papers_summary}

Research Gaps: {json.dumps(review.get('research_gaps', []))}

State of the Art: {review.get('state_of_the_art', '')[:1000]}

Available Datasets: {json.dumps(review.get('available_datasets', []))}

Suggested Methodology: {review.get('suggested_methodology', '')[:1000]}

Design a complete research plan. Respond with ONLY valid JSON (no markdown, no code blocks):
{{
    "research_title": "Proposed paper title",
    "research_objective": "Clear statement of the research objective",
    "hypotheses": ["hypothesis1", "hypothesis2"],
    "methodology": {{
        "approach": "Overall methodological approach description",
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
        "evaluation_metrics": ["metric1", "metric2"],
        "baseline_comparisons": ["baseline1", "baseline2"]
    }},
    "data_plan": {{
        "primary_dataset": {{
            "name": "Dataset name",
            "source": "Where to get it or how to generate synthetic data",
            "preprocessing": "Preprocessing steps"
        }},
        "data_splits": {{"train": 0.7, "validation": 0.15, "test": 0.15}},
        "augmentation": "Data augmentation strategy if applicable"
    }},
    "expected_results": "What results are anticipated",
    "novelty_statement": "What makes this research novel",
    "paper_outline": {{
        "introduction_points": ["point1", "point2"],
        "methodology_sections": ["section1", "section2"],
        "expected_figures": ["fig1 description", "fig2 description"],
        "expected_tables": ["table1 description", "table2 description"]
    }}
}}"""

    design_raw = call_claude(design_prompt, system=SYSTEM_PROMPT, timeout=600)
    try:
        design = json.loads(_clean_json(design_raw))
    except (json.JSONDecodeError, ValueError):
        logger.error(f"Failed to parse design JSON. Raw: {design_raw[:500]}")
        design = {"raw_design": design_raw}

    if progress_callback:
        title = design.get("research_title", "N/A")
        progress_callback(f"Research design completed: {title}")

    return {
        "design": design,
        "status": "completed",
    }
