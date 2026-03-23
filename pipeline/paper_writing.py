import json
import logging
from utils.claude_cli import call_claude
from utils.word_generator import generate_word
from utils.code_runner import list_generated_figures

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert academic writer specializing in structural engineering research papers.
You write in formal, precise academic English suitable for top-tier journals.
Your writing is clear, concise, and follows the conventions of the target journal.
You cite references properly and maintain logical flow throughout the paper.
Always respond in English."""


def run(topic: str, literature: dict, design: dict, code_results: dict,
        analysis: dict, journal_key: str, journal_guideline: dict, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback("Writing paper draft...")

    research_design = design.get("design", {})
    review = literature.get("review", {})
    analysis_data = analysis.get("analysis", {})
    ref_style = journal_guideline.get("references", {})
    manuscript = journal_guideline.get("manuscript_structure", {})

    sections_order = manuscript.get("sections", [
        "Title Page", "Abstract", "Introduction", "Literature Review",
        "Methodology", "Results", "Discussion", "Conclusion",
        "Acknowledgments", "References"
    ])

    abstract_limit = manuscript.get("abstract_word_limit", 250)
    keywords_count = manuscript.get("keywords_count", "4-6")

    paper_prompt = f"""Write a COMPLETE academic research paper on the following topic.

Topic: "{topic}"
Proposed Title: {research_design.get('research_title', topic)}
Target Journal: {journal_guideline.get('journal_name', '')}

JOURNAL REQUIREMENTS:
- Required sections (in order): {json.dumps(sections_order)}
- Abstract word limit: {abstract_limit} words
- Keywords: {keywords_count}
- Reference style: {ref_style.get('style', 'Author-Year')}
- Reference format: {ref_style.get('format', '')}

RESEARCH CONTENT:
- Objective: {research_design.get('research_objective', '')}
- Novelty: {research_design.get('novelty_statement', '')}
- Hypotheses: {json.dumps(research_design.get('hypotheses', []))}

LITERATURE REVIEW DATA:
- Summary: {review.get('summary', '')}
- Key Papers: {json.dumps(review.get('key_papers', [])[:10])}
- Research Gaps: {json.dumps(review.get('research_gaps', []))}

METHODOLOGY:
{json.dumps(research_design.get('methodology', {}), indent=2)}

RESULTS:
{code_results.get('stdout', '')[:5000]}

ANALYSIS:
- Summary: {analysis_data.get('results_summary', '')}
- Findings: {json.dumps(analysis_data.get('detailed_findings', []))}
- Hypothesis Evaluation: {json.dumps(analysis_data.get('hypothesis_evaluation', []))}
- Discussion Points: {json.dumps(analysis_data.get('discussion_points', []))}
- Limitations: {json.dumps(analysis_data.get('limitations', []))}
- Future Work: {json.dumps(analysis_data.get('future_work', []))}

FIGURES: {json.dumps(analysis_data.get('figure_descriptions', []))}

Respond in JSON format:
{{
    "title": "Paper title",
    "authors": "Author Name(s)",
    "abstract": "Abstract text ({abstract_limit} words max)",
    "keywords": "keyword1; keyword2; keyword3; ...",
    "sections": [
        {{
            "heading": "INTRODUCTION",
            "content": "Full text of the introduction section..."
        }},
        {{
            "heading": "LITERATURE REVIEW",
            "content": "Full text..."
        }},
        {{
            "heading": "METHODOLOGY",
            "content": "Full text..."
        }},
        {{
            "heading": "RESULTS AND DISCUSSION",
            "content": "Full text..."
        }},
        {{
            "heading": "CONCLUSION",
            "content": "Full text..."
        }}
    ],
    "references": [
        "Formatted reference 1",
        "Formatted reference 2"
    ]
}}

IMPORTANT:
- Write FULL, DETAILED content for each section (not summaries or outlines)
- Each section should be multiple paragraphs
- Reference figures as Fig. 1, Fig. 2, etc.
- Include in-text citations in the journal's required format
- The paper should be publication-ready quality
- Introduction should be at least 4-5 paragraphs
- Methodology should thoroughly describe the approach
- Results should present and interpret all key findings
- References should follow {ref_style.get('style', '')} format exactly"""

    paper_raw = call_claude(paper_prompt, system=SYSTEM_PROMPT, timeout=600)
    try:
        cleaned = paper_raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        paper_content = json.loads(cleaned)
    except json.JSONDecodeError:
        paper_content = _fallback_parse(paper_raw, research_design)

    if progress_callback:
        progress_callback("Generating Word document...")

    figures = list_generated_figures()
    output_path = generate_word(paper_content, journal_key, figures)

    if progress_callback:
        progress_callback(f"Paper saved to {output_path}")

    return {
        "paper_content": paper_content,
        "output_path": output_path,
        "figures_included": len(figures),
        "status": "completed",
    }


def _fallback_parse(raw: str, design: dict) -> dict:
    return {
        "title": design.get("research_title", "Research Paper"),
        "authors": "",
        "abstract": "",
        "keywords": "",
        "sections": [{"heading": "PAPER CONTENT", "content": raw}],
        "references": [],
    }
