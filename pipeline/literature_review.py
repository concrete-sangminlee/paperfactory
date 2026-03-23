import json
import logging
from utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior research scientist specializing in structural engineering and AI/ML applications.
Your task is to conduct a thorough literature review on the given topic.
You must search for and analyze relevant academic papers, identify research gaps, and summarize findings.
Focus on recent publications (last 5 years) while including seminal works.
Always respond in English."""


def run(topic: str, journal_guideline: dict, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback("Generating search keywords...")

    keywords_prompt = f"""Given the research topic: "{topic}"
Target journal: {journal_guideline.get('journal_name', '')}
Journal scope: {journal_guideline.get('scope', '')}

Generate a comprehensive list of search keywords and key phrases for literature review.
Respond in JSON format:
{{
    "primary_keywords": ["keyword1", "keyword2", ...],
    "secondary_keywords": ["keyword1", "keyword2", ...],
    "key_phrases": ["phrase1", "phrase2", ...],
    "related_fields": ["field1", "field2", ...]
}}"""

    keywords_raw = call_claude(keywords_prompt, system=SYSTEM_PROMPT, timeout=120)
    try:
        cleaned = keywords_raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        keywords = json.loads(cleaned)
    except json.JSONDecodeError:
        keywords = {
            "primary_keywords": [topic],
            "secondary_keywords": [],
            "key_phrases": [],
            "related_fields": [],
        }

    if progress_callback:
        progress_callback("Conducting literature survey...")

    review_prompt = f"""Conduct a comprehensive literature review on the topic: "{topic}"

Search keywords: {json.dumps(keywords, indent=2)}
Target journal: {journal_guideline.get('journal_name', '')}
Journal scope: {journal_guideline.get('scope', '')}

Please provide a thorough literature review covering:
1. Key foundational works in this area
2. Recent advances and state-of-the-art methods (focus on last 5 years)
3. Relevant datasets commonly used
4. Identified research gaps and opportunities
5. Potential novel contributions

Respond in JSON format:
{{
    "summary": "Brief overview of the field (2-3 paragraphs)",
    "key_papers": [
        {{
            "authors": "Author names",
            "year": 2024,
            "title": "Paper title",
            "journal": "Journal name",
            "key_findings": "Main contributions",
            "relevance": "How it relates to our topic"
        }}
    ],
    "research_gaps": ["gap1", "gap2", ...],
    "state_of_the_art": "Description of current best approaches",
    "available_datasets": [
        {{
            "name": "Dataset name",
            "description": "What it contains",
            "source": "Where to find it",
            "size": "Approximate size"
        }}
    ],
    "potential_contributions": ["contribution1", "contribution2", ...],
    "suggested_methodology": "Recommended approach based on literature"
}}"""

    review_raw = call_claude(review_prompt, system=SYSTEM_PROMPT, timeout=300)
    try:
        cleaned = review_raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        review = json.loads(cleaned)
    except json.JSONDecodeError:
        review = {
            "summary": review_raw,
            "key_papers": [],
            "research_gaps": [],
            "state_of_the_art": "",
            "available_datasets": [],
            "potential_contributions": [],
            "suggested_methodology": "",
        }

    if progress_callback:
        progress_callback("Literature review completed.")

    return {
        "keywords": keywords,
        "review": review,
        "status": "completed",
    }
