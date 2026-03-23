import json
import re
import logging
from utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

WEB_TOOLS = ["WebSearch", "WebFetch"]

SEARCH_SYSTEM = """You are a research scientist. Your ONLY job is to use WebSearch to find real academic papers.
You MUST use the WebSearch tool multiple times with different queries.
For each paper found, provide: title, authors, year, journal, DOI or URL.
ONLY include papers you actually found via web search. Never fabricate papers.
Always respond in English."""

ANALYSIS_SYSTEM = """You are a senior research scientist specializing in structural engineering and AI/ML.
Analyze the provided literature search results and produce a structured review.
Always respond in English.
You MUST respond with valid JSON only. No markdown code blocks, no extra text."""


def _clean_json(raw: str) -> str:
    cleaned = raw.strip()
    # Remove markdown code blocks
    match = re.search(r'```(?:json)?\s*\n(.*?)```', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    # Try to find JSON object
    if not cleaned.startswith('{'):
        brace_start = cleaned.find('{')
        if brace_start != -1:
            cleaned = cleaned[brace_start:]
    # Find matching closing brace
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


def run(topic: str, journal_guideline: dict, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback("Step 1/3: Generating search keywords...")

    keywords_prompt = f"""Given the research topic: "{topic}"
Target journal: {journal_guideline.get('journal_name', '')}

Generate search keywords. Respond in JSON:
{{"primary_keywords": ["kw1", "kw2"], "secondary_keywords": ["kw3"], "key_phrases": ["phrase1"]}}"""

    keywords_raw = call_claude(keywords_prompt, system=ANALYSIS_SYSTEM, timeout=120)
    try:
        keywords = json.loads(_clean_json(keywords_raw))
    except (json.JSONDecodeError, ValueError):
        keywords = {"primary_keywords": [topic], "secondary_keywords": [], "key_phrases": []}

    if progress_callback:
        progress_callback("Step 2/3: Searching academic databases for papers...")

    all_kw = keywords.get("primary_keywords", []) + keywords.get("key_phrases", [])
    search_terms = all_kw[:4] if all_kw else [topic]

    search_prompt = f"""Search the web for academic papers related to: "{topic}"

Perform these searches:
{chr(10).join(f'{i+1}. WebSearch for: "{q}"' for i, q in enumerate(search_terms))}
{len(search_terms)+1}. WebSearch for: "{topic} structural engineering"
{len(search_terms)+2}. WebSearch for: "{topic} {journal_guideline.get('journal_name', '')}"
{len(search_terms)+3}. WebSearch for: "{topic} deep learning machine learning"

For each paper you find, record:
- Title (exact)
- Authors
- Year
- Journal name
- DOI or URL

List ALL papers you find. Aim for at least 10-15 papers."""

    search_raw = call_claude(
        search_prompt,
        system=SEARCH_SYSTEM,
        timeout=600,
        allowed_tools=WEB_TOOLS,
        max_turns=20,
    )

    if progress_callback:
        progress_callback("Step 3/3: Analyzing literature and identifying research gaps...")

    analysis_prompt = f"""Below are academic papers found via web search about: "{topic}"
Target journal: {journal_guideline.get('journal_name', '')}
Journal scope: {journal_guideline.get('scope', '')}

=== SEARCH RESULTS ===
{search_raw[:8000]}
=== END SEARCH RESULTS ===

Analyze these papers and respond with ONLY this JSON (no markdown, no code blocks):
{{
    "summary": "2-3 paragraph overview of the field and current research landscape",
    "key_papers": [
        {{
            "authors": "Author names as found",
            "year": 2024,
            "title": "Exact paper title as found",
            "journal": "Journal name",
            "doi_or_url": "DOI or URL",
            "key_findings": "Brief summary of contributions",
            "relevance": "Why relevant to our topic"
        }}
    ],
    "research_gaps": ["gap1", "gap2", "gap3"],
    "state_of_the_art": "Current best approaches and methods",
    "available_datasets": [
        {{
            "name": "Dataset name",
            "description": "What it contains",
            "source": "URL or source",
            "size": "Approximate size"
        }}
    ],
    "potential_contributions": ["contribution1", "contribution2"],
    "suggested_methodology": "Recommended approach based on gaps and state of the art"
}}"""

    analysis_raw = call_claude(analysis_prompt, system=ANALYSIS_SYSTEM, timeout=300)
    try:
        review = json.loads(_clean_json(analysis_raw))
    except (json.JSONDecodeError, ValueError):
        logger.error(f"Failed to parse analysis JSON. Raw: {analysis_raw[:500]}")
        review = {
            "summary": analysis_raw,
            "key_papers": [],
            "research_gaps": [],
            "state_of_the_art": "",
            "available_datasets": [],
            "potential_contributions": [],
            "suggested_methodology": "",
        }

    papers_count = len(review.get("key_papers", []))
    if progress_callback:
        progress_callback(f"Literature review completed. Found {papers_count} relevant papers.")

    return {
        "keywords": keywords,
        "review": review,
        "status": "completed",
    }
