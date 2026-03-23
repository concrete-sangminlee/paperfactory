import json
import logging
from utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

WEB_TOOLS = ["WebSearch", "WebFetch"]

SYSTEM_PROMPT = """You are a senior research scientist specializing in structural engineering and AI/ML applications.
Your task is to conduct a thorough literature review using web search.
You MUST use the WebSearch tool to find real, published academic papers.
Search Google Scholar, ResearchGate, ScienceDirect, ASCE Library, and other academic databases.
DO NOT fabricate or hallucinate any paper titles, authors, or DOIs.
Only include papers you have actually found through web search.
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
        progress_callback("Searching academic databases for relevant papers...")

    all_keywords = keywords.get("primary_keywords", []) + keywords.get("key_phrases", [])
    search_queries = all_keywords[:5]

    review_prompt = f"""Conduct a comprehensive literature review on the topic: "{topic}"

You MUST perform the following web searches to find REAL published papers:

1. Search for each of these queries on the web:
{chr(10).join(f'   - "{q}"' for q in search_queries)}

2. Also search for:
   - "{topic} site:scholar.google.com"
   - "{topic} structural engineering machine learning"
   - "{topic} {journal_guideline.get('journal_name', '')}"

3. For each paper you find, visit its page to get accurate details (title, authors, year, journal, DOI).

4. After searching, compile your findings.

IMPORTANT:
- ONLY include papers you actually found through web search
- Include the DOI or URL for each paper
- Do NOT make up paper titles or authors
- If you cannot find enough papers, say so honestly

Target journal: {journal_guideline.get('journal_name', '')}
Journal scope: {journal_guideline.get('scope', '')}

After completing all searches, respond in JSON format:
{{
    "summary": "Brief overview of the field (2-3 paragraphs)",
    "key_papers": [
        {{
            "authors": "Author names",
            "year": 2024,
            "title": "Paper title",
            "journal": "Journal name",
            "doi_or_url": "DOI or URL",
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
            "source": "Where to find it (URL)",
            "size": "Approximate size"
        }}
    ],
    "potential_contributions": ["contribution1", "contribution2", ...],
    "suggested_methodology": "Recommended approach based on literature"
}}"""

    review_raw = call_claude(
        review_prompt,
        system=SYSTEM_PROMPT,
        timeout=600,
        allowed_tools=WEB_TOOLS,
        max_turns=15,
    )
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
        papers_found = len(review.get("key_papers", []))
        progress_callback(f"Literature review completed. Found {papers_found} relevant papers.")

    return {
        "keywords": keywords,
        "review": review,
        "status": "completed",
    }
