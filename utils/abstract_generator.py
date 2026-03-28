"""Generate or improve paper abstracts from body content.

Extracts key information from each section to compose a structured abstract
following the standard pattern: purpose, methods, results, conclusions.
"""

import re


def generate_abstract(paper_content: dict, max_words: int = 250) -> str:
    """Generate an abstract from paper body sections.

    Extracts the first meaningful sentence from each major section
    and combines them into a structured abstract.

    Parameters
    ----------
    paper_content : dict
        Paper content with sections.
    max_words : int
        Maximum word count for the abstract.

    Returns
    -------
    str
        Generated abstract text.
    """
    sections = paper_content.get("sections", [])
    parts = []

    # Purpose (from Introduction)
    intro = _find_section(sections, "INTRODUCTION")
    if intro:
        purpose = _extract_purpose(intro)
        if purpose:
            parts.append(purpose)

    # Methods (from Methodology)
    method = _find_section(sections, "METHOD")
    if method:
        methods = _extract_methods(method)
        if methods:
            parts.append(methods)

    # Results (from Results)
    results = _find_section(sections, "RESULT")
    if results:
        findings = _extract_results(results)
        if findings:
            parts.append(findings)

    # Conclusions
    conclusion = _find_section(sections, "CONCLUSION")
    if conclusion:
        conc = _extract_conclusions(conclusion)
        if conc:
            parts.append(conc)

    abstract = " ".join(parts)

    # Trim to max_words
    words = abstract.split()
    if len(words) > max_words:
        abstract = " ".join(words[:max_words])
        # End at last complete sentence
        last_period = abstract.rfind(".")
        if last_period > len(abstract) * 0.5:
            abstract = abstract[: last_period + 1]

    return abstract


def improve_abstract(abstract: str, paper_content: dict, max_words: int = 250) -> dict:
    """Analyze and suggest improvements for an existing abstract.

    Returns dict with 'score', 'suggestions', and 'improved' (if possible).
    """
    suggestions = []
    words = abstract.split()
    sentences = [s.strip() for s in abstract.split(".") if s.strip()]

    # Length check
    if len(words) < 100:
        suggestions.append(
            f"Abstract is short ({len(words)} words). Aim for 150-{max_words} words."
        )
    elif len(words) > max_words:
        suggestions.append(
            f"Abstract exceeds limit ({len(words)}/{max_words} words). Trim unnecessary details."
        )

    # Structure check
    has_purpose = any(
        w in abstract.lower()
        for w in ["this study", "this paper", "this work", "we present", "we propose"]
    )
    has_method = any(
        w in abstract.lower()
        for w in ["method", "approach", "model", "framework", "technique", "algorithm"]
    )
    has_results = any(
        w in abstract.lower()
        for w in [
            "result",
            "found",
            "achieved",
            "show",
            "demonstrate",
            "accuracy",
            "r-squared",
            "rmse",
        ]
    )
    has_conclusion = any(
        w in abstract.lower()
        for w in ["conclud", "significant", "effective", "practical", "implication", "potential"]
    )

    if not has_purpose:
        suggestions.append(
            "Missing purpose statement. Start with 'This study...' or 'This paper...'"
        )
    if not has_method:
        suggestions.append("Missing methodology description. Mention the approach/models used.")
    if not has_results:
        suggestions.append(
            "Missing quantitative results. Include key metrics (R², accuracy, etc.)."
        )
    if not has_conclusion:
        suggestions.append("Missing conclusion/significance statement.")

    # Specificity check
    numbers = re.findall(r"\d+\.?\d*", abstract)
    if len(numbers) < 2:
        suggestions.append(
            "Add more specific quantitative results (numbers, percentages, metrics)."
        )

    score = 100 - len(suggestions) * 15
    score = max(score, 0)

    return {
        "score": score,
        "word_count": len(words),
        "suggestions": suggestions,
        "has_purpose": has_purpose,
        "has_method": has_method,
        "has_results": has_results,
        "has_conclusion": has_conclusion,
    }


def _find_section(sections: list, keyword: str):
    for s in sections:
        if keyword in s.get("heading", "").upper():
            return s
    return None


def _get_text(section: dict) -> str:
    text = section.get("content", "")
    for sub in section.get("subsections", []):
        text += " " + sub.get("content", "")
    return text


def _extract_purpose(section: dict) -> str:
    """Extract purpose/objective from introduction."""
    text = _get_text(section)
    sentences = _get_sentences(text)

    # Look for objective sentences
    for s in sentences:
        s_lower = s.lower()
        if any(
            phrase in s_lower
            for phrase in [
                "this study",
                "this paper",
                "this work",
                "the present",
                "objective",
                "aim",
                "purpose",
                "addresses",
            ]
        ):
            return s
    # Fallback: last sentence of intro (often states objectives)
    if sentences:
        return sentences[-1]
    return ""


def _extract_methods(section: dict) -> str:
    """Extract key methodology from methods section."""
    text = _get_text(section)
    sentences = _get_sentences(text)

    method_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if any(
            w in s_lower
            for w in [
                "model",
                "method",
                "approach",
                "framework",
                "algorithm",
                "trained",
                "compared",
                "cross-validation",
            ]
        ):
            method_sentences.append(s)
            if len(method_sentences) >= 2:
                break
    return " ".join(method_sentences) if method_sentences else (sentences[0] if sentences else "")


def _extract_results(section: dict) -> str:
    """Extract key results."""
    text = _get_text(section)
    sentences = _get_sentences(text)

    result_sentences = []
    for s in sentences:
        # Prioritize sentences with numbers
        if re.search(r"\d+\.\d+", s):
            result_sentences.append(s)
            if len(result_sentences) >= 2:
                break
    return " ".join(result_sentences) if result_sentences else (sentences[0] if sentences else "")


def _extract_conclusions(section: dict) -> str:
    """Extract key conclusions."""
    text = _get_text(section)
    sentences = _get_sentences(text)
    if sentences:
        return sentences[0]
    return ""


def _get_sentences(text: str) -> list:
    """Split text into clean sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]
