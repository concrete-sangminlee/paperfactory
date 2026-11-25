"""Generate or improve paper abstracts from body content.

Extracts key information from each section to compose a structured abstract
following the standard pattern: purpose, methods, results, conclusions.
"""

import re

from utils.paper_analytics import split_sentences

# Section headings (upper-cased substrings) searched for each abstract part, in
# order of preference.  The fallbacks cover the headings used by the
# experimental and numerical templates in ``utils.template_system``.
_PURPOSE_HEADINGS = ("INTRODUCTION",)
_METHOD_HEADINGS = (
    "METHOD",
    "EXPERIMENTAL PROGRAM",
    "EXPERIMENTAL SETUP",
    "EXPERIMENTAL PROCEDURE",
    "NUMERICAL MODEL",
    "FINITE ELEMENT MODEL",
    "MATERIALS AND",
)
_RESULT_HEADINGS = ("RESULT", "PARAMETRIC STUDY", "DISCUSSION")
_CONCLUSION_HEADINGS = ("CONCLUSION",)

# Word-boundary patterns, so that e.g. "claim" does not match "aim" and
# "foundation" does not match "found".
_PURPOSE_STRONG_RE = re.compile(
    r"\b(?:this|the current)\s+(?:study|paper|work|research|article)\b"
    r"|\bthe present\b|\bwe\s+(?:propose|present|develop|investigate)\b",
    re.IGNORECASE,
)
_PURPOSE_WEAK_RE = re.compile(
    r"\b(?:objectives?|aims?|aimed|purpose|address(?:es)?)\b", re.IGNORECASE
)
_METHOD_SENTENCE_RE = re.compile(
    r"\b(?:model|method|approach|framework|algorithm|trained|compared|cross-validation)",
    re.IGNORECASE,
)
_RESULT_NUMBER_RE = re.compile(r"\d+\.\d+|\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)

_HAS_PURPOSE_RE = re.compile(
    r"\b(?:this study|this paper|this work|we present|we propose)\b", re.IGNORECASE
)
_HAS_METHOD_RE = re.compile(
    r"\b(?:method|approach|model|framework|technique|algorithm)", re.IGNORECASE
)
_HAS_RESULTS_RE = re.compile(
    r"\b(?:result|found\b|achieved|show|demonstrate|accuracy|r-squared|rmse\b|mae\b|r2\b"
    r"|outperform)|r²",
    re.IGNORECASE,
)
_HAS_CONCLUSION_RE = re.compile(
    r"\b(?:conclud|significant|effective|practical|implication|potential)", re.IGNORECASE
)


def generate_abstract(paper_content: dict, max_words: int = 250) -> str:
    """Generate an abstract from paper body sections.

    Extracts the first meaningful sentence(s) from each major section
    (purpose, methods, results, conclusions) and combines them into a
    structured abstract.  Only whole sentences are used, so the abstract never
    ends mid-sentence, and it never exceeds ``max_words`` words.

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
    return _fit_to_word_limit(_abstract_sentences(paper_content), max_words)


def improve_abstract(abstract: str, paper_content: dict, max_words: int = 250) -> dict:
    """Analyze and suggest improvements for an existing abstract.

    Returns dict with 'score', 'word_count', 'suggestions', the 'has_*' flags
    and 'improved': a revised abstract (missing components added from the
    paper body, trimmed to ``max_words`` at sentence boundaries), or ``None``
    when no automatic improvement is possible.
    """
    abstract = abstract or ""
    suggestions = []
    words = abstract.split()
    sentences = split_sentences(abstract)

    # Length check (the lower target scales down for journals with short limits)
    target_min = min(150, round(max_words * 0.6))
    if len(words) < min(100, target_min):
        suggestions.append(
            f"Abstract is short ({len(words)} words). Aim for {target_min}-{max_words} words."
        )
    elif len(words) > max_words:
        suggestions.append(
            f"Abstract exceeds limit ({len(words)}/{max_words} words). Trim unnecessary details."
        )

    # Structure check
    has_purpose = bool(_HAS_PURPOSE_RE.search(abstract))
    has_method = bool(_HAS_METHOD_RE.search(abstract))
    has_results = bool(_HAS_RESULTS_RE.search(abstract))
    has_conclusion = bool(_HAS_CONCLUSION_RE.search(abstract))

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

    # Build an improved version: add missing parts from the body, then trim.
    sections = (paper_content or {}).get("sections") or []
    revised = list(sentences)
    if not has_purpose:
        revised = _split(_extract_part(sections, _PURPOSE_HEADINGS, _extract_purpose)) + revised
    if not has_method:
        revised += _split(_extract_part(sections, _METHOD_HEADINGS, _extract_methods))
    if not has_results:
        revised += _split(_extract_part(sections, _RESULT_HEADINGS, _extract_results))
    if not has_conclusion:
        revised += _split(_extract_part(sections, _CONCLUSION_HEADINGS, _extract_conclusions))
    improved = _fit_to_word_limit(_dedupe(revised), max_words)
    if not improved or improved == " ".join(sentences):
        improved = None

    return {
        "score": score,
        "word_count": len(words),
        "suggestions": suggestions,
        "has_purpose": has_purpose,
        "has_method": has_method,
        "has_results": has_results,
        "has_conclusion": has_conclusion,
        "improved": improved,
    }


def _abstract_sentences(paper_content: dict) -> list:
    """Return the ordered, de-duplicated candidate sentences for an abstract."""
    sections = (paper_content or {}).get("sections") or []
    sentences = []
    for headings, extractor in (
        (_PURPOSE_HEADINGS, _extract_purpose),
        (_METHOD_HEADINGS, _extract_methods),
        (_RESULT_HEADINGS, _extract_results),
        (_CONCLUSION_HEADINGS, _extract_conclusions),
    ):
        sentences += _split(_extract_part(sections, headings, extractor))
    return _dedupe(sentences)


def _extract_part(sections: list, headings: tuple, extractor) -> str:
    section = _find_section(sections, headings)
    return extractor(section) if section else ""


def _fit_to_word_limit(sentences: list, max_words: int) -> str:
    """Join whole sentences, in order, while the total stays within ``max_words``.

    Sentences that would overflow the limit are skipped (a later, shorter one
    may still fit).  If not even one sentence fits, the first is truncated to
    ``max_words`` words and marked with an ellipsis.
    """
    if max_words is None or max_words <= 0 or not sentences:
        return ""
    kept, count = [], 0
    for sentence in sentences:
        n = len(sentence.split())
        if count + n <= max_words:
            kept.append(sentence)
            count += n
    if kept:
        return " ".join(kept)
    head = sentences[0].split()[:max_words]
    return " ".join(head).rstrip(".,;:!?") + "..."


def _split(text: str) -> list:
    return split_sentences(text) if text else []


def _dedupe(sentences: list) -> list:
    seen, unique = set(), []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _find_section(sections: list, keyword):
    """Return the first section whose heading contains ``keyword``.

    ``keyword`` may be a string or a tuple of strings tried in order of
    preference (all sections are searched for the first keyword before the
    next keyword is tried).
    """
    keywords = (keyword,) if isinstance(keyword, str) else tuple(keyword)
    valid = [s for s in sections or [] if isinstance(s, dict)]
    for kw in keywords:
        for s in valid:
            if kw in str(s.get("heading") or "").upper():
                return s
    return None


def _get_text(section: dict) -> str:
    parts = [section.get("content") or ""]
    for sub in section.get("subsections") or []:
        if isinstance(sub, dict):
            parts.append(_get_text(sub))
    return "\n\n".join(p for p in parts if p)


def _extract_purpose(section: dict) -> str:
    """Extract purpose/objective from introduction."""
    sentences = _get_sentences(_get_text(section))

    # Look for explicit objective statements first ("This study..."), then
    # weaker cues ("The objective...", "...aims to...").
    for pattern in (_PURPOSE_STRONG_RE, _PURPOSE_WEAK_RE):
        for s in sentences:
            if pattern.search(s):
                return s
    # Fallback: last sentence of intro (often states objectives)
    if sentences:
        return sentences[-1]
    return ""


def _extract_methods(section: dict) -> str:
    """Extract key methodology from methods section."""
    sentences = _get_sentences(_get_text(section))

    method_sentences = []
    for s in sentences:
        if _METHOD_SENTENCE_RE.search(s):
            method_sentences.append(s)
            if len(method_sentences) >= 2:
                break
    return " ".join(method_sentences) if method_sentences else (sentences[0] if sentences else "")


def _extract_results(section: dict) -> str:
    """Extract key results."""
    sentences = _get_sentences(_get_text(section))

    result_sentences = []
    for s in sentences:
        # Prioritize sentences with numbers (decimals or percentages)
        if _RESULT_NUMBER_RE.search(s):
            result_sentences.append(s)
            if len(result_sentences) >= 2:
                break
    return " ".join(result_sentences) if result_sentences else (sentences[0] if sentences else "")


def _extract_conclusions(section: dict) -> str:
    """Extract key conclusions."""
    sentences = _get_sentences(_get_text(section))
    if sentences:
        return sentences[0]
    return ""


def _get_sentences(text: str) -> list:
    """Split text into clean sentences (abbreviation/decimal aware)."""
    return [s for s in split_sentences(text) if len(s) > 20]
