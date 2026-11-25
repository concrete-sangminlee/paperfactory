"""AI-powered pre-submission paper review simulation.

Analyzes paper_content structure and generates reviewer-style critique
using rule-based heuristics (no LLM API required).  The expected section
structure, abstract limit and highlights rules come from the journal
guideline when ``journal_key`` is known.
"""

from __future__ import annotations

import datetime

from utils.guidelines import UnknownJournalError, load_guideline
from utils.quality_checker import (
    analyze_sections,
    classify_heading,
    highlight_rules,
    iter_sections,
    list_of,
    parse_highlights,
    text_of,
)
from utils.reference_utils import extract_year, normalize_references

#: Sections every research paper needs: (category, label used in comments).
_CORE_SECTIONS = (
    ("introduction", "INTRODUCTION"),
    ("methodology", "METHODOLOGY"),
    ("results", "RESULTS"),
    ("conclusions", "CONCLUSION"),
)

_METRIC_TERMS = (
    "r2",
    "r²",
    "r^2",
    "r-squared",
    "coefficient of determination",
    "rmse",
    "mae",
    "mse",
    "mape",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "error",
)


def review_paper(paper_content: dict, journal_key: str, figures: list = None) -> dict:
    """Simulate a peer review of the paper.

    Returns a structured review with comments organized by section,
    severity (major/minor), and category.
    """
    paper_content = paper_content or {}
    try:
        guideline = load_guideline(journal_key)
    except UnknownJournalError:
        guideline = None  # generic review without journal-specific rules
    ms = (guideline or {}).get("manuscript_structure", {}) or {}
    journal_name = (guideline or {}).get("journal_name", "the journal")

    comments = []
    sections = iter_sections(paper_content.get("sections"))
    abstract = text_of(paper_content.get("abstract"))
    refs = normalize_references(paper_content.get("references"))

    # 1. Structural completeness (synonyms, numbered and Korean headings understood)
    generic = analyze_sections(paper_content, None, figures=figures)
    found = generic["found"]
    for category, label in _CORE_SECTIONS:
        if category not in found:
            comments.append(
                _comment(
                    "major",
                    "Structure",
                    f"Missing '{label}' section. Most journals require this section.",
                )
            )
    if guideline:
        journal_analysis = analyze_sections(paper_content, guideline, figures=figures)
        for name in journal_analysis["missing"]:
            categories = classify_heading(name)
            if name.lower() == "main body" or {"methodology", "results"} & set(categories):
                continue  # already reported as a core section above
            comments.append(
                _comment(
                    "minor",
                    "Structure",
                    f"{journal_name} expects a '{name}' section in its manuscript structure.",
                )
            )

    # 2. Introduction quality
    intro = _section_for(sections, found.get("introduction"))
    if intro:
        intro_text = _get_full_text(intro)
        intro_words = len(intro_text.split())
        if intro_words < 500:
            comments.append(
                _comment(
                    "major",
                    "Introduction",
                    f"Introduction is too brief ({intro_words} words). A thorough literature "
                    "review typically requires 800-1500 words to establish context, review "
                    "prior work, identify research gaps, and state objectives.",
                )
            )
        if not any(
            phrase in intro_text.lower()
            for phrase in ["this study", "this paper", "this work", "the present"]
        ):
            comments.append(
                _comment(
                    "minor",
                    "Introduction",
                    "Introduction should explicitly state the objectives/contributions of this "
                    "study.",
                )
            )
        if not any(
            phrase in intro_text.lower()
            for phrase in ["gap", "limitation", "however", "despite", "lacking"]
        ):
            comments.append(
                _comment(
                    "major",
                    "Introduction",
                    "Introduction should identify clear research gaps that motivate this work.",
                )
            )

    # 3. Methodology
    method = _section_for(sections, found.get("methodology"))
    if method:
        method_text = _get_full_text(method)
        if "random" in method_text.lower() and "seed" not in method_text.lower():
            comments.append(
                _comment(
                    "minor",
                    "Methodology",
                    "If random processes are used, specify the random seed for reproducibility.",
                )
            )
        if not any(
            w in method_text.lower()
            for w in ["cross-validation", "validation", "test set", "holdout"]
        ):
            comments.append(
                _comment(
                    "major",
                    "Methodology",
                    "No validation strategy described. Cross-validation or train-test split "
                    "should be specified.",
                )
            )
        if len(method_text.split()) < 300:
            comments.append(
                _comment(
                    "major",
                    "Methodology",
                    "Methodology section is too brief. Provide sufficient detail for "
                    "reproducibility.",
                )
            )

    # 4. Results quality
    results = _section_for(sections, found.get("results"))
    if results:
        results_text = _get_full_text(results).lower()
        if not any(w in results_text for w in ["table", "fig.", "figure"]):
            comments.append(
                _comment(
                    "major",
                    "Results",
                    "Results should reference specific tables and figures to support claims.",
                )
            )
        if not any(
            w in results_text for w in ["compar", "outperform", "superior", "baseline", "prior"]
        ):
            comments.append(
                _comment(
                    "major", "Results", "Results should compare with existing methods or baselines."
                )
            )
        if not any(w in results_text for w in _METRIC_TERMS):
            comments.append(
                _comment(
                    "minor",
                    "Results",
                    "Include quantitative performance metrics (R², RMSE, accuracy, F1, etc.).",
                )
            )

    # 5. Conclusions
    conclusion = _section_for(sections, found.get("conclusions"))
    if conclusion:
        conc_text = _get_full_text(conclusion)
        if not any(w in conc_text.lower() for w in ["limitation", "future", "further"]):
            comments.append(
                _comment(
                    "minor",
                    "Conclusions",
                    "Conclusions should discuss limitations and suggest future research "
                    "directions.",
                )
            )
        if len(conc_text.split()) < 150:
            comments.append(
                _comment(
                    "minor",
                    "Conclusions",
                    "Conclusions section is brief. Summarize key findings and their implications.",
                )
            )

    # 6. References
    if len(refs) < 15:
        comments.append(
            _comment(
                "major",
                "References",
                f"Only {len(refs)} references cited. A comprehensive literature review "
                "typically requires 20-40 references for a full-length research paper.",
            )
        )
    recent_count = sum(1 for r in refs if _has_recent_year(r))
    if refs and recent_count / len(refs) < 0.3:
        comments.append(
            _comment(
                "minor",
                "References",
                f"Only {recent_count}/{len(refs)} references are from the last 5 years. "
                "Include more recent publications to demonstrate awareness of current "
                "developments.",
            )
        )

    # 7. Figures and tables
    figure_list = list_of(figures)
    n_figs = len(figure_list) if figure_list else len(list_of(paper_content.get("figure_captions")))
    n_tables = len(list_of(paper_content.get("tables")))
    if n_figs < 4:
        comments.append(
            _comment(
                "minor",
                "Figures",
                f"Only {n_figs} figures. Consider adding more visualizations to support your "
                "analysis.",
            )
        )
    if n_tables < 2:
        comments.append(
            _comment(
                "minor",
                "Tables",
                f"Only {n_tables} tables. Include comparison tables for model performance.",
            )
        )

    # 8. Abstract
    abstract_words = len(abstract.split())
    if abstract_words < 100:
        comments.append(
            _comment(
                "minor",
                "Abstract",
                f"Abstract is very brief ({abstract_words} words). Include purpose, methods, key "
                "results, and conclusions.",
            )
        )
    limit = ms.get("abstract_word_limit")
    if limit and abstract_words > limit:
        comments.append(
            _comment(
                "minor",
                "Abstract",
                f"Abstract has {abstract_words} words, exceeding the {journal_name} limit of "
                f"{limit} words.",
            )
        )
    if not any(
        w in abstract.lower() for w in ["result", "found", "achieved", "show", "demonstrate"]
    ):
        comments.append(
            _comment("minor", "Abstract", "Abstract should include specific quantitative results.")
        )

    # 9. Novelty
    all_text = " ".join(_get_full_text(s) for s in sections)
    if not any(
        phrase in all_text.lower()
        for phrase in [
            "novel",
            "first time",
            "for the first",
            "contribution",
            "unique",
            "new approach",
            "proposed",
        ]
    ):
        comments.append(
            _comment(
                "major",
                "Novelty",
                "The manuscript should clearly state its novel contributions compared to "
                "existing work.",
            )
        )
    if (
        ms.get("novelty_statement_required")
        and not text_of(paper_content.get("novelty_statement")).strip()
    ):
        comments.append(
            _comment(
                "minor",
                "Novelty",
                f"{journal_name} requires a separate novelty statement (bulleted list of the "
                "new contributions).",
            )
        )

    # 10. Highlights (journal-specific)
    if ms.get("highlights_required"):
        rules = highlight_rules(guideline)
        highlights = parse_highlights(paper_content.get("highlights"))
        too_long = [i + 1 for i, h in enumerate(highlights) if len(h) > rules["max_chars"]]
        if not rules["min"] <= len(highlights) <= rules["max"] or too_long:
            comments.append(
                _comment(
                    "minor",
                    "Highlights",
                    f"{journal_name} requires {rules['min']}-{rules['max']} highlights of at most "
                    f"{rules['max_chars']} characters ({len(highlights)} given"
                    + (f"; too long: {too_long}" if too_long else "")
                    + ").",
                )
            )

    # Score
    major_count = sum(1 for c in comments if c["severity"] == "major")
    minor_count = sum(1 for c in comments if c["severity"] == "minor")

    if major_count == 0 and minor_count <= 2:
        decision = "Accept with minor revisions"
    elif major_count == 0:
        decision = "Minor revision required"
    elif major_count <= 2:
        decision = "Major revision required"
    elif major_count <= 5:
        decision = "Major revision required (significant concerns)"
    else:
        decision = "Reject and resubmit"

    return {
        "decision": decision,
        "major_issues": major_count,
        "minor_issues": minor_count,
        "total_issues": len(comments),
        "comments": comments,
        "summary": (
            f"Review Decision: {decision}\n"
            f"Issues found: {major_count} major, {minor_count} minor\n"
            + (
                "\n".join(
                    f"  [{c['severity'].upper()}] ({c['section']}) {c['comment']}" for c in comments
                )
            )
        ),
    }


def _comment(severity: str, section: str, comment: str) -> dict:
    return {"severity": severity, "section": section, "comment": comment}


def _find_section(sections: list, keyword: str) -> dict:
    """Return the first section whose heading contains *keyword* (case-insensitive)."""
    for s in iter_sections(sections):
        if keyword.upper() in text_of(s.get("heading")).upper():
            return s
    return None


def _section_for(sections: list, heading: str | None) -> dict | None:
    """Return the section (or nested subsection) whose heading is *heading*."""
    if not heading:
        return None
    stack = list(iter_sections(sections))
    while stack:
        sec = stack.pop(0)
        if text_of(sec.get("heading")).strip() == heading:
            return sec
        stack.extend(iter_sections(sec.get("subsections")))
    return None


def _get_full_text(section: dict) -> str:
    """Section content plus all nested subsection content."""
    parts = [text_of(section.get("content"))]
    for sub in iter_sections(section.get("subsections")):
        parts.append(_get_full_text(sub))
    return " ".join(p for p in parts if p)


def _has_recent_year(ref: str, years: int = 5) -> bool:
    current_year = datetime.date.today().year
    year = extract_year(ref, current_year)
    return year is not None and year >= current_year - years
