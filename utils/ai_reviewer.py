"""AI-powered pre-submission paper review simulation.

Analyzes paper_content structure and generates reviewer-style critique
using rule-based heuristics (no LLM API required).
"""

import re


def review_paper(paper_content: dict, journal_key: str, figures: list = None) -> dict:
    """Simulate a peer review of the paper.

    Returns a structured review with comments organized by section,
    severity (major/minor), and category.
    """
    comments = []
    sections = paper_content.get("sections", [])
    abstract = paper_content.get("abstract", "")
    refs = paper_content.get("references", [])
    if isinstance(refs, str):
        refs = [r for r in refs.split("\n") if r.strip()]

    # 1. Structural completeness
    section_names = [s.get("heading", "").upper() for s in sections]
    expected = ["INTRODUCTION", "METHODOLOGY", "RESULT", "CONCLUSION"]
    for exp in expected:
        if not any(exp in sn for sn in section_names):
            comments.append(
                _comment(
                    "major",
                    "Structure",
                    f"Missing '{exp}' section. Most journals require this section.",
                )
            )

    # 2. Introduction quality
    intro = _find_section(sections, "INTRODUCTION")
    if intro:
        intro_text = _get_full_text(intro)
        intro_words = len(intro_text.split())
        if intro_words < 500:
            comments.append(
                _comment(
                    "major",
                    "Introduction",
                    f"Introduction is too brief ({intro_words} words). A thorough literature review "
                    "typically requires 800-1500 words to establish context, review prior work, "
                    "identify research gaps, and state objectives.",
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
                    "Introduction should explicitly state the objectives/contributions of this study.",
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
    method = _find_section(sections, "METHODOLOGY") or _find_section(sections, "METHOD")
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
                    "No validation strategy described. Cross-validation or train-test split should be specified.",
                )
            )
        if len(method_text.split()) < 300:
            comments.append(
                _comment(
                    "major",
                    "Methodology",
                    "Methodology section is too brief. Provide sufficient detail for reproducibility.",
                )
            )

    # 4. Results quality
    results = _find_section(sections, "RESULT")
    if results:
        results_text = _get_full_text(results)
        if not any(w in results_text.lower() for w in ["table", "fig.", "figure"]):
            comments.append(
                _comment(
                    "major",
                    "Results",
                    "Results should reference specific tables and figures to support claims.",
                )
            )
        if not any(
            w in results_text.lower()
            for w in ["compar", "outperform", "superior", "baseline", "prior"]
        ):
            comments.append(
                _comment(
                    "major", "Results", "Results should compare with existing methods or baselines."
                )
            )
        if not any(
            w in results_text.lower()
            for w in ["r2", "r-squared", "rmse", "accuracy", "f1", "mae", "mse"]
        ):
            comments.append(
                _comment(
                    "minor",
                    "Results",
                    "Include quantitative performance metrics (R², RMSE, accuracy, F1, etc.).",
                )
            )

    # 5. Conclusions
    conclusion = _find_section(sections, "CONCLUSION")
    if conclusion:
        conc_text = _get_full_text(conclusion)
        if not any(w in conc_text.lower() for w in ["limitation", "future", "further"]):
            comments.append(
                _comment(
                    "minor",
                    "Conclusions",
                    "Conclusions should discuss limitations and suggest future research directions.",
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
                f"Only {len(refs)} references cited. A comprehensive literature review typically "
                "requires 20-40 references for a full-length research paper.",
            )
        )
    recent_count = sum(1 for r in refs if _has_recent_year(r))
    if refs and recent_count / len(refs) < 0.3:
        comments.append(
            _comment(
                "minor",
                "References",
                f"Only {recent_count}/{len(refs)} references are from the last 5 years. "
                "Include more recent publications to demonstrate awareness of current developments.",
            )
        )

    # 7. Figures and tables
    n_figs = len(figures) if figures else len(paper_content.get("figure_captions", []))
    n_tables = len(paper_content.get("tables", []))
    if n_figs < 4:
        comments.append(
            _comment(
                "minor",
                "Figures",
                f"Only {n_figs} figures. Consider adding more visualizations to support your analysis.",
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
                f"Abstract is very brief ({abstract_words} words). Include purpose, methods, key results, "
                "and conclusions.",
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
                "The manuscript should clearly state its novel contributions compared to existing work.",
            )
        )

    # Score
    major_count = sum(1 for c in comments if c["severity"] == "major")
    minor_count = sum(1 for c in comments if c["severity"] == "minor")

    if major_count == 0 and minor_count <= 2:
        decision = "Accept with minor revisions"
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
    for s in sections:
        if keyword in s.get("heading", "").upper():
            return s
    return None


def _get_full_text(section: dict) -> str:
    text = section.get("content", "")
    for sub in section.get("subsections", []):
        text += " " + sub.get("content", "")
    return text


def _has_recent_year(ref: str) -> bool:
    import datetime

    current_year = datetime.datetime.now().year
    years = re.findall(r"\b(?:19|20)\d{2}\b", ref)
    return bool(years) and int(years[-1]) >= current_year - 5
