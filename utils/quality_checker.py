"""Validate generated paper content against journal quality criteria."""

import json
import os
import re

GUIDELINES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guidelines"
)

# Minimum thresholds from CLAUDE.md pipeline requirements
MIN_BODY_WORDS = 6000
MIN_FIGURES = 6
MIN_TABLES = 3
MIN_REFERENCES = 15
MIN_RECENT_RATIO = 0.5  # 50% of references within 5 years
MIN_COMPARISONS = 3
MIN_LIMITATIONS = 3


def load_guideline(journal_key: str) -> dict:
    """Load a journal guideline JSON file."""
    path = os.path.join(GUIDELINES_DIR, f"{journal_key}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Guideline not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_paper(paper_content: dict, journal_key: str, figures: list = None) -> dict:
    """Run all quality checks on a paper_content dict.

    Parameters
    ----------
    paper_content:
        Dict with keys: title, authors, abstract, keywords, sections,
        references, tables, figure_captions, highlights, etc.
    journal_key:
        Key matching a journal guideline (e.g. "jweia").
    figures:
        Optional list of figure file paths.

    Returns
    -------
    dict with keys:
        - passed (bool): True if all critical checks pass
        - score (int): 0-100 quality score
        - checks (list[dict]): individual check results
        - summary (str): human-readable summary
    """
    guideline = load_guideline(journal_key)
    ms = guideline.get("manuscript_structure", {})
    checks = []

    # 1. Abstract word count
    abstract = paper_content.get("abstract", "")
    abstract_words = len(abstract.split())
    abstract_limit = ms.get("abstract_word_limit", 300)
    checks.append({
        "name": "abstract_word_count",
        "passed": 0 < abstract_words <= abstract_limit,
        "value": abstract_words,
        "limit": abstract_limit,
        "severity": "critical",
        "message": f"Abstract: {abstract_words} words (limit: {abstract_limit})",
    })

    # 2. Body word count
    body_words = _count_body_words(paper_content)
    checks.append({
        "name": "body_word_count",
        "passed": body_words >= MIN_BODY_WORDS,
        "value": body_words,
        "limit": MIN_BODY_WORDS,
        "severity": "critical",
        "message": f"Body: {body_words} words (min: {MIN_BODY_WORDS})",
    })

    # 3. Required sections
    required_sections = ms.get("sections", [])
    paper_sections = [s.get("heading", "").upper() for s in paper_content.get("sections", [])]
    core_sections = ["INTRODUCTION", "CONCLUSION"]
    missing = [s for s in core_sections if not any(s in ps for ps in paper_sections)]
    checks.append({
        "name": "required_sections",
        "passed": len(missing) == 0,
        "value": paper_sections,
        "missing": missing,
        "severity": "critical",
        "message": f"Sections: {len(paper_sections)} found"
                   + (f", missing: {missing}" if missing else ""),
    })

    # 4. References count
    refs = paper_content.get("references", [])
    if isinstance(refs, str):
        refs = [r for r in refs.split("\n") if r.strip()]
    n_refs = len(refs)
    checks.append({
        "name": "reference_count",
        "passed": n_refs >= MIN_REFERENCES,
        "value": n_refs,
        "limit": MIN_REFERENCES,
        "severity": "critical",
        "message": f"References: {n_refs} (min: {MIN_REFERENCES})",
    })

    # 5. Recent references ratio
    import datetime
    current_year = datetime.datetime.now().year
    recent_count = 0
    for ref in refs:
        years = re.findall(r"\b(?:19|20)\d{2}\b", str(ref))
        if years and int(years[-1]) >= current_year - 5:
            recent_count += 1
    recent_ratio = recent_count / max(n_refs, 1)
    checks.append({
        "name": "recent_references",
        "passed": recent_ratio >= MIN_RECENT_RATIO,
        "value": f"{recent_ratio:.0%}",
        "limit": f"{MIN_RECENT_RATIO:.0%}",
        "severity": "warning",
        "message": f"Recent refs (last 5yr): {recent_count}/{n_refs} ({recent_ratio:.0%})",
    })

    # 6. Figures count
    n_figures = len(figures) if figures else len(paper_content.get("figure_captions", []))
    checks.append({
        "name": "figure_count",
        "passed": n_figures >= MIN_FIGURES,
        "value": n_figures,
        "limit": MIN_FIGURES,
        "severity": "critical",
        "message": f"Figures: {n_figures} (min: {MIN_FIGURES})",
    })

    # 7. Tables count
    n_tables = len(paper_content.get("tables", []))
    checks.append({
        "name": "table_count",
        "passed": n_tables >= MIN_TABLES,
        "value": n_tables,
        "limit": MIN_TABLES,
        "severity": "warning",
        "message": f"Tables: {n_tables} (min: {MIN_TABLES})",
    })

    # 8. Keywords
    keywords = paper_content.get("keywords", "")
    kw_list = [k.strip() for k in keywords.split(";") if k.strip()] if keywords else []
    checks.append({
        "name": "keywords",
        "passed": len(kw_list) >= 1,
        "value": len(kw_list),
        "severity": "warning",
        "message": f"Keywords: {len(kw_list)}",
    })

    # 9. Highlights (if required by journal)
    highlights_required = ms.get("highlights_required", False)
    highlights = paper_content.get("highlights", [])
    if highlights_required:
        checks.append({
            "name": "highlights",
            "passed": len(highlights) >= 3,
            "value": len(highlights),
            "severity": "warning",
            "message": f"Highlights: {len(highlights)} (required by journal)",
        })

    # 10. Title present
    title = paper_content.get("title", "")
    checks.append({
        "name": "title",
        "passed": len(title) > 10,
        "value": len(title),
        "severity": "critical",
        "message": f"Title: {len(title)} chars",
    })

    # 11. Data availability statement
    data_avail = paper_content.get("data_availability", "")
    checks.append({
        "name": "data_availability",
        "passed": len(data_avail) > 0,
        "value": bool(data_avail),
        "severity": "info",
        "message": f"Data availability: {'present' if data_avail else 'missing'}",
    })

    # Score calculation
    critical_checks = [c for c in checks if c["severity"] == "critical"]
    warning_checks = [c for c in checks if c["severity"] == "warning"]
    info_checks = [c for c in checks if c["severity"] == "info"]

    critical_pass = sum(1 for c in critical_checks if c["passed"])
    warning_pass = sum(1 for c in warning_checks if c["passed"])
    info_pass = sum(1 for c in info_checks if c["passed"])

    score = int(
        (critical_pass / max(len(critical_checks), 1)) * 70
        + (warning_pass / max(len(warning_checks), 1)) * 25
        + (info_pass / max(len(info_checks), 1)) * 5
    )

    all_critical_pass = all(c["passed"] for c in critical_checks)

    # Summary
    failed = [c for c in checks if not c["passed"]]
    lines = [f"Quality Score: {score}/100"]
    lines.append(f"Status: {'PASS' if all_critical_pass else 'FAIL'}")
    lines.append(f"Checks: {sum(1 for c in checks if c['passed'])}/{len(checks)} passed")
    if failed:
        lines.append("Issues:")
        for c in failed:
            lines.append(f"  [{c['severity'].upper()}] {c['message']}")

    return {
        "passed": all_critical_pass,
        "score": score,
        "checks": checks,
        "summary": "\n".join(lines),
    }


def _count_body_words(paper_content: dict) -> int:
    """Count words in all section content (excluding abstract, references)."""
    total = 0
    for sec in paper_content.get("sections", []):
        content = sec.get("content", "")
        total += len(content.split())
        for subsec in sec.get("subsections", []):
            total += len(subsec.get("content", "").split())
    return total
