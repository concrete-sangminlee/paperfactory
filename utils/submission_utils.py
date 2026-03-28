"""Submission workflow utilities: checklist, cover letter, journal reformatting."""

import json
import os
from datetime import datetime

GUIDELINES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guidelines"
)


def submission_checklist(paper_content: dict, journal_key: str, figures: list = None) -> dict:
    """Generate a journal-specific submission checklist.

    Returns dict with 'items' (list of check dicts) and 'ready' (bool).
    """
    guideline = _load_guideline(journal_key)
    ms = guideline.get("manuscript_structure", {})
    items = []

    # Manuscript format checks
    items.append(_check("Title page present", bool(paper_content.get("title"))))
    items.append(_check("Authors listed", bool(paper_content.get("authors"))))
    items.append(
        _check(
            "Abstract within word limit",
            len(paper_content.get("abstract", "").split()) <= ms.get("abstract_word_limit", 300),
        )
    )
    items.append(_check("Keywords provided", bool(paper_content.get("keywords"))))

    # Highlights
    if ms.get("highlights_required"):
        hl = paper_content.get("highlights", [])
        items.append(_check("Highlights provided (3-5 items)", 3 <= len(hl) <= 5))
        if hl:
            items.append(_check("Highlights within character limit", all(len(h) <= 85 for h in hl)))

    # Graphical abstract
    if ms.get("graphical_abstract_required"):
        items.append(
            _check(
                "Graphical abstract prepared",
                bool(paper_content.get("graphical_abstract")),
                note="Required by journal — use PaperBanana to generate",
            )
        )

    # Sections
    required_headings = ["INTRODUCTION", "CONCLUSION"]
    section_headings = [s.get("heading", "").upper() for s in paper_content.get("sections", [])]
    for req in required_headings:
        items.append(_check(f"Section '{req}' present", any(req in h for h in section_headings)))

    # References
    refs = paper_content.get("references", [])
    if isinstance(refs, str):
        refs = refs.split("\n")
    items.append(_check("References present (15+)", len(refs) >= 15))
    doi_count = sum(1 for r in refs if "doi" in r.lower() or "10." in r)
    items.append(_check("References include DOIs", doi_count >= len(refs) * 0.5))

    # Figures
    n_figs = len(figures) if figures else len(paper_content.get("figure_captions", []))
    items.append(_check("Figures included (6+)", n_figs >= 6))
    items.append(
        _check("Figure captions provided", len(paper_content.get("figure_captions", [])) >= n_figs)
    )

    # Tables
    items.append(_check("Tables included (3+)", len(paper_content.get("tables", [])) >= 3))

    # Data availability
    items.append(
        _check("Data availability statement", bool(paper_content.get("data_availability")))
    )

    # Ethical statements
    items.append(
        _check(
            "Conflict of interest statement",
            bool(paper_content.get("conflict_of_interest", paper_content.get("acknowledgments"))),
            note="Can be included in acknowledgments",
        )
    )

    ready = all(item["passed"] for item in items)
    return {"items": items, "ready": ready, "journal": guideline.get("journal_name", journal_key)}


def generate_cover_letter(
    paper_content: dict, journal_key: str, editor_name: str = "Editor-in-Chief"
) -> str:
    """Generate a cover letter for journal submission."""
    guideline = _load_guideline(journal_key)
    journal_name = guideline.get("journal_name", journal_key)
    title = paper_content.get("title", "")
    authors = paper_content.get("authors", "")
    abstract = paper_content.get("abstract", "")

    # Extract key findings from abstract (last 2 sentences)
    sentences = [s.strip() for s in abstract.replace("\n", " ").split(".") if s.strip()]
    key_findings = ". ".join(sentences[-3:]) + "." if len(sentences) >= 3 else abstract

    # Extract keywords
    keywords = paper_content.get("keywords", "")

    date_str = datetime.now().strftime("%B %d, %Y")

    letter = f"""Dear {editor_name},

We are pleased to submit our manuscript entitled "{title}" for consideration for publication in {journal_name}.

{_get_scope_statement(guideline, keywords)}

{key_findings}

We believe this work makes a significant contribution to the field and is well-suited for the readership of {journal_name}. The manuscript has not been published previously and is not under consideration for publication elsewhere. All authors have approved the manuscript and agree with its submission.

We confirm that the manuscript complies with the journal's guidelines for authors. {_get_highlights_note(paper_content, guideline)}

We look forward to your favorable consideration.

Sincerely,
{authors}

Date: {date_str}
"""
    return letter.strip()


def reformat_paper(paper_content: dict, from_journal: str, to_journal: str) -> dict:
    """Reformat paper_content from one journal's format to another.

    Returns a new paper_content dict adjusted for the target journal.
    """
    to_guideline = _load_guideline(to_journal)
    to_ms = to_guideline.get("manuscript_structure", {})

    new_content = dict(paper_content)

    # Adjust abstract length if needed
    abstract = new_content.get("abstract", "")
    to_limit = to_ms.get("abstract_word_limit", 300)
    abstract_words = abstract.split()
    if len(abstract_words) > to_limit:
        new_content["_warnings"] = new_content.get("_warnings", [])
        new_content["_warnings"].append(
            f"Abstract ({len(abstract_words)} words) exceeds {to_journal} limit ({to_limit} words). "
            "Manual trimming required."
        )

    # Add highlights if required by target but missing
    if to_ms.get("highlights_required") and not new_content.get("highlights"):
        new_content["_warnings"] = new_content.get("_warnings", [])
        new_content["_warnings"].append(
            f"{to_journal} requires highlights (3-5 bullet points, max 85 chars each)."
        )

    # Update reference style note
    to_ref_style = to_guideline.get("references", {}).get("style", "")
    new_content["_ref_style"] = to_ref_style

    # Add reformatting metadata
    new_content["_reformatted"] = {
        "from": from_journal,
        "to": to_journal,
        "date": datetime.now().isoformat(),
        "target_journal": to_guideline.get("journal_name", to_journal),
    }

    return new_content


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_guideline(journal_key: str) -> dict:
    path = os.path.join(GUIDELINES_DIR, f"{journal_key}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Guideline not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check(description: str, passed: bool, note: str = "") -> dict:
    return {"description": description, "passed": passed, "note": note}


def _get_scope_statement(guideline: dict, keywords: str) -> str:
    scope = guideline.get("scope", "")
    if not scope:
        return "This work falls within the scope of the journal."
    # Match keywords to scope
    kw_list = [k.strip().lower() for k in keywords.split(";") if k.strip()]
    scope_lower = scope.lower()
    matching = [kw for kw in kw_list if any(w in scope_lower for w in kw.split())]
    if matching:
        return (
            f"This work addresses topics within the journal's scope, specifically in the areas of "
            f"{', '.join(matching[:3])}."
        )
    return "This work falls within the scope of the journal."


def _get_highlights_note(paper_content: dict, guideline: dict) -> str:
    ms = guideline.get("manuscript_structure", {})
    notes = []
    if ms.get("highlights_required") and paper_content.get("highlights"):
        notes.append("Highlights are included as required.")
    if ms.get("graphical_abstract_required"):
        notes.append("A graphical abstract is provided.")
    return " ".join(notes) if notes else ""
