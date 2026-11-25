"""Submission workflow utilities: checklist, cover letter, journal reformatting."""

from __future__ import annotations

import copy
import re
from datetime import datetime

from utils.citation_converter import _parse_ref, convert_style, first_author_family
from utils.guidelines import (  # noqa: F401  (re-exported)
    GUIDELINES_DIR,
    UnknownJournalError,
    resolve_journal_key,
)
from utils.guidelines import load_guideline as _load_guideline
from utils.quality_checker import (
    analyze_sections,
    highlight_rules,
    list_of,
    normalize_heading,
    parse_highlights,
    parse_keywords,
    parse_range,
    text_of,
)
from utils.reference_utils import (
    check_duplicates,
    detect_citation_style,
    extract_year,
    has_doi,
    normalize_references,
)

_COI_FIELDS = (
    "conflict_of_interest",
    "conflicts_of_interest",
    "competing_interests",
    "declaration_of_interest",
    "declaration_of_competing_interest",
)
_CREDIT_FIELDS = ("credit_statement", "credit_author_statement", "author_contributions")


def submission_checklist(paper_content: dict, journal_key: str, figures: list = None) -> dict:
    """Generate a journal-specific submission checklist.

    Returns dict with 'items' (list of check dicts), 'ready' (bool), 'journal'
    (full name) and 'journal_key'.  Each item has 'description', 'passed',
    'note' and 'required'; advisory items (``required=False``, e.g. optional
    sections of the journal's structure) are reported but do not affect
    'ready'.
    """
    paper_content = paper_content or {}
    guideline = _load_guideline(journal_key)
    ms = guideline.get("manuscript_structure", {}) or {}
    items = []

    # Manuscript format checks
    title = text_of(paper_content.get("title")).strip()
    items.append(_check("Title page present", bool(title)))
    items.append(_check("Authors listed", bool(_authors_text(paper_content.get("authors")))))
    n_abstract = len(text_of(paper_content.get("abstract")).split())
    limit = ms.get("abstract_word_limit") or 300
    items.append(
        _check(
            "Abstract within word limit",
            0 < n_abstract <= limit,
            note=f"{n_abstract}/{limit} words" if n_abstract else "Abstract is empty",
        )
    )
    if ms.get("abstract_word_minimum"):
        minimum = ms["abstract_word_minimum"]
        items.append(
            _check(
                f"Abstract meets journal minimum ({minimum}+ words)",
                n_abstract >= minimum,
                note=f"{n_abstract} words",
            )
        )
    keywords = parse_keywords(paper_content.get("keywords"))
    items.append(_check("Keywords provided", bool(keywords)))
    kw_min, kw_max = parse_range(ms.get("keywords_count"))
    if keywords and (kw_min or kw_max):
        items.append(
            _check(
                f"Keywords count within journal range ({ms['keywords_count']})",
                (kw_min is None or len(keywords) >= kw_min)
                and (kw_max is None or len(keywords) <= kw_max),
                note=f"{len(keywords)} keywords",
            )
        )
    if ms.get("title_max_characters"):
        max_chars = ms["title_max_characters"]
        items.append(
            _check(
                f"Title within {max_chars} characters",
                0 < len(title) <= max_chars,
                note=f"{len(title)} characters",
            )
        )

    # Highlights (count and length rules come from the guideline's highlights_note)
    if ms.get("highlights_required"):
        hl = parse_highlights(paper_content.get("highlights"))
        rules = highlight_rules(guideline)
        items.append(
            _check(
                f"Highlights provided ({rules['min']}-{rules['max']} items)",
                rules["min"] <= len(hl) <= rules["max"],
                note=f"{len(hl)} provided",
            )
        )
        if hl:
            over = [f"#{i + 1} ({len(h)})" for i, h in enumerate(hl) if len(h) > rules["max_chars"]]
            items.append(
                _check(
                    "Highlights within character limit",
                    not over,
                    note=f"max {rules['max_chars']} characters each"
                    + (f"; too long: {', '.join(over)}" if over else ""),
                )
            )

    # Graphical abstract
    if ms.get("graphical_abstract_required"):
        items.append(
            _check(
                "Graphical abstract prepared",
                bool(paper_content.get("graphical_abstract")),
                note="Required by journal — use PaperBanana to generate",
            )
        )

    # Practical Applications (ASCE)
    if ms.get("practical_applications"):
        items.append(
            _check(
                "Practical Applications section provided",
                bool(text_of(paper_content.get("practical_applications")).strip()),
                note=str(ms["practical_applications"]),
            )
        )

    # Sections: Introduction/Conclusion are required, the rest of the journal's
    # core structure is advisory.
    analysis = analyze_sections(paper_content, guideline, figures=figures)
    for req in ("INTRODUCTION", "CONCLUSION"):
        items.append(_check(f"Section '{req}' present", req not in analysis["critical_missing"]))
    for name in analysis["required"]:
        found = analysis["matched"].get(name)
        items.append(
            _check(
                f"Section '{name}' present (journal structure)",
                name not in analysis["missing"],
                note=f"Matched: {found}" if found else "Advisory — does not affect 'ready'",
                required=False,
            )
        )

    # References
    refs = normalize_references(paper_content.get("references"))
    items.append(_check("References present (15+)", len(refs) >= 15, note=f"{len(refs)} found"))
    doi_count = sum(1 for r in refs if has_doi(r))
    items.append(
        _check(
            "References include DOIs",
            bool(refs) and doi_count >= len(refs) * 0.5,
            note=f"{doi_count}/{len(refs)} with DOI",
        )
    )
    duplicates = check_duplicates(refs)
    items.append(
        _check(
            "No duplicate references",
            not duplicates,
            note=", ".join(f"#{d['index_a'] + 1} = #{d['index_b'] + 1}" for d in duplicates),
        )
    )
    journal_style = guideline.get("references", {}).get("style", "")
    family = _style_family(journal_style)
    detected = _detected_family(refs)
    if family and detected:
        items.append(
            _check(
                f"Reference list style matches journal ({journal_style})",
                detected == family,
                note=f"Detected {detected.replace('_', '-')} references",
            )
        )

    # Figures
    figure_list = list_of(figures)
    captions = list_of(paper_content.get("figure_captions"))
    n_figs = len(figure_list) if figure_list else len(captions)
    items.append(_check("Figures included (6+)", n_figs >= 6, note=f"{n_figs} found"))
    items.append(_check("Figure captions provided", len(captions) >= n_figs))

    # Tables
    n_tables = len(list_of(paper_content.get("tables")))
    items.append(_check("Tables included (3+)", n_tables >= 3, note=f"{n_tables} found"))

    # Data availability
    headings = [normalize_heading(h) for h in analysis["headings"]]
    items.append(
        _check(
            "Data availability statement",
            bool(text_of(paper_content.get("data_availability")).strip())
            or any("data availability" in h for h in headings),
        )
    )

    # Ethical statements
    coi = any(text_of(paper_content.get(f)).strip() for f in _COI_FIELDS) or any(
        re.search(r"competing interest|conflicts? of interest", h) for h in headings
    )
    items.append(
        _check(
            "Conflict of interest statement",
            coi or bool(text_of(paper_content.get("acknowledgments")).strip()),
            note="Can be included in acknowledgments",
        )
    )
    if ms.get("credit_author_statement"):
        items.append(
            _check(
                "CRediT author statement",
                any(text_of(paper_content.get(f)).strip() for f in _CREDIT_FIELDS),
                note="Recommended by the journal (author contributions)",
                required=False,
            )
        )

    ready = all(item["passed"] for item in items if item.get("required", True))
    return {
        "items": items,
        "ready": ready,
        "journal": guideline.get("journal_name", journal_key),
        "journal_key": resolve_journal_key(journal_key),
    }


def generate_cover_letter(
    paper_content: dict, journal_key: str, editor_name: str = "Editor-in-Chief"
) -> str:
    """Generate a cover letter for journal submission."""
    paper_content = paper_content or {}
    guideline = _load_guideline(journal_key)
    journal_name = guideline.get("journal_name", journal_key)
    title = text_of(paper_content.get("title")).strip()
    authors = _authors_text(paper_content.get("authors")) or "The Authors"
    abstract = text_of(paper_content.get("abstract")).strip()
    editor = text_of(editor_name).strip() or "Editor-in-Chief"

    # Key findings: the last three sentences of the abstract.
    sentences = _sentences(abstract)
    key_findings = " ".join(sentences[-3:]) if len(sentences) >= 3 else abstract

    keywords = parse_keywords(paper_content.get("keywords"))
    now = datetime.now()
    date_str = f"{now:%B} {now.day}, {now.year}"

    paragraphs = [
        f"Dear {editor},",
        f'We are pleased to submit our manuscript entitled "{title}" for consideration '
        f"for publication in {journal_name}.",
        _get_scope_statement(guideline, keywords),
        key_findings,
        "We believe this work makes a significant contribution to the field and is "
        f"well-suited for the readership of {journal_name}. The manuscript has not been "
        "published previously and is not under consideration for publication elsewhere. "
        "All authors have approved the manuscript and agree with its submission.",
        (
            "We confirm that the manuscript complies with the journal's guidelines for authors. "
            + _get_highlights_note(paper_content, guideline)
        ).strip(),
        "We look forward to your favorable consideration.",
        f"Sincerely,\n{authors}",
        f"Date: {date_str}",
    ]
    return "\n\n".join(p for p in paragraphs if p).strip()


def reformat_paper(
    paper_content: dict, from_journal: str, to_journal: str, convert_references: bool = True
) -> dict:
    """Reformat paper_content from one journal's format to another.

    Returns a new paper_content dict (the input is never modified) adjusted for
    the target journal: journal-specific problems are listed in ``_warnings``
    and, when the target journal uses a different reference style family
    (numbered vs author-date), the references are converted (set
    ``convert_references=False`` to keep them); the originals are kept in
    ``_original_references``.
    """
    to_guideline = _load_guideline(to_journal)
    to_key = resolve_journal_key(to_journal)
    to_ms = to_guideline.get("manuscript_structure", {}) or {}
    to_name = to_guideline.get("journal_name", to_key)

    new_content = copy.deepcopy(paper_content or {})
    warnings = list(new_content.get("_warnings") or [])

    try:
        from_guideline = _load_guideline(from_journal)
        from_key = resolve_journal_key(from_journal)
    except UnknownJournalError:
        from_guideline, from_key = None, from_journal
        warnings.append(
            f"Unknown source journal '{from_journal}'; reference style detected from the list."
        )

    # Abstract length
    abstract_words = text_of(new_content.get("abstract")).split()
    to_limit = to_ms.get("abstract_word_limit") or 300
    if len(abstract_words) > to_limit:
        warnings.append(
            f"Abstract ({len(abstract_words)} words) exceeds the {to_name} limit "
            f"({to_limit} words). Manual trimming required."
        )
    to_min = to_ms.get("abstract_word_minimum")
    if to_min and 0 < len(abstract_words) < to_min:
        warnings.append(
            f"Abstract ({len(abstract_words)} words) is shorter than the {to_name} "
            f"minimum ({to_min} words)."
        )

    # Highlights
    if to_ms.get("highlights_required"):
        rules = highlight_rules(to_guideline)
        hl = parse_highlights(new_content.get("highlights"))
        rule_text = (
            f"{rules['min']}-{rules['max']} bullet points, max {rules['max_chars']} chars each"
        )
        if not hl:
            warnings.append(f"{to_name} requires highlights ({rule_text}).")
        else:
            too_long = [i + 1 for i, h in enumerate(hl) if len(h) > rules["max_chars"]]
            if not rules["min"] <= len(hl) <= rules["max"] or too_long:
                warnings.append(
                    f"Highlights do not meet the {to_name} rule ({rule_text}): {len(hl)} given"
                    + (f", too long: {too_long}" if too_long else "")
                    + "."
                )
    if to_ms.get("graphical_abstract_required") and not new_content.get("graphical_abstract"):
        warnings.append(f"{to_name} requires a graphical abstract.")
    if (
        to_ms.get("practical_applications")
        and not text_of(new_content.get("practical_applications")).strip()
    ):
        warnings.append(
            f"{to_name} requires a Practical Applications section "
            f"({to_ms['practical_applications']})."
        )
    kw = parse_keywords(new_content.get("keywords"))
    kw_min, kw_max = parse_range(to_ms.get("keywords_count"))
    if kw and ((kw_min and len(kw) < kw_min) or (kw_max and len(kw) > kw_max)):
        warnings.append(f"{len(kw)} keywords given; {to_name} allows {to_ms['keywords_count']}.")
    title_max = to_ms.get("title_max_characters")
    title = text_of(new_content.get("title")).strip()
    if title_max and len(title) > title_max:
        warnings.append(f"Title ({len(title)} chars) exceeds the {to_name} limit ({title_max}).")

    # Section structure of the target journal
    analysis = analyze_sections(new_content, to_guideline)
    missing_sections = analysis["critical_missing"] + analysis["missing"]
    if missing_sections:
        warnings.append(f"{to_name} structure: missing section(s) {missing_sections}.")

    # References: convert when the style family differs
    to_ref_style = to_guideline.get("references", {}).get("style", "")
    from_ref_style = (from_guideline or {}).get("references", {}).get("style", "")
    to_family = _style_family(to_ref_style)
    refs = normalize_references(new_content.get("references"))
    src_family = _detected_family(refs) or _style_family(from_ref_style)
    converted = False
    if refs and to_family and src_family and src_family != to_family:
        if convert_references:
            new_refs = convert_style(refs, to_family)
            list_format = str(to_guideline.get("references", {}).get("list_format", ""))
            if to_family == "author_date" and "alphabetical" in list_format.lower():
                new_refs = sorted(
                    new_refs,
                    key=lambda r: (first_author_family(r).casefold(), extract_year(r) or 0),
                )
            new_content["_original_references"] = copy.deepcopy(paper_content.get("references"))
            new_content["references"] = new_refs
            converted = True
            unparsed = sum(1 for r in refs if not _parse_ref(r)["_parsed"])
            warnings.append(
                f"References converted from {src_family.replace('_', '-')} to "
                f"{to_family.replace('_', '-')} style for {to_name} ({to_ref_style}); "
                "update in-text citations accordingly."
                + (
                    f" {unparsed} reference(s) could not be parsed and were left unchanged."
                    if unparsed
                    else ""
                )
            )
            if to_family == "numbered":
                warnings.append(
                    "References were numbered in their existing order; renumber them in "
                    "order of first citation."
                )
        else:
            warnings.append(
                f"{to_name} uses {to_ref_style} references but the list is "
                f"{src_family.replace('_', '-')}; conversion was skipped."
            )
    elif (
        refs
        and from_ref_style
        and to_ref_style
        and _style_key(from_ref_style) != _style_key(to_ref_style)
    ):
        warnings.append(
            f"Reference formatting differs ({from_ref_style} -> {to_ref_style}); "
            "check each entry against the journal's examples."
        )

    new_content["_ref_style"] = to_ref_style
    new_content["_reformatted"] = {
        "from": from_key,
        "to": to_key,
        "date": datetime.now().isoformat(),
        "target_journal": to_name,
        "source_journal": (from_guideline or {}).get("journal_name", from_journal),
        "references_converted": converted,
    }
    if warnings:
        new_content["_warnings"] = warnings

    return new_content


# ── Helpers ──────────────────────────────────────────────────────────────────


def _check(description: str, passed: bool, note: str = "", required: bool = True) -> dict:
    return {"description": description, "passed": bool(passed), "note": note, "required": required}


def _authors_text(authors) -> str:
    if isinstance(authors, (list, tuple)):
        names = []
        for a in authors:
            if isinstance(a, dict):
                a = a.get("name") or " ".join(
                    str(a.get(k, "")) for k in ("given", "family") if a.get(k)
                )
            if a:
                names.append(str(a).strip())
        return ", ".join(n for n in names if n)
    return text_of(authors).strip()


def _style_family(style: str) -> str | None:
    """Map a guideline reference style to a converter style family."""
    low = str(style or "").lower()
    if "numbered" in low or "vancouver" in low:
        return "numbered"
    if "author-date" in low or "author date" in low or "harvard" in low or "apa" in low:
        return "author_date"
    return None


def _style_key(style: str) -> str:
    """Normalise a style name so cosmetic variants compare equal."""
    return re.sub(r"\s+with titles\b", "", str(style or "").lower()).strip()


def _detected_family(refs: list) -> str | None:
    detected = detect_citation_style(refs)
    return {"numbered": "numbered", "author-date": "author_date"}.get(detected)


_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "al.",
    "fig.",
    "figs.",
    "eq.",
    "eqs.",
    "ref.",
    "refs.",
    "no.",
    "vs.",
    "approx.",
    "ca.",
    "cf.",
    "resp.",
    "dr.",
    "prof.",
    "st.",
    "vol.",
    "pp.",
    "sec.",
    "tab.",
}
_SENTENCE_START = r"[A-Z0-9\"'(\[" + chr(0xAC00) + "-" + chr(0xD7A3) + r"]"


def _sentences(text: str) -> list[str]:
    """Split text into sentences without breaking decimals ("0.95") or "e.g."."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    pieces = re.split(rf"(?<=[.!?])\s+(?={_SENTENCE_START})", text)
    sentences: list[str] = []
    for piece in pieces:
        if sentences:
            last_word = sentences[-1].rsplit(" ", 1)[-1]
            if last_word.lower() in _ABBREVIATIONS or re.fullmatch(
                r"(?:[A-Za-z]\.){2,}", last_word
            ):
                sentences[-1] += " " + piece
                continue
        sentences.append(piece)
    return sentences


_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "into",
    "using",
    "based",
    "under",
    "over",
    "via",
    "its",
    "their",
    "this",
    "that",
    "of",
    "in",
    "on",
    "to",
    "a",
    "an",
    "by",
    "or",
}


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def _get_scope_statement(guideline: dict, keywords) -> str:
    scope = text_of(guideline.get("scope", ""))
    if not scope:
        return "This work falls within the scope of the journal."
    if isinstance(keywords, str) or keywords is None:
        keywords = parse_keywords(keywords)
    scope_words = {_stem(w) for w in re.findall(r"[a-z0-9]+", scope.lower())}
    matching = []
    for kw in keywords:
        words = [w for w in re.findall(r"[a-z0-9]+", kw.lower()) if w not in _STOPWORDS]
        if any(len(w) >= 3 and _stem(w) in scope_words for w in words):
            matching.append(kw.strip())
    if matching:
        return (
            f"This work addresses topics within the journal's scope, specifically in the areas of "
            f"{', '.join(matching[:3])}."
        )
    return "This work falls within the scope of the journal."


def _get_highlights_note(paper_content: dict, guideline: dict) -> str:
    ms = guideline.get("manuscript_structure", {}) or {}
    notes = []
    if ms.get("highlights_required") and parse_highlights(paper_content.get("highlights")):
        notes.append("Highlights are included as required.")
    if paper_content.get("graphical_abstract"):
        notes.append("A graphical abstract is provided.")
    return " ".join(notes) if notes else ""
