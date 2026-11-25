"""Validate generated paper content against journal quality criteria.

Besides :func:`check_paper`, this module hosts the paper_content helpers the
other checkers (submission checklist, AI reviewer) share:

* tolerant accessors -- :func:`text_of`, :func:`list_of`, :func:`iter_sections`,
  :func:`parse_keywords`, :func:`parse_highlights` (``None`` values, keywords
  given as a list or as a ``"a; b"`` / ``"a, b"`` string, highlights given as a
  newline-separated string, ...);
* journal rules read from the guideline -- :func:`parse_range`,
  :func:`highlight_rules`;
* :func:`analyze_sections` -- maps the guideline's
  ``manuscript_structure.sections`` onto the paper's headings using synonyms
  ("Methods", "Materials and Methods", "Proposed framework", numbered
  headings, Korean headings, a combined "Results and Discussion", ...), and
  maps front/back matter (Abstract, Highlights, References, Data
  Availability, ...) onto the matching paper_content fields.
"""

from __future__ import annotations

import datetime
import re

from utils.guidelines import GUIDELINES_DIR, load_guideline  # noqa: F401  (re-exported)
from utils.reference_utils import extract_year, normalize_references

# Minimum thresholds from CLAUDE.md pipeline requirements
MIN_BODY_WORDS = 6000
MIN_FIGURES = 6
MIN_TABLES = 3
MIN_REFERENCES = 15
MIN_RECENT_RATIO = 0.5  # 50% of references within 5 years
MIN_COMPARISONS = 3
MIN_LIMITATIONS = 3
RECENT_YEARS = 5

#: Highlights rule used when a guideline's ``highlights_note`` does not say otherwise.
DEFAULT_HIGHLIGHT_RULES = {"min": 3, "max": 5, "max_chars": 85}


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
        - journal (str): the journal's full name
    """
    paper_content = paper_content or {}
    guideline = load_guideline(journal_key)
    ms = guideline.get("manuscript_structure", {}) or {}
    checks = []

    # 1. Abstract word count
    abstract_words = len(text_of(paper_content.get("abstract")).split())
    abstract_limit = ms.get("abstract_word_limit") or 300
    checks.append(
        {
            "name": "abstract_word_count",
            "passed": 0 < abstract_words <= abstract_limit,
            "value": abstract_words,
            "limit": abstract_limit,
            "severity": "critical",
            "message": f"Abstract: {abstract_words} words (limit: {abstract_limit})",
        }
    )
    abstract_min = ms.get("abstract_word_minimum")
    if abstract_min:
        checks.append(
            {
                "name": "abstract_word_minimum",
                "passed": abstract_words >= abstract_min,
                "value": abstract_words,
                "limit": abstract_min,
                "severity": "warning",
                "message": f"Abstract: {abstract_words} words (journal minimum: {abstract_min})",
            }
        )

    # 2. Body word count
    body_words = _count_body_words(paper_content)
    checks.append(
        {
            "name": "body_word_count",
            "passed": body_words >= MIN_BODY_WORDS,
            "value": body_words,
            "limit": MIN_BODY_WORDS,
            "severity": "critical",
            "message": f"Body: {body_words} words (min: {MIN_BODY_WORDS})",
        }
    )

    # 3. Required sections: Introduction and Conclusions are critical; the other
    #    core body sections of the journal's structure are a separate warning.
    analysis = analyze_sections(paper_content, guideline, figures=figures)
    paper_sections = analysis["headings"]
    missing = analysis["critical_missing"]
    checks.append(
        {
            "name": "required_sections",
            "passed": len(missing) == 0,
            "value": paper_sections,
            "missing": missing,
            "severity": "critical",
            "message": f"Sections: {len(paper_sections)} found"
            + (f", missing: {missing}" if missing else ""),
        }
    )
    core_missing = analysis["missing"]
    n_core = len(analysis["required"])
    message = f"Journal structure: {n_core - len(core_missing)}/{n_core} core body sections"
    if core_missing:
        message += f", missing: {core_missing}"
    if analysis["inferred"]:
        message += f" (inferred: {analysis['inferred']})"
    checks.append(
        {
            "name": "journal_sections",
            "passed": not core_missing,
            "value": analysis["matched"],
            "missing": core_missing,
            "optional_missing": analysis["optional_missing"],
            "inferred": analysis["inferred"],
            "front_back_matter": analysis["matter"],
            "severity": "warning",
            "message": message,
        }
    )

    # 4. References count
    refs = normalize_references(paper_content.get("references"))
    n_refs = len(refs)
    checks.append(
        {
            "name": "reference_count",
            "passed": n_refs >= MIN_REFERENCES,
            "value": n_refs,
            "limit": MIN_REFERENCES,
            "severity": "critical",
            "message": f"References: {n_refs} (min: {MIN_REFERENCES})",
        }
    )

    # 5. Recent references ratio (publication year, not DOI/page digits)
    current_year = datetime.date.today().year
    years = [extract_year(ref, current_year) for ref in refs]
    recent_count = sum(1 for y in years if y is not None and y >= current_year - RECENT_YEARS)
    recent_ratio = recent_count / max(n_refs, 1)
    checks.append(
        {
            "name": "recent_references",
            "passed": recent_ratio >= MIN_RECENT_RATIO,
            "value": f"{recent_ratio:.0%}",
            "limit": f"{MIN_RECENT_RATIO:.0%}",
            "severity": "warning",
            "message": f"Recent refs (last {RECENT_YEARS}yr): {recent_count}/{n_refs} "
            f"({recent_ratio:.0%})",
        }
    )

    # 6. Figures count
    figure_list = list_of(figures)
    n_figures = (
        len(figure_list) if figure_list else len(list_of(paper_content.get("figure_captions")))
    )
    checks.append(
        {
            "name": "figure_count",
            "passed": n_figures >= MIN_FIGURES,
            "value": n_figures,
            "limit": MIN_FIGURES,
            "severity": "critical",
            "message": f"Figures: {n_figures} (min: {MIN_FIGURES})",
        }
    )

    # 7. Tables count
    n_tables = len(list_of(paper_content.get("tables")))
    checks.append(
        {
            "name": "table_count",
            "passed": n_tables >= MIN_TABLES,
            "value": n_tables,
            "limit": MIN_TABLES,
            "severity": "warning",
            "message": f"Tables: {n_tables} (min: {MIN_TABLES})",
        }
    )

    # 8. Keywords (count range from the guideline, e.g. "1-7", "up to 9")
    kw_list = parse_keywords(paper_content.get("keywords"))
    kw_min, kw_max = parse_range(ms.get("keywords_count"))
    kw_min = kw_min or 1
    kw_ok = len(kw_list) >= kw_min and (kw_max is None or len(kw_list) <= kw_max)
    kw_limit = f"{kw_min}-{kw_max}" if kw_max else f"{kw_min}+"
    checks.append(
        {
            "name": "keywords",
            "passed": kw_ok,
            "value": len(kw_list),
            "limit": kw_limit,
            "severity": "warning",
            "message": f"Keywords: {len(kw_list)}"
            + ("" if kw_ok else f" (journal requires {kw_limit})"),
        }
    )

    # 9. Highlights (if required by journal)
    if ms.get("highlights_required", False):
        highlights = parse_highlights(paper_content.get("highlights"))
        rules = highlight_rules(guideline)
        n_hl = len(highlights)
        count_ok = n_hl >= rules["min"] and (rules["max"] is None or n_hl <= rules["max"])
        over = [
            i + 1
            for i, h in enumerate(highlights)
            if rules["max_chars"] and len(h) > rules["max_chars"]
        ]
        message = f"Highlights: {n_hl} (required by journal: {rules['min']}-{rules['max']})"
        if over:
            message += (
                f"; highlight(s) {over} exceed {rules['max_chars']} characters "
                "(see submission_checklist)"
            )
        checks.append(
            {
                "name": "highlights",
                "passed": count_ok,
                "value": n_hl,
                "limit": f"{rules['min']}-{rules['max']}",
                "char_limit": rules["max_chars"],
                "over_char_limit": over,
                "severity": "warning",
                "message": message,
            }
        )

    # 10. Title present
    title = text_of(paper_content.get("title")).strip()
    checks.append(
        {
            "name": "title",
            "passed": len(title) > 10,
            "value": len(title),
            "severity": "critical",
            "message": f"Title: {len(title)} chars",
        }
    )
    title_max = ms.get("title_max_characters")
    if title_max:
        checks.append(
            {
                "name": "title_length",
                "passed": len(title) <= title_max,
                "value": len(title),
                "limit": title_max,
                "severity": "warning",
                "message": f"Title: {len(title)} chars (journal maximum: {title_max})",
            }
        )

    # 11. Practical Applications (ASCE journals)
    if ms.get("practical_applications"):
        pa_words = len(text_of(paper_content.get("practical_applications")).split())
        pa_min, pa_max = parse_range(ms.get("practical_applications"))
        pa_ok = pa_words > 0 and (
            (pa_min is None or pa_words >= pa_min) and (pa_max is None or pa_words <= pa_max)
        )
        checks.append(
            {
                "name": "practical_applications",
                "passed": pa_ok,
                "value": pa_words,
                "limit": f"{pa_min}-{pa_max}" if pa_min and pa_max else None,
                "severity": "warning",
                "message": f"Practical Applications: {pa_words} words"
                + (f" (journal requires {pa_min}-{pa_max})" if pa_min and pa_max else ""),
            }
        )

    # 12. Data availability statement
    data_avail = bool(text_of(paper_content.get("data_availability")).strip()) or any(
        "data availability" in normalize_heading(h) for h in analysis["headings"]
    )
    checks.append(
        {
            "name": "data_availability",
            "passed": data_avail,
            "value": data_avail,
            "severity": "info",
            "message": f"Data availability: {'present' if data_avail else 'missing'}",
        }
    )

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
        "journal": guideline.get("journal_name", journal_key),
    }


def _count_body_words(paper_content: dict) -> int:
    """Count words in all section content (excluding abstract, references).

    Subsections are counted at any nesting depth (the Word generator renders
    three heading levels).
    """

    def count(sec: dict) -> int:
        total = len(text_of(sec.get("content")).split())
        for sub in iter_sections(sec.get("subsections")):
            total += count(sub)
        return total

    return sum(count(sec) for sec in iter_sections((paper_content or {}).get("sections")))


# ── paper_content accessors ──────────────────────────────────────────────────


def text_of(value) -> str:
    """Return *value* as text: ``None`` -> ``""``, a list of paragraphs is joined."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(text_of(v) for v in value if v is not None)
    return str(value)


def list_of(value) -> list:
    """Return *value* as a list: ``None`` -> ``[]``, a string -> its non-empty lines."""
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None and not (isinstance(v, str) and not v.strip())]
    return [value]


def iter_sections(value) -> list[dict]:
    """Return the section dicts of *value* (``None`` and non-dict items tolerated)."""
    out = []
    for item in list_of(value):
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"heading": "", "content": item})
    return out


def parse_keywords(value) -> list[str]:
    """Return keywords from a list or a ``"a; b"`` / ``"a, b"`` / multi-line string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(k).strip() for k in value if k is not None and str(k).strip()]
    text = re.sub(r"^\s*(?:key\s*words?|index terms)\s*[:\-]\s*", "", str(value), flags=re.I)
    separator = r"[;\n]" if ";" in text else r"[,\n]"
    return [k.strip().rstrip(".") for k in re.split(separator, text) if k.strip().rstrip(".")]


_BULLET_RE = re.compile(
    r"^\s*(?:[-*" + "".join(map(chr, (0x2022, 0x2023, 0x25AA, 0x25CF, 0x2013, 0xB7))) + r"]"
    r"|\(?\d{1,2}[.)])\s*"
)


def parse_highlights(value) -> list[str]:
    """Return highlights from a list or a newline-separated (optionally bulleted) string."""
    if value is None:
        return []
    items = list_of(value) if isinstance(value, str) else list_of(list(value))
    out = []
    for item in items:
        text = _BULLET_RE.sub("", text_of(item)).strip()
        if text:
            out.append(text)
    return out


def parse_range(spec, default: tuple = (None, None)) -> tuple:
    """Parse a guideline count spec: ``"1-7"`` -> (1, 7), ``"up to 9"`` -> (None, 9),
    ``"150-200 words"`` -> (150, 200), ``5`` -> (None, 5)."""
    if spec is None or spec is False:
        return default
    if isinstance(spec, bool):
        return default
    if isinstance(spec, int):
        return (None, spec)
    text = str(spec).lower()
    m = re.search(r"(\d+)\s*(?:-|\u2013|to)\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(
        r"(?:up to|max(?:imum)?\.?|at most|no more than|not exceed(?:ing)?)\s*(\d+)", text
    )
    if m:
        return None, int(m.group(1))
    m = re.search(r"(?:at least|min(?:imum)?\.?)\s*(\d+)", text)
    if m:
        return int(m.group(1)), None
    m = re.fullmatch(r"\s*(\d+)\s*", text)
    if m:
        return None, int(m.group(1))
    return default


def highlight_rules(guideline: dict) -> dict:
    """Return ``{"min", "max", "max_chars"}`` for highlights from the guideline.

    Reads ``manuscript_structure.highlights_note`` (e.g. "3-5 bullet points,
    max 85 characters each including spaces"); optional explicit keys
    ``highlights_count`` / ``highlights_max_characters`` take precedence.
    """
    ms = (guideline or {}).get("manuscript_structure", {}) or {}
    rules = dict(DEFAULT_HIGHLIGHT_RULES)
    note = str(ms.get("highlights_note") or "")
    m = re.search(r"(\d+)\s*(?:-|\u2013|to)\s*(\d+)\s*(?:bullet|highlight|item|point)", note, re.I)
    if m:
        rules["min"], rules["max"] = int(m.group(1)), int(m.group(2))
    m = re.search(
        r"(?:max(?:imum)?\.?|up to|at most|no more than)\s*(\d+)\s*char", note, re.IGNORECASE
    )
    if m:
        rules["max_chars"] = int(m.group(1))
    if ms.get("highlights_count") is not None:
        lo, hi = parse_range(ms["highlights_count"])
        rules["min"], rules["max"] = lo or rules["min"], hi or rules["max"]
    if ms.get("highlights_max_characters"):
        rules["max_chars"] = int(ms["highlights_max_characters"])
    return rules


# ── Section analysis ─────────────────────────────────────────────────────────

#: Unambiguous heading synonyms per body-section category (normalised, lower case).
_STRONG_PATTERNS = {
    "introduction": r"\bintroduction\b|서론|머리말",
    "conclusions": r"\bconclu(?:sion|sions|ding)\b|\b(?:closing|final) remarks\b|^summary$|"
    r"^summary and\b|결론|맺음말",
    "results": r"\bresults?\b|\bfindings\b|결과",
    "discussion": r"\bdiscussions?\b|고찰|논의|토의",
    "methodology": r"\bmethods?\b|\bmethodolog(?:y|ies)\b|\bmaterials?\b|"
    r"\bexperimental (?:program(?:me)?|setup|set-up|procedure|design|campaign|details|work)\b|"
    r"\banalytical program(?:me)?\b|\btest (?:program(?:me)?|setup|set-up|procedure|specimens?)\b|"
    r"\bresearch design\b|\bproposed (?:framework|method|methodology|approach|model|system)\b|"
    r"방법",
    "theory": r"\btheor(?:y|ies|etical)\b|\bformulation\b|\bgoverning equations?\b|이론",
    "literature_review": r"\bliterature\b|\brelated (?:work|works|studies|research)\b|"
    r"\bstate[- ]of[- ](?:the[- ])?art\b|\bprevious (?:studies|research|work)\b|선행\s*연구|문헌",
    "research_significance": r"\bsignificance\b|의의",
}

#: Weaker hints, used (in this priority order) only when no strong pattern matches.
_WEAK_PATTERNS = (
    (
        "methodology",
        r"\bexperiment(?:s|al)?\b|\bnumerical\b|\bmodel(?:s|l?ing)?\b|\bsimulations?\b|"
        r"\bframework\b|\bapproach\b|\bdata(?:sets?)?\b|\bfinite[- ]element\b|\bprocedures?\b|"
        r"\bspecimens?\b|\bsetup\b|\bdesign\b|실험|해석|모델",
    ),
    (
        "results",
        r"\bcase stud(?:y|ies)\b|\bparametric\b|\bvalidation\b|\bcomparisons?\b|"
        r"\bperformance\b|\bevaluation\b|\bapplications?\b|\bsensitivity\b",
    ),
    ("theory", r"\bbackground\b|\bpreliminar(?:y|ies)\b|\bfundamentals?\b|\bconcepts?\b"),
    ("literature_review", r"\breview\b"),
)

#: Front/back matter: (regex on the guideline section name, paper_content fields,
#: regex on a body heading that also satisfies it).
_MATTER = (
    (r"^title( page)?$", ("title",), None),
    (r"author names|affiliation", ("authors", "affiliations"), None),
    (r"biograph", ("author_bios", "biographies", "author_biographies"), r"biograph"),
    (r"^abstract$", ("abstract",), r"^abstract$"),
    (r"^key ?words$", ("keywords",), r"^key ?words$"),
    (r"^highlights$", ("highlights",), r"^highlights$"),
    (r"graphical abstract", ("graphical_abstract",), r"graphical abstract"),
    (r"practical applications?", ("practical_applications",), r"practical applications?"),
    (r"nomenclature|notation", ("notation", "nomenclature"), r"nomenclature|notation"),
    (r"acknowledg", ("acknowledgments", "acknowledgements"), r"acknowledg"),
    (r"^references$|bibliography", ("references",), r"^references$|bibliography"),
    (r"appendi", ("appendices", "appendix"), r"^appendi"),
    (r"data availability", ("data_availability",), r"data availability"),
    (
        r"competing interest|conflicts? of interest",
        (
            "conflict_of_interest",
            "conflicts_of_interest",
            "competing_interests",
            "declaration_of_interest",
            "declaration_of_competing_interest",
        ),
        r"competing interest|conflicts? of interest",
    ),
    (
        r"generative ai",
        ("ai_declaration", "generative_ai_declaration", "declaration_of_generative_ai"),
        r"generative ai",
    ),
    (
        r"credit|author contributions?",
        ("credit_statement", "credit_author_statement", "author_contributions"),
        r"credit|author contributions?",
    ),
    (r"^funding$", ("funding",), r"^funding$"),
    (
        r"supplement",
        ("supplemental_materials", "supplementary_materials", "supplementary_material"),
        r"supplement",
    ),
    (r"^tables$", ("tables",), None),
    (r"figure captions", ("figure_captions",), None),
    (r"^figures$", ("figures", "figure_captions"), None),
)

_CRITICAL_LABELS = {"introduction": "INTRODUCTION", "conclusions": "CONCLUSION"}

#: Structure assumed when no guideline is available.
DEFAULT_SECTIONS = ["Introduction", "Methodology", "Results", "Conclusions"]

_HEADING_NUMBER_RE = re.compile(
    r"^\s*(?:(?:section|chapter)\s+)?(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.|[a-z]\.)\s*[:.)\-]?\s+",
    re.IGNORECASE,
)


def normalize_heading(heading) -> str:
    """'2.1. Materials & Methods' -> 'materials and methods'; '제1장 서론' -> '서론'."""
    text = text_of(heading).strip()
    text = re.sub(r"^\s*제?\s*\d+\s*장\s*", "", text)
    text = _HEADING_NUMBER_RE.sub("", text, count=1)
    text = text.replace("&", " and ").replace("/", " / ").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:;-")


def classify_heading(heading) -> list[str]:
    """Return the body-section categories a heading belongs to (may be several)."""
    text = normalize_heading(heading)
    if not text:
        return []
    strong = [cat for cat, pattern in _STRONG_PATTERNS.items() if re.search(pattern, text)]
    if strong:
        return strong
    for cat, pattern in _WEAK_PATTERNS:
        if re.search(pattern, text):
            return [cat]
    return []


def _matter_rule(name: str):
    low = normalize_heading(name)
    for name_re, fields, heading_re in _MATTER:
        if re.search(name_re, low):
            return fields, heading_re
    return None


def _is_matter_heading(heading: str) -> bool:
    low = normalize_heading(heading)
    return any(h_re and re.search(h_re, low) for _n, _f, h_re in _MATTER)


def _requirement(name: str, categories: list[str], ms: dict) -> str:
    """'critical', 'core' or 'optional' for a guideline body section."""
    if any(c in _CRITICAL_LABELS for c in categories):
        return "critical"
    if "methodology" in categories or "results" in categories:
        return "core"
    if categories == ["discussion"]:
        return "core"
    if "research_significance" in categories:
        return "core" if ms.get("research_significance_required") else "optional"
    return "optional"


def analyze_sections(paper_content: dict, guideline: dict | None = None, figures=None) -> dict:
    """Map the guideline's section list onto the paper's headings and fields.

    Returns a dict with:

    * ``headings`` -- the paper's top-level headings (upper case);
    * ``critical_missing`` -- ``"INTRODUCTION"`` / ``"CONCLUSION"`` if absent;
    * ``required`` / ``missing`` -- core body sections of the journal structure
      (e.g. Methodology, Results, Discussion, Main Body) and those not found;
    * ``optional_missing`` -- optional body sections not found (Theory,
      Literature Review, ...);
    * ``matched`` -- guideline section -> paper heading (or ``field:<name>``);
    * ``inferred`` -- guideline section -> heading assigned by position (an
      unrecognised heading between Introduction and Results counts as the
      methods section, one between methods and Conclusions as results);
    * ``matter`` -- front/back-matter guideline entries -> present (bool);
      these are satisfied by paper_content fields and never fail the check.
    """
    paper_content = paper_content or {}
    ms = (guideline or {}).get("manuscript_structure", {}) or {}
    names = [str(s) for s in (ms.get("sections") or DEFAULT_SECTIONS)]

    sections = iter_sections(paper_content.get("sections"))
    headings = [text_of(s.get("heading")).strip() for s in sections]
    top_cats = [[] if _is_matter_heading(h) else classify_heading(h) for h in headings]
    sub_headings = []
    for sec in sections:
        stack = list(iter_sections(sec.get("subsections")))
        while stack:
            sub = stack.pop(0)
            sub_headings.append(text_of(sub.get("heading")).strip())
            stack.extend(iter_sections(sub.get("subsections")))

    def find(category: str, include_subsections: bool = True) -> str | None:
        for heading, cats in zip(headings, top_cats, strict=True):
            if category in cats:
                return heading
        if include_subsections:
            for heading in sub_headings:
                if category in classify_heading(heading):
                    return heading
        return None

    found = {cat: find(cat, cat not in _CRITICAL_LABELS) for cat in _STRONG_PATTERNS}

    # Positional inference for unrecognised top-level headings.
    inferred_cats: dict[str, str] = {}
    unclassified = [
        i for i, (h, cats) in enumerate(zip(headings, top_cats, strict=True)) if h and not cats
    ]
    unclassified = [i for i in unclassified if not _is_matter_heading(headings[i])]

    def index_of(category: str) -> int | None:
        return next((i for i, cats in enumerate(top_cats) if category in cats), None)

    intro_i = index_of("introduction")
    concl_i = index_of("conclusions")
    lower = intro_i if intro_i is not None else -1
    upper = concl_i if concl_i is not None else len(headings)
    pool = [i for i in unclassified if lower < i < upper]
    if found["methodology"] is None and pool:
        results_i = index_of("results")
        candidates = [i for i in pool if results_i is None or i < results_i]
        if candidates:
            inferred_cats["methodology"] = headings[candidates[0]]
            pool.remove(candidates[0])
    if found["results"] is None and pool:
        method_i = index_of("methodology")
        candidates = [i for i in pool if method_i is None or i > method_i]
        if candidates:
            inferred_cats["results"] = headings[candidates[-1]]
    for cat, heading in inferred_cats.items():
        found[cat] = heading

    critical_missing = [label for cat, label in _CRITICAL_LABELS.items() if found.get(cat) is None]

    body_headings = [
        h
        for h, cats in zip(headings, top_cats, strict=True)
        if h and not _is_matter_heading(h) and not any(c in _CRITICAL_LABELS for c in cats)
    ]

    required, missing, optional_missing = [], [], []
    matched: dict = {}
    inferred: dict = {}
    matter: dict = {}
    for name in names:
        rule = _matter_rule(name)
        if rule is not None:
            fields, heading_re = rule
            present = any(_has_value(paper_content.get(f)) for f in fields)
            if "figures" in fields and list_of(figures):
                present = True
            if not present and heading_re:
                present = any(
                    re.search(heading_re, normalize_heading(h)) for h in headings + sub_headings
                )
            matter[name] = present
            if present:
                field = next((f for f in fields if _has_value(paper_content.get(f))), None)
                matched[name] = f"field:{field}" if field else "heading"
            continue

        low = normalize_heading(name)
        if low == "main body":
            required.append(name)
            if body_headings:
                matched[name] = body_headings[0]
            else:
                missing.append(name)
            continue

        categories = [c for c in _STRONG_PATTERNS if re.search(_STRONG_PATTERNS[c], low)]
        if not categories:
            categories = classify_heading(name)
        requirement = _requirement(name, categories, ms)
        if requirement == "critical":
            cat = next(c for c in categories if c in _CRITICAL_LABELS)
            if found.get(cat):
                matched[name] = found[cat]
            continue
        # "Results and Discussion" is satisfied by a results section; a separately
        # listed "Discussion" needs a discussion heading (or a combined one).
        needed = ["results"] if "results" in categories else categories[:1] or [low]
        heading = next((found.get(c) for c in needed if found.get(c)), None)
        if requirement == "core":
            required.append(name)
        if heading:
            matched[name] = heading
            if any(inferred_cats.get(c) == heading for c in needed):
                inferred[name] = heading
        elif requirement == "core":
            missing.append(name)
        else:
            optional_missing.append(name)

    return {
        "headings": [h.upper() for h in headings],
        "critical_missing": critical_missing,
        "required": required,
        "missing": missing,
        "optional_missing": optional_missing,
        "matched": matched,
        "inferred": inferred,
        "matter": matter,
        "found": {k: v for k, v in found.items() if v},
    }


def _has_value(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True
