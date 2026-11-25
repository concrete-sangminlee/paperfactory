"""Word document generator for PaperFactory.

Supports all journals defined in guidelines/*.json.

Besides :func:`generate_word`, this module holds the small helpers that
normalise a ``paper_content`` dict (``None`` values, lists given as strings,
ragged tables, ...) and interpret journal guideline rules (heading numbering,
caption style, back-matter order).  :mod:`utils.latex_generator` and
:mod:`utils.pdf_generator` import them so that all three exporters agree.
"""

from __future__ import annotations

import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.image.image import Image as DocxImage
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

from utils.guidelines import (  # noqa: F401  (GUIDELINES_DIR/load_guideline are re-exported)
    GUIDELINES_DIR,
    load_guideline,
    resolve_journal_key,
)

DEFAULT_PAPERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "papers"
)

#: Font used for Hangul text in Word output (Times New Roman has no Hangul glyphs).
KOREAN_FONT = "Batang"


# ---------------------------------------------------------------------------
# Shared helpers: paper_content normalisation (used by all three generators)
# ---------------------------------------------------------------------------

_HANGUL_RE = re.compile("[ᄀ-ᇿ㄰-㆏ꥠ-꥿가-힯ힰ-퟿]")
_BULLET_RE = re.compile(r"^\s*(?:[•●▪‣⁃*-])\s+")
# Characters that are invalid in XML 1.0 (python-docx raises ValueError on them).
# Form feed / vertical tab are common in text copied from PDFs and become spaces.
_PAGE_BREAK_CHARS_RE = re.compile("[\x0b\x0c]")
_XML_INVALID_RE = re.compile("[\x00-\x08\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _contains_hangul(text) -> bool:
    """Return True if *text* contains Korean (Hangul) characters."""
    return isinstance(text, str) and bool(_HANGUL_RE.search(text))


def _as_text(value, sep: str = ", ") -> str:
    """Coerce an optional ``paper_content`` value to a plain string.

    ``None`` becomes ``""``; lists/tuples are joined with *sep* (empty items are
    skipped); dicts contribute their ``name`` (author records) or their values.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, dict):
        if "name" in value:
            return _as_text(value["name"], sep)
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        parts = (_as_text(v, sep).strip() for v in value)
        return sep.join(p for p in parts if p)
    return _clean_text(str(value))


def _clean_text(text: str) -> str:
    """Drop characters that cannot be stored in .docx XML or PDF text."""
    return _XML_INVALID_RE.sub("", _PAGE_BREAK_CHARS_RE.sub(" ", text))


def _all_text(value) -> str:
    """All strings inside a (nested) ``paper_content`` value, newline-joined."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_all_text(v) for v in value)
    return "" if value is None else str(value)


def _as_list(value) -> list:
    """Return *value* as a list: ``None`` -> ``[]``, scalars -> ``[value]``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_items(value, split_lines: bool = True) -> list[str]:
    """Normalise a list-like value to a list of non-empty, stripped strings.

    A single string is split into lines when *split_lines* is true (the
    documented way of passing ``references`` as one string).
    """
    if isinstance(value, str):
        raw = value.splitlines() if split_lines else [value]
    else:
        raw = _as_list(value)
    items = []
    for item in raw:
        text = _as_text(item).strip()
        if text:
            items.append(text)
    return items


def _highlight_items(value) -> list[str]:
    """Highlights as a list of strings, without leading bullet characters."""
    return [_BULLET_RE.sub("", item) for item in _as_items(value)]


def _as_sections(value) -> list[dict]:
    """Normalise ``sections``/``subsections`` to a list of dicts.

    Plain strings are kept as heading-less sections instead of being dropped.
    """
    if isinstance(value, dict):
        value = [value]
    elif isinstance(value, str):
        value = [value]
    sections = []
    for sec in _as_list(value):
        if isinstance(sec, dict):
            sections.append(sec)
        else:
            text = _as_text(sec, "\n\n")
            if text.strip():
                sections.append({"heading": "", "content": text})
    return sections


def _section_parts(sec: dict) -> tuple[str, str, list[dict]]:
    """Return ``(heading, content, subsections)`` for one section dict."""
    heading = _as_text(sec.get("heading")).strip()
    content = _as_text(sec.get("content"), "\n\n")
    return heading, content, _as_sections(sec.get("subsections"))


def _notation_entries(value) -> list[tuple[str, str]]:
    """Normalise ``notation`` to ``[(symbol, definition), ...]``.

    Accepts the documented list of ``{"symbol", "definition"}`` dicts as well as
    ``(symbol, definition)`` pairs or a ``{symbol: definition}`` mapping.
    """
    if isinstance(value, dict):
        if set(value) <= {"symbol", "definition"}:
            raw = [value]
        else:
            raw = list(value.items())
    else:
        raw = _as_list(value)
    entries = []
    for entry in raw:
        if isinstance(entry, dict):
            symbol, definition = entry.get("symbol"), entry.get("definition")
        elif isinstance(entry, (list, tuple)):
            symbol = entry[0] if entry else ""
            definition = entry[1] if len(entry) > 1 else ""
        else:
            symbol, definition = entry, ""
        symbol, definition = _as_text(symbol).strip(), _as_text(definition).strip()
        if symbol or definition:
            entries.append((symbol, definition))
    return entries


def _table_parts(tbl_info) -> tuple[str, list[str], list[list[str]]]:
    """Return ``(caption, headers, rows)`` with every row padded to the same width."""
    if not isinstance(tbl_info, dict):
        return "", [], []
    caption = _as_text(tbl_info.get("caption")).strip()
    headers = [_as_text(h) for h in _as_list(tbl_info.get("headers"))]
    rows = []
    for row in _as_list(tbl_info.get("rows")):
        if isinstance(row, dict):
            row = list(row.values())
        rows.append([_as_text(cell) for cell in _as_list(row)])
    n_cols = max([len(headers)] + [len(r) for r in rows])
    if n_cols == 0:
        return caption, [], []
    if headers:
        headers += [""] * (n_cols - len(headers))
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    return caption, headers, rows


def _figure_entries(figures, captions) -> list[tuple[str | None, str]]:
    """Pair figure paths with user captions, position by position.

    Returns one ``(path_or_None, caption_or_"")`` tuple per figure *or* caption,
    whichever list is longer, so neither surplus figures nor surplus captions
    are dropped.  A single path string is treated as a one-element list.
    """
    if isinstance(figures, (str, os.PathLike)):
        figures = [figures]
    paths = []
    for fig in _as_list(figures):
        if fig is None or fig == "":
            paths.append(None)
        elif isinstance(fig, (str, os.PathLike)):
            paths.append(os.fspath(fig))
        else:
            paths.append(str(fig))
    caps = [_as_text(c).strip() for c in (_as_list(captions) if captions else [])]
    count = max(len(paths), len(caps))
    return [
        (paths[i] if i < len(paths) else None, caps[i] if i < len(caps) else "")
        for i in range(count)
    ]


_FIG_STEM_PREFIX_RE = re.compile(
    r"^(?:fig(?:ure)?|tab(?:le)?)(?:[\s_-]*\d+[a-z]?)?(?:[\s_-]+|$)", re.IGNORECASE
)


def _describe_figure(path) -> str:
    """Human-readable description derived from a figure file name.

    ``outputs/figures/fig_1_model_comparison.png`` -> ``"Model comparison"``.
    """
    if not path:
        return ""
    stem = os.path.splitext(os.path.basename(os.fspath(path)))[0]
    stem = _FIG_STEM_PREFIX_RE.sub("", stem)
    text = re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", stem)).strip()
    return text[:1].upper() + text[1:]


_CAPTION_PREFIX_RE = re.compile(r"^(.*?\{n\}[^\w\s{]*\s*)")


def _default_caption(style, number: int, desc: str = "") -> str:
    """Build a caption from a guideline style such as ``"Fig. {n}. Description."``.

    Guideline styles describe the caption *format*; the trailing example text
    ("Description.", "Caption text in sentence case.") must not leak into the
    manuscript, so only the label prefix up to ``{n}`` and its punctuation is
    kept.  An explicit ``{desc}`` placeholder is honoured when present.
    """
    style = _as_text(style) or "Fig. {n}. {desc}"
    if "{n}" not in style:
        style = "Fig. {n}. {desc}"
    if "{desc}" in style:
        return style.replace("{n}", str(number)).replace("{desc}", desc).strip()
    match = _CAPTION_PREFIX_RE.match(style)
    prefix = match.group(1).replace("{n}", str(number))
    if not desc:
        return prefix.rstrip().rstrip("—–-:").rstrip()
    if style.rstrip().endswith(".") and not desc.endswith((".", "?", "!")):
        desc += "."
    return prefix + desc


def _safe_stem(title, fallback: str = "paper", max_len: int = 50) -> str:
    """Filesystem-safe file stem derived from *title* (never empty)."""
    safe = re.sub(r"[^\w\s-]", "", _as_text(title))[:max_len].strip()
    safe = re.sub(r"\s+", "_", safe).strip("_-")
    return safe or fallback


def _output_stem(paper_content: dict, journal_key: str) -> str:
    """``<safe title>_<canonical journal key>`` used for every output file."""
    return f"{_safe_stem(paper_content.get('title'))}_{resolve_journal_key(journal_key)}"


def _resolve_output_dir(output_dir, default: str | None = None) -> str:
    """Absolute output directory (created if needed).

    *default* is the calling module's ``DEFAULT_PAPERS_DIR`` (looked up at call
    time so that it can be monkeypatched per module).
    """
    if output_dir is None or output_dir == "":
        path = os.path.abspath(default or DEFAULT_PAPERS_DIR)
    else:
        path = os.path.abspath(os.path.expanduser(os.fspath(output_dir)))
    os.makedirs(path, exist_ok=True)
    return path


def _check_content(paper_content) -> dict:
    if not isinstance(paper_content, dict):
        raise TypeError(f"paper_content must be a dict, got {type(paper_content).__name__}")
    return paper_content


# ---------------------------------------------------------------------------
# Shared helpers: guideline interpretation
# ---------------------------------------------------------------------------


def _primary_font(font) -> str:
    """First font of a guideline value such as ``"Times New Roman or Arial"``."""
    text = _as_text(font).strip()
    first = re.split(r"\s+or\s+|[,/;]", text, maxsplit=1)[0].strip()
    return first or "Times New Roman"


def _journal_features(guideline: dict, journal_key: str) -> dict:
    """Which optional front-matter blocks this journal uses."""
    publisher = _as_text(guideline.get("publisher")).lower()
    ms = guideline.get("manuscript_structure") or {}
    sections = [_as_text(s).strip().lower() for s in _as_list(ms.get("sections"))]
    key = _as_text(journal_key).lower()
    is_elsevier = "elsevier" in publisher
    is_asce = "american society of civil engineers" in publisher or "asce" in key
    return {
        "is_elsevier": is_elsevier,
        "is_asce": is_asce,
        "highlights": is_elsevier
        or bool(ms.get("highlights_required"))
        or any(s.startswith("highlights") for s in sections),
        "practical_applications": is_asce
        or "practical_applications" in ms
        or any(s.startswith("practical applications") for s in sections),
    }


def _heading_rules(guideline: dict) -> dict:
    """Interpret ``manuscript_structure.heading_format``.

    Returns flags for automatic numbering ("1., 1.1., 1.1.1."), upper-casing of
    level-1 headings and bold/italic for levels 2 and 3.  Without a heading
    format the historical PaperFactory style (ASCE-like) is used.
    """
    hf = (guideline.get("manuscript_structure") or {}).get("heading_format") or {}
    if not isinstance(hf, dict):
        hf = {"style": _as_text(hf)}
    if not hf:
        return {
            "numbered": False,
            "upper": True,
            "l2_bold": True,
            "l2_italic": False,
            "l3_bold": False,
            "l3_italic": True,
        }
    joined = " ".join(_as_text(v) for v in hf.values()).lower()
    level1 = _as_text(hf.get("level_1") or hf.get("major_headings")).lower()
    level2 = _as_text(hf.get("level_2")).lower()
    level3 = _as_text(hf.get("level_3")).lower()
    l2_italic = "italic" in level2
    return {
        "numbered": "numbered" in joined.replace("unnumbered", ""),
        "upper": any(w in level1 for w in ("capital", "uppercase", "upper case")),
        "l2_bold": "bold" in level2 or not l2_italic,
        "l2_italic": l2_italic,
        "l3_bold": "bold" in level3,
        "l3_italic": "italic" in level3 or not level3,
    }


_HEADING_NUMBER_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.|\d+(?:\.\d+)+)\s+")
_UNNUMBERED_HEADINGS = (
    "abstract",
    "highlights",
    "graphical abstract",
    "keywords",
    "practical applications",
    "nomenclature",
    "notation",
    "abbreviations",
    "acknowledg",
    "data availability",
    "declaration",
    "credit",
    "author contributions",
    "funding",
    "conflict",
    "supplementa",
    "appendix",
    "appendices",
    "references",
    "bibliography",
)


def _has_heading_number(text: str) -> bool:
    return bool(_HEADING_NUMBER_RE.match(text or ""))


def _strip_heading_number(text: str) -> str:
    """Remove a manual ``"1."``/``"2.3"`` prefix from a heading."""
    return _HEADING_NUMBER_RE.sub("", text or "", count=1)


def _is_unnumbered_heading(text: str) -> bool:
    """Back-matter style headings that journals never number."""
    return _strip_heading_number(text).strip().lower().startswith(_UNNUMBERED_HEADINGS)


def _all_headings(sections: list[dict]):
    for sec in sections:
        heading, _content, subs = _section_parts(sec)
        yield heading
        yield from _all_headings(subs)


class _HeadingNumberer:
    """Produce ``"1."``, ``"1.1."``, ``"1.1.1."`` labels for numbered journals.

    Numbering is disabled when the author already numbered any heading, and
    back-matter headings (Acknowledgements, Appendix, ...) and their children
    stay unnumbered.
    """

    def __init__(self, enabled: bool, sections: list[dict] = ()):
        self.enabled = enabled and not any(_has_heading_number(h) for h in _all_headings(sections))
        self._counters = [0, 0, 0]
        self._active = [False, False, False]

    def label(self, level: int, text: str) -> str:
        idx = min(max(level, 1), 3) - 1
        if not self.enabled or not text:
            return text
        if _is_unnumbered_heading(text) or (idx > 0 and not self._active[idx - 1]):
            self._active[idx:] = [False] * (3 - idx)
            return text
        self._counters[idx] += 1
        self._active[idx] = True
        for j in range(idx + 1, 3):
            self._counters[j] = 0
            self._active[j] = False
        return ".".join(str(c) for c in self._counters[: idx + 1]) + ". " + text


_BLOCK_ALIASES = {
    "notation": ("notation", "nomenclature"),
    "data_availability": ("data availability",),
    "acknowledgments": ("acknowledg",),
    "competing_interests": (
        "declaration of competing",
        "declaration of interest",
        "competing interest",
        "conflict of interest",
        "conflicts of interest",
    ),
    "references": ("references",),
    "tables": ("tables",),
    "figures": ("figure captions", "figures"),
}
_DEFAULT_LABELS = {
    "notation": "Notation",
    "data_availability": "Data Availability Statement",
    "acknowledgments": "Acknowledgments",
    "competing_interests": "Declaration of Competing Interest",
    "references": "References",
    "tables": "Tables",
    "figures": "Figures",
}


#: paper_content keys accepted for the competing-interest declaration.
COMPETING_INTEREST_KEYS = (
    "conflict_of_interest",
    "conflicts_of_interest",
    "competing_interests",
    "declaration_of_competing_interest",
)


def _competing_interests(paper_content: dict) -> str:
    """Return the competing-interest declaration from whichever key the paper uses."""
    for key in COMPETING_INTEREST_KEYS:
        text = _as_text(paper_content.get(key), "\n\n").strip()
        if text:
            return text
    return ""


def _document_layout(guideline: dict) -> dict:
    """Where the back-matter blocks go, following ``manuscript_structure.sections``.

    Returns ``order`` (block names after the body), ``front_notation`` (True
    when the journal puts its Nomenclature before the Introduction),
    ``floats_at_end`` (tables/figures at the end of the manuscript rather than
    embedded after the body) and ``labels`` (heading text spelled as in the
    guideline, e.g. "Acknowledgements" vs "Acknowledgments").
    """
    ms = guideline.get("manuscript_structure") or {}
    raw_sections = [_as_text(s).strip() for s in _as_list(ms.get("sections"))]
    sections = [s.lower() for s in raw_sections]

    def position(block):
        for i, name in enumerate(sections):
            if name.startswith(_BLOCK_ALIASES[block]):
                return i
        return None

    placement = _as_text(
        (guideline.get("figures_tables") or {}).get("placement", "end of manuscript")
    ).lower()
    floats_at_end = "end" in placement or "separate" in placement or "embedded" not in placement
    if floats_at_end:
        order = ["notation", "data_availability", "acknowledgments", "competing_interests"]
        order += ["references", "tables", "figures"]
    else:
        order = ["tables", "figures", "notation", "data_availability", "acknowledgments"]
        order += ["competing_interests", "references"]

    intro = next((i for i, s in enumerate(sections) if s.startswith("introduction")), None)
    notation_pos = position("notation")
    front_notation = notation_pos is not None and intro is not None and notation_pos < intro
    if front_notation:
        order.remove("notation")

    # Re-order the blocks the guideline lists, keeping unlisted ones in their default slot.
    listed = [b for b in order if position(b) is not None]
    slots = [i for i, b in enumerate(order) if position(b) is not None]
    for slot, block in zip(slots, sorted(listed, key=position), strict=True):
        order[slot] = block

    labels = dict(_DEFAULT_LABELS)
    for block in ("notation", "data_availability", "acknowledgments", "competing_interests"):
        pos = position(block)
        if pos is not None:
            labels[block] = raw_sections[pos]
    for block in ("references",):
        pos = position(block)
        if pos is not None:
            labels[block] = raw_sections[pos]
    return {
        "order": order,
        "front_notation": front_notation,
        "floats_at_end": floats_at_end,
        "labels": labels,
    }


def _parse_margin(margin_str) -> float:
    """Return margin in inches parsed from a guideline margin string."""
    s = _as_text(margin_str).lower()
    if "1 inch" in s or "2.54" in s or "1in" in s:
        return 1.0
    match = re.search(r"([\d.]+)\s*(inch|in|cm|mm)", s)
    if match:
        try:
            val = float(match.group(1))
        except ValueError:
            return 1.0
        unit = match.group(2)
        if unit == "cm":
            return val / 2.54
        if unit == "mm":
            return val / 25.4
        return val
    return 1.0


def _line_spacing_value(spacing) -> float:
    """Line-spacing multiple for a guideline value ("single", "1.5", "double", 2)."""
    if isinstance(spacing, (int, float)) and not isinstance(spacing, bool):
        return float(spacing) if spacing > 0 else 2.0
    s = _as_text(spacing).strip().lower()
    if s.startswith("single"):
        return 1.0
    if s.startswith(("double", "two")):
        return 2.0
    if "half" in s:
        return 1.5
    match = re.match(r"(\d+(?:\.\d+)?)", s)
    if match and float(match.group(1)) > 0:
        return float(match.group(1))
    return 2.0


# ---------------------------------------------------------------------------
# Internal helpers (Word specific)
# ---------------------------------------------------------------------------


def _font_size(formatting: dict) -> float:
    size = formatting.get("font_size", 12)
    try:
        return float(size)
    except (TypeError, ValueError):
        match = re.match(r"\s*(\d+(?:\.\d+)?)", _as_text(size))
        return float(match.group(1)) if match else 12.0


def _apply_font(run, formatting: dict) -> None:
    run.font.name = _primary_font(formatting.get("font", "Times New Roman"))
    run.font.size = Pt(_font_size(formatting))
    if _contains_hangul(run.text):
        # Word draws Hangul with the East-Asian font slot, not w:ascii/w:hAnsi.
        run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), KOREAN_FONT)


def _set_spacing(paragraph, formatting: dict) -> None:
    paragraph.paragraph_format.line_spacing = _line_spacing_value(
        formatting.get("line_spacing", "double")
    )


def _add_heading_l1(doc, text: str, formatting: dict, upper: bool = True) -> None:
    """Level-1 heading: bold, left-aligned, ALL CAPS when *upper*."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = para.add_run(text.upper() if upper else text)
    run.bold = True
    _apply_font(run, formatting)
    _set_spacing(para, formatting)


def _add_heading_l2(doc, text: str, formatting: dict, bold: bool = True, italic: bool = False):
    """Level-2 subsection heading: Title Case, bold (default), left-aligned."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    _apply_font(run, formatting)
    _set_spacing(para, formatting)


def _add_heading_l3(doc, text: str, formatting: dict, bold: bool = False, italic: bool = True):
    """Level-3 subsection heading: Sentence case, italic (default), left-aligned."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    _apply_font(run, formatting)
    _set_spacing(para, formatting)


def _add_body_text(doc, text: str, formatting: dict) -> None:
    """Add one or more body paragraphs, splitting on blank lines."""
    justified = _as_text(formatting.get("text_alignment")).lower().startswith("justif")
    indent = formatting.get("paragraph_indent")
    indent_in = _parse_margin(indent) if indent and "first" in _as_text(indent).lower() else 0
    for chunk in re.split(r"\n\s*\n", _as_text(text, "\n\n")):
        chunk = chunk.strip()
        if not chunk:
            continue
        para = doc.add_paragraph()
        run = para.add_run(chunk)
        _apply_font(run, formatting)
        _set_spacing(para, formatting)
        if justified:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent_in:
            para.paragraph_format.first_line_indent = Inches(indent_in)


def _set_borders(table, values: dict) -> None:
    tbl_xml = table._tbl
    tbl_pr = tbl_xml.tblPr if tbl_xml.tblPr is not None else tbl_xml._add_tblPr()
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = borders.find(qn(f"w:{name}"))
        if b is None:
            b = OxmlElement(f"w:{name}")
            borders.append(b)
        if values[name]:
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), "4")
            b.set(qn("w:color"), "000000")
        else:
            b.set(qn("w:val"), "none")


def _table_no_vertical_borders(table) -> None:
    """Remove vertical rules from *table* (horizontal-only style)."""
    _set_borders(
        table,
        {"top": 1, "bottom": 1, "insideH": 1, "left": 0, "right": 0, "insideV": 0},
    )


def _table_full_borders(table) -> None:
    """Apply full grid borders to *table*."""
    _set_borders(table, dict.fromkeys(("top", "left", "bottom", "right", "insideH", "insideV"), 1))


def _write_cell(cell, text: str, formatting: dict, bold: bool = False, size_delta: float = -2):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    _apply_font(run, formatting)
    run.font.size = Pt(_font_size(formatting) + size_delta)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _insert_table(
    doc, tbl_info: dict, formatting: dict, no_vertical: bool, number: int = 0, caption_style=None
) -> None:
    """Render one table dict (caption above, data, optional border style)."""
    caption, headers, rows = _table_parts(tbl_info)
    if not caption and number:
        caption = _default_caption(caption_style or "Table {n}. {desc}", number)

    cap_para = doc.add_paragraph()
    cap_run = cap_para.add_run(caption)
    cap_run.bold = True
    _apply_font(cap_run, formatting)
    cap_run.font.size = Pt(_font_size(formatting) - 1)
    _set_spacing(cap_para, formatting)

    if not headers and not rows:
        return

    n_cols = len(headers) if headers else len(rows[0])
    table = doc.add_table(rows=(1 if headers else 0) + len(rows), cols=n_cols)
    table.style = "Table Grid"

    offset = 0
    if headers:
        for j, header in enumerate(headers):
            _write_cell(table.rows[0].cells[j], header, formatting, bold=True)
        offset = 1
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            _write_cell(table.rows[i + offset].cells[j], val, formatting)

    if no_vertical:
        _table_no_vertical_borders(table)
    else:
        _table_full_borders(table)

    doc.add_paragraph()  # spacing after table


def _enable_line_numbers(section) -> None:
    """Continuous line numbering (reviewers cite line numbers)."""
    sect_pr = section._sectPr
    if sect_pr.find(qn("w:lnNumType")) is not None:
        return
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    # Schema order: ... pgSz, pgMar, paperSrc, pgBorders, lnNumType, pgNumType, cols ...
    anchor = None
    for tag in ("w:pgBorders", "w:paperSrc", "w:pgMar", "w:pgSz"):
        anchor = sect_pr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addnext(ln)
    else:
        sect_pr.insert(0, ln)


def _add_page_number_footer(section, formatting: dict) -> None:
    """Centered ``PAGE`` field in the footer."""
    para = section.footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        run = para.add_run()
        _apply_font(run, formatting)
        if kind:
            fld = OxmlElement("w:fldChar")
            fld.set(qn("w:fldCharType"), kind)
            run._r.append(fld)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = text
            run._r.append(instr)


def _picture_width(path: str, max_w: float, max_h: float):
    """Width (as a docx Length) that fits the image in ``max_w`` x ``max_h`` inches."""
    img = DocxImage.from_file(path)
    if not img.px_width or not img.px_height:
        return Inches(max_w)
    aspect = img.px_height / img.px_width
    width = max_w if max_w * aspect <= max_h else max_h / aspect
    return Inches(width)


def _add_caption(doc, text: str, formatting: dict) -> None:
    para = doc.add_paragraph()
    run = para.add_run(text)
    _apply_font(run, formatting)
    _set_spacing(para, formatting)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def generate_word(
    paper_content: dict,
    journal_key: str,
    figures: list = None,
    output_dir: str = None,
) -> str:
    """Generate a Word .docx file from *paper_content* following the journal guidelines.

    Parameters
    ----------
    paper_content:
        Dict with keys: title, authors, abstract, keywords, sections, tables,
        references, figure_captions, practical_applications, highlights,
        data_availability, acknowledgments, notation.  Every key is optional;
        ``keywords``/``authors`` may be lists, ``references``/``highlights`` a
        newline-separated string.
    journal_key:
        Key matching a file in guidelines/ (e.g. "asce_jse", "eng_structures");
        aliases accepted by :func:`utils.guidelines.resolve_journal_key` work too.
    figures:
        Optional list of paths to figure image files.  Missing or unreadable
        files are replaced by a visible placeholder; their captions are kept.
    output_dir:
        Directory in which to save the .docx file.  Defaults to
        outputs/papers/ inside the project root.

    Returns
    -------
    str
        Absolute path to the saved .docx file.

    Raises
    ------
    utils.guidelines.UnknownJournalError
        If *journal_key* does not match any guideline.
    """
    paper_content = _check_content(paper_content)
    guideline = load_guideline(journal_key)
    output_dir = _resolve_output_dir(output_dir, DEFAULT_PAPERS_DIR)

    formatting = dict(guideline.get("formatting") or {})
    formatting["font"] = _primary_font(formatting.get("font"))
    fig_config = guideline.get("figures_tables") or {}
    features = _journal_features(guideline, journal_key)
    rules = _heading_rules(guideline)
    layout = _document_layout(guideline)
    labels = layout["labels"]
    upper = rules["upper"]

    # Determine table border style from guideline
    no_vertical_borders = "no vertical" in _as_text(fig_config.get("table_rules")).lower()

    # ------------------------------------------------------------------
    # Document setup
    # ------------------------------------------------------------------
    doc = Document()

    section = doc.sections[0]
    page_size = _as_text(formatting.get("page_size", "US Letter"))
    if "A4" in page_size.upper():
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
    else:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)

    margin_str = _as_text(formatting.get("margins", "1 inch all sides"))
    # "standard" is used by some guidelines — default to 1 inch
    margin_val = 1.0 if margin_str.lower() in ("standard", "") else _parse_margin(margin_str)
    section.top_margin = Inches(margin_val)
    section.bottom_margin = Inches(margin_val)
    section.left_margin = Inches(margin_val)
    section.right_margin = Inches(margin_val)
    usable_w = section.page_width.inches - 2 * margin_val
    usable_h = section.page_height.inches - 2 * margin_val

    if formatting.get("line_numbers"):
        _enable_line_numbers(section)
    if formatting.get("page_numbers"):
        _add_page_number_footer(section, formatting)

    # Global Normal style
    normal_style = doc.styles["Normal"]
    normal_style.font.name = formatting["font"]
    normal_style.font.size = Pt(_font_size(formatting))

    sections = _as_sections(paper_content.get("sections"))
    numberer = _HeadingNumberer(rules["numbered"], sections)

    # ------------------------------------------------------------------
    # Highlights (Elsevier-specific) — before title
    # ------------------------------------------------------------------
    highlights = _highlight_items(paper_content.get("highlights"))
    if features["highlights"] and highlights:
        _add_heading_l1(doc, "Highlights", formatting, upper)
        for bullet in highlights:
            para = doc.add_paragraph(style="List Bullet")
            run = para.add_run(bullet)
            _apply_font(run, formatting)
            _set_spacing(para, formatting)
        doc.add_paragraph()

    # ------------------------------------------------------------------
    # Title and authors
    # ------------------------------------------------------------------
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(_as_text(paper_content.get("title")).strip() or "Untitled")
    title_run.bold = True
    _apply_font(title_run, formatting)
    title_run.font.size = Pt(_font_size(formatting) + 2)
    _set_spacing(title_para, formatting)

    authors = _as_text(paper_content.get("authors"), ", ").strip()
    if authors:
        authors_para = doc.add_paragraph()
        authors_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        authors_run = authors_para.add_run(authors)
        _apply_font(authors_run, formatting)
        _set_spacing(authors_para, formatting)

    # ------------------------------------------------------------------
    # Abstract, keywords, practical applications (ASCE)
    # ------------------------------------------------------------------
    abstract = _as_text(paper_content.get("abstract"), "\n\n")
    if abstract.strip():
        doc.add_paragraph()
        _add_heading_l1(doc, "Abstract", formatting, upper)
        _add_body_text(doc, abstract, formatting)

    keywords = _as_text(paper_content.get("keywords"), "; ").strip()
    if keywords:
        kw_para = doc.add_paragraph()
        kw_label = kw_para.add_run("Keywords: ")
        kw_label.bold = True
        _apply_font(kw_label, formatting)
        kw_text = kw_para.add_run(keywords)
        _apply_font(kw_text, formatting)
        _set_spacing(kw_para, formatting)

    practical = _as_text(paper_content.get("practical_applications"), "\n\n")
    if features["practical_applications"] and practical.strip():
        doc.add_paragraph()
        _add_heading_l1(doc, "Practical Applications", formatting, upper)
        _add_body_text(doc, practical, formatting)

    notation = _notation_entries(paper_content.get("notation"))

    def add_notation():
        doc.add_paragraph()
        _add_heading_l1(doc, labels["notation"], formatting, upper)
        tbl = doc.add_table(rows=len(notation), cols=2)
        tbl.style = "Table Grid"
        for i, (symbol, definition) in enumerate(notation):
            for cell, text in zip(tbl.rows[i].cells, (symbol, definition), strict=True):
                cell.text = ""
                _apply_font(cell.paragraphs[0].add_run(text), formatting)
        if no_vertical_borders:
            _table_no_vertical_borders(tbl)
        doc.add_paragraph()

    if layout["front_notation"] and notation:
        add_notation()

    # ------------------------------------------------------------------
    # Body sections (with subsection support)
    # ------------------------------------------------------------------
    for sec in sections:
        heading_text, body_text, subsections = _section_parts(sec)

        doc.add_paragraph()
        if heading_text:
            _add_heading_l1(doc, numberer.label(1, heading_text), formatting, upper)
        if body_text.strip():
            _add_body_text(doc, body_text, formatting)

        for subsec in subsections:
            sub_heading, sub_content, sub_subsections = _section_parts(subsec)
            if sub_heading:
                _add_heading_l2(
                    doc,
                    numberer.label(2, sub_heading),
                    formatting,
                    bold=rules["l2_bold"],
                    italic=rules["l2_italic"],
                )
            if sub_content.strip():
                _add_body_text(doc, sub_content, formatting)

            for subsubsec in sub_subsections:
                h3, c3, _ = _section_parts(subsubsec)
                if h3:
                    _add_heading_l3(
                        doc,
                        numberer.label(3, h3),
                        formatting,
                        bold=rules["l3_bold"],
                        italic=rules["l3_italic"],
                    )
                if c3.strip():
                    _add_body_text(doc, c3, formatting)

    # ------------------------------------------------------------------
    # Back matter, in the order the journal guideline lists it
    # ------------------------------------------------------------------
    tables_data = [t for t in _as_list(paper_content.get("tables")) if isinstance(t, dict)]
    fig_entries = _figure_entries(figures, paper_content.get("figure_captions"))
    references = _as_items(paper_content.get("references"))
    at_end = layout["floats_at_end"]

    def add_tables():
        if at_end:
            doc.add_page_break()
            _add_heading_l1(doc, labels["tables"], formatting, upper)
        for n, tbl_info in enumerate(tables_data, 1):
            _insert_table(
                doc,
                tbl_info,
                formatting,
                no_vertical_borders,
                number=n,
                caption_style=fig_config.get("table_caption_style"),
            )

    def add_figures():
        if at_end:
            doc.add_page_break()
            _add_heading_l1(doc, labels["figures"], formatting, upper)
        cap_style = fig_config.get("figure_caption_style", "Fig. {n}. {desc}")
        max_w = max(1.0, min(5.5, usable_w))
        max_h = max(1.0, usable_h - 1.5)
        for n, (fig_path, custom_caption) in enumerate(fig_entries, 1):
            if fig_path:
                fig_para = doc.add_paragraph()
                fig_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                name = os.path.basename(fig_path)
                if os.path.isfile(fig_path):
                    try:
                        width = _picture_width(fig_path, max_w, max_h)
                        fig_para.add_run().add_picture(fig_path, width=width)
                    except Exception:
                        fig_para.add_run(f"[Figure {n}: {name} could not be embedded]")
                else:
                    fig_para.add_run(f"[Figure {n}: {name} not found]")
                _set_spacing(fig_para, formatting)
            caption = custom_caption or _default_caption(cap_style, n, _describe_figure(fig_path))
            _add_caption(doc, caption, formatting)

    def add_data_availability():
        doc.add_paragraph()
        _add_heading_l1(doc, labels["data_availability"], formatting, upper)
        _add_body_text(doc, data_avail, formatting)

    def add_acknowledgments():
        doc.add_paragraph()
        _add_heading_l1(doc, labels["acknowledgments"], formatting, upper)
        _add_body_text(doc, acknowledgments, formatting)

    def add_competing_interests():
        doc.add_paragraph()
        _add_heading_l1(doc, labels["competing_interests"], formatting, upper)
        _add_body_text(doc, competing, formatting)

    def add_references():
        doc.add_page_break()
        _add_heading_l1(doc, labels["references"], formatting, upper)
        for ref in references:
            ref_para = doc.add_paragraph()
            ref_run = ref_para.add_run(ref)
            _apply_font(ref_run, formatting)
            _set_spacing(ref_para, formatting)

    data_avail = _as_text(paper_content.get("data_availability"), "\n\n")
    acknowledgments = _as_text(paper_content.get("acknowledgments"), "\n\n")
    competing = _competing_interests(paper_content)
    blocks = {
        "notation": (add_notation, bool(notation) and not layout["front_notation"]),
        "data_availability": (add_data_availability, bool(data_avail.strip())),
        "acknowledgments": (add_acknowledgments, bool(acknowledgments.strip())),
        "competing_interests": (add_competing_interests, bool(competing)),
        "references": (add_references, bool(references)),
        "tables": (add_tables, bool(tables_data)),
        "figures": (add_figures, bool(fig_entries)),
    }
    for block in layout["order"]:
        render, present = blocks[block]
        if present:
            render()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = os.path.join(output_dir, f"{_output_stem(paper_content, journal_key)}.docx")
    doc.save(output_path)
    return output_path
