"""Generate journal-formatted PDF documents for PaperFactory.

Uses reportlab for direct PDF generation without LaTeX compilation.

Page size, margins, section order, caption style and the journal-specific
blocks (highlights, practical applications, nomenclature) follow the journal
guideline.  Text that the standard PDF fonts cannot encode (Greek letters,
math symbols, Korean) switches the document to a Unicode-capable font instead
of being silently replaced by wrong glyphs.
"""

from __future__ import annotations

import importlib.util
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

from utils.guidelines import load_guideline
from utils.word_generator import (
    _all_text,
    _as_items,
    _as_list,
    _as_sections,
    _as_text,
    _check_content,
    _competing_interests,
    _contains_hangul,
    _default_caption,
    _describe_figure,
    _document_layout,
    _figure_entries,
    _heading_rules,
    _HeadingNumberer,
    _highlight_items,
    _journal_features,
    _notation_entries,
    _output_stem,
    _parse_margin,
    _resolve_output_dir,
    _section_parts,
    _table_parts,
)

DEFAULT_PAPERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "papers"
)

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

_BASE_FONTS = {
    "regular": "Times-Roman",
    "bold": "Times-Bold",
    "italic": "Times-Italic",
    "bold_italic": "Times-BoldItalic",
}
# Adobe Asian fonts built into reportlab (no font file needed) for Hangul text.
_KOREAN_FONTS = {
    "regular": "HYSMyeongJo-Medium",
    "bold": "HYGothic-Medium",
    "italic": "HYSMyeongJo-Medium",
    "bold_italic": "HYGothic-Medium",
}
# DejaVu Serif ships with matplotlib (a hard dependency) and covers Greek and math.
_DEJAVU_FILES = {
    "regular": "DejaVuSerif.ttf",
    "bold": "DejaVuSerif-Bold.ttf",
    "italic": "DejaVuSerif-Italic.ttf",
    "bold_italic": "DejaVuSerif-BoldItalic.ttf",
}
_UNICODE_FONTS = {k: f"PF-{os.path.splitext(v)[0]}" for k, v in _DEJAVU_FILES.items()}
_registered: dict = {}


def _register_family(fonts: dict) -> None:
    pdfmetrics.registerFontFamily(
        fonts["regular"],
        normal=fonts["regular"],
        bold=fonts["bold"],
        italic=fonts["italic"],
        boldItalic=fonts["bold_italic"],
    )


def _dejavu_dirs() -> list[str]:
    dirs = []
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            dirs.append(os.path.join(loc, "mpl-data", "fonts", "ttf"))
    dirs.append("/usr/share/fonts/truetype/dejavu")
    return dirs


def _unicode_fonts():
    """Register DejaVu Serif once; return its font names or None if unavailable."""
    if "unicode" not in _registered:
        _registered["unicode"] = None
        for directory in _dejavu_dirs():
            paths = {k: os.path.join(directory, f) for k, f in _DEJAVU_FILES.items()}
            if all(os.path.isfile(p) for p in paths.values()):
                try:
                    for k, path in paths.items():
                        pdfmetrics.registerFont(TTFont(_UNICODE_FONTS[k], path))
                    _register_family(_UNICODE_FONTS)
                    _registered["unicode"] = _UNICODE_FONTS
                except Exception:  # pragma: no cover - corrupt font file
                    continue
                break
    return _registered["unicode"]


def _korean_fonts() -> dict:
    if "korean" not in _registered:
        for name in set(_KOREAN_FONTS.values()):
            pdfmetrics.registerFont(UnicodeCIDFont(name))
        _register_family(_KOREAN_FONTS)
        _registered["korean"] = _KOREAN_FONTS
    return _registered["korean"]


def _needs_unicode_font(text: str) -> bool:
    """True if the standard (WinAnsi-encoded) PDF fonts cannot show *text*."""
    try:
        text.encode("cp1252")
    except UnicodeEncodeError:
        return True
    return False


def _select_fonts(text: str) -> dict:
    """Font names able to render every character of *text*."""
    if _contains_hangul(text):
        return _korean_fonts()
    if _needs_unicode_font(text):
        return _unicode_fonts() or _BASE_FONTS
    return _BASE_FONTS


# ---------------------------------------------------------------------------
# Styles and small builders
# ---------------------------------------------------------------------------


def _build_styles(fonts: dict = None):
    """Create publication-quality paragraph styles."""
    fonts = fonts or _BASE_FONTS
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "PaperTitle",
            parent=styles["Title"],
            fontSize=16,
            leading=20,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName=fonts["bold"],
        )
    )
    styles.add(
        ParagraphStyle(
            "Authors",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            fontName=fonts["regular"],
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "AbstractHeading",
            parent=styles["Heading2"],
            fontSize=11,
            fontName=fonts["bold"],
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "AbstractBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            fontName=fonts["italic"],
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading1"],
            fontSize=12,
            fontName=fonts["bold"],
            spaceBefore=16,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubsectionHeading",
            parent=styles["Heading2"],
            fontSize=11,
            fontName=fonts["bold_italic"],
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubsubsectionHeading",
            parent=styles["Heading3"],
            fontSize=10,
            fontName=fonts["italic"],
            spaceBefore=8,
            spaceAfter=3,
        )
    )
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 13
    styles["BodyText"].fontName = fonts["regular"]
    styles["BodyText"].alignment = TA_JUSTIFY
    styles["BodyText"].spaceAfter = 6
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            fontName=fonts["italic"],
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "Reference",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            fontName=fonts["regular"],
            leftIndent=20,
            firstLineIndent=-20,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            "Keywords",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            fontName=fonts["regular"],
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "HighlightBullet",
            parent=styles["BodyText"],
            leftIndent=14,
            bulletIndent=4,
            bulletFontName="Times-Roman",
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            fontName=fonts["regular"],
            alignment=TA_CENTER,
        )
    )
    styles.add(ParagraphStyle("TableHeader", parent=styles["TableCell"], fontName=fonts["bold"]))
    return styles


_XML_INVALID_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _safe_xml(text) -> str:
    """Escape XML special characters for reportlab Paragraph (``None``-safe)."""
    text = _XML_INVALID_RE.sub("", _as_text(text))
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _paragraphs(text, style) -> list:
    """One Paragraph per blank-line separated chunk of *text*."""
    return [
        Paragraph(_safe_xml(chunk.strip()), style)
        for chunk in re.split(r"\n\s*\n", _as_text(text, "\n\n"))
        if chunk.strip()
    ]


def _column_widths(data, style, available: float) -> list[float]:
    """Natural column widths, shrunk proportionally to fit *available* points."""
    n_cols = len(data[0])
    padding = 14  # Table's default 6 pt left + right cell padding, plus slack
    natural = [
        max(pdfmetrics.stringWidth(row[j], style.fontName, style.fontSize) for row in data)
        + padding
        for j in range(n_cols)
    ]
    total = sum(natural)
    if total <= available:
        return natural
    minimum = available / (2 * n_cols)
    widths = [max(minimum, available * w / total) for w in natural]
    scale = available / sum(widths)
    return [w * scale for w in widths]


def _build_table(headers, rows, styles, available: float):
    """Booktabs-style table (no shading, no vertical rules) with wrapping cells."""
    data = ([headers] if headers else []) + rows
    if not data or not data[0]:
        return None
    widths = _column_widths(data, styles["TableCell"], available)
    cells = [
        [
            Paragraph(_safe_xml(c), styles["TableHeader" if headers and i == 0 else "TableCell"])
            for c in row
        ]
        for i, row in enumerate(data)
    ]
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
    ]
    if headers:
        commands.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black))
    table = Table(cells, colWidths=widths, repeatRows=1 if headers else 0)
    table.setStyle(TableStyle(commands))
    return table


def _figure_flowable(path: str, max_w: float, max_h: float):
    """Image scaled to fit ``max_w`` x ``max_h`` points with its aspect ratio kept.

    Raises if the file is not a readable raster image, so the caller can show a
    placeholder instead of failing at ``doc.build`` time.
    """
    width, height = ImageReader(path).getSize()
    if not width or not height:
        raise ValueError(f"Image has no size: {path}")
    scale = min(max_w / width, max_h / height)
    img = Image(path, width=width * scale, height=height * scale)
    img.hAlign = "CENTER"
    return img


def _page_number_drawer(font: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 9)
        canvas.drawCentredString(doc.pagesize[0] / 2, doc.bottomMargin / 2, str(doc.page))
        canvas.restoreState()

    return draw


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def generate_pdf(
    paper_content: dict,
    journal_key: str,
    figures: list = None,
    output_dir: str = None,
) -> str:
    """Generate a PDF document from paper content.

    Parameters
    ----------
    paper_content : dict
        Dict with keys: title, authors, abstract, keywords, sections,
        references, tables, figure_captions, highlights, etc.  Every key is
        optional; ``keywords``/``authors`` may be lists.
    journal_key : str
        Journal key (e.g. "jweia", "eng_structures") or an alias accepted by
        :func:`utils.guidelines.resolve_journal_key`.
    figures : list, optional
        List of figure image file paths.  Missing or unreadable files are
        replaced by a placeholder; their captions are kept.
    output_dir : str, optional
        Output directory. Defaults to outputs/papers/.

    Returns
    -------
    str
        Absolute path to the generated PDF file.

    Raises
    ------
    utils.guidelines.UnknownJournalError
        If *journal_key* does not match any guideline.
    """
    paper_content = _check_content(paper_content)
    guideline = load_guideline(journal_key)
    output_dir = _resolve_output_dir(output_dir, DEFAULT_PAPERS_DIR)

    formatting = guideline.get("formatting") or {}
    fig_config = guideline.get("figures_tables") or {}
    features = _journal_features(guideline, journal_key)
    rules = _heading_rules(guideline)
    layout = _document_layout(guideline)
    labels = layout["labels"]
    fonts = _select_fonts(_all_text(paper_content))
    styles = _build_styles(fonts)

    def heading(text):
        return text.upper() if rules["upper"] else text

    title = _as_text(paper_content.get("title")).strip() or "Untitled"
    authors = _as_text(paper_content.get("authors"), ", ").strip()
    pdf_path = os.path.join(output_dir, f"{_output_stem(paper_content, journal_key)}.pdf")

    pagesize = letter if "letter" in _as_text(formatting.get("page_size")).lower() else A4
    margins = _as_text(formatting.get("margins", "standard"))
    margin = 25 * mm if margins.lower().startswith("standard") else _parse_margin(margins) * inch
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=pagesize,
        topMargin=margin,
        bottomMargin=margin,
        leftMargin=margin,
        rightMargin=margin,
        title=title,
        author=authors,
    )
    frame_w = pagesize[0] - 2 * margin
    frame_h = pagesize[1] - 2 * margin

    story = []

    # Title
    story.append(Paragraph(_safe_xml(title), styles["PaperTitle"]))

    # Authors
    if authors:
        story.append(Paragraph(_safe_xml(authors), styles["Authors"]))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 6))

    # Abstract
    abstract = paper_content.get("abstract")
    abstract_paras = _paragraphs(abstract, styles["AbstractBody"])
    if abstract_paras:
        story.append(Paragraph(_safe_xml(heading("Abstract")), styles["AbstractHeading"]))
        story.extend(abstract_paras)

    # Keywords
    keywords = _as_text(paper_content.get("keywords"), "; ").strip()
    if keywords:
        story.append(Paragraph(f"<b>Keywords:</b> {_safe_xml(keywords)}", styles["Keywords"]))

    # Highlights (only for journals that use them)
    highlights = _highlight_items(paper_content.get("highlights"))
    if features["highlights"] and highlights:
        story.append(Paragraph(_safe_xml(heading("Highlights")), styles["SectionHeading"]))
        for hl in highlights:
            story.append(Paragraph(_safe_xml(hl), styles["HighlightBullet"], bulletText="•"))
        story.append(Spacer(1, 8))

    # Practical Applications (ASCE)
    practical = _paragraphs(paper_content.get("practical_applications"), styles["BodyText"])
    if features["practical_applications"] and practical:
        story.append(
            Paragraph(_safe_xml(heading("Practical Applications")), styles["SectionHeading"])
        )
        story.extend(practical)

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))

    notation = _notation_entries(paper_content.get("notation"))

    def add_notation():
        story.append(Paragraph(_safe_xml(heading(labels["notation"])), styles["SectionHeading"]))
        table = _build_table([], [list(e) for e in notation], styles, frame_w)
        story.append(table)

    if layout["front_notation"] and notation:
        add_notation()

    # Sections
    sections = _as_sections(paper_content.get("sections"))
    numberer = _HeadingNumberer(rules["numbered"], sections)
    for sec in sections:
        sec_heading, content, subsections = _section_parts(sec)
        if sec_heading:
            text = heading(numberer.label(1, sec_heading))
            story.append(Paragraph(_safe_xml(text), styles["SectionHeading"]))
        story.extend(_paragraphs(content, styles["BodyText"]))

        for subsec in subsections:
            sub_heading, sub_content, sub_subsections = _section_parts(subsec)
            if sub_heading:
                text = numberer.label(2, sub_heading)
                story.append(Paragraph(_safe_xml(text), styles["SubsectionHeading"]))
            story.extend(_paragraphs(sub_content, styles["BodyText"]))
            for subsub in sub_subsections:
                h3, c3, _ = _section_parts(subsub)
                if h3:
                    text = numberer.label(3, h3)
                    story.append(Paragraph(_safe_xml(text), styles["SubsubsectionHeading"]))
                story.extend(_paragraphs(c3, styles["BodyText"]))

    # Back matter and floats, in the order the journal guideline lists them
    tables = [t for t in _as_list(paper_content.get("tables")) if isinstance(t, dict)]
    fig_entries = _figure_entries(figures, paper_content.get("figure_captions"))
    references = _as_items(paper_content.get("references"))
    data_avail = _paragraphs(paper_content.get("data_availability"), styles["BodyText"])
    acknowledgments = _paragraphs(paper_content.get("acknowledgments"), styles["BodyText"])
    competing = _paragraphs(_competing_interests(paper_content), styles["BodyText"])

    def add_tables():
        cap_style = fig_config.get("table_caption_style", "Table {n}. {desc}")
        for n, tbl_info in enumerate(tables, 1):
            caption, headers, rows = _table_parts(tbl_info)
            block = [
                Spacer(1, 10),
                Paragraph(_safe_xml(caption or _default_caption(cap_style, n)), styles["Caption"]),
            ]
            table = _build_table(headers, rows, styles, frame_w)
            if table is not None:
                block.append(table)
            story.append(KeepTogether(block))
            story.append(Spacer(1, 10))

    def add_figures():
        cap_style = fig_config.get("figure_caption_style", "Fig. {n}. {desc}")
        max_w = min(6.0 * inch, frame_w)
        max_h = frame_h * 0.65
        for n, (fig_path, custom_caption) in enumerate(fig_entries, 1):
            block = [Spacer(1, 10)]
            if fig_path:
                name = _safe_xml(os.path.basename(fig_path))
                if os.path.isfile(fig_path):
                    try:
                        block.append(_figure_flowable(fig_path, max_w, max_h))
                    except Exception:
                        text = f"[Figure {n}: {name} could not be loaded]"
                        block.append(Paragraph(text, styles["Caption"]))
                else:
                    block.append(Paragraph(f"[Figure {n}: {name} not found]", styles["Caption"]))
            caption = custom_caption or _default_caption(cap_style, n, _describe_figure(fig_path))
            block.append(Paragraph(_safe_xml(caption), styles["Caption"]))
            story.append(KeepTogether(block))

    def add_section(label, flowables):
        def render():
            story.append(Spacer(1, 10))
            story.append(Paragraph(_safe_xml(heading(label)), styles["SectionHeading"]))
            story.extend(flowables)

        return render

    def add_references():
        story.append(Paragraph(_safe_xml(heading(labels["references"])), styles["SectionHeading"]))
        for ref in references:
            story.append(Paragraph(_safe_xml(ref), styles["Reference"]))

    blocks = {
        "notation": (add_notation, bool(notation) and not layout["front_notation"]),
        "data_availability": (
            add_section(labels["data_availability"], data_avail),
            bool(data_avail),
        ),
        "acknowledgments": (
            add_section(labels["acknowledgments"], acknowledgments),
            bool(acknowledgments),
        ),
        "competing_interests": (
            add_section(labels["competing_interests"], competing),
            bool(competing),
        ),
        "references": (add_references, bool(references)),
        "tables": (add_tables, bool(tables)),
        "figures": (add_figures, bool(fig_entries)),
    }
    for block in layout["order"]:
        render, present = blocks[block]
        if present:
            render()

    draw_page_number = _page_number_drawer(fonts["regular"])
    if formatting.get("page_numbers", True):
        doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    else:
        doc.build(story)
    return pdf_path
