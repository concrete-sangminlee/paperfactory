"""Generate journal-formatted LaTeX documents for PaperFactory.

The ``.tex`` file follows the journal guideline (document class, paper size,
spacing, line numbers, heading numbering, back-matter order) and cites every
entry of the companion ``.bib`` file via ``\\nocite{*}``.  Free-text references
are converted to structured BibTeX entries when they can be parsed reliably
and are otherwise kept verbatim (``@misc`` with a ``note``), so no reference
text is ever lost or garbled.
"""

from __future__ import annotations

import os
import re
import unicodedata

from utils.guidelines import UnknownJournalError, load_guideline, resolve_journal_key
from utils.word_generator import (
    _all_text,
    _as_items,
    _as_list,
    _as_sections,
    _as_text,
    _check_content,
    _competing_interests,
    _contains_hangul,
    _describe_figure,
    _document_layout,
    _figure_entries,
    _heading_rules,
    _highlight_items,
    _is_unnumbered_heading,
    _journal_features,
    _line_spacing_value,
    _notation_entries,
    _parse_margin,
    _resolve_output_dir,
    _safe_stem,
    _section_parts,
    _strip_heading_number,
    _table_parts,
)


def _latex_stem(paper_content: dict, journal_key: str) -> str:
    """ASCII-only ``<title>_<journal>`` stem for the .tex/.bib pair.

    BibTeX cannot read a ``\\bibdata`` name containing non-ASCII characters (kotex
    turns Hangul into ``\\unihangulchar`` macros), so accents are folded to ASCII,
    other scripts are dropped, and an empty result falls back to ``paper``.
    """
    title = unicodedata.normalize("NFKD", _safe_stem(paper_content.get("title"), fallback=""))
    title = title.encode("ascii", "ignore").decode("ascii")
    title = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "", title)).strip("_-")
    return f"{title or 'paper'}_{resolve_journal_key(journal_key)}"


DEFAULT_PAPERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "papers"
)

#: Document class per journal.  Journals not listed here (e.g. custom guidelines
#: from ``PAPERFACTORY_GUIDELINES_DIR``) are mapped from their publisher.
_DOCUMENT_CLASSES = {
    "asce_jse": "ascelike",
    "aci_sj": "article",
    "jweia": "elsarticle",
    "jbe": "elsarticle",
    "eng_structures": "elsarticle",
    "eesd": "article",
    "thin_walled": "elsarticle",
    "cem_con_comp": "elsarticle",
    "comput_struct": "elsarticle",
    "autom_constr": "elsarticle",
    "struct_safety": "elsarticle",
    "const_build_mat": "elsarticle",
    "steel_comp_struct": "elsarticle",
    "ksce_jce": "article",
    "buildings_mdpi": "article",
}

_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # OT1/T1 text fonts render these ASCII characters as other glyphs (e.g. "<" as "¡").
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}

# Characters that pdfLaTeX's utf8 input encoding does not know (Greek letters and
# math symbols are common in engineering text: "σ ≤ 0.5 μm").  Latin-1 characters
# such as ° ± × ² µ are supported natively and need no mapping.
# fmt: off
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "varepsilon",
    "ϵ": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "ϑ": "vartheta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi",
    "π": "pi", "ρ": "rho", "ς": "varsigma", "σ": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "varphi", "ϕ": "phi", "χ": "chi", "ψ": "psi",
    "ω": "omega", "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda",
    "Ξ": "Xi", "Π": "Pi", "Σ": "Sigma", "Υ": "Upsilon", "Φ": "Phi", "Ψ": "Psi",
    "Ω": "Omega",
    "\u2206": "Delta",  # INCREMENT
    "\u2126": "Omega",  # OHM SIGN
}
_MATH_SYMBOLS = {
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "≅": r"\cong",
    "≃": r"\simeq", "≡": r"\equiv", "∼": r"\sim", "∝": r"\propto", "∞": r"\infty",
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "√": r"\surd", "∂": r"\partial",
    "∇": r"\nabla", "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "∩": r"\cap",
    "∪": r"\cup", "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow", "↑": r"\uparrow", "↓": r"\downarrow",
    "∀": r"\forall", "∃": r"\exists", "⋅": r"\cdot", "∘": r"\circ", "−": "-",
    "∓": r"\mp", "≪": r"\ll", "≫": r"\gg", "∠": r"\angle", "⊥": r"\perp",
    "∥": r"\parallel", "′": r"^{\prime}", "″": r"^{\prime\prime}",
    "℃": r"^{\circ}\mathrm{C}",
}
# fmt: on
_UNICODE_LATEX = {ch: f"\\ensuremath{{\\{name}}}" for ch, name in _GREEK.items()}
_UNICODE_LATEX.update({ch: f"\\ensuremath{{{cmd}}}" for ch, cmd in _MATH_SYMBOLS.items()})
_UNICODE_LATEX.update({"‰": r"\textperthousand{}", "✓": r"\ensuremath{\checkmark}"})
for _digit, _sup in zip("0456789+-", "⁰⁴⁵⁶⁷⁸⁹⁺⁻", strict=True):
    _UNICODE_LATEX[_sup] = f"\\textsuperscript{{{_digit}}}"
for _digit, _sub in zip("0123456789", "₀₁₂₃₄₅₆₇₈₉", strict=True):
    _UNICODE_LATEX[_sub] = f"\\textsubscript{{{_digit}}}"
_UNICODE_RE = re.compile("|".join(re.escape(ch) for ch in _UNICODE_LATEX))

_MATH_PATTERN = re.compile(r"(\$[^$\s][^$]*[^$\s]\$|\$[^$\s]\$)")
_DISPLAY_MATH_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def _escape_text_segment(part: str) -> str:
    escaped = part.replace("\\", "\x00BACKSLASH\x00")
    escaped = escaped.replace("{", r"\{").replace("}", r"\}")
    escaped = escaped.replace("$", r"\$")
    for char, replacement in _LATEX_SPECIAL.items():
        escaped = escaped.replace(char, replacement)
    escaped = escaped.replace("\x00BACKSLASH\x00", r"\textbackslash{}")
    return _UNICODE_RE.sub(lambda m: _UNICODE_LATEX[m.group()], escaped)


def _escape_inline(text: str) -> str:
    result = []
    for part in _MATH_PATTERN.split(text):
        if part.startswith("$") and part.endswith("$") and len(part) >= 3:
            result.append(part)
        else:
            result.append(_escape_text_segment(part))
    return "".join(result)


def _escape_latex(text) -> str:
    """Escape LaTeX special characters while preserving math.

    Inline math is detected as $<content>$ where content does not start or end
    with whitespace (to avoid matching standalone dollar signs like "$5").
    Display math ``$$...$$`` is emitted as ``\\[...\\]``.  Greek letters and
    common math symbols outside math are mapped to ``\\ensuremath`` commands so
    that pdfLaTeX can typeset them.  ``None`` and non-strings are accepted.
    """
    parts = _DISPLAY_MATH_PATTERN.split(_as_text(text))
    result = []
    for i, part in enumerate(parts):
        if i % 2:
            result.append("\\[" + part.strip() + "\\]")
        else:
            result.append(_escape_inline(part))
    return "".join(result)


def _escape_line(text) -> str:
    """Escape text for a single-line context (headings, captions, table cells).

    Blank lines would end the paragraph inside a macro argument or a tabular
    cell, which is a LaTeX error, so all whitespace runs collapse to one space.
    """
    return _escape_latex(" ".join(_as_text(text).split()))


def _get_document_class(journal_key: str) -> str:
    """Return the LaTeX document class for a journal (``"article"`` if unknown)."""
    try:
        key = resolve_journal_key(journal_key)
    except UnknownJournalError:
        return _DOCUMENT_CLASSES.get(journal_key, "article")
    if key in _DOCUMENT_CLASSES:
        return _DOCUMENT_CLASSES[key]
    publisher = _as_text(load_guideline(key).get("publisher")).lower()
    if "elsevier" in publisher:
        return "elsarticle"
    if "american society of civil engineers" in publisher:
        return "ascelike"
    return "article"


def _bibliography_style(doc_class: str, guideline: dict) -> str:
    refs = guideline.get("references") or {}
    style = f"{_as_text(refs.get('style'))} {_as_text(refs.get('format'))}".lower()
    numbered = "number" in style or "[1]" in style
    if doc_class == "elsarticle":
        return "elsarticle-num" if numbered else "elsarticle-harv"
    if doc_class == "ascelike":
        return "ascelike"
    # unsrt keeps citation order; plain would re-sort numbered lists alphabetically.
    return "unsrt" if numbered else "apalike"


_FIG_LABEL_RE = re.compile(
    r"^\s*fig(?:ure)?s?\.?\s*\d+(?-i:[a-z])?(?=[\s.:—–|-]|$)\s*(?:[.:—–|-]\s*)?",
    re.IGNORECASE,
)
_TAB_LABEL_RE = re.compile(
    r"^\s*tables?\.?\s*\d+(?-i:[a-z])?(?=[\s.:—–|-]|$)\s*(?:[.:—–|-]\s*)?", re.IGNORECASE
)


def _strip_caption_label(caption: str, kind: str = "figure") -> str:
    """Drop a manual "Fig. 1." / "Table 2." prefix; LaTeX numbers floats itself."""
    pattern = _TAB_LABEL_RE if kind == "table" else _FIG_LABEL_RE
    return pattern.sub("", caption or "", count=1).strip()


def _keyword_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return _as_items(value)
    text = _as_text(value)
    sep = ";" if ";" in text else ","
    return [kw.strip() for kw in text.split(sep) if kw.strip()]


def _graphics_dirs(paths, output_dir: str) -> list[str]:
    """``\\graphicspath`` entries (relative to *output_dir*) for the figure files."""
    dirs = []
    for path in paths:
        if not path:
            continue
        directory = os.path.dirname(os.path.abspath(path))
        if directory == output_dir:
            continue
        try:
            rel = os.path.relpath(directory, output_dir)
        except ValueError:  # different drive on Windows
            rel = directory
        rel = rel.replace(os.sep, "/").rstrip("/") + "/"
        if rel not in dirs:
            dirs.append(rel)
    return dirs


def _latex_heading(command: str, text: str) -> str:
    clean = _strip_heading_number(text).strip()
    star = "*" if _is_unnumbered_heading(clean) else ""
    return f"\\{command}{star}{{{_escape_line(clean)}}}"


def _latex_table(tbl_info: dict, number: int, caption_below: bool = False) -> list[str]:
    caption, headers, rows = _table_parts(tbl_info)
    caption_lines = [
        f"\\caption{{{_escape_line(_strip_caption_label(caption, 'table'))}}}",
        f"\\label{{tab:{number}}}",
    ]
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if not caption_below:
        lines += caption_lines
    n_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    if n_cols:
        lines.append(f"\\begin{{tabular}}{{l{'c' * (n_cols - 1)}}}")
        lines.append(r"\toprule")
        if headers:
            lines.append(" & ".join(_escape_line(h) for h in headers) + r" \\")
            lines.append(r"\midrule")
        for row in rows:
            lines.append(" & ".join(_escape_line(c) for c in row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    if caption_below:
        lines += caption_lines
    lines += [r"\end{table}", ""]
    return lines


def _latex_figure(path, custom_caption: str, number: int) -> list[str]:
    caption = (
        _strip_caption_label(custom_caption, "figure") if custom_caption else _describe_figure(path)
    )
    lines = [r"\begin{figure}[htbp]", r"\centering"]
    if path:
        lines.append(
            "\\includegraphics[width=0.9\\textwidth,height=0.8\\textheight,keepaspectratio]"
            f"{{{os.path.basename(path)}}}"
        )
    lines += [f"\\caption{{{_escape_line(caption)}}}", f"\\label{{fig:{number}}}"]
    lines += [r"\end{figure}", ""]
    return lines


def _latex_notation(label: str, entries) -> list[str]:
    lines = [f"\\section*{{{_escape_line(label)}}}", r"\begin{tabular}{@{}lp{0.75\linewidth}@{}}"]
    for symbol, definition in entries:
        lines.append(f"{_escape_line(symbol)} & {_escape_line(definition)} \\\\")
    lines += [r"\end{tabular}", ""]
    return lines


def generate_latex(
    paper_content: dict,
    journal_key: str,
    figures: list = None,
    output_dir: str = None,
) -> tuple:
    """Generate LaTeX .tex and .bib files.

    Parameters
    ----------
    paper_content:
        Dict with keys: title, authors, abstract, keywords, sections (with
        nested subsections), tables, references, figure_captions, highlights,
        practical_applications, notation, data_availability, acknowledgments.
        Every key is optional.
    journal_key:
        Key matching a journal (e.g. "asce_jse", "eng_structures"); aliases
        accepted by :func:`utils.guidelines.resolve_journal_key` work too.
    figures:
        Optional list of paths to figure image files.  The figures are
        referenced by file name and their directories added to
        ``\\graphicspath`` relative to *output_dir*.
    output_dir:
        Directory in which to save output files. Defaults to
        outputs/papers/ inside the project root.

    Returns
    -------
    tuple[str, str]
        (tex_path, bib_path) absolute paths to the generated files.  Both share
        the ``<title>_<journal>`` stem, so exports for different journals never
        overwrite each other's bibliography.

    Raises
    ------
    utils.guidelines.UnknownJournalError
        If *journal_key* does not match any guideline.
    """
    paper_content = _check_content(paper_content)
    guideline = load_guideline(journal_key)
    key = resolve_journal_key(journal_key)
    output_dir = _resolve_output_dir(output_dir, DEFAULT_PAPERS_DIR)
    stem = _latex_stem(paper_content, key)

    doc_class = _get_document_class(key)
    formatting = guideline.get("formatting") or {}
    fig_config = guideline.get("figures_tables") or {}
    features = _journal_features(guideline, key)
    rules = _heading_rules(guideline)
    layout = _document_layout(guideline)
    labels = layout["labels"]

    title = _as_text(paper_content.get("title")).strip() or "Untitled"
    authors = _as_text(paper_content.get("authors"), ", ").strip()
    abstract = _as_text(paper_content.get("abstract"), "\n\n").strip()
    keywords = _keyword_list(paper_content.get("keywords"))
    highlights = _highlight_items(paper_content.get("highlights"))
    practical = _as_text(paper_content.get("practical_applications"), "\n\n").strip()
    sections = _as_sections(paper_content.get("sections"))
    notation = _notation_entries(paper_content.get("notation"))
    tables = [t for t in _as_list(paper_content.get("tables")) if isinstance(t, dict)]
    fig_entries = _figure_entries(figures, paper_content.get("figure_captions"))
    references = _as_items(paper_content.get("references"))
    data_avail = _as_text(paper_content.get("data_availability"), "\n\n").strip()
    acknowledgments = _as_text(paper_content.get("acknowledgments"), "\n\n").strip()
    competing = _competing_interests(paper_content)

    lines = []

    # ------------------------------------------------------------------
    # Preamble
    # ------------------------------------------------------------------
    if doc_class == "elsarticle":
        lines.append(r"\documentclass[review,3p]{elsarticle}")
    elif doc_class == "ascelike":
        lines.append(r"\documentclass[Journal]{ascelike}")
    else:
        size = _as_text(formatting.get("font_size", 12)).strip()
        size = size if size in ("10", "11", "12") else "12"
        letter = "letter" in _as_text(formatting.get("page_size")).lower()
        paper = "letterpaper" if letter else "a4paper"
        lines.append(f"\\documentclass[{size}pt,{paper}]{{article}}")

    lines.append(r"\usepackage[T1]{fontenc}")
    lines.append(r"\usepackage[utf8]{inputenc}")
    if _contains_hangul(_all_text(paper_content)):
        lines.append(r"\usepackage{kotex}")
    lines.append(r"\usepackage{amsmath,amssymb}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{booktabs}")
    if doc_class == "article":
        margins = _as_text(formatting.get("margins", "1 inch"))
        margin = 1.0 if margins.lower() in ("standard", "") else _parse_margin(margins)
        lines.append(f"\\usepackage[margin={margin:.2f}in]{{geometry}}")
        spacing = _line_spacing_value(formatting.get("line_spacing", "double"))
        if spacing >= 2:
            lines += [r"\usepackage{setspace}", r"\doublespacing"]
        elif spacing > 1:
            lines += [r"\usepackage{setspace}", r"\onehalfspacing"]
    if formatting.get("line_numbers"):
        lines.append(r"\usepackage{lineno}")
        lines.append(r"\linenumbers")
    lines.append(r"\usepackage{hyperref}")

    graphics_dirs = _graphics_dirs([p for p, _ in fig_entries], output_dir)
    if graphics_dirs:
        lines.append("\\graphicspath{" + "".join(f"{{{d}}}" for d in graphics_dirs) + "}")
    if not rules["numbered"]:
        lines.append(r"\setcounter{secnumdepth}{0}")
    if references:
        lines.append(f"\\bibliographystyle{{{_bibliography_style(doc_class, guideline)}}}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")

    # ------------------------------------------------------------------
    # Title block
    # ------------------------------------------------------------------
    show_highlights = features["highlights"] and highlights
    if doc_class == "elsarticle":
        lines.append(r"\begin{frontmatter}")
        lines.append(f"\\title{{{_escape_line(title)}}}")
        if authors:
            lines.append(f"\\author{{{_escape_line(authors)}}}")
        if abstract:
            lines += [r"\begin{abstract}", _escape_latex(abstract), r"\end{abstract}"]
        if show_highlights:
            lines.append(r"\begin{highlights}")
            lines += [f"\\item {_escape_line(h)}" for h in highlights]
            lines.append(r"\end{highlights}")
        if keywords:
            lines.append(r"\begin{keyword}")
            lines.append(" \\sep ".join(_escape_line(kw) for kw in keywords))
            lines.append(r"\end{keyword}")
        lines.append(r"\end{frontmatter}")
    else:
        lines.append(f"\\title{{{_escape_line(title)}}}")
        lines.append(f"\\author{{{_escape_line(authors)}}}")
        lines.append(r"\maketitle")
        if abstract:
            lines += [r"\begin{abstract}", _escape_latex(abstract), r"\end{abstract}"]
        if keywords:
            kw_text = "; ".join(_escape_line(kw) for kw in keywords)
            lines.append(f"\\noindent\\textbf{{Keywords:}} {kw_text}")
        if show_highlights:
            lines += ["", r"\section*{Highlights}", r"\begin{itemize}"]
            lines += [f"\\item {_escape_line(h)}" for h in highlights]
            lines.append(r"\end{itemize}")
    lines.append("")

    if features["practical_applications"] and practical:
        lines += [r"\section*{Practical Applications}", _escape_latex(practical), ""]
    if layout["front_notation"] and notation:
        lines += _latex_notation(labels["notation"], notation)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    for sec in sections:
        heading, content, subsections = _section_parts(sec)
        if heading:
            lines.append(_latex_heading("section", heading))
        if content.strip():
            lines.append(_escape_latex(content.strip()))
        lines.append("")
        for subsec in subsections:
            sub_heading, sub_content, sub_subsections = _section_parts(subsec)
            if sub_heading:
                lines.append(_latex_heading("subsection", sub_heading))
            if sub_content.strip():
                lines.append(_escape_latex(sub_content.strip()))
            lines.append("")
            for subsub in sub_subsections:
                h3, c3, _ = _section_parts(subsub)
                if h3:
                    lines.append(_latex_heading("subsubsection", h3))
                if c3.strip():
                    lines.append(_escape_latex(c3.strip()))
                lines.append("")

    # ------------------------------------------------------------------
    # Back matter and floats, in the order the journal guideline lists them
    # ------------------------------------------------------------------
    table_caption_below = _as_text(fig_config.get("caption_position_table")).lower() == "below"
    for block in layout["order"]:
        if block == "tables":
            for n, tbl in enumerate(tables, 1):
                lines += _latex_table(tbl, n, caption_below=table_caption_below)
        elif block == "figures":
            for n, (path, caption) in enumerate(fig_entries, 1):
                lines += _latex_figure(path, caption, n)
        elif block == "notation" and notation and not layout["front_notation"]:
            lines += _latex_notation(labels["notation"], notation)
        elif block == "data_availability" and data_avail:
            lines += [f"\\section*{{{_escape_line(labels[block])}}}", _escape_latex(data_avail), ""]
        elif block == "acknowledgments" and acknowledgments:
            lines += [
                f"\\section*{{{_escape_line(labels[block])}}}",
                _escape_latex(acknowledgments),
                "",
            ]
        elif block == "competing_interests" and competing:
            lines += [f"\\section*{{{_escape_line(labels[block])}}}", _escape_latex(competing), ""]
        elif block == "references" and references:
            # Nothing in the body uses \cite, so without \nocite BibTeX would
            # produce an empty reference list.
            lines += [r"\nocite{*}", f"\\bibliography{{{stem}}}", ""]

    lines.append(r"\end{document}")

    # Write .tex
    tex_path = os.path.join(output_dir, f"{stem}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Write .bib
    bib_path = os.path.join(output_dir, f"{stem}.bib")
    bib_lines = []
    for i, ref in enumerate(references, 1):
        bib_lines += _bib_entry(f"ref{i}", ref)
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write("\n".join(bib_lines))

    return tex_path, bib_path


# ---------------------------------------------------------------------------
# References -> BibTeX
# ---------------------------------------------------------------------------

_REF_LABEL_RE = re.compile(r"^\s*(?:\[\d+\]|\(\d+\)|\d+\.(?=\s))\s*")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"]+")
_INITIALS_ONLY_RE = re.compile(r"^(?:[A-Z]\.?(?:\s*-\s*|\s+)?)+$")
_AUTHOR_PART_RE = re.compile(
    r"^(?:[A-Z]\.\s*(?:-\s*)?)+\s*[^\W\d_][^\s,]*(?:\s+[^\s,]+){0,5}$|^et\s+al\.?$"
)
_QUOTED_TITLE_RE = re.compile(
    r"^(?P<author>.+?)[.,]?\s*\(?(?P<year>(?:19|20)\d{2}[a-z]?)\)?[.,]?\s*"
    r"[\"“](?P<title>.+?)[.,]?[\"”][.,]?\s*(?P<rest>.*)$"
)
_JOURNAL_VOLUME_YEAR_RE = re.compile(
    r",\s*(?P<journal>[^,]+?)\s+(?P<volume>\d+)\s*\((?P<year>(?:19|20)\d{2})\)"
    r"\s*(?P<pages>[A-Za-z]?\d+(?:\s*[-–]+\s*[A-Za-z]?\d+)?)?"
)


def _bibtex_authors(text: str) -> str:
    """Convert a reference's author list to BibTeX's ``A and B and C`` form.

    Handles "A.B. Smith, C.D. Jones", "Kim, S., Lee, J., and Park, H." and
    ASCE's "Evans, G. M., and J. C. Furlong"; "et al." becomes "others".
    """
    text = re.sub(r"\s*,?\s*et\s+al\.?\s*$", ", et al", text.strip())
    tokens = [t.strip() for t in re.split(r",\s*(?:and\s+|&\s*)?|\s+and\s+|\s+&\s+", text)]
    names = []
    for token in tokens:
        if not token:
            continue
        if token.lower().startswith("et al"):
            names.append("others")
        elif (
            names
            and names[-1] != "others"
            and "," not in names[-1]
            and _INITIALS_ONLY_RE.match(token)
        ):
            names[-1] = f"{names[-1]}, {token}"
        else:
            names.append(token)
    # "Park, H" (period eaten by the year separator) -> "Park, H."
    names = [re.sub(r"(\b[A-Z])$", r"\1.", name) for name in names]
    return " and ".join(names)


def _split_journal_details(rest: str) -> dict:
    """Parse "Journal, Vol. 27, No. 5, pp. 1-10" / "Appl. Mech. Rev., 40 (1), 1-23"."""
    rest = rest.strip().rstrip(".,; ")
    if not rest:
        return {}
    journal, _, details = rest.partition(",")
    fields = {"journal": journal.strip()}
    volume = re.search(r"\bV(?:ol)?\.\s*(\d+)", details) or re.match(r"\s*(\d+)", details)
    if volume:
        fields["volume"] = volume.group(1)
    number = re.search(r"\bNo\.\s*([\w-]+)", details) or re.search(r"\d+\s*\(([\w-]+)\)", details)
    if number:
        fields["number"] = number.group(1)
    pages = re.search(r"\bpp?\.\s*([A-Za-z]?\d+(?:\s*[-–]+\s*[A-Za-z]?\d+)?)", details)
    if not pages:
        pages = re.search(r"\b([A-Za-z]?\d+\s*[-–]+\s*[A-Za-z]?\d+)\b", details)
    if pages:
        fields["pages"] = pages.group(1)
    return fields


def _parse_quoted_title(body: str) -> dict:
    """ASCE / Springer / ACI styles: ``Authors. (Year). "Title." Journal, ...``."""
    match = _QUOTED_TITLE_RE.match(body)
    if not match:
        return {}
    fields = {
        "author": _bibtex_authors(match.group("author")),
        "year": match.group("year"),
        "title": match.group("title").strip(),
    }
    fields.update(_split_journal_details(match.group("rest")))
    return fields


def _parse_elsevier(body: str) -> dict:
    """Elsevier numbered style: ``A.B. Name, C.D. Name, Title, J. Name 50 (2020) 1-10``."""
    parts = [p.strip() for p in body.split(",")]
    n_authors = 0
    for part in parts:
        if _AUTHOR_PART_RE.match(part):
            n_authors += 1
        else:
            break
    if not n_authors or n_authors >= len(parts):
        return {}
    remaining = ", " + ", ".join(parts[n_authors:])
    matches = list(_JOURNAL_VOLUME_YEAR_RE.finditer(remaining))
    if not matches:
        return {}
    match = matches[-1]
    title = remaining[: match.start()].strip(" ,")
    if not title:
        return {}
    fields = {
        "author": _bibtex_authors(", ".join(parts[:n_authors])),
        "title": title,
        "journal": match.group("journal").strip(),
        "volume": match.group("volume"),
        "year": match.group("year"),
    }
    if match.group("pages"):
        fields["pages"] = match.group("pages")
    return fields


def _parse_reference(ref_text: str) -> dict:
    """Parse a reference string into BibTeX fields (best-effort heuristic).

    Always returns ``doi``/``year`` when present.  ``author``/``title``/
    ``journal`` are only returned when the whole reference matched a known
    style; callers fall back to keeping the text verbatim otherwise.
    """
    text = _as_text(ref_text).strip()
    fields = {}
    doi_match = _DOI_RE.search(text)
    if doi_match:
        fields["doi"] = doi_match.group().rstrip(".")
    year_match = re.search(r"\b((?:19|20)\d{2})\b", _DOI_RE.sub("", text))
    if year_match:
        fields["year"] = year_match.group()

    # Strip URL / DOI noise so it is not mistaken for volume or page numbers.
    body = re.sub(r"https?://\S+", "", text)
    body = re.sub(r"\b(?:doi|DOI)\s*:?\s*(?=10\.)", "", body)
    body = _DOI_RE.sub("", body)
    body = re.sub(r"[\s,;.]+$", "", body).strip()

    parsed = _parse_quoted_title(body) or _parse_elsevier(body)
    if parsed:
        fields.update({k: v for k, v in parsed.items() if v})
    return fields


def _bib_escape(text: str) -> str:
    """Escape for a BibTeX field value.

    BibTeX counts every ``{``/``}`` (it does not understand ``\\{``), so literal
    braces become ``\\textbraceleft{}``/``\\textbraceright{}`` to keep the value
    balanced; one stray brace would otherwise break the whole .bib file.
    """
    escaped = _escape_line(text)
    return escaped.replace(r"\{", r"\textbraceleft{}").replace(r"\}", r"\textbraceright{}")


def _bib_value(name: str, value: str) -> str:
    if name == "doi":
        return re.sub(r"[\s{}%\\]", "", value)
    if name == "pages":
        return re.sub(r"\s*[-–]+\s*", "--", value)
    escaped = _bib_escape(value)
    # Double braces keep capitalisation ("RC", "Monte Carlo") under BibTeX styles.
    return "{" + escaped + "}" if name == "title" else escaped


def _bib_entry(key: str, ref: str) -> list[str]:
    """One BibTeX entry for a free-text reference (structured or verbatim)."""
    clean = _REF_LABEL_RE.sub("", _as_text(ref), count=1).strip()
    fields = _parse_reference(clean)
    if all(fields.get(k) for k in ("author", "title", "journal")):
        lines = [f"@article{{{key},"]
        for name in ("author", "title", "journal", "volume", "number", "pages", "year", "doi"):
            if fields.get(name):
                lines.append(f"  {name} = {{{_bib_value(name, fields[name])}}},")
    else:
        lines = [f"@misc{{{key},", f"  note = {{{_bib_escape(clean)}}},"]
    lines += ["}", ""]
    return lines


def _safe_filename(title: str) -> str:
    """Return a filesystem-safe stem derived from *title* (never empty)."""
    return _safe_stem(title)
