"""Generate journal-formatted LaTeX documents for PaperFactory."""

import os
import re

DEFAULT_PAPERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "papers"
)

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
}

_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_MATH_PATTERN = re.compile(r"(\$[^$\s][^$]*[^$\s]\$|\$[^$\s]\$)")


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters while preserving inline math ($...$).

    Inline math is detected as $<content>$ where content does not start or end
    with whitespace (to avoid matching standalone dollar signs like "$5").
    """
    parts = _MATH_PATTERN.split(text)
    result = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) >= 3:
            result.append(part)
        else:
            escaped = part.replace("\\", "\x00BACKSLASH\x00")
            escaped = escaped.replace("{", r"\{").replace("}", r"\}")
            escaped = escaped.replace("$", r"\$")
            for char, replacement in _LATEX_SPECIAL.items():
                escaped = escaped.replace(char, replacement)
            escaped = escaped.replace("\x00BACKSLASH\x00", r"\textbackslash{}")
            result.append(escaped)
    return "".join(result)


def _get_document_class(journal_key: str) -> str:
    """Return the LaTeX document class for a journal."""
    return _DOCUMENT_CLASSES.get(journal_key, "article")


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
        Dict with keys: title, authors, abstract, keywords, sections,
        references, figure_captions.
    journal_key:
        Key matching a journal (e.g. "asce_jse", "eng_structures").
    figures:
        Optional list of absolute paths to figure image files.
    output_dir:
        Directory in which to save output files. Defaults to
        outputs/papers/ inside the project root.

    Returns
    -------
    tuple[str, str]
        (tex_path, bib_path) absolute paths to the generated files.
    """
    if output_dir is None:
        output_dir = DEFAULT_PAPERS_DIR
    os.makedirs(output_dir, exist_ok=True)

    doc_class = _get_document_class(journal_key)
    title = paper_content.get("title", "Untitled")
    authors = paper_content.get("authors", "")
    abstract = paper_content.get("abstract", "")
    keywords = paper_content.get("keywords", "")

    lines = []

    # Document class
    if doc_class == "elsarticle":
        lines.append(r"\documentclass[review,3p]{elsarticle}")
    elif doc_class == "ascelike":
        lines.append(r"\documentclass[Journal]{ascelike}")
    else:
        lines.append(r"\documentclass[12pt,a4paper]{article}")

    # Packages
    lines.append(r"\usepackage[utf8]{inputenc}")
    lines.append(r"\usepackage{amsmath,amssymb}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{hyperref}")
    if doc_class == "elsarticle":
        lines.append(r"\usepackage{lineno}")
        lines.append(r"\linenumbers")
    lines.append("")

    bib_name = _safe_filename(title)
    lines.append(
        r"\bibliographystyle{elsarticle-num}"
        if doc_class == "elsarticle"
        else r"\bibliographystyle{ascelike}"
        if doc_class == "ascelike"
        else r"\bibliographystyle{plain}"
    )
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append("")

    # Title block
    if doc_class == "elsarticle":
        lines.append(r"\begin{frontmatter}")
        lines.append(f"\\title{{{_escape_latex(title)}}}")
        lines.append(f"\\author{{{_escape_latex(authors)}}}")
        lines.append(r"\begin{abstract}")
        lines.append(_escape_latex(abstract))
        lines.append(r"\end{abstract}")
        if keywords:
            kws = [kw.strip() for kw in keywords.split(";")]
            lines.append(r"\begin{keyword}")
            lines.append(" \\sep ".join(_escape_latex(kw) for kw in kws))
            lines.append(r"\end{keyword}")
        lines.append(r"\end{frontmatter}")
    else:
        lines.append(f"\\title{{{_escape_latex(title)}}}")
        lines.append(f"\\author{{{_escape_latex(authors)}}}")
        lines.append(r"\maketitle")
        lines.append(r"\begin{abstract}")
        lines.append(_escape_latex(abstract))
        lines.append(r"\end{abstract}")
        if keywords:
            lines.append(f"\\noindent\\textbf{{Keywords:}} {_escape_latex(keywords)}")
    lines.append("")

    # Sections
    for sec in paper_content.get("sections", []):
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        lines.append(f"\\section{{{_escape_latex(heading)}}}")
        if content:
            lines.append(_escape_latex(content))
        lines.append("")
        for subsec in sec.get("subsections", []):
            sub_heading = subsec.get("heading", "")
            sub_content = subsec.get("content", "")
            lines.append(f"\\subsection{{{_escape_latex(sub_heading)}}}")
            if sub_content:
                lines.append(_escape_latex(sub_content))
            lines.append("")

    # Figures
    if figures:
        custom_captions = paper_content.get("figure_captions", [])
        for i, fig_path in enumerate(figures, 1):
            fig_basename = os.path.basename(fig_path)
            cap = custom_captions[i - 1] if i <= len(custom_captions) else f"Figure {i}"
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(f"\\includegraphics[width=0.9\\textwidth]{{{fig_basename}}}")
            lines.append(f"\\caption{{{_escape_latex(cap)}}}")
            lines.append(f"\\label{{fig:{i}}}")
            lines.append(r"\end{figure}")
            lines.append("")

    lines.append(f"\\bibliography{{{bib_name}}}")
    lines.append("")
    lines.append(r"\end{document}")

    # Write .tex
    safe_name = _safe_filename(title)
    tex_path = os.path.join(output_dir, f"{safe_name}_{journal_key}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Write .bib
    bib_path = os.path.join(output_dir, f"{bib_name}.bib")
    bib_lines = []
    refs = paper_content.get("references", [])
    if isinstance(refs, str):
        refs = refs.split("\n")
    for i, ref in enumerate(refs, 1):
        ref = ref.strip()
        if ref:
            clean_ref = re.sub(r"^\s*\[?\d+\]?\s*\.?\s*", "", ref)
            fields = _parse_reference(clean_ref)
            bib_lines.append(f"@article{{ref{i},")
            for key in ("author", "title", "journal", "volume", "pages", "year", "doi"):
                if fields.get(key):
                    bib_lines.append(f"  {key} = {{{fields[key]}}},")
            if not any(fields.get(k) for k in ("author", "title")):
                bib_lines.append(f"  note = {{{clean_ref}}},")
            bib_lines.append("}")
            bib_lines.append("")
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write("\n".join(bib_lines))

    return tex_path, bib_path


def _parse_reference(ref_text: str) -> dict:
    """Parse a reference string into BibTeX fields (best-effort heuristic)."""
    fields = {}

    # DOI
    doi_match = re.search(r"(10\.\d{4,}/[^\s,]+)", ref_text)
    if doi_match:
        fields["doi"] = doi_match.group().rstrip(".")

    # Year
    year_match = re.search(r"\b((?:19|20)\d{2})\b", ref_text)
    if year_match:
        fields["year"] = year_match.group()

    # Try to split "Authors, Title, Journal Vol (Year) Pages"
    # Common pattern: "A.B. Name, C.D. Name, Title of paper, J. Name Vol (Year) Pages."
    # Remove DOI/URL suffix for cleaner parsing
    clean = re.sub(r"https?://\S+", "", ref_text).strip().rstrip(".")

    # Split on comma-separated segments
    parts = [p.strip() for p in clean.split(",")]

    if len(parts) >= 3:
        # Heuristic: author names typically contain initials (e.g., "A.B. Name")
        author_parts = []
        title_start = 0
        for j, part in enumerate(parts):
            # Author part: contains initials like "A." or "A.B."
            if re.search(r"\b[A-Z]\.\s*[A-Z]?\.", part) or (
                j == 0 and re.search(r"\b[A-Z]\.", part)
            ):
                author_parts.append(part)
                title_start = j + 1
            else:
                break

        if author_parts:
            fields["author"] = ", ".join(author_parts)

        remaining = ", ".join(parts[title_start:])

        # Extract title: text before "J. " or "Eng. " or volume pattern
        title_match = re.match(
            r"(.+?),\s*([A-Z][a-z]*[\.\s].*?(?:\d+\s*\(|\d{4}))",
            remaining,
        )
        if title_match:
            fields["title"] = title_match.group(1).strip()
            journal_etc = title_match.group(2).strip()
            # Extract journal name (before volume number)
            jrnl_match = re.match(r"(.+?)\s+(\d+)", journal_etc)
            if jrnl_match:
                fields["journal"] = jrnl_match.group(1).strip().rstrip(".")
                fields["volume"] = jrnl_match.group(2)
            # Extract pages
            pages_match = re.search(r"(\d+[-–]\d+|\d{5,})", remaining)
            if pages_match:
                fields["pages"] = pages_match.group().replace("–", "--")
        elif title_start < len(parts):
            fields["title"] = parts[title_start].strip()

    return fields


def _safe_filename(title: str) -> str:
    """Return a filesystem-safe stem derived from *title*."""
    safe = re.sub(r"[^\w\s-]", "", title)[:50].strip()
    return re.sub(r"\s+", "_", safe)
