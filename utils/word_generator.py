import json
import os
import re
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT

GUIDELINES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guidelines")
PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "papers")


def load_guideline(journal_key: str) -> dict:
    path = os.path.join(GUIDELINES_DIR, f"{journal_key}.json")
    with open(path, "r") as f:
        return json.load(f)


def _apply_font(run, formatting: dict):
    run.font.name = formatting.get("font", "Times New Roman")
    run.font.size = Pt(formatting.get("font_size", 12))


def _set_spacing(paragraph, formatting: dict):
    spacing = formatting.get("line_spacing", "double")
    pf = paragraph.paragraph_format
    if spacing == "double":
        pf.line_spacing = 2.0
    elif spacing == "1.5":
        pf.line_spacing = 1.5
    else:
        pf.line_spacing = 2.0


def _parse_margin(margin_str: str) -> float:
    if "1 inch" in margin_str.lower() or "2.54" in margin_str or "1in" in margin_str.lower():
        return 1.0
    match = re.search(r"([\d.]+)\s*(inch|in|cm|mm)", margin_str.lower())
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit in ("cm",):
            return val / 2.54
        if unit in ("mm",):
            return val / 25.4
        return val
    return 1.0


def generate_word(paper_content: dict, journal_key: str, figures: list[str] = None) -> str:
    os.makedirs(PAPERS_DIR, exist_ok=True)
    guideline = load_guideline(journal_key)
    formatting = guideline.get("formatting", {})

    doc = Document()

    # Page setup
    section = doc.sections[0]
    page_size = formatting.get("page_size", "US Letter")
    if "A4" in page_size.upper():
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
    else:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)

    margin_val = _parse_margin(formatting.get("margins", "1 inch all sides"))
    section.top_margin = Inches(margin_val)
    section.bottom_margin = Inches(margin_val)
    section.left_margin = Inches(margin_val)
    section.right_margin = Inches(margin_val)

    # Style defaults
    style = doc.styles["Normal"]
    font = style.font
    font.name = formatting.get("font", "Times New Roman")
    font.size = Pt(formatting.get("font_size", 12))

    # Title
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(paper_content.get("title", "Untitled"))
    title_run.bold = True
    title_run.font.size = Pt(formatting.get("font_size", 12) + 2)
    _apply_font(title_run, formatting)
    _set_spacing(title_para, formatting)

    # Authors
    if paper_content.get("authors"):
        authors_para = doc.add_paragraph()
        authors_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        authors_run = authors_para.add_run(paper_content["authors"])
        _apply_font(authors_run, formatting)
        _set_spacing(authors_para, formatting)

    # Abstract
    if paper_content.get("abstract"):
        doc.add_paragraph()
        abs_heading = doc.add_paragraph()
        abs_run = abs_heading.add_run("ABSTRACT")
        abs_run.bold = True
        _apply_font(abs_run, formatting)
        _set_spacing(abs_heading, formatting)

        abs_para = doc.add_paragraph()
        abs_text_run = abs_para.add_run(paper_content["abstract"])
        _apply_font(abs_text_run, formatting)
        _set_spacing(abs_para, formatting)

    # Keywords
    if paper_content.get("keywords"):
        kw_para = doc.add_paragraph()
        kw_label = kw_para.add_run("Keywords: ")
        kw_label.bold = True
        _apply_font(kw_label, formatting)
        kw_text = kw_para.add_run(paper_content["keywords"])
        _apply_font(kw_text, formatting)
        _set_spacing(kw_para, formatting)

    # Body sections
    sections_content = paper_content.get("sections", [])
    for sec in sections_content:
        heading_text = sec.get("heading", "")
        body_text = sec.get("content", "")

        h_para = doc.add_paragraph()
        h_run = h_para.add_run(heading_text.upper())
        h_run.bold = True
        _apply_font(h_run, formatting)
        _set_spacing(h_para, formatting)

        if body_text:
            paragraphs = body_text.split("\n\n")
            for p_text in paragraphs:
                p_text = p_text.strip()
                if not p_text:
                    continue
                body_para = doc.add_paragraph()
                body_run = body_para.add_run(p_text)
                _apply_font(body_run, formatting)
                _set_spacing(body_para, formatting)

    # Tables
    tables_data = paper_content.get("tables", [])
    if tables_data:
        doc.add_page_break()
        tables_heading = doc.add_paragraph()
        t_h_run = tables_heading.add_run("TABLES")
        t_h_run.bold = True
        _apply_font(t_h_run, formatting)

        for tbl_info in tables_data:
            # Table caption (above)
            cap_para = doc.add_paragraph()
            cap_run = cap_para.add_run(tbl_info.get("caption", ""))
            cap_run.bold = True
            _apply_font(cap_run, formatting)
            cap_run.font.size = Pt(formatting.get("font_size", 12) - 1)
            _set_spacing(cap_para, formatting)

            headers = tbl_info.get("headers", [])
            rows = tbl_info.get("rows", [])
            if headers and rows:
                n_cols = len(headers)
                table = doc.add_table(rows=1 + len(rows), cols=n_cols)
                table.style = 'Table Grid'

                # Header row
                for j, header in enumerate(headers):
                    cell = table.rows[0].cells[j]
                    cell.text = ""
                    p = cell.paragraphs[0]
                    run = p.add_run(header)
                    run.bold = True
                    run.font.name = formatting.get("font", "Times New Roman")
                    run.font.size = Pt(formatting.get("font_size", 12) - 2)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                # Data rows
                for i, row in enumerate(rows):
                    for j, val in enumerate(row):
                        cell = table.rows[i + 1].cells[j]
                        cell.text = ""
                        p = cell.paragraphs[0]
                        run = p.add_run(str(val))
                        run.font.name = formatting.get("font", "Times New Roman")
                        run.font.size = Pt(formatting.get("font_size", 12) - 2)
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                # Remove vertical borders (ASCE style: horizontal rules only)
                from docx.oxml.ns import qn
                tbl_xml = table._tbl
                tbl_pr = tbl_xml.tblPr if tbl_xml.tblPr is not None else tbl_xml._add_tblPr()
                borders = tbl_pr.find(qn('w:tblBorders'))
                if borders is None:
                    from docx.oxml import OxmlElement
                    borders = OxmlElement('w:tblBorders')
                    tbl_pr.append(borders)
                for border_name in ['top', 'bottom', 'insideH']:
                    b = borders.find(qn(f'w:{border_name}'))
                    if b is None:
                        from docx.oxml import OxmlElement
                        b = OxmlElement(f'w:{border_name}')
                        borders.append(b)
                    b.set(qn('w:val'), 'single')
                    b.set(qn('w:sz'), '4')
                    b.set(qn('w:color'), '000000')
                for border_name in ['left', 'right', 'insideV']:
                    b = borders.find(qn(f'w:{border_name}'))
                    if b is None:
                        from docx.oxml import OxmlElement
                        b = OxmlElement(f'w:{border_name}')
                        borders.append(b)
                    b.set(qn('w:val'), 'none')

            doc.add_paragraph()  # spacing between tables

    # Figures
    fig_config = guideline.get("figures_tables", {})
    placement = fig_config.get("placement", "end of manuscript")

    if figures and "end" in placement.lower():
        doc.add_page_break()
        fig_heading = doc.add_paragraph()
        fig_h_run = fig_heading.add_run("FIGURES")
        fig_h_run.bold = True
        _apply_font(fig_h_run, formatting)

    if figures:
        caption_style = fig_config.get("caption_style", "Fig. {n}. {desc}")
        for i, fig_path in enumerate(figures, 1):
            if os.path.exists(fig_path):
                fig_para = doc.add_paragraph()
                fig_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                try:
                    fig_para.add_run().add_picture(fig_path, width=Inches(5.5))
                except Exception:
                    fig_para.add_run(f"[Figure {i}: {os.path.basename(fig_path)}]")
                _set_spacing(fig_para, formatting)

                cap_text = caption_style.replace("{n}", str(i)).replace(
                    "{desc}", os.path.splitext(os.path.basename(fig_path))[0]
                )
                cap_para = doc.add_paragraph()
                cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap_run = cap_para.add_run(cap_text)
                cap_run.italic = True
                _apply_font(cap_run, formatting)
                _set_spacing(cap_para, formatting)

    # References
    if paper_content.get("references"):
        doc.add_page_break()
        ref_heading = doc.add_paragraph()
        ref_h_run = ref_heading.add_run("REFERENCES")
        ref_h_run.bold = True
        _apply_font(ref_h_run, formatting)
        _set_spacing(ref_heading, formatting)

        refs = paper_content["references"]
        if isinstance(refs, str):
            refs = refs.split("\n")
        for ref in refs:
            ref = ref.strip()
            if ref:
                ref_para = doc.add_paragraph()
                ref_run = ref_para.add_run(ref)
                _apply_font(ref_run, formatting)
                _set_spacing(ref_para, formatting)

    # Save
    safe_title = re.sub(r"[^\w\s-]", "", paper_content.get("title", "paper"))[:50].strip()
    safe_title = re.sub(r"\s+", "_", safe_title)
    filename = f"{safe_title}_{journal_key}.docx"
    output_path = os.path.join(PAPERS_DIR, filename)
    doc.save(output_path)
    return output_path
