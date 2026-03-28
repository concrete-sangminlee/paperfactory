"""Generate journal-formatted PDF documents for PaperFactory.

Uses reportlab for direct PDF generation without LaTeX compilation.
"""

import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
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

DEFAULT_PAPERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "papers"
)


def _build_styles():
    """Create publication-quality paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "PaperTitle",
            parent=styles["Title"],
            fontSize=16,
            leading=20,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName="Times-Bold",
        )
    )
    styles.add(
        ParagraphStyle(
            "Authors",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            fontName="Times-Roman",
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "AbstractHeading",
            parent=styles["Heading2"],
            fontSize=11,
            fontName="Times-Bold",
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "AbstractBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            fontName="Times-Italic",
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
            fontName="Times-Bold",
            spaceBefore=16,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubsectionHeading",
            parent=styles["Heading2"],
            fontSize=11,
            fontName="Times-BoldItalic",
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 13
    styles["BodyText"].fontName = "Times-Roman"
    styles["BodyText"].alignment = TA_JUSTIFY
    styles["BodyText"].spaceAfter = 6
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            fontName="Times-Italic",
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
            fontName="Times-Roman",
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
            fontName="Times-Roman",
            spaceAfter=12,
        )
    )
    return styles


def _safe_xml(text: str) -> str:
    """Escape XML special characters for reportlab Paragraph."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


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
        references, tables, figure_captions, highlights, etc.
    journal_key : str
        Journal key (e.g. "jweia", "eng_structures").
    figures : list, optional
        List of figure image file paths.
    output_dir : str, optional
        Output directory. Defaults to outputs/papers/.

    Returns
    -------
    str
        Path to the generated PDF file.
    """
    if output_dir is None:
        output_dir = DEFAULT_PAPERS_DIR
    os.makedirs(output_dir, exist_ok=True)

    styles = _build_styles()
    title = paper_content.get("title", "Untitled")
    safe_name = re.sub(r"[^\w\s-]", "", title)[:50].strip()
    safe_name = re.sub(r"\s+", "_", safe_name)
    pdf_path = os.path.join(output_dir, f"{safe_name}_{journal_key}.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
    )

    story = []

    # Title
    story.append(Paragraph(_safe_xml(title), styles["PaperTitle"]))

    # Authors
    authors = paper_content.get("authors", "")
    if authors:
        story.append(Paragraph(_safe_xml(authors), styles["Authors"]))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 6))

    # Abstract
    abstract = paper_content.get("abstract", "")
    if abstract:
        story.append(Paragraph("Abstract", styles["AbstractHeading"]))
        for para in abstract.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(_safe_xml(para), styles["AbstractBody"]))

    # Keywords
    keywords = paper_content.get("keywords", "")
    if keywords:
        story.append(Paragraph(f"<b>Keywords:</b> {_safe_xml(keywords)}", styles["Keywords"]))

    # Highlights
    highlights = paper_content.get("highlights", [])
    if highlights:
        story.append(Paragraph("<b>Highlights</b>", styles["SectionHeading"]))
        for hl in highlights:
            story.append(Paragraph(f"• {_safe_xml(hl)}", styles["BodyText"]))
        story.append(Spacer(1, 8))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))

    # Sections
    for sec in paper_content.get("sections", []):
        heading = sec.get("heading", "")
        content = sec.get("content", "")

        story.append(Paragraph(_safe_xml(heading), styles["SectionHeading"]))
        if content:
            for para in content.split("\n\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(_safe_xml(para), styles["BodyText"]))

        for subsec in sec.get("subsections", []):
            sub_heading = subsec.get("heading", "")
            sub_content = subsec.get("content", "")
            story.append(Paragraph(_safe_xml(sub_heading), styles["SubsectionHeading"]))
            if sub_content:
                for para in sub_content.split("\n\n"):
                    para = para.strip()
                    if para:
                        story.append(Paragraph(_safe_xml(para), styles["BodyText"]))

    # Tables
    for tbl in paper_content.get("tables", []):
        story.append(Spacer(1, 10))
        cap = tbl.get("caption", "")
        headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])

        if cap:
            story.append(Paragraph(_safe_xml(cap), styles["Caption"]))

        data = [headers] + rows
        t = Table(data, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, 0), 0.5, colors.white),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
                    ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8f9fa")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(KeepTogether([t]))
        story.append(Spacer(1, 10))

    # Figures
    if figures:
        captions = paper_content.get("figure_captions", [])
        for i, fig_path in enumerate(figures):
            if os.path.exists(fig_path):
                story.append(Spacer(1, 10))
                try:
                    img = Image(fig_path, width=6.0 * inch, height=3.5 * inch)
                    img.hAlign = "CENTER"
                    story.append(img)
                except Exception:
                    story.append(
                        Paragraph(f"[Figure: {os.path.basename(fig_path)}]", styles["Caption"])
                    )
                cap = captions[i] if i < len(captions) else f"Fig. {i + 1}."
                story.append(Paragraph(_safe_xml(cap), styles["Caption"]))

    # References
    refs = paper_content.get("references", [])
    if refs:
        story.append(Paragraph("REFERENCES", styles["SectionHeading"]))
        if isinstance(refs, str):
            refs = [r for r in refs.split("\n") if r.strip()]
        for ref in refs:
            ref = ref.strip()
            if ref:
                story.append(Paragraph(_safe_xml(ref), styles["Reference"]))

    # Data availability
    da = paper_content.get("data_availability", "")
    if da:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Data Availability", styles["SectionHeading"]))
        story.append(Paragraph(_safe_xml(da), styles["BodyText"]))

    # Acknowledgments
    ack = paper_content.get("acknowledgments", "")
    if ack:
        story.append(Paragraph("Acknowledgments", styles["SectionHeading"]))
        story.append(Paragraph(_safe_xml(ack), styles["BodyText"]))

    doc.build(story)
    return pdf_path
