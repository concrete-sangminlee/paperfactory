"""Regression tests for bugs fixed in utils/pdf_generator.py.

The generated PDFs are inspected with pdfplumber (text, page size, image boxes).
"""

import os

import pdfplumber
import pytest
from PIL import Image

from utils import pdf_generator, word_generator
from utils.guidelines import UnknownJournalError, available_journal_keys
from utils.pdf_generator import _build_styles, _needs_unicode_font, _safe_xml, generate_pdf


@pytest.fixture(autouse=True)
def _no_writes_to_outputs(tmp_path, monkeypatch):
    """Redirect the default output directory so no test can write into outputs/."""
    for module in (pdf_generator, word_generator):
        monkeypatch.setattr(module, "DEFAULT_PAPERS_DIR", str(tmp_path / "default_papers"))


BASE = {
    "title": "Regression Paper",
    "authors": "A. Author",
    "abstract": "Abstract text.",
    "keywords": "a; b",
    "sections": [{"heading": "Introduction", "content": "Body text."}],
    "references": ["[1] A. Author, Title, J. Eng. 1 (2024) 1-2."],
}


def _png(path, size=(400, 300)):
    Image.new("RGB", size, "white").save(path)
    return str(path)


def _pdf(tmp_path, content=None, journal="jweia", **kw):
    return generate_pdf(content or BASE, journal, output_dir=str(tmp_path), **kw)


def _text(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def test_unknown_journal_raises(tmp_path):
    # Any key used to be accepted silently (the guideline was never loaded).
    with pytest.raises(UnknownJournalError):
        _pdf(tmp_path, journal="no_such_journal")


@pytest.mark.parametrize("journal,size", [("asce_jse", (612, 792)), ("jweia", (595, 842))])
def test_page_size_follows_guideline(tmp_path, journal, size):
    with pdfplumber.open(_pdf(tmp_path, journal=journal)) as pdf:
        page = pdf.pages[0]
        assert (round(page.width), round(page.height)) == size


def test_references_come_after_data_availability_and_acknowledgments(tmp_path):
    content = {**BASE, "data_availability": "DA-TEXT", "acknowledgments": "ACK-TEXT"}
    text = _text(_pdf(tmp_path, content))
    assert text.index("DA-TEXT") < text.index("References")
    assert text.index("ACK-TEXT") < text.index("References")


def test_journal_specific_blocks(tmp_path):
    content = {
        **BASE,
        "highlights": ["HL-ONE"],
        "practical_applications": "PRACTICAL-TEXT",
        "notation": [{"symbol": "f_c", "definition": "NOTATION-DEF"}],
        "sections": [
            {
                "heading": "Intro",
                "content": "x",
                "subsections": [
                    {"heading": "Sub", "subsections": [{"heading": "SUBSUB-H", "content": "SS"}]}
                ],
            }
        ],
    }
    asce = _text(_pdf(tmp_path, content, journal="asce_jse"))
    assert "HL-ONE" not in asce  # ASCE has no highlights
    for marker in ("PRACTICAL-TEXT", "NOTATION-DEF", "SUBSUB-H", "SS"):
        assert marker in asce
    jweia = _text(_pdf(tmp_path / "j", content))
    assert "HL-ONE" in jweia and "PRACTICAL-TEXT" not in jweia
    assert jweia.index("Nomenclature") < jweia.index("1. Intro")
    assert "1.1. Sub" in jweia and "1.1.1. SUBSUB-H" in jweia


@pytest.mark.parametrize(
    "override",
    [
        {"title": None},
        {"keywords": ["a", "b"]},
        {"authors": ["A", "B"]},
        {"sections": None},
        {"sections": [{"heading": "H", "subsections": None}]},
        {"references": ["ok", None]},
        {"tables": [{"caption": "T", "headers": [], "rows": []}]},
        {"tables": [{"caption": "T", "headers": ["a", "b"], "rows": [["1"], ["1", "2", "3"]]}]},
        {"abstract": "Form\x0cfeed and NUL\x00 and <b>markup</b> & more"},
    ],
)
def test_plausible_inputs_do_not_crash(tmp_path, override):
    assert os.path.getsize(_pdf(tmp_path, {**BASE, **override})) > 500


def test_list_keywords_and_highlight_string(tmp_path):
    content = {**BASE, "keywords": ["wind", "ML"], "highlights": "Only one highlight"}
    text = _text(_pdf(tmp_path, content))
    assert "Keywords: wind; ML" in text
    assert "Only one highlight" in text and "• O\n" not in text


def test_greek_and_math_symbols_are_not_mangled(tmp_path):
    # Standard Type-1 fonts used to turn "σ ≤ 0.5 μm" into "s £ 0.5 mm".
    content = {**BASE, "abstract": "Stress σ ≤ 0.5 μm and Δ ± 2"}
    assert "Stress σ ≤ 0.5 μm and Δ ± 2" in _text(_pdf(tmp_path, content))


def test_korean_text_is_rendered(tmp_path):
    content = {**BASE, "title": "철근콘크리트 보의 전단강도", "abstract": "한국어 초록 σ"}
    path = _pdf(tmp_path, content, journal="ksce_jce")
    text = _text(path)
    assert "철근콘크리트 보의 전단강도" in text and "한국어 초록" in text
    assert os.path.basename(path) == "철근콘크리트_보의_전단강도_ksce_jce.pdf"


def test_figure_aspect_ratio_is_preserved(tmp_path):
    # Every figure used to be stretched to 6 x 3.5 in.
    path = _pdf(tmp_path, figures=[_png(tmp_path / "square.png", (300, 300))])
    with pdfplumber.open(path) as pdf:
        (img,) = [im for page in pdf.pages for im in page.images]
    assert img["width"] == pytest.approx(img["height"], rel=0.01)


def test_tall_figure_fits_inside_page(tmp_path):
    path = _pdf(tmp_path, figures=[_png(tmp_path / "tall.png", (200, 2000))])
    with pdfplumber.open(path) as pdf:
        (img,) = [im for page in pdf.pages for im in page.images]
        assert img["height"] < pdf.pages[0].height * 0.7


def test_corrupt_figure_gets_placeholder_instead_of_crash(tmp_path):
    bad = tmp_path / "corrupt.png"
    bad.write_text("not an image", encoding="utf-8")
    content = {**BASE, "figure_captions": ["Fig. 1. Kept caption."]}
    text = _text(_pdf(tmp_path, content, figures=[str(bad)]))
    assert "corrupt.png could not be loaded" in text and "Kept caption" in text


def test_missing_figure_keeps_caption(tmp_path):
    content = {**BASE, "figure_captions": ["Fig. 1. MISSING-CAPTION."]}
    text = _text(_pdf(tmp_path, content, figures=[str(tmp_path / "nope.png"), None]))
    assert "MISSING-CAPTION" in text and "nope.png not found" in text
    assert "Fig. 2." in text  # None entry still numbered


def test_default_caption_follows_journal_style(tmp_path):
    fig = _png(tmp_path / "fig_1_load_path.png")
    assert "Figure 1. Load path." in _text(_pdf(tmp_path, journal="buildings_mdpi", figures=[fig]))
    assert "Fig. 1. Load path." in _text(_pdf(tmp_path / "e", journal="jweia", figures=[fig]))


def test_wide_table_wraps_inside_page(tmp_path):
    table = {
        "caption": "Table 1. Wide.",
        "headers": ["Model", "Description"],
        "rows": [["RF", "A very long description " * 12]],
    }
    path = _pdf(tmp_path, {**BASE, "tables": [table]})
    with pdfplumber.open(path) as pdf:
        width = pdf.pages[0].width
        right_edges = [w["x1"] for page in pdf.pages for w in page.extract_words()]
    assert max(right_edges) < width


def test_page_numbers_are_drawn(tmp_path):
    content = {**BASE, "sections": [{"heading": "Long", "content": "Text. " * 2000}]}
    with pdfplumber.open(_pdf(tmp_path, content)) as pdf:
        assert len(pdf.pages) > 1
        last_lines = [(page.extract_text() or "").splitlines()[-1] for page in pdf.pages]
    assert last_lines[:2] == ["1", "2"]


def test_numbered_and_uppercase_headings_follow_guideline(tmp_path):
    content = {**BASE, "data_availability": "On request."}
    asce = _text(_pdf(tmp_path, content, journal="asce_jse"))
    assert "INTRODUCTION" in asce and "REFERENCES" in asce and "1. " not in asce
    eesd = _text(_pdf(tmp_path / "e", content, journal="eesd"))
    assert "1. INTRODUCTION" in eesd


def test_relative_output_dir_returns_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = generate_pdf(BASE, "jweia", output_dir="rel")
    assert os.path.isabs(path) and os.path.exists(path)


def test_helpers():
    assert _safe_xml(None) == ""
    assert _safe_xml("a < b & c\x00") == "a &lt; b &amp; c"
    assert not _needs_unicode_font("Plain – “quoted” text €")
    assert _needs_unicode_font("σ")
    assert _build_styles()["BodyText"].fontName == "Times-Roman"


@pytest.mark.parametrize("journal_key", available_journal_keys())
def test_full_paper_for_every_journal(tmp_path, journal_key):
    fig = _png(tmp_path / "fig_1_results.png")
    content = {
        **BASE,
        "highlights": ["H1"],
        "tables": [{"caption": "Table 1. T.", "headers": ["a", "b"], "rows": [["1", "2"]]}],
        "figure_captions": ["Fig. 1. Results."],
        "notation": [{"symbol": "x", "definition": "y"}],
        "data_availability": "On request.",
        "acknowledgments": "Thanks.",
    }
    path = _pdf(tmp_path, content, journal=journal_key, figures=[fig])
    text = _text(path)
    assert path.endswith(f"_{journal_key}.pdf")
    assert "Regression Paper" in text and "Results." in text and "Table 1. T." in text


def test_default_output_dir_is_module_constant(tmp_path):
    path = generate_pdf(BASE, "jweia")
    assert os.path.dirname(path) == str(tmp_path / "default_papers")
