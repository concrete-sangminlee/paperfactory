"""Regression tests for bugs fixed in utils/word_generator.py.

Each test documents the symptom it guards against; all of them failed on the
previous implementation.
"""

import os

import pytest
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from PIL import Image

from utils import word_generator
from utils.guidelines import UnknownJournalError, available_journal_keys
from utils.word_generator import generate_word


@pytest.fixture(autouse=True)
def _no_writes_to_outputs(tmp_path, monkeypatch):
    """Redirect the default output directory so no test can write into outputs/."""
    for module in (word_generator,):
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


def _texts(path):
    return [p.text for p in Document(path).paragraphs]


def _index(texts, needle):
    return next(i for i, t in enumerate(texts) if needle in t)


def _body_paragraph(doc, text):
    return next(p for p in doc.paragraphs if p.text == text)


# ---------------------------------------------------------------------------
# Crashes and mis-rendering on plausible input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"title": None},
        {"sections": None},
        {"sections": [{"heading": None, "content": "x"}]},
        {"sections": [{"heading": "H", "content": None, "subsections": None}]},
        {"references": ["[1] ok", None, ""]},
        {"tables": [{"caption": "T", "headers": ["a"], "rows": [["1", "2", "3"]]}]},
        {"tables": [{"caption": "T", "headers": [1, 2.5], "rows": [[1, None]]}]},
        {"tables": [{"caption": "T", "rows": [[]]}], "notation": None},
        {"abstract": "Text copied from a PDF\x0cnext page\x00."},
    ],
    ids=[
        "title-none",
        "sections-none",
        "heading-none",
        "subsections-none",
        "reference-none",
        "row-longer-than-header",
        "non-str-cells",
        "empty-row",
        "control-chars",
    ],
)
def test_plausible_inputs_do_not_crash(tmp_path, override):
    path = generate_word({**BASE, **override}, "jweia", output_dir=str(tmp_path))
    assert os.path.exists(path)
    Document(path)  # still a valid .docx


def test_figure_list_with_none_entry_does_not_crash(tmp_path):
    path = generate_word(BASE, "jweia", figures=[None], output_dir=str(tmp_path))
    assert "Fig. 1." in _texts(path)


def test_list_keywords_and_authors_are_joined_with_separators(tmp_path):
    # Previously rendered as "windML" / "A. AuthorB. Author".
    content = {**BASE, "keywords": ["wind", "ML"], "authors": ["A. Author", "B. Author"]}
    texts = _texts(generate_word(content, "jweia", output_dir=str(tmp_path)))
    assert "Keywords: wind; ML" in texts
    assert "A. Author, B. Author" in texts


def test_highlights_string_is_not_split_into_characters(tmp_path):
    content = {**BASE, "highlights": "Only one highlight"}
    doc = Document(generate_word(content, "jweia", output_dir=str(tmp_path)))
    bullets = [p.text for p in doc.paragraphs if p.style.name == "List Bullet"]
    assert bullets == ["Only one highlight"]


def test_table_row_longer_than_headers_keeps_all_cells(tmp_path):
    content = {**BASE, "tables": [{"headers": ["A"], "rows": [["1", "2"]]}]}
    doc = Document(generate_word(content, "jweia", output_dir=str(tmp_path)))
    assert [c.text for c in doc.tables[0].rows[1].cells] == ["1", "2"]


def test_table_without_headers_is_still_rendered(tmp_path):
    content = {**BASE, "tables": [{"caption": "Table 1. X.", "rows": [["r1", "r2"]]}]}
    doc = Document(generate_word(content, "jweia", output_dir=str(tmp_path)))
    assert [c.text for c in doc.tables[0].rows[0].cells] == ["r1", "r2"]


# ---------------------------------------------------------------------------
# Guideline rules that were ignored or misapplied
# ---------------------------------------------------------------------------


def test_single_line_spacing_is_honoured(tmp_path):
    # buildings_mdpi specifies "single"; it used to fall through to double.
    doc = Document(generate_word(BASE, "buildings_mdpi", output_dir=str(tmp_path)))
    assert _body_paragraph(doc, "Body text.").paragraph_format.line_spacing == 1.0


def test_font_alternatives_use_first_font(tmp_path):
    # jweia: "Times New Roman or Arial" was written verbatim as the font name.
    doc = Document(generate_word(BASE, "jweia", output_dir=str(tmp_path)))
    fonts = {r.font.name for p in doc.paragraphs for r in p.runs if r.text}
    assert fonts == {"Times New Roman"}


def test_title_is_larger_than_body_text(tmp_path):
    # The +2 pt title size used to be overwritten by _apply_font.
    doc = Document(generate_word(BASE, "asce_jse", output_dir=str(tmp_path)))
    title_run = _body_paragraph(doc, "Regression Paper").runs[0]
    assert title_run.font.size == Pt(14)


@pytest.mark.parametrize(
    "journal,expected",
    [
        ("asce_jse", "Fig. 1. Model comparison."),
        ("aci_sj", "Fig. 1—Model comparison"),
        ("eng_structures", "Fig. 1. Model comparison."),
        ("buildings_mdpi", "Figure 1. Model comparison."),
    ],
)
def test_default_caption_does_not_leak_guideline_placeholder(tmp_path, journal, expected):
    # Used to produce "Fig. 1. Caption text in sentence case." / "Fig. 1—Description".
    fig = _png(tmp_path / "fig_1_model_comparison.png")
    texts = _texts(generate_word(BASE, journal, figures=[fig], output_dir=str(tmp_path)))
    assert expected in texts
    assert not any("Description" in t or "Caption text" in t for t in texts)


def test_surplus_captions_are_kept(tmp_path):
    fig = _png(tmp_path / "a.png")
    content = {**BASE, "figure_captions": ["Fig. 1. A.", "Fig. 2. B."]}
    texts = _texts(generate_word(content, "jweia", figures=[fig], output_dir=str(tmp_path)))
    assert "Fig. 1. A." in texts and "Fig. 2. B." in texts


def test_captions_without_figures_are_kept(tmp_path):
    content = {**BASE, "figure_captions": ["Fig. 1. A."]}
    assert "Fig. 1. A." in _texts(generate_word(content, "jweia", output_dir=str(tmp_path)))


def test_missing_figure_gets_visible_placeholder(tmp_path):
    texts = _texts(
        generate_word(BASE, "jweia", figures=[str(tmp_path / "nope.png")], output_dir=str(tmp_path))
    )
    assert any("nope.png not found" in t for t in texts)


def test_unreadable_figure_gets_placeholder(tmp_path):
    bad = tmp_path / "corrupt.png"
    bad.write_text("not an image", encoding="utf-8")
    texts = _texts(generate_word(BASE, "jweia", figures=[str(bad)], output_dir=str(tmp_path)))
    assert any("could not be embedded" in t for t in texts)


def test_tall_figure_fits_on_the_page(tmp_path):
    # Width was fixed at 5.5 in, so a portrait figure ran off the page.
    fig = _png(tmp_path / "tall.png", size=(300, 1200))
    doc = Document(generate_word(BASE, "asce_jse", figures=[fig], output_dir=str(tmp_path)))
    shape = doc.inline_shapes[0]
    usable_height = doc.sections[0].page_height - 2 * doc.sections[0].top_margin
    assert shape.height <= usable_height
    assert abs(shape.height / shape.width - 4.0) < 0.01  # aspect ratio preserved


def test_tables_and_figures_follow_references_when_guideline_says_end(tmp_path):
    # ACI lists References, Notation, Tables, Figure Captions, Figures in that order.
    content = {
        **BASE,
        "tables": [{"caption": "Table 1—X", "headers": ["a"], "rows": [["1"]]}],
        "figure_captions": ["Fig. 1—Y"],
        "notation": [{"symbol": "f_c", "definition": "strength"}],
        "acknowledgments": "Thanks.",
    }
    texts = _texts(generate_word(content, "aci_sj", output_dir=str(tmp_path)))
    order = [_index(texts, h) for h in ("ACKNOWLEDGMENTS", "REFERENCES", "NOTATION", "TABLES")]
    assert order == sorted(order)
    assert _index(texts, "TABLES") < _index(texts, "FIGURES")


def test_embedded_placement_keeps_floats_before_back_matter(tmp_path):
    content = {
        **BASE,
        "tables": [{"caption": "Table 1. X.", "headers": ["a"], "rows": [["1"]]}],
        "data_availability": "On request.",
    }
    texts = _texts(generate_word(content, "buildings_mdpi", output_dir=str(tmp_path)))
    assert _index(texts, "Table 1. X.") < _index(texts, "Data Availability Statement")
    assert _index(texts, "Data Availability Statement") < _index(texts, "References")


def test_back_matter_order_and_spelling_follow_guideline(tmp_path):
    content = {**BASE, "data_availability": "On request.", "acknowledgments": "Thanks."}
    # eng_structures lists Acknowledgements before Data Availability Statement.
    texts = _texts(generate_word(content, "eng_structures", output_dir=str(tmp_path)))
    assert _index(texts, "Acknowledgements") < _index(texts, "Data Availability Statement")
    # ASCE puts the Data Availability Statement before Acknowledgments.
    texts = _texts(generate_word(content, "asce_jse", output_dir=str(tmp_path / "asce")))
    assert _index(texts, "DATA AVAILABILITY STATEMENT") < _index(texts, "ACKNOWLEDGMENTS")


def test_nomenclature_precedes_introduction_when_guideline_says_so(tmp_path):
    content = {**BASE, "notation": [{"symbol": "U", "definition": "wind speed"}]}
    texts = _texts(generate_word(content, "jweia", output_dir=str(tmp_path)))
    assert _index(texts, "Nomenclature") < _index(texts, "Introduction")
    texts = _texts(generate_word(content, "asce_jse", output_dir=str(tmp_path / "asce")))
    assert _index(texts, "INTRODUCTION") < _index(texts, "NOTATION")


def test_numbered_journals_get_numbered_headings(tmp_path):
    content = {
        **BASE,
        "sections": [
            {
                "heading": "Introduction",
                "content": "x",
                "subsections": [
                    {
                        "heading": "Background",
                        "subsections": [{"heading": "Details", "content": "y"}],
                    }
                ],
            },
            {"heading": "Conclusions", "content": "z"},
            {"heading": "Acknowledgements", "content": "t", "subsections": [{"heading": "Sub"}]},
        ],
    }
    texts = _texts(generate_word(content, "eng_structures", output_dir=str(tmp_path)))
    for heading in ("1. Introduction", "1.1. Background", "1.1.1. Details", "2. Conclusions"):
        assert heading in texts
    assert "Acknowledgements" in texts and "Sub" in texts  # back matter stays unnumbered
    texts = _texts(generate_word(content, "asce_jse", output_dir=str(tmp_path / "asce")))
    assert "INTRODUCTION" in texts and "Background" in texts  # ASCE: unnumbered


def test_existing_manual_numbers_are_not_doubled(tmp_path):
    content = {**BASE, "sections": [{"heading": "1. Introduction", "content": "x"}]}
    texts = _texts(generate_word(content, "eng_structures", output_dir=str(tmp_path)))
    assert "1. Introduction" in texts and "1. 1. Introduction" not in texts


def test_eesd_level2_headings_are_italic(tmp_path):
    content = {
        **BASE,
        "sections": [{"heading": "Method", "subsections": [{"heading": "Ground motions"}]}],
    }
    doc = Document(generate_word(content, "eesd", output_dir=str(tmp_path)))
    run = _body_paragraph(doc, "1.1. Ground motions").runs[0]
    assert run.italic and not run.bold
    assert "1. METHOD" in [p.text for p in doc.paragraphs]


def test_line_and_page_numbers_follow_guideline(tmp_path):
    asce = Document(generate_word(BASE, "asce_jse", output_dir=str(tmp_path)))
    assert asce.sections[0]._sectPr.find(qn("w:lnNumType")) is not None
    footer_xml = asce.sections[0].footer._element.xml
    assert "PAGE" in footer_xml and "fldChar" in footer_xml
    ksce = Document(generate_word(BASE, "ksce_jce", output_dir=str(tmp_path / "k")))
    assert ksce.sections[0]._sectPr.find(qn("w:lnNumType")) is None  # line_numbers: false


def test_asce_body_is_justified_with_first_line_indent(tmp_path):
    doc = Document(generate_word(BASE, "asce_jse", output_dir=str(tmp_path)))
    body = _body_paragraph(doc, "Body text.")
    assert body.alignment == 3  # WD_ALIGN_PARAGRAPH.JUSTIFY
    assert body.paragraph_format.first_line_indent.inches == pytest.approx(0.5)


def test_korean_text_gets_east_asian_font(tmp_path):
    content = {**BASE, "title": "철근콘크리트 보의 전단강도", "sections": []}
    doc = Document(generate_word(content, "ksce_jce", output_dir=str(tmp_path)))
    run = _body_paragraph(doc, "철근콘크리트 보의 전단강도").runs[0]
    assert run._element.rPr.rFonts.get(qn("w:eastAsia")) == "Batang"
    assert os.path.basename(
        generate_word(content, "ksce_jce", output_dir=str(tmp_path))
    ).startswith("철근콘크리트")


@pytest.mark.parametrize("journal,expected", [("jweia", "Table 1"), ("asce_jse", "Table 1.")])
def test_missing_table_caption_gets_journal_style_default(tmp_path, journal, expected):
    # Used to be an empty bold paragraph.  JWEIA style is "Table {n} Description."
    content = {**BASE, "tables": [{"headers": ["a"], "rows": [["1"]]}]}
    assert expected in _texts(generate_word(content, journal, output_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("title", ["???", "", None, "   "])
def test_title_without_word_characters_gets_fallback_filename(tmp_path, title):
    path = generate_word({**BASE, "title": title}, "jweia", output_dir=str(tmp_path))
    assert os.path.basename(path) == "paper_jweia.docx"


def test_alias_journal_key_uses_canonical_key_in_filename(tmp_path):
    path = generate_word(BASE, " JWEIA.json ", output_dir=str(tmp_path))
    assert os.path.basename(path) == "Regression_Paper_jweia.docx"
    path = generate_word(BASE, "jcsr", output_dir=str(tmp_path))
    assert os.path.basename(path) == "Regression_Paper_steel_comp_struct.docx"


def test_long_unsafe_title_is_sanitised(tmp_path):
    title = 'A/very\\long:title*with?"<unsafe>|chars ' * 5
    path = generate_word({**BASE, "title": title}, "jweia", output_dir=str(tmp_path))
    name = os.path.basename(path)
    stem = name[: -len("_jweia.docx")]
    assert len(stem) <= 50
    assert not set('/\\:*?"<>|') & set(name)


def test_relative_output_dir_returns_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = generate_word(BASE, "jweia", output_dir="papers")
    assert os.path.isabs(path)
    assert os.path.exists(path)


def test_unknown_journal_raises_before_creating_output_dir(tmp_path):
    target = tmp_path / "never"
    with pytest.raises(UnknownJournalError):
        generate_word(BASE, "no_such_journal", output_dir=str(target))
    assert not target.exists()


def test_non_dict_content_raises_type_error(tmp_path):
    with pytest.raises(TypeError):
        generate_word(["not", "a", "dict"], "jweia", output_dir=str(tmp_path))


@pytest.mark.parametrize("journal_key", available_journal_keys())
def test_full_paper_for_every_journal(tmp_path, journal_key):
    fig = _png(tmp_path / "fig_1_results.png")
    content = {
        **BASE,
        "highlights": ["H1", "H2"],
        "practical_applications": "Useful.",
        "tables": [{"caption": "Table 1. T.", "headers": ["a", "b"], "rows": [["1", "2"]]}],
        "figure_captions": ["Fig. 1. Results."],
        "notation": [{"symbol": "x", "definition": "y"}],
        "data_availability": "On request.",
        "acknowledgments": "Thanks.",
    }
    path = generate_word(content, journal_key, figures=[fig], output_dir=str(tmp_path))
    doc = Document(path)
    texts = [p.text for p in doc.paragraphs]
    assert len(doc.inline_shapes) == 1 and len(doc.tables) == 2
    assert any("Results." in t for t in texts)
    assert any(t.lower() == "references" for t in texts)


def test_default_output_dir_is_module_constant(tmp_path):
    path = generate_word(BASE, "jweia")
    assert os.path.dirname(path) == str(tmp_path / "default_papers")
