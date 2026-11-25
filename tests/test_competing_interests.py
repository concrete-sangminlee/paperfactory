"""The competing-interest declaration is rendered by every generator, in guideline order."""

import docx
import pdfplumber
import pytest

from utils.latex_generator import generate_latex
from utils.pdf_generator import generate_pdf
from utils.word_generator import COMPETING_INTEREST_KEYS, generate_word

STATEMENT = "The authors declare no known competing financial interests."


def _paper(**extra):
    paper = {
        "title": "Declaration rendering check",
        "authors": "A. Author",
        "abstract": "Short abstract.",
        "keywords": "a; b",
        "sections": [
            {"heading": "Introduction", "content": "Intro text."},
            {"heading": "Conclusions", "content": "Conclusion text."},
        ],
        "data_availability": "Data available on request.",
        "acknowledgments": "Thanks to the lab.",
        "references": ["[1] A. Author, A title, J. Test. 1 (2024) 1-2."],
    }
    paper.update(extra)
    return paper


def _docx_paragraphs(path):
    return [p.text for p in docx.Document(path).paragraphs if p.text.strip()]


@pytest.mark.parametrize("key", COMPETING_INTEREST_KEYS)
def test_every_accepted_key_is_rendered_in_word(tmp_path, key):
    path = generate_word(_paper(**{key: STATEMENT}), "eng_structures", output_dir=str(tmp_path))
    assert STATEMENT in _docx_paragraphs(path)


def test_word_heading_and_order_follow_guideline(tmp_path):
    # Engineering Structures: Acknowledgements -> Declaration of Competing Interest -> Data
    # Availability -> References
    path = generate_word(
        _paper(conflict_of_interest=STATEMENT), "eng_structures", output_dir=str(tmp_path)
    )
    paras = _docx_paragraphs(path)
    heading = next(p for p in paras if "competing interest" in p.lower())
    assert heading.lower().endswith("declaration of competing interest")
    order = [
        paras.index(p) for p in ("Thanks to the lab.", STATEMENT, "Data available on request.")
    ]
    assert order == sorted(order)


def test_mdpi_uses_its_own_heading(tmp_path):
    path = generate_word(
        _paper(conflict_of_interest=STATEMENT), "buildings_mdpi", output_dir=str(tmp_path)
    )
    paras = _docx_paragraphs(path)
    assert any(p.lower().endswith("conflicts of interest") for p in paras)


def test_absent_statement_adds_no_heading(tmp_path):
    path = generate_word(_paper(), "eng_structures", output_dir=str(tmp_path))
    assert not any("competing interest" in p.lower() for p in _docx_paragraphs(path))


def test_pdf_renders_statement(tmp_path):
    path = generate_pdf(_paper(competing_interests=STATEMENT), "jbe", output_dir=str(tmp_path))
    with pdfplumber.open(path) as pdf:
        text = " ".join((page.extract_text() or "") for page in pdf.pages)
    assert "Declaration of Competing Interest" in text
    assert "competing financial interests" in " ".join(text.split())


def test_latex_renders_statement(tmp_path):
    tex, _ = generate_latex(
        _paper(declaration_of_competing_interest=STATEMENT), "thin_walled", output_dir=str(tmp_path)
    )
    with open(tex, encoding="utf-8") as f:
        source = f.read()
    assert r"\section*{Declaration of Competing Interest}" in source
    assert STATEMENT in source
    assert source.index("Declaration of Competing Interest") < source.index(r"\bibliography{")
