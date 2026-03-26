import os
import pytest
from utils.latex_generator import generate_latex, _get_document_class, _escape_latex


class TestGetDocumentClass:
    def test_asce_returns_ascelike(self):
        assert "asce" in _get_document_class("asce_jse").lower()

    def test_elsevier_returns_elsarticle(self):
        assert _get_document_class("eng_structures") == "elsarticle"

    def test_aci_returns_article(self):
        assert _get_document_class("aci_sj") == "article"


class TestEscapeLatex:
    def test_escapes_ampersand(self):
        assert _escape_latex("A & B") == r"A \& B"

    def test_escapes_percent(self):
        assert _escape_latex("50%") == r"50\%"

    def test_escapes_underscore(self):
        assert _escape_latex("var_name") == r"var\_name"

    def test_preserves_normal_text(self):
        assert _escape_latex("Hello world") == "Hello world"


class TestGenerateLatex:
    def test_creates_tex_file(self, tmp_path):
        content = {
            "title": "Test Paper",
            "authors": "A. Author",
            "abstract": "Abstract text.",
            "keywords": "kw1; kw2",
            "sections": [{"heading": "Introduction", "content": "Intro text."}],
            "references": ["[1] A. Author, Title, J. Eng. 50 (2024) 1-10."],
        }
        tex_path, bib_path = generate_latex(content, "eng_structures", output_dir=str(tmp_path))
        assert os.path.exists(tex_path)
        assert tex_path.endswith(".tex")

    def test_creates_bib_file(self, tmp_path):
        content = {
            "title": "Test Paper",
            "authors": "A. Author",
            "abstract": "Abstract text.",
            "keywords": "kw1; kw2",
            "sections": [{"heading": "Introduction", "content": "Intro text."}],
            "references": ["[1] A. Author, Title, J. Eng. 50 (2024) 1-10."],
        }
        tex_path, bib_path = generate_latex(content, "eng_structures", output_dir=str(tmp_path))
        assert os.path.exists(bib_path)
        assert bib_path.endswith(".bib")

    def test_tex_contains_title(self, tmp_path):
        content = {
            "title": "My Great Paper",
            "authors": "A. Author",
            "abstract": "Abstract.",
            "keywords": "kw1",
            "sections": [],
            "references": [],
        }
        tex_path, _ = generate_latex(content, "eng_structures", output_dir=str(tmp_path))
        with open(tex_path) as f:
            text = f.read()
        assert "My Great Paper" in text

    def test_tex_contains_documentclass(self, tmp_path):
        content = {
            "title": "Test",
            "authors": "A. Author",
            "abstract": "Abstract.",
            "keywords": "kw1",
            "sections": [],
            "references": [],
        }
        tex_path, _ = generate_latex(content, "eng_structures", output_dir=str(tmp_path))
        with open(tex_path) as f:
            text = f.read()
        assert "\\documentclass" in text
