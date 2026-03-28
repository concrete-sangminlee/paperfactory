import os

from utils.latex_generator import _escape_latex, _get_document_class, generate_latex


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

    def test_escapes_backslash(self):
        assert _escape_latex("a\\b") == r"a\textbackslash{}b"

    def test_escapes_curly_braces(self):
        assert _escape_latex("{x}") == r"\{x\}"

    def test_escapes_hash(self):
        assert _escape_latex("#1") == r"\#1"

    def test_escapes_tilde(self):
        assert _escape_latex("~") == r"\textasciitilde{}"

    def test_escapes_caret(self):
        assert _escape_latex("^") == r"\textasciicircum{}"

    def test_preserves_inline_math(self):
        assert _escape_latex("value $\\alpha$ here") == r"value $\alpha$ here"

    def test_escapes_dollar_outside_math(self):
        assert _escape_latex("costs $5") == r"costs \$5"

    def test_escapes_multiple_dollars_outside_math(self):
        result = _escape_latex("costs $5, also $10 extra")
        assert r"\$5" in result
        assert r"\$10" in result

    def test_backslash_does_not_double_escape_braces(self):
        result = _escape_latex("\\")
        assert result == r"\textbackslash{}"
        assert r"\textbackslash\{" not in result


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
