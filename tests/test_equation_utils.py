import os

from utils.equation_utils import (
    _latex_to_readable,
    equations_to_text,
    extract_equations,
    number_equations,
    render_equation_image,
    save_equation_image,
)


class TestRenderEquation:
    def test_returns_bytes(self):
        data = render_equation_image(r"\alpha + \beta = \gamma")
        assert isinstance(data, bytes)
        assert len(data) > 100

    def test_png_header(self):
        data = render_equation_image("x^2")
        assert data[:4] == b"\x89PNG"


class TestSaveEquation:
    def test_saves_file(self, tmp_path):
        path = save_equation_image("E = mc^2", "eq_1", output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestExtractEquations:
    def test_inline(self):
        eqs = extract_equations("The value $\\alpha$ is important.")
        assert len(eqs) == 1
        assert eqs[0]["type"] == "inline"
        assert "alpha" in eqs[0]["latex"]

    def test_display(self):
        eqs = extract_equations("The equation is: $$E = mc^2$$")
        assert len(eqs) == 1
        assert eqs[0]["type"] == "display"

    def test_mixed(self):
        text = "Inline $x$ and display $$y = mx + b$$"
        eqs = extract_equations(text)
        assert len(eqs) == 2

    def test_no_equations(self):
        eqs = extract_equations("No equations here.")
        assert len(eqs) == 0

    def test_sorted_by_position(self):
        text = "First $a$ then $$b = c$$ then $d$"
        eqs = extract_equations(text)
        positions = [e["position"] for e in eqs]
        assert positions == sorted(positions)


class TestNumberEquations:
    def test_adds_numbers(self):
        text = "First: $$a = b$$ Second: $$c = d$$"
        result, count = number_equations(text)
        assert "(1)" in result
        assert "(2)" in result
        assert count == 2

    def test_custom_start(self):
        text = "Equation: $$x = y$$"
        result, count = number_equations(text, start_num=5)
        assert "(5)" in result

    def test_no_display_equations(self):
        text = "Only inline $x$."
        result, count = number_equations(text)
        assert count == 0
        assert result == text


class TestEquationsToText:
    def test_converts_inline(self):
        result = equations_to_text("Value $\\alpha$ here")
        assert "α" in result

    def test_converts_display(self):
        result = equations_to_text("Equation: $$E = mc^2$$")
        assert "E = mc^2" in result or "E" in result

    def test_no_equations_unchanged(self):
        text = "No equations here."
        assert equations_to_text(text) == text


class TestLatexToReadable:
    def test_greek_letters(self):
        assert "α" in _latex_to_readable(r"\alpha")
        assert "β" in _latex_to_readable(r"\beta")
        assert "θ" in _latex_to_readable(r"\theta")

    def test_fraction(self):
        result = _latex_to_readable(r"\frac{a}{b}")
        assert "a" in result and "b" in result

    def test_operators(self):
        assert "×" in _latex_to_readable(r"\times")
        assert "±" in _latex_to_readable(r"\pm")
        assert "≈" in _latex_to_readable(r"\approx")
