import os
import warnings

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.equation_utils import (
    _latex_to_readable,
    equations_to_text,
    extract_equations,
    number_equations,
    render_equation_image,
    save_equation_image,
)
from utils.figure_utils import setup_style


@pytest.fixture(autouse=True)
def _isolate_matplotlib_state():
    with matplotlib.rc_context():
        yield


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


class TestExtractEquationsRegressions:
    def test_escaped_dollars_are_not_delimiters(self):
        assert extract_equations(r"The cost rose from \$5 to \$10 per unit.") == []

    def test_escaped_dollar_next_to_real_equation(self):
        eqs = extract_equations(r"Costs \$3 and the stress $f_c$ is known.")
        assert [(e["type"], e["latex"]) for e in eqs] == [("inline", "f_c")]

    def test_currency_without_escapes(self):
        assert extract_equations("It costs $5 and $10 respectively.") == []

    def test_bracket_delimiters(self):
        eqs = extract_equations(r"Display \[ E = mc^2 \] here and \( x \) inline.")
        assert [(e["type"], e["latex"]) for e in eqs] == [("display", "E = mc^2"), ("inline", "x")]
        assert eqs[0]["original"] == r"\[ E = mc^2 \]"

    def test_multiline_display(self):
        eqs = extract_equations("$$\na = b\n+ c\n$$")
        assert eqs[0]["latex"] == "a = b\n+ c"

    def test_adjacent_inline(self):
        eqs = extract_equations("pair $a$,$b$.")
        assert [e["latex"] for e in eqs] == ["a", "b"]

    def test_inline_positions_and_originals(self):
        text = "x $$a$$ y $b$"
        eqs = extract_equations(text)
        for e in eqs:
            assert text[e["position"] :].startswith(e["original"])


class TestNumberEquationsRegressions:
    def test_idempotent(self):
        once, n1 = number_equations("A: $$a=b$$ B: $$c=d$$")
        twice, n2 = number_equations(once)
        assert twice == once == "A: $$a=b$$ (1) B: $$c=d$$ (2)"
        assert n1 == n2 == 2

    def test_renumbers_after_insertion(self):
        text, _ = number_equations("$$a$$ (1) and $$b$$ (2)")
        text, _ = number_equations("$$z$$ " + text)
        assert text == "$$z$$ (1) $$a$$ (2) and $$b$$ (3)"

    def test_bracket_display_keeps_its_delimiters(self):
        text, count = number_equations(r"See \[ a = b \] now.")
        assert text == r"See \[a = b\] (1) now."
        assert count == 1

    def test_year_in_parentheses_is_not_an_equation_number(self):
        text, _ = number_equations("$$a$$ (2010) showed")
        assert text == "$$a$$ (1) (2010) showed"


class TestEquationsToTextRegressions:
    def test_escaped_dollars_become_plain(self):
        assert equations_to_text(r"Costs \$5 and $\alpha$.") == "Costs $5 and α."

    def test_bracket_display(self):
        assert equations_to_text(r"Eq: \[ \sigma = E \varepsilon \]") == "Eq: \n    σ = E ε\n"


class TestLatexToReadableRegressions:
    @pytest.mark.parametrize(
        "latex,expected",
        [
            (r"a \rightarrow b", "a → b"),  # was "a arrow b" (prefix match of \right)
            (r"\leftarrow", "←"),
            (r"x_1, \cdots, x_n", "x_1, ⋯, x_n"),  # was "·s"
            (r"\sqrt{\frac{a}{b}}", "√((a)/(b))"),  # nested braces
            (r"\frac{x^{2}}{y}", "(x^2)/(y)"),
            (r"e^{-t/T}", "e^(-t/T)"),
            (r"\ddot{u} + 2\xi\omega\dot{u}", "ü + 2ξωu̇"),
            (r"\varepsilon_{y}", "ε_y"),
            (r"\sin(\theta)", "sin(θ)"),
            (r"\sin\theta", "sin θ"),
            (r"f_{\text{max}}", "f_max"),
            (r"\mathbf{K}\mathbf{u} = \mathbf{F}", "Ku = F"),
            (r"\left( \frac{a}{b} \right)", "( (a)/(b) )"),
            (r"\left\{ x \right\}", "{ x }"),
            (r"100\%", "100%"),
            (r"\sqrt[3]{x}", "3√(x)"),
            (r"\nu \leq 0.5", "ν ≤ 0.5"),
        ],
    )
    def test_conversion(self, latex, expected):
        assert _latex_to_readable(latex) == expected


class TestRenderEquationRegressions:
    def test_no_figure_leak_on_parse_error(self):
        before = set(plt.get_fignums())
        with pytest.raises(ValueError):
            render_equation_image(r"\frac{a}{")
        assert set(plt.get_fignums()) == before

    def test_no_layout_warning_after_setup_style(self):
        setup_style()  # enables constrained layout globally
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data = render_equation_image("x^2")
        assert data[:4] == b"\x89PNG"

    def test_delimiters_are_tolerated(self):
        assert render_equation_image("$x^2$")[:4] == b"\x89PNG"
        assert render_equation_image(r"\[ a = b \]")[:4] == b"\x89PNG"

    def test_empty_equation_raises(self):
        with pytest.raises(ValueError, match="empty"):
            render_equation_image("  ")

    def test_save_creates_directory(self, tmp_path):
        path = save_equation_image("a+b", "eq", output_dir=str(tmp_path / "new" / "dir"))
        assert os.path.exists(path)
