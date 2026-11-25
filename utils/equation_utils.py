"""Equation rendering utilities for Word and PDF output.

Supports inline LaTeX-style equations in paper content by converting
them to appropriate format for each output type.

Recognised delimiters: display ``$$...$$`` and ``\\[...\\]``; inline ``$...$``
and ``\\(...\\)``.  An escaped dollar (``\\$5``) is a literal dollar sign, never a
delimiter.  Following Pandoc, an inline ``$`` must not be followed by a space
when opening, nor preceded by a space or followed by a digit when closing, so
"costs $5 and $10" contains no equation.
"""

import os
import re
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Pattern for display equations: $$...$$ (group 1) or \[...\] (group 2)
DISPLAY_EQ_PATTERN = re.compile(
    r"(?<!\\)\$\$((?:[^$\\]|\\.)+?)\$\$|(?<!\\)\\\[(.+?)\\\]", re.DOTALL
)
# Pattern for inline equations: $...$ (group 1) or \(...\) (group 2)
INLINE_EQ_PATTERN = re.compile(
    r"(?<![\\$])\$(?![\s$])((?:[^$\\]|\\.)*?(?:[^$\s\\]|\\.))\$(?![$\d])"
    r"|(?<!\\)\\\((.+?)\\\)",
    re.DOTALL,
)
# A display equation optionally followed by an existing equation number " (n)".
_NUMBERED_DISPLAY_PATTERN = re.compile(
    rf"(?:{DISPLAY_EQ_PATTERN.pattern})(?:[ \t]*\(\d{{1,3}}\))?", re.DOTALL
)


def _match_latex(match: re.Match) -> str:
    """LaTeX source of a DISPLAY/INLINE pattern match (whichever group matched)."""
    latex = match.group(1)
    return latex if latex is not None else (match.group(2) or "")


def _strip_delimiters(latex_str: str) -> str:
    latex = str(latex_str or "").strip()
    for opening, closing in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if len(latex) >= len(opening) + len(closing) and (
            latex.startswith(opening) and latex.endswith(closing)
        ):
            return latex[len(opening) : len(latex) - len(closing)].strip()
    return latex


def render_equation_image(latex_str: str, fontsize: int = 14, dpi: int = 300) -> bytes:
    """Render a LaTeX equation string to PNG image bytes.

    Parameters
    ----------
    latex_str : str
        LaTeX math expression (surrounding ``$``/``$$``/``\\[ \\]`` delimiters
        are tolerated and removed).
    fontsize : int
        Font size for rendering.
    dpi : int
        Image resolution.

    Returns
    -------
    bytes
        PNG image data.

    Raises
    ------
    ValueError
        If the expression is empty or cannot be parsed by matplotlib mathtext.
    """
    latex = _strip_delimiters(latex_str)
    if not latex:
        raise ValueError("Cannot render an empty equation")

    # A standalone Figure (not pyplot) is never registered globally, so nothing
    # leaks when mathtext raises; layout="none" ignores a global
    # constrained-layout setting (e.g. from figure_utils.setup_style()).
    fig = Figure(figsize=(0.01, 0.01), layout="none")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.axis("off")
    text = ax.text(
        0,
        0,
        f"${latex}$",
        fontsize=fontsize,
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="left",
    )

    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer)

    # Resize figure to fit text (the window extent is in pixels at fig.dpi)
    fig.set_size_inches(bbox.width / fig.dpi + 0.1, bbox.height / fig.dpi + 0.1)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
    return buf.getvalue()


def save_equation_image(
    latex_str: str, name: str, output_dir: str = None, fontsize: int = 14, dpi: int = 300
) -> str:
    """Render and save equation as PNG file.

    Returns path to saved image.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures"
        )

    img_data = render_equation_image(latex_str, fontsize, dpi)
    path = os.path.join(output_dir, f"{name}.png")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(img_data)
    return path


def extract_equations(text: str) -> list:
    """Extract all equations from text.

    Returns list of dicts with 'latex', 'type' (inline/display), 'position',
    'original', sorted by position.
    """
    text = text or ""
    equations = []

    masked = list(text)
    for match in DISPLAY_EQ_PATTERN.finditer(text):
        equations.append(
            {
                "latex": _match_latex(match).strip(),
                "type": "display",
                "position": match.start(),
                "original": match.group(),
            }
        )
        # Blank out display equations so inline matching cannot overlap them.
        masked[match.start() : match.end()] = " " * (match.end() - match.start())

    for match in INLINE_EQ_PATTERN.finditer("".join(masked)):
        latex = _match_latex(match).strip()
        if latex:
            equations.append(
                {
                    "latex": latex,
                    "type": "inline",
                    "position": match.start(),
                    "original": text[match.start() : match.end()],
                }
            )

    equations.sort(key=lambda x: x["position"])
    return equations


def number_equations(text: str, start_num: int = 1) -> tuple:
    """Add equation numbers to display equations in text.

    Existing numbers directly after an equation ("$$a=b$$ (3)") are replaced,
    so the function is idempotent and renumbers after edits.  The original
    delimiter style ($$...$$ or \\[...\\]) is preserved.

    Returns (modified_text, equation_count).
    """
    counter = [start_num]

    def replacer(match):
        num = counter[0]
        counter[0] += 1
        if match.group(1) is not None:
            return f"$${match.group(1).strip()}$$ ({num})"
        return f"\\[{match.group(2).strip()}\\] ({num})"

    modified = _NUMBERED_DISPLAY_PATTERN.sub(replacer, text)
    return modified, counter[0] - start_num


def equations_to_text(text: str) -> str:
    """Convert LaTeX equations to plain text representation for non-LaTeX outputs.

    Useful for Word documents where LaTeX rendering is not available.  Escaped
    dollars (``\\$``) outside equations become plain ``$``.
    """

    # Display equations: add visual separation
    def display_repl(match):
        readable = _latex_to_readable(_match_latex(match).strip())
        return f"\n    {readable}\n"

    result = DISPLAY_EQ_PATTERN.sub(display_repl, text)

    # Inline equations: convert to readable
    def inline_repl(match):
        latex = _match_latex(match)
        if latex:
            return _latex_to_readable(latex.strip())
        return match.group()

    result = INLINE_EQ_PATTERN.sub(inline_repl, result)
    return result.replace("\\$", "$")


_COMMANDS = {
    # Greek letters
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    # Operators, relations and symbols
    "cdot": "·",
    "cdots": "⋯",
    "ldots": "…",
    "dots": "…",
    "times": "×",
    "div": "÷",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "sim": "~",
    "simeq": "≃",
    "equiv": "≡",
    "propto": "∝",
    "pm": "±",
    "mp": "∓",
    "infty": "∞",
    "sum": "Σ",
    "prod": "Π",
    "int": "∫",
    "oint": "∮",
    "partial": "∂",
    "nabla": "∇",
    "in": "∈",
    "forall": "∀",
    "exists": "∃",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "leftrightarrow": "↔",
    "Leftrightarrow": "⇔",
    "circ": "∘",
    "degree": "°",
    "prime": "′",
    "quad": " ",
    "qquad": " ",
    # Size/spacing commands that only affect typesetting
    "left": "",
    "right": "",
    "big": "",
    "Big": "",
    "bigg": "",
    "Bigg": "",
    "displaystyle": "",
    "limits": "",
    "nolimits": "",
}
# Commands whose single brace argument is kept as plain text.
_TEXT_COMMANDS = (
    "text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|"
    "boldsymbol|operatorname|mbox"
)
# Accent commands -> Unicode combining characters.
_ACCENTS = {
    "bar": "\u0304",
    "overline": "\u0305",
    "hat": "\u0302",
    "widehat": "\u0302",
    "tilde": "\u0303",
    "widetilde": "\u0303",
    "dot": "\u0307",
    "ddot": "\u0308",
    "vec": "\u20d7",
}

_BRACED = r"\{([^{}]*)\}"
_FRAC_RE = re.compile(r"\\[dt]?frac\s*" + _BRACED + r"\s*" + _BRACED)
_SQRT_RE = re.compile(r"\\sqrt\s*(?:\[([^\[\]]*)\])?\s*" + _BRACED)
_TEXT_RE = re.compile(rf"\\(?:{_TEXT_COMMANDS})\s*" + _BRACED)
_ACCENT_RE = re.compile(r"\\(" + "|".join(_ACCENTS) + r")(?![a-zA-Z])\s*(?:" + _BRACED + r"|(\w))")
_SUPERSCRIPT_RE = re.compile(r"\^\s*" + _BRACED)
_SUBSCRIPT_RE = re.compile(r"_\s*" + _BRACED)
_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")


_ESCAPE_PLACEHOLDERS = {c: chr(0xE000 + i) for i, c in enumerate("{}%$&#_")}
_STRUCTURAL_COMMANDS = frozenset(
    ["frac", "dfrac", "tfrac", "sqrt", *_TEXT_COMMANDS.split("|"), *_ACCENTS]
)


def _resolve_command(match: re.Match) -> str:
    """Map a whole command name (\\cdots is not \\cdot + "s") to readable text.

    Structural commands are left for the brace pass; unknown ones such as
    ``\\sin`` or ``\\log`` keep their name (plus a space before a following letter).
    """
    name = match.group(1)
    if name in _COMMANDS:
        return _COMMANDS[name]
    if name in _STRUCTURAL_COMMANDS:
        return match.group()
    following = match.string[match.end() : match.end() + 1]
    return name + (" " if following.isalpha() or following == "\\" else "")


def _latex_to_readable(latex: str) -> str:
    """Convert common LaTeX math to readable text."""

    def accent(m):
        body = m.group(2) if m.group(2) is not None else m.group(3)
        return body + _ACCENTS[m.group(1)]

    def superscript(m):
        body = m.group(1).strip()
        return f"^{body}" if len(body) <= 1 else f"^({body})"

    def sqrt(m):
        index = (m.group(1) or "").strip()
        return f"{index}√({m.group(2)})"

    result = re.sub(r"\\(?:left|right)\.", "", latex)  # invisible delimiters
    # Symbol commands first, so that text produced below (e.g. an accented "u")
    # cannot merge with a preceding command name ("\\omega\\dot{u}").
    result = _COMMAND_RE.sub(_resolve_command, result)
    # Resolve brace constructs from the innermost outwards until nothing changes,
    # so that nested forms like \sqrt{\frac{a}{b}} or \frac{x^{2}}{y} work.
    for _ in range(50):
        previous = result
        result = _TEXT_RE.sub(r"\1", result)
        result = _ACCENT_RE.sub(accent, result)
        result = _SUPERSCRIPT_RE.sub(superscript, result)
        result = _SUBSCRIPT_RE.sub(r"_\1", result)
        result = _FRAC_RE.sub(r"(\1)/(\2)", result)
        result = _SQRT_RE.sub(sqrt, result)
        if result == previous:
            break

    # Escaped characters survive the removal of grouping braces / alignment "&".
    result = re.sub(r"\\([{}%$&#_])", lambda m: _ESCAPE_PLACEHOLDERS[m.group(1)], result)
    result = re.sub(r"\\[,;:! ]", " ", result)
    result = result.replace("\\\\", " ")
    # Structural commands left without a brace argument ("\\frac12") keep their name.
    result = _COMMAND_RE.sub(lambda m: m.group(1), result)
    result = result.replace("{", "").replace("}", "").replace("&", "").replace("~", " ")
    for char, placeholder in _ESCAPE_PLACEHOLDERS.items():
        result = result.replace(placeholder, char)
    return re.sub(r"[ \t]{2,}", " ", result).strip()
