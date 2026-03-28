"""Equation rendering utilities for Word and PDF output.

Supports inline LaTeX-style equations in paper content by converting
them to appropriate format for each output type.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
from io import BytesIO

import matplotlib.pyplot as plt

# Pattern for display equations: $$...$$
DISPLAY_EQ_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Pattern for inline equations: $...$
INLINE_EQ_PATTERN = re.compile(r"(?<!\$)\$([^$\s][^$]*?[^$\s])\$(?!\$)|\$([^$\s])\$")


def render_equation_image(latex_str: str, fontsize: int = 14, dpi: int = 300) -> bytes:
    """Render a LaTeX equation string to PNG image bytes.

    Parameters
    ----------
    latex_str : str
        LaTeX math expression (without $ delimiters).
    fontsize : int
        Font size for rendering.
    dpi : int
        Image resolution.

    Returns
    -------
    bytes
        PNG image data.
    """
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    ax.axis("off")
    text = ax.text(
        0,
        0,
        f"${latex_str}$",
        fontsize=fontsize,
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="left",
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer)

    # Resize figure to fit text
    fig_width = bbox.width / dpi + 0.1
    fig_height = bbox.height / dpi + 0.1
    fig.set_size_inches(fig_width, fig_height)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


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
    os.makedirs(output_dir, exist_ok=True)

    img_data = render_equation_image(latex_str, fontsize, dpi)
    path = os.path.join(output_dir, f"{name}.png")
    with open(path, "wb") as f:
        f.write(img_data)
    return path


def extract_equations(text: str) -> list:
    """Extract all equations from text.

    Returns list of dicts with 'latex', 'type' (inline/display), 'position'.
    """
    equations = []

    for match in DISPLAY_EQ_PATTERN.finditer(text):
        equations.append(
            {
                "latex": match.group(1).strip(),
                "type": "display",
                "position": match.start(),
                "original": match.group(),
            }
        )

    for match in INLINE_EQ_PATTERN.finditer(text):
        latex = match.group(1) or match.group(2)
        if latex:
            equations.append(
                {
                    "latex": latex.strip(),
                    "type": "inline",
                    "position": match.start(),
                    "original": match.group(),
                }
            )

    equations.sort(key=lambda x: x["position"])
    return equations


def number_equations(text: str, start_num: int = 1) -> tuple:
    """Add equation numbers to display equations in text.

    Returns (modified_text, equation_count).
    """
    counter = [start_num]

    def replacer(match):
        latex = match.group(1).strip()
        num = counter[0]
        counter[0] += 1
        return f"$${latex}$$ ({num})"

    modified = DISPLAY_EQ_PATTERN.sub(replacer, text)
    return modified, counter[0] - start_num


def equations_to_text(text: str) -> str:
    """Convert LaTeX equations to plain text representation for non-LaTeX outputs.

    Useful for Word documents where LaTeX rendering is not available.
    """

    # Display equations: add visual separation
    def display_repl(match):
        latex = match.group(1).strip()
        readable = _latex_to_readable(latex)
        return f"\n    {readable}\n"

    result = DISPLAY_EQ_PATTERN.sub(display_repl, text)

    # Inline equations: convert to readable
    def inline_repl(match):
        latex = match.group(1) or match.group(2)
        if latex:
            return _latex_to_readable(latex.strip())
        return match.group()

    result = INLINE_EQ_PATTERN.sub(inline_repl, result)
    return result


def _latex_to_readable(latex: str) -> str:
    """Convert common LaTeX math to readable text."""
    replacements = [
        (r"\\alpha", "α"),
        (r"\\beta", "β"),
        (r"\\gamma", "γ"),
        (r"\\delta", "δ"),
        (r"\\epsilon", "ε"),
        (r"\\theta", "θ"),
        (r"\\lambda", "λ"),
        (r"\\mu", "μ"),
        (r"\\sigma", "σ"),
        (r"\\phi", "φ"),
        (r"\\psi", "ψ"),
        (r"\\omega", "ω"),
        (r"\\pi", "π"),
        (r"\\rho", "ρ"),
        (r"\\tau", "τ"),
        (r"\\Delta", "Δ"),
        (r"\\Sigma", "Σ"),
        (r"\\Omega", "Ω"),
        (r"\\sqrt\{([^}]+)\}", r"√(\1)"),
        (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)"),
        (r"\^(\{[^}]+\}|\w)", lambda m: "^" + m.group(1).strip("{}")),
        (r"_(\{[^}]+\}|\w)", lambda m: "_" + m.group(1).strip("{}")),
        (r"\\cdot", "·"),
        (r"\\times", "×"),
        (r"\\leq", "≤"),
        (r"\\geq", "≥"),
        (r"\\neq", "≠"),
        (r"\\approx", "≈"),
        (r"\\pm", "±"),
        (r"\\infty", "∞"),
        (r"\\sum", "Σ"),
        (r"\\int", "∫"),
        (r"\\partial", "∂"),
        (r"\\text\{([^}]+)\}", r"\1"),
        (r"\\mathrm\{([^}]+)\}", r"\1"),
        (r"\\left", ""),
        (r"\\right", ""),
        (r"\{", ""),
        (r"\}", ""),
    ]
    result = latex
    for pattern, repl in replacements:
        if callable(repl):
            result = re.sub(pattern, repl, result)
        else:
            result = re.sub(pattern, repl, result)
    return result.strip()
