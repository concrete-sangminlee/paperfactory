"""Standardized figure styling for PaperFactory research outputs."""

import os

import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures"
)

_PALETTES = {
    "default": [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ],
    "muted": [
        "#4878d0",
        "#ee854a",
        "#6acc64",
        "#d65f5f",
        "#956cb4",
        "#8c613c",
        "#dc7ec0",
        "#797979",
    ],
    "grayscale": [
        "#000000",
        "#333333",
        "#555555",
        "#777777",
        "#999999",
        "#bbbbbb",
        "#dddddd",
        "#444444",
    ],
}

_FIGSIZES = {
    "single": (3.5, 2.8),
    "double": (7.0, 4.5),
    "square": (3.5, 3.5),
    "single_tall": (3.5, 5.0),
    "double_tall": (7.0, 7.0),
}


def setup_style():
    """Apply publication-quality matplotlib style. Call once at top of script.

    Uses Times New Roman with DejaVu Serif as fallback for Linux/CI environments
    where Times New Roman is not installed.
    """
    plt.rcParams.update(
        {
            "font.family": ["serif"],
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.dpi": 100,
            "figure.constrained_layout.use": True,
            "mathtext.fontset": "dejavuserif",
        }
    )


def get_colors(palette: str = "default") -> list[str]:
    """Return a list of 8 colors for the given palette name."""
    if palette not in _PALETTES:
        raise ValueError(f"Unknown palette '{palette}'. Choose from: {list(_PALETTES.keys())}")
    return list(_PALETTES[palette])


def get_figsize(size: str = "single") -> tuple[float, float]:
    """Return (width, height) in inches for standard figure sizes."""
    if size not in _FIGSIZES:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(_FIGSIZES.keys())}")
    return _FIGSIZES[size]


def save_figure(fig, name: str, output_dir: str = None, fmt: str = "png") -> str:
    """Save figure with standard settings. Returns the saved file path."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    return path
