"""Standardized academic table generation for PaperFactory."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TABLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "data"
)


def create_table_figure(
    headers: list,
    rows: list,
    caption: str = "",
    col_widths: list = None,
    highlight_best: str = None,
    highlight_col: int = None,
    output_dir: str = None,
    name: str = "table",
) -> str:
    """Render a table as a publication-quality figure image.

    Parameters
    ----------
    headers : list[str]
        Column header labels.
    rows : list[list[str]]
        Table data rows.
    caption : str
        Table caption (displayed below).
    col_widths : list[float], optional
        Relative column widths. Auto-calculated if None.
    highlight_best : str, optional
        "max" or "min" — highlight the best value in highlight_col.
    highlight_col : int, optional
        Column index (0-based) to apply highlight_best.
    output_dir : str, optional
        Directory for output. Defaults to outputs/data/.
    name : str
        Filename stem (without extension).

    Returns
    -------
    str
        Path to saved PNG file.
    """
    if output_dir is None:
        output_dir = TABLES_DIR
    os.makedirs(output_dir, exist_ok=True)

    n_cols = len(headers)
    n_rows = len(rows)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    fig_width = 7.0
    row_height = 0.35
    fig_height = (n_rows + 1.5) * row_height + (0.4 if caption else 0.1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_edgecolor("white")
        cell.set_height(row_height / fig_height)

    # Style data rows
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor("#f8f9fa" if i % 2 == 0 else "white")
            cell.set_edgecolor("#dee2e6")
            cell.set_height(row_height / fig_height)
            cell.set_text_props(fontsize=10)

    # Highlight best value
    if highlight_best and highlight_col is not None:
        try:
            vals = [float(rows[i][highlight_col]) for i in range(n_rows)]
            best_idx = np.argmax(vals) if highlight_best == "max" else np.argmin(vals)
            cell = table[best_idx + 1, highlight_col]
            cell.set_text_props(fontweight="bold", color="#c0392b")
        except (ValueError, IndexError):
            pass

    if caption:
        fig.text(0.5, 0.02, caption, ha="center", fontsize=9, style="italic")

    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    return path


def save_table_csv(
    headers: list,
    rows: list,
    name: str = "table",
    output_dir: str = None,
) -> str:
    """Save table data as a CSV file.

    Returns
    -------
    str
        Path to saved CSV file.
    """
    if output_dir is None:
        output_dir = TABLES_DIR
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def format_number(value: float, decimals: int = 4) -> str:
    """Format a number for table display."""
    if abs(value) >= 1000:
        return f"{value:,.{decimals}f}"
    return f"{value:.{decimals}f}"


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format mean +/- std for table display."""
    return f"{mean:.{decimals}f} \\u00b1 {std:.{decimals}f}"
