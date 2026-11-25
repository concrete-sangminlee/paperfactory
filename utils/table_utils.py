"""Standardized academic table generation for PaperFactory."""

import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

TABLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "data"
)

# Leading number of a table cell: "0.95", "-1,234.5", "1e-3", "95%", "0.91 ± 0.02".
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_HIGHLIGHT_COLOR = "#c0392b"


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

    n_cols = len(headers)
    if n_cols == 0:
        raise ValueError("headers must not be empty")
    rows = _normalize_rows(rows, n_cols)
    n_rows = len(rows)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    elif len(col_widths) != n_cols:
        raise ValueError(f"col_widths has {len(col_widths)} entries but there are {n_cols} columns")

    mode = None
    if highlight_best and highlight_col is not None:
        mode = str(highlight_best).strip().lower()
        if mode not in ("max", "min"):
            raise ValueError(f"highlight_best must be 'max' or 'min', got {highlight_best!r}")
        if not -n_cols <= highlight_col < n_cols:
            raise ValueError(f"highlight_col {highlight_col} is out of range for {n_cols} columns")
        highlight_col %= n_cols

    os.makedirs(output_dir, exist_ok=True)

    fig_width = 7.0
    row_height = 0.35
    fig_height = (n_rows + 1.5) * row_height + (0.4 if caption else 0.1)

    # An explicit Figure (not pyplot) is never registered globally, so nothing can
    # leak even if rendering fails, and it is safe to use from worker threads.
    fig = Figure(figsize=(fig_width, fig_height))
    ax = fig.subplots()
    ax.axis("off")

    if n_rows:
        table = ax.table(
            cellText=rows,
            colLabels=list(headers),
            colWidths=col_widths,
            loc="center",
            cellLoc="center",
        )
    else:
        # Header-only table (matplotlib cannot render colLabels without cellText).
        table = ax.table(
            cellText=[[str(h) for h in headers]],
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

    # Highlight best value(s): non-numeric/NaN cells are ignored, ties all highlighted
    if mode:
        for i in _best_row_indices(rows, highlight_col, mode):
            table[i + 1, highlight_col].set_text_props(fontweight="bold", color=_HIGHLIGHT_COLOR)

    if caption:
        fig.text(0.5, 0.02, caption, ha="center", fontsize=9, style="italic")

    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    return path


def _normalize_rows(rows: list, n_cols: int) -> list:
    """Convert cells to strings and pad short rows to ``n_cols`` cells."""
    normalized = []
    for i, row in enumerate(rows or []):
        cells = ["" if c is None else str(c) for c in row]
        if len(cells) > n_cols:
            raise ValueError(f"Row {i} has {len(cells)} cells but there are {n_cols} headers")
        normalized.append(cells + [""] * (n_cols - len(cells)))
    return normalized


def _parse_number(cell) -> float:
    """Return the leading finite number in a cell, or None if there is none."""
    text = str(cell).strip().replace("\u2212", "-")
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    try:
        value = float(match.group().replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _best_row_indices(rows: list, col: int, mode: str) -> list:
    values = [(i, _parse_number(row[col])) for i, row in enumerate(rows)]
    values = [(i, v) for i, v in values if v is not None]
    if not values:
        return []
    best = max(v for _, v in values) if mode == "max" else min(v for _, v in values)
    return [i for i, v in values if v == best]


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


def _format_value(value, decimals: int, thousands: bool) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if math.isnan(value):
        return "N/A"
    if math.isinf(value):
        return "\u221e" if value > 0 else "-\u221e"
    text = f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"
    if text.startswith("-") and float(text[1:].replace(",", "")) == 0:
        text = text[1:]  # values that round to zero print as "0.0000", not "-0.0000"
    return text


def format_number(value: float, decimals: int = 4) -> str:
    """Format a number for table display.

    Thousands separators are used for magnitudes of 1,000 or more (including
    values that round up to 1,000), negative zero is printed without a sign and
    NaN/None as "N/A".
    """
    return _format_value(value, decimals, thousands=True)


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format mean ± std for table display."""
    return f"{_format_value(mean, decimals, False)} \u00b1 {_format_value(std, decimals, False)}"
