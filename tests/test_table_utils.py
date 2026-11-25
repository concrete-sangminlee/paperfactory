import csv
import os

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.table_utils import create_table_figure, format_mean_std, format_number, save_table_csv


class TestCreateTableFigure:
    def test_creates_png(self, tmp_path):
        path = create_table_figure(
            headers=["Model", "R2", "RMSE"],
            rows=[["RF", "0.99", "0.01"], ["GBR", "0.98", "0.02"]],
            output_dir=str(tmp_path),
            name="test_table",
        )
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_with_caption(self, tmp_path):
        path = create_table_figure(
            headers=["A", "B"],
            rows=[["1", "2"]],
            caption="Table 1. Test caption.",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)

    def test_highlight_max(self, tmp_path):
        path = create_table_figure(
            headers=["Model", "R2"],
            rows=[["RF", "0.99"], ["GBR", "0.95"], ["DNN", "0.90"]],
            highlight_best="max",
            highlight_col=1,
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)

    def test_highlight_min(self, tmp_path):
        path = create_table_figure(
            headers=["Model", "RMSE"],
            rows=[["RF", "0.01"], ["GBR", "0.02"], ["DNN", "0.05"]],
            highlight_best="min",
            highlight_col=1,
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)

    def test_custom_col_widths(self, tmp_path):
        path = create_table_figure(
            headers=["Name", "Value"],
            rows=[["long name here", "123"]],
            col_widths=[0.7, 0.3],
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)


class TestSaveTableCsv:
    def test_creates_csv(self, tmp_path):
        path = save_table_csv(
            headers=["A", "B", "C"],
            rows=[["1", "2", "3"], ["4", "5", "6"]],
            name="test_csv",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)
        assert path.endswith(".csv")

    def test_csv_content(self, tmp_path):
        path = save_table_csv(
            headers=["X", "Y"],
            rows=[["a", "b"]],
            output_dir=str(tmp_path),
        )
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["X", "Y"]
        assert rows[1] == ["a", "b"]


class TestFormatNumber:
    def test_small_number(self):
        assert format_number(0.9876, 4) == "0.9876"

    def test_large_number(self):
        assert "," in format_number(12345.678, 2)

    def test_custom_decimals(self):
        assert format_number(0.1, 2) == "0.10"


class TestFormatMeanStd:
    def test_basic(self):
        result = format_mean_std(0.99, 0.01, 4)
        assert "0.9900" in result
        assert "0.0100" in result


HIGHLIGHT = "#c0392b"


def _highlighted(monkeypatch, tmp_path, rows, best="max", col=1, headers=("Model", "Value")):
    """Render a table and return the texts of the highlighted cells."""
    captured = []
    original = Figure.savefig

    def spy(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    create_table_figure(
        list(headers), rows, highlight_best=best, highlight_col=col, output_dir=str(tmp_path)
    )
    table = captured[-1].axes[0].tables[0]
    return [
        cell.get_text().get_text()
        for (r, _c), cell in sorted(table.get_celld().items())
        if r > 0 and cell.get_text().get_color() == HIGHLIGHT
    ]


class TestHighlightRegressions:
    def test_highlights_max(self, monkeypatch, tmp_path):
        rows = [["RF", "0.99"], ["GBR", "0.95"], ["DNN", "0.90"]]
        assert _highlighted(monkeypatch, tmp_path, rows, "max") == ["0.99"]
        assert _highlighted(monkeypatch, tmp_path, rows, "min") == ["0.90"]

    def test_nan_is_never_best(self, monkeypatch, tmp_path):
        rows = [["A", "0.5"], ["B", "nan"], ["C", "0.9"]]
        assert _highlighted(monkeypatch, tmp_path, rows, "max") == ["0.9"]

    def test_non_numeric_cells_are_skipped(self, monkeypatch, tmp_path):
        rows = [["A", "0.5"], ["B", "N/A"], ["C", "0.9"], ["D", "—"]]
        assert _highlighted(monkeypatch, tmp_path, rows, "max") == ["0.9"]

    def test_ties_are_all_highlighted(self, monkeypatch, tmp_path):
        rows = [["A", "0.9"], ["B", "0.9"], ["C", "0.1"]]
        assert _highlighted(monkeypatch, tmp_path, rows, "max") == ["0.9", "0.9"]

    def test_formatted_numbers_are_parsed(self, monkeypatch, tmp_path):
        rows = [
            ["A", format_number(1234.5, 1)],
            ["B", "999.0"],
            ["C", "\u22122,000.0"],  # typographic minus
            ["D", format_mean_std(0.5, 0.1, 2)],
        ]
        assert _highlighted(monkeypatch, tmp_path, rows, "max") == ["1,234.5"]
        assert _highlighted(monkeypatch, tmp_path, rows, "min") == ["\u22122,000.0"]

    def test_mode_is_case_insensitive(self, monkeypatch, tmp_path):
        # Anything but exactly "max" used to be treated as "min".
        assert _highlighted(monkeypatch, tmp_path, [["A", "0.1"], ["B", "0.9"]], "Max") == ["0.9"]

    def test_negative_column_index(self, monkeypatch, tmp_path):
        assert _highlighted(monkeypatch, tmp_path, [["A", "0.1"], ["B", "0.9"]], col=-1) == ["0.9"]

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="highlight_best"):
            create_table_figure(
                ["A", "B"],
                [["x", "1"]],
                highlight_best="best",
                highlight_col=1,
                output_dir=str(tmp_path),
            )

    def test_out_of_range_column_raises(self, tmp_path):
        with pytest.raises(ValueError, match="out of range"):
            create_table_figure(
                ["A", "B"],
                [["x", "1"]],
                highlight_best="max",
                highlight_col=5,
                output_dir=str(tmp_path),
            )


class TestCreateTableFigureRobustness:
    def test_header_only_table(self, tmp_path):
        path = create_table_figure(["Model", "R2"], [], output_dir=str(tmp_path), name="empty")
        assert os.path.exists(path)

    def test_short_rows_are_padded(self, tmp_path):
        path = create_table_figure(["A", "B", "C"], [["1"], ["2", "3"]], output_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_too_long_row_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Row 0"):
            create_table_figure(["A"], [["1", "2"]], output_dir=str(tmp_path))

    def test_non_string_cells(self, tmp_path):
        path = create_table_figure(
            ["A", "B"], [[1, None], [2.5, float("nan")]], output_dir=str(tmp_path)
        )
        assert os.path.exists(path)

    def test_empty_headers_raise(self, tmp_path):
        with pytest.raises(ValueError):
            create_table_figure([], [], output_dir=str(tmp_path))

    def test_no_figure_leak_even_if_saving_fails(self, monkeypatch, tmp_path):
        before = set(plt.get_fignums())

        def fail(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Figure, "savefig", fail)
        with pytest.raises(OSError):
            create_table_figure(["A"], [["1"]], output_dir=str(tmp_path))
        assert set(plt.get_fignums()) == before


class TestFormatRegressions:
    def test_mean_std_uses_plus_minus_sign(self):
        # The f-string contained a literal "\\u00b1" instead of the ± character.
        assert format_mean_std(0.99, 0.01, 4) == "0.9900 \u00b1 0.0100"

    def test_negative_zero(self):
        assert format_number(-0.00004, 4) == "0.0000"
        assert format_number(-0.0, 2) == "0.00"
        assert format_mean_std(-0.00001, 0.5, 2) == "0.00 \u00b1 0.50"

    def test_negative_numbers(self):
        assert format_number(-0.5, 2) == "-0.50"
        assert format_number(-1234.5, 2) == "-1,234.50"

    def test_rounding_up_to_thousand_uses_separator(self):
        assert format_number(999.99996, 4) == "1,000.0000"

    def test_nan_inf_none(self):
        assert format_number(float("nan")) == "N/A"
        assert format_number(None) == "N/A"
        assert format_number(float("inf"), 2) == "\u221e"
        assert format_number(float("-inf"), 2) == "-\u221e"
        assert format_mean_std(float("nan"), 0.1, 2) == "N/A \u00b1 0.10"

    def test_zero_decimals(self):
        assert format_number(12345.678, 0) == "12,346"
