import os
import csv
import pytest
import matplotlib
matplotlib.use("Agg")

from utils.table_utils import create_table_figure, save_table_csv, format_number, format_mean_std


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
