import os
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.figure_utils import setup_style, save_figure, get_colors, get_figsize


class TestSetupStyle:
    def test_sets_times_new_roman(self):
        setup_style()
        assert plt.rcParams["font.family"] == ["serif"]
        assert "Times New Roman" in plt.rcParams["font.serif"]

    def test_sets_dpi_300(self):
        setup_style()
        assert plt.rcParams["savefig.dpi"] == 300

    def test_sets_tick_direction_in(self):
        setup_style()
        assert plt.rcParams["xtick.direction"] == "in"
        assert plt.rcParams["ytick.direction"] == "in"

    def test_sets_font_sizes(self):
        setup_style()
        assert plt.rcParams["font.size"] == 11
        assert plt.rcParams["axes.labelsize"] == 12
        assert plt.rcParams["legend.fontsize"] == 10


class TestGetColors:
    def test_returns_8_colors_default(self):
        colors = get_colors()
        assert len(colors) == 8

    def test_returns_list_of_strings(self):
        colors = get_colors()
        assert all(isinstance(c, str) for c in colors)

    def test_muted_palette(self):
        colors = get_colors("muted")
        assert len(colors) == 8

    def test_invalid_palette_raises(self):
        with pytest.raises(ValueError):
            get_colors("nonexistent")


class TestGetFigsize:
    def test_single_column(self):
        w, h = get_figsize("single")
        assert w == pytest.approx(3.5)
        assert isinstance(h, float)

    def test_double_column(self):
        w, h = get_figsize("double")
        assert w == pytest.approx(7.0)

    def test_square(self):
        w, h = get_figsize("square")
        assert w == h

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_figsize("triple")


class TestSaveFigure:
    def test_saves_png_to_outputs(self, tmp_path):
        setup_style()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        path = save_figure(fig, "test_fig", output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".png")
        plt.close(fig)

    def test_default_filename_format(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        path = save_figure(fig, "fig_1_example", output_dir=str(tmp_path))
        assert "fig_1_example.png" in path
        plt.close(fig)
