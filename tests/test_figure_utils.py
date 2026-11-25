import os

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.figure_utils as figure_utils
from utils.figure_utils import get_colors, get_figsize, save_figure, setup_style


@pytest.fixture(autouse=True)
def _isolate_matplotlib_state():
    """setup_style() mutates global rcParams; restore them (and close figures) after each test."""
    with matplotlib.rc_context():
        yield
    plt.close("all")


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


class TestSaveFigureRegressions:
    def test_closes_figure_by_default(self, tmp_path):
        fig, ax = plt.subplots()
        save_figure(fig, "closed", output_dir=str(tmp_path))
        assert not plt.fignum_exists(fig.number)

    def test_closed_figure_can_be_saved_again(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        save_figure(fig, "fig", output_dir=str(tmp_path))
        pdf = save_figure(fig, "fig", output_dir=str(tmp_path), fmt="pdf")
        assert os.path.getsize(pdf) > 0

    def test_close_false_keeps_figure_open(self, tmp_path):
        fig, ax = plt.subplots()
        save_figure(fig, "open", output_dir=str(tmp_path), close=False)
        assert plt.fignum_exists(fig.number)

    def test_no_figures_accumulate(self, tmp_path):
        before = len(plt.get_fignums())
        for i in range(25):
            fig, ax = plt.subplots()
            save_figure(fig, f"fig_{i}", output_dir=str(tmp_path))
        assert len(plt.get_fignums()) == before

    def test_extension_is_not_duplicated(self, tmp_path):
        fig, ax = plt.subplots()
        assert (
            os.path.basename(save_figure(fig, "fig_2.png", output_dir=str(tmp_path))) == "fig_2.png"
        )
        assert (
            os.path.basename(save_figure(fig, "fig_3", output_dir=str(tmp_path), fmt=".PDF"))
            == "fig_3.pdf"
        )

    def test_creates_subdirectories(self, tmp_path):
        fig, ax = plt.subplots()
        path = save_figure(fig, "sub/dir/fig_4", output_dir=str(tmp_path))
        assert path == os.path.join(str(tmp_path), "sub", "dir", "fig_4.png")
        assert os.path.exists(path)

    def test_default_output_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(figure_utils, "FIGURES_DIR", str(tmp_path / "figs"))
        fig, ax = plt.subplots()
        path = save_figure(fig, "fig_5")
        assert path == os.path.join(str(tmp_path / "figs"), "fig_5.png")
        assert os.path.exists(path)

    def test_saves_at_300_dpi(self, tmp_path):
        from PIL import Image

        fig, ax = plt.subplots(figsize=(1, 1))
        path = save_figure(fig, "dpi", output_dir=str(tmp_path))
        with Image.open(path) as image:
            assert image.info["dpi"] == pytest.approx((300, 300), abs=1)


class TestSetupStyleIsolation:
    def test_rcparams_restored_between_tests_part1(self):
        setup_style()
        assert plt.rcParams["figure.constrained_layout.use"] is True

    def test_rcparams_restored_between_tests_part2(self):
        assert (
            plt.rcParams["figure.constrained_layout.use"]
            == matplotlib.rcParamsDefault["figure.constrained_layout.use"]
        )
