import numpy as np
import pytest
from scipy import stats

from utils.research_advisor import recommend_figure_type, recommend_statistical_tests

# All data is deterministic.  "Clearly normal" / "clearly non-normal" samples are
# exact distribution quantiles in a seeded random order, so a normality test
# accepts / rejects them regardless of sampling luck (unseeded np.random data
# made these tests flaky: ~5% of runs rejected normality of normal data).


def _normal(n, loc=0.0, scale=1.0, seed=0):
    q = stats.norm.ppf((np.arange(n) + 0.5) / n, loc=loc, scale=scale)
    return np.random.default_rng(seed).permutation(q)


def _exponential(n, seed=0):
    q = stats.expon.ppf((np.arange(n) + 0.5) / n)
    return np.random.default_rng(seed).permutation(q)


def _test_names(recs):
    return [r["test"] for r in recs]


def _group_rec(recs):
    """The group-comparison recommendation (the one carrying group_sizes)."""
    matches = [r for r in recs if "group_sizes" in r]
    assert len(matches) == 1, recs
    return matches[0]


class TestRecommendStatisticalTests:
    def test_normal_data_single_sample(self):
        recs = recommend_statistical_tests(_normal(100))
        assert len(recs) >= 2
        assert any("Normality" in r.get("purpose", "") for r in recs)
        assert recs[0]["conclusion"] == "Normal"

    def test_two_groups_normal(self):
        data = np.concatenate([_normal(50, seed=1), _normal(50, seed=2)])
        groups = np.array([0] * 50 + [1] * 50)
        recs = recommend_statistical_tests(data, groups)
        test_names = _test_names(recs)
        assert any("t-test" in t for t in test_names)

    def test_two_groups_non_normal(self):
        data = np.concatenate([_exponential(50, seed=1), _exponential(50, seed=2)])
        groups = np.array([0] * 50 + [1] * 50)
        recs = recommend_statistical_tests(data, groups)
        test_names = _test_names(recs)
        assert any("Mann-Whitney" in t or "Wilcoxon" in t for t in test_names)

    def test_three_groups(self):
        data = np.concatenate([_normal(50, seed=s) for s in range(3)])
        groups = np.array([0] * 50 + [1] * 50 + [2] * 50)
        recs = recommend_statistical_tests(data, groups)
        test_names = _test_names(recs)
        assert "One-way ANOVA" in test_names

    def test_three_groups_non_normal(self):
        data = np.concatenate([_exponential(50, seed=s) for s in range(3)])
        groups = np.repeat([0, 1, 2], 50)
        recs = recommend_statistical_tests(data, groups)
        assert "Kruskal-Wallis H" in _test_names(recs)

    def test_2d_data(self):
        data = np.random.default_rng(42).normal(size=(100, 5))
        recs = recommend_statistical_tests(data)
        assert any("correlation" in r["test"].lower() or "PCA" in r["test"] for r in recs)

    def test_includes_effect_size(self):
        recs = recommend_statistical_tests(_normal(100))
        assert any("Cohen" in r["test"] for r in recs)

    def test_paired_groups(self):
        data = np.concatenate([_normal(50, seed=1), _normal(50, seed=2)])
        groups = np.array([0] * 50 + [1] * 50)
        recs = recommend_statistical_tests(data, groups, paired=True)
        test_names = _test_names(recs)
        assert any("Paired" in t or "Wilcoxon" in t for t in test_names)


class TestStatisticalTestsRegressions:
    def test_separated_normal_groups_use_parametric_test(self):
        # Two normal groups with different means: the pooled data is bimodal, so
        # testing normality on pooled data wrongly chose Mann-Whitney U.
        data = np.concatenate([_normal(60, loc=0), _normal(60, loc=6, seed=1)])
        groups = np.array([0] * 60 + [1] * 60)
        recs = recommend_statistical_tests(data, groups)
        assert recs[0]["conclusion"] == "Non-normal"  # pooled data
        assert _group_rec(recs)["test"] == "Independent t-test"

    def test_unequal_variances_use_welch(self):
        data = np.concatenate([_normal(40, 0, 1), _normal(120, 0.5, 6, seed=1)])
        groups = np.array([0] * 40 + [1] * 120)
        rec = _group_rec(recommend_statistical_tests(data, groups))
        assert rec["test"] == "Welch's t-test"
        assert "equal_var=False" in rec["scipy"]
        assert rec["group_sizes"] == {0: 40, 1: 120}

    def test_unequal_variances_three_groups_use_welch_anova(self):
        data = np.concatenate([_normal(40, 0, 1), _normal(40, 0, 5, seed=1), _normal(40, 1, 9)])
        groups = np.repeat([0, 1, 2], 40)
        assert _group_rec(recommend_statistical_tests(data, groups))["test"] == "Welch's ANOVA"

    def test_large_non_normal_sample_reports_p_value(self):
        # p underflows to exactly 0.0; a truthiness check reported "n too small".
        recs = recommend_statistical_tests(_exponential(20000))
        assert recs[0]["result"] == "p<0.0001"
        assert recs[0]["conclusion"] == "Non-normal"
        assert recs[0]["test"] == "D'Agostino-Pearson"
        assert "normaltest" in recs[0]["scipy"]

    def test_nan_and_inf_values_are_excluded(self):
        data = _normal(50)
        data[[3, 7]] = [np.nan, np.inf]
        recs = recommend_statistical_tests(data)
        assert recs[0]["conclusion"] == "Normal"
        assert recs[0]["n"] == 48
        assert "2 non-finite" in recs[0]["note"]

    def test_nan_values_are_excluded_with_their_group_labels(self):
        data = np.concatenate([_normal(30, loc=0), _normal(30, loc=8, seed=1)])
        data[0] = np.nan
        groups = np.array([0] * 30 + [1] * 30)
        rec = _group_rec(recommend_statistical_tests(data, groups))
        assert rec["test"] == "Independent t-test"
        assert rec["group_sizes"] == {0: 29, 1: 30}

    def test_constant_data_is_flagged_not_called_non_normal(self):
        recs = recommend_statistical_tests(np.full(30, 2.5), np.array([0, 1] * 15))
        assert recs[0]["conclusion"] == "Constant data"
        assert "warning" in recs[0]
        names = _test_names(recs)
        assert "Spearman rho" not in names and "Pearson r" not in names
        assert not any("group_sizes" in r for r in recs)

    def test_paired_with_unequal_group_sizes_falls_back_to_independent(self):
        data = np.concatenate([_normal(30), _normal(70, seed=1)])
        groups = np.array([0] * 30 + [1] * 70)
        rec = _group_rec(recommend_statistical_tests(data, groups, paired=True))
        assert rec["test"] == "Independent t-test"
        assert "group sizes differ" in rec["warning"]

    def test_paired_normality_is_assessed_on_differences(self):
        before = _exponential(40)
        after = before + _normal(40, loc=0.5, scale=0.1, seed=3)  # normal differences
        data = np.concatenate([before, after])
        groups = np.array([0] * 40 + [1] * 40)
        assert _group_rec(recommend_statistical_tests(data, groups, paired=True))["test"] == (
            "Paired t-test"
        )

    def test_paired_three_groups_use_repeated_measures_tests(self):
        data = np.concatenate([_normal(30, seed=s) for s in range(3)])
        groups = np.repeat([0, 1, 2], 30)
        rec = _group_rec(recommend_statistical_tests(data, groups, paired=True))
        assert rec["test"] == "Repeated-measures ANOVA"
        data = np.concatenate([_exponential(30, seed=s) for s in range(3)])
        rec = _group_rec(recommend_statistical_tests(data, groups, paired=True))
        assert rec["test"] == "Friedman test"

    def test_group_with_single_observation_warns(self):
        data = np.append(_normal(20), 0.3)
        groups = np.array([0] * 20 + [1])
        rec = _group_rec(recommend_statistical_tests(data, groups))
        assert "fewer than 2 observations" in rec["warning"]

    def test_groups_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="one label per data value"):
            recommend_statistical_tests(_normal(10), np.array([0, 1]))

    def test_accepts_python_lists(self):
        data = list(_normal(20))
        recs = recommend_statistical_tests(data, [0, 1] * 10)
        assert recs[0]["n"] == 20
        assert any("t-test" in t for t in _test_names(recs))

    def test_string_group_labels(self):
        data = np.concatenate([_normal(30), _normal(30, loc=4, seed=1)])
        groups = np.array(["RF"] * 30 + ["GBR"] * 30)
        rec = _group_rec(recommend_statistical_tests(data, groups))
        assert rec["group_sizes"] == {"GBR": 30, "RF": 30}

    def test_normal_groups_are_rarely_rejected_regardless_of_group_count(self):
        # Requiring every group to pass its own alpha=0.05 test would reject
        # normal data in ~23% of 5-group cases; the per-group level is adjusted.
        groups = np.repeat(np.arange(5), 30)
        non_parametric = 0
        for seed in range(100):
            data = np.random.default_rng(seed).normal(size=150)
            test = _group_rec(recommend_statistical_tests(data, groups))["test"]
            non_parametric += test == "Kruskal-Wallis H"
        assert non_parametric <= 10


class TestRecommendFigureType:
    def setup_method(self):
        self.rng = np.random.default_rng(7)

    def test_continuous_data(self):
        data = self.rng.normal(size=(100, 3))
        recs = recommend_figure_type(data, data_type="continuous")
        types = [r["type"] for r in recs]
        assert any("Box" in t or "Scatter" in t for t in types)

    def test_group_comparison(self):
        data = self.rng.normal(size=(100, 1))
        recs = recommend_figure_type(data, data_type="continuous", comparison="groups")
        types = [r["type"] for r in recs]
        assert any("Box" in t for t in types)

    def test_time_series(self):
        data = self.rng.normal(size=(1000, 1))
        recs = recommend_figure_type(data, data_type="time_series")
        types = [r["type"] for r in recs]
        assert any("Line" in t for t in types)
        assert any("PSD" in t for t in types)

    def test_spatial(self):
        data = self.rng.normal(size=(100, 2))
        recs = recommend_figure_type(data, data_type="spatial")
        types = [r["type"] for r in recs]
        assert any("Contour" in t or "Polar" in t for t in types)

    def test_categorical(self):
        data = self.rng.normal(size=(100, 1))
        recs = recommend_figure_type(data, data_type="categorical")
        types = [r["type"] for r in recs]
        assert any("Confusion" in t for t in types)

    def test_many_features_suggests_importance(self):
        data = self.rng.normal(size=(100, 10))
        recs = recommend_figure_type(data, data_type="continuous")
        types = [r["type"] for r in recs]
        assert any("importance" in t.lower() for t in types)

    def test_all_have_code(self):
        data = self.rng.normal(size=(100, 3))
        recs = recommend_figure_type(data, data_type="continuous", comparison="correlation")
        for r in recs:
            assert "code" in r or "formula" in r

    def test_trend_comparison_is_supported(self):
        # "trend" is a documented comparison but used to return no recommendation.
        recs = recommend_figure_type(self.rng.normal(size=100), comparison="trend")
        assert recs and any("Line" in r["type"] for r in recs)
        assert all("code" in r for r in recs)

    def test_accepts_python_lists(self):
        recs = recommend_figure_type([[1.0, 2.0], [3.0, 4.0]])
        assert any("Scatter" in r["type"] for r in recs)
