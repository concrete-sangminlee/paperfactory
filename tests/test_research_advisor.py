import numpy as np

from utils.research_advisor import recommend_figure_type, recommend_statistical_tests


class TestRecommendStatisticalTests:
    def test_normal_data_single_sample(self):
        data = np.random.randn(100)
        recs = recommend_statistical_tests(data)
        assert len(recs) >= 2
        assert any("Normality" in r.get("purpose", "") for r in recs)

    def test_two_groups_normal(self):
        data = np.random.randn(100)
        groups = np.array([0] * 50 + [1] * 50)
        recs = recommend_statistical_tests(data, groups)
        test_names = [r["test"] for r in recs]
        assert any("t-test" in t for t in test_names)

    def test_two_groups_non_normal(self):
        data = np.random.exponential(1, 100)
        groups = np.array([0] * 50 + [1] * 50)
        recs = recommend_statistical_tests(data, groups)
        test_names = [r["test"] for r in recs]
        assert any("Mann-Whitney" in t or "Wilcoxon" in t for t in test_names)

    def test_three_groups(self):
        data = np.random.randn(150)
        groups = np.array([0] * 50 + [1] * 50 + [2] * 50)
        recs = recommend_statistical_tests(data, groups)
        test_names = [r["test"] for r in recs]
        assert any("ANOVA" in t or "Kruskal" in t for t in test_names)

    def test_2d_data(self):
        data = np.random.randn(100, 5)
        recs = recommend_statistical_tests(data)
        assert any("correlation" in r["test"].lower() or "PCA" in r["test"] for r in recs)

    def test_includes_effect_size(self):
        data = np.random.randn(100)
        recs = recommend_statistical_tests(data)
        assert any("Cohen" in r["test"] for r in recs)

    def test_paired_groups(self):
        data = np.random.randn(100)
        groups = np.array([0] * 50 + [1] * 50)
        recs = recommend_statistical_tests(data, groups, paired=True)
        test_names = [r["test"] for r in recs]
        assert any("Paired" in t or "Wilcoxon" in t for t in test_names)


class TestRecommendFigureType:
    def test_continuous_data(self):
        data = np.random.randn(100, 3)
        recs = recommend_figure_type(data, data_type="continuous")
        types = [r["type"] for r in recs]
        assert any("Box" in t or "Scatter" in t for t in types)

    def test_group_comparison(self):
        data = np.random.randn(100, 1)
        recs = recommend_figure_type(data, data_type="continuous", comparison="groups")
        types = [r["type"] for r in recs]
        assert any("Box" in t for t in types)

    def test_time_series(self):
        data = np.random.randn(1000, 1)
        recs = recommend_figure_type(data, data_type="time_series")
        types = [r["type"] for r in recs]
        assert any("Line" in t for t in types)
        assert any("PSD" in t for t in types)

    def test_spatial(self):
        data = np.random.randn(100, 2)
        recs = recommend_figure_type(data, data_type="spatial")
        types = [r["type"] for r in recs]
        assert any("Contour" in t or "Polar" in t for t in types)

    def test_categorical(self):
        data = np.random.randn(100, 1)
        recs = recommend_figure_type(data, data_type="categorical")
        types = [r["type"] for r in recs]
        assert any("Confusion" in t for t in types)

    def test_many_features_suggests_importance(self):
        data = np.random.randn(100, 10)
        recs = recommend_figure_type(data, data_type="continuous")
        types = [r["type"] for r in recs]
        assert any("importance" in t.lower() for t in types)

    def test_all_have_code(self):
        data = np.random.randn(100, 3)
        recs = recommend_figure_type(data, data_type="continuous", comparison="correlation")
        for r in recs:
            assert "code" in r or "formula" in r
