import pytest
from utils.abstract_generator import generate_abstract, improve_abstract


def _make_paper():
    return {
        "sections": [
            {"heading": "INTRODUCTION", "content": (
                "Wind loads are critical for building design. "
                "Previous studies have used neural networks for prediction. "
                "This study develops ML models for peak wind pressure prediction using the TPU database."
            )},
            {"heading": "METHODOLOGY", "content": (
                "Three models were compared: Random Forest, Gradient Boosting, and DNN. "
                "The models were trained with 10-fold cross-validation on 5184 samples."
            )},
            {"heading": "RESULTS AND DISCUSSION", "content": (
                "Random Forest achieved R-squared of 0.9999 and RMSE of 0.012. "
                "Feature importance analysis shows statistical features dominate. "
                "The model outperforms baseline approaches by 15 percent."
            )},
            {"heading": "CONCLUSIONS", "content": (
                "This study demonstrated that ML models effectively predict peak wind pressures. "
                "Random Forest is recommended for tabular wind pressure data. "
                "Future work should validate with real TPU records."
            )},
        ],
    }


class TestGenerateAbstract:
    def test_generates_nonempty(self):
        abstract = generate_abstract(_make_paper())
        assert len(abstract) > 50

    def test_respects_max_words(self):
        abstract = generate_abstract(_make_paper(), max_words=50)
        assert len(abstract.split()) <= 55  # small tolerance for sentence boundary

    def test_contains_purpose(self):
        abstract = generate_abstract(_make_paper())
        assert any(w in abstract.lower() for w in ["this study", "develops", "prediction"])

    def test_contains_results(self):
        abstract = generate_abstract(_make_paper())
        assert any(w in abstract for w in ["0.9999", "0.012", "Random Forest"])

    def test_empty_paper(self):
        abstract = generate_abstract({"sections": []})
        assert abstract == ""


class TestImproveAbstract:
    def test_good_abstract_scores_high(self):
        abstract = (
            "This study investigates ML models for wind pressure prediction. "
            "Three models (RF, GBR, DNN) were compared using 10-fold cross-validation. "
            "Results show RF achieved R-squared of 0.999 and RMSE of 0.012. "
            "The framework provides an effective and practical surrogate for wind tunnel testing."
        )
        result = improve_abstract(abstract, _make_paper())
        assert result["score"] >= 70
        assert result["has_purpose"]
        assert result["has_method"]
        assert result["has_results"]

    def test_poor_abstract_gets_suggestions(self):
        result = improve_abstract("Short abstract.", _make_paper())
        assert len(result["suggestions"]) >= 3
        assert result["score"] < 50

    def test_detects_missing_results(self):
        abstract = "This study proposes a new method for structural analysis."
        result = improve_abstract(abstract, _make_paper())
        assert not result["has_results"]
        assert any("result" in s.lower() for s in result["suggestions"])

    def test_word_count(self):
        abstract = "word " * 100
        result = improve_abstract(abstract, _make_paper(), max_words=50)
        assert result["word_count"] == 100
        assert any("exceeds" in s.lower() for s in result["suggestions"])
