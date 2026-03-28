import pytest
from utils.ai_reviewer import review_paper


def _make_good_paper():
    return {
        "title": "ML Wind Pressure Prediction",
        "authors": "A. Author",
        "abstract": (
            "This study proposes a novel ML framework for wind pressure prediction. "
            "Results show that Random Forest achieved R-squared of 0.999 and RMSE of 0.01. "
            "The proposed approach demonstrates significant improvement over existing methods."
        ),
        "keywords": "wind; ML",
        "sections": [
            {"heading": "INTRODUCTION", "content": (
                "Wind loads are critical. " * 100 +
                "Despite advances, gaps remain in this area. However, existing methods have limitations. "
                "This study addresses these gaps by proposing a novel approach."
            )},
            {"heading": "METHODOLOGY", "content": (
                "The proposed method uses cross-validation with random seed 42. " * 40
            )},
            {"heading": "RESULTS AND DISCUSSION", "content": (
                "Table 1 shows results. Fig. 1 compares models. "
                "The proposed method outperforms baseline approaches. "
                "R-squared values exceed 0.99 for all models." + " Analysis text." * 30
            )},
            {"heading": "CONCLUSIONS", "content": (
                "This study demonstrated effective wind pressure prediction. "
                "Limitations include synthetic data. Future work should extend to real data. " * 10
            )},
        ],
        "tables": [{"caption": "Table 1.", "headers": ["A"], "rows": [["1"]]}] * 3,
        "references": [
            f"[{i}] Author, Title ({2024}). doi:10.1000/{i}" for i in range(1, 25)
        ],
        "figure_captions": [f"Fig. {i}." for i in range(1, 8)],
    }


def _make_weak_paper():
    return {
        "title": "Short Paper",
        "authors": "A. Author",
        "abstract": "Brief abstract.",
        "keywords": "test",
        "sections": [
            {"heading": "INTRODUCTION", "content": "Short intro."},
            {"heading": "RESULTS", "content": "Some results."},
        ],
        "tables": [],
        "references": ["[1] Old ref (2010)."],
        "figure_captions": [],
    }


class TestReviewPaper:
    def test_good_paper_gets_minor_or_accept(self):
        result = review_paper(_make_good_paper(), "jweia", figures=["f"] * 7)
        assert result["major_issues"] <= 2
        assert "Accept" in result["decision"] or "Minor" in result["decision"] or "Major" in result["decision"]

    def test_weak_paper_gets_major_or_reject(self):
        result = review_paper(_make_weak_paper(), "jweia")
        assert result["major_issues"] >= 3
        assert "Major" in result["decision"] or "Reject" in result["decision"]

    def test_missing_methodology_flagged(self):
        result = review_paper(_make_weak_paper(), "jweia")
        comments = [c["comment"] for c in result["comments"]]
        assert any("METHODOLOGY" in c for c in comments)

    def test_missing_conclusion_flagged(self):
        result = review_paper(_make_weak_paper(), "jweia")
        comments = [c["comment"] for c in result["comments"]]
        assert any("CONCLUSION" in c for c in comments)

    def test_few_references_flagged(self):
        result = review_paper(_make_weak_paper(), "jweia")
        comments = [c["comment"] for c in result["comments"]]
        assert any("reference" in c.lower() for c in comments)

    def test_summary_contains_decision(self):
        result = review_paper(_make_good_paper(), "jweia")
        assert "Decision" in result["summary"]

    def test_returns_structured_comments(self):
        result = review_paper(_make_good_paper(), "jweia")
        for c in result["comments"]:
            assert "severity" in c
            assert "section" in c
            assert "comment" in c
            assert c["severity"] in ("major", "minor")
