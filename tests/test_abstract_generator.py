import pytest

from utils.abstract_generator import generate_abstract, improve_abstract


def _make_paper():
    return {
        "sections": [
            {
                "heading": "INTRODUCTION",
                "content": (
                    "Wind loads are critical for building design. "
                    "Previous studies have used neural networks for prediction. "
                    "This study develops ML models for peak wind pressure prediction using the TPU database."
                ),
            },
            {
                "heading": "METHODOLOGY",
                "content": (
                    "Three models were compared: Random Forest, Gradient Boosting, and DNN. "
                    "The models were trained with 10-fold cross-validation on 5184 samples."
                ),
            },
            {
                "heading": "RESULTS AND DISCUSSION",
                "content": (
                    "Random Forest achieved R-squared of 0.9999 and RMSE of 0.012. "
                    "Feature importance analysis shows statistical features dominate. "
                    "The model outperforms baseline approaches by 15 percent."
                ),
            },
            {
                "heading": "CONCLUSIONS",
                "content": (
                    "This study demonstrated that ML models effectively predict peak wind pressures. "
                    "Random Forest is recommended for tabular wind pressure data. "
                    "Future work should validate with real TPU records."
                ),
            },
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


def _long_paper():
    return {
        "sections": [
            {
                "heading": "INTRODUCTION",
                "content": "This study develops machine learning models for predicting peak wind pressure coefficients on low-rise gable roof buildings.",
            },
            {
                "heading": "METHODOLOGY",
                "content": "Random forest and gradient boosting models were trained with ten-fold cross-validation on the full aerodynamic database.",
            },
            {
                "heading": "RESULTS",
                "content": "The random forest model achieved a coefficient of determination of 0.987 and a root mean square error of 0.0123 on the held-out test set.",
            },
        ]
    }


class TestGenerateAbstractRegressions:
    def test_trim_does_not_cut_at_decimal_point(self):
        # Trimming used rfind("."), which found the "." of "0.987" -> "... of 0."
        abstract = generate_abstract(_long_paper(), max_words=46)
        assert not abstract.endswith("0.")
        assert abstract.endswith("database.")
        assert len(abstract.split()) <= 46

    def test_trim_never_ends_mid_sentence(self):
        paper = {
            "sections": [
                {"heading": "INTRODUCTION", "content": "This study addresses roof wind loads."},
                {
                    "heading": "METHODOLOGY",
                    "content": "A random forest model with five hundred trees and a gradient boosting model with tuned learning rate and depth were trained using ten-fold cross-validation on the aerodynamic database.",
                },
            ]
        }
        abstract = generate_abstract(paper, max_words=20)
        assert abstract == "This study addresses roof wind loads."

    @pytest.mark.parametrize("max_words", [10, 25, 33, 50, 60, 250])
    def test_never_exceeds_max_words(self, max_words):
        for paper in (_make_paper(), _long_paper()):
            abstract = generate_abstract(paper, max_words=max_words)
            assert len(abstract.split()) <= max_words
            assert abstract.endswith((".", "!", "?", "..."))

    def test_single_overlong_sentence_is_truncated_with_ellipsis(self):
        abstract = generate_abstract(_long_paper(), max_words=5)
        assert abstract == "This study develops machine learning..."

    def test_non_positive_max_words(self):
        assert generate_abstract(_make_paper(), max_words=0) == ""

    def test_abbreviations_do_not_break_sentences(self):
        paper = {
            "sections": [
                {
                    "heading": "RESULTS",
                    "content": "As shown in Fig. 3, the random forest achieved an R2 of 0.95 on the test set. Other results follow here.",
                }
            ]
        }
        abstract = generate_abstract(paper)
        assert abstract.startswith("As shown in Fig. 3, the random forest")

    def test_purpose_does_not_match_claim(self):
        # "aim" used to match inside "claim" and picked the wrong sentence.
        paper = {
            "sections": [
                {
                    "heading": "INTRODUCTION",
                    "content": "Several researchers claim that deep networks outperform tree ensembles. "
                    "This study compares both families on a common wind pressure benchmark.",
                }
            ]
        }
        assert generate_abstract(paper).startswith("This study compares")

    def test_none_heading_content_and_subsections(self):
        paper = {
            "sections": [
                {"heading": None, "content": None},
                {"heading": "INTRODUCTION", "content": None, "subsections": None},
                None,
            ]
        }
        assert generate_abstract(paper) == ""

    def test_nested_subsection_text_is_used(self):
        paper = {
            "sections": [
                {
                    "heading": "METHODOLOGY",
                    "subsections": [
                        {
                            "heading": "Models",
                            "subsections": [
                                {
                                    "heading": "Ensembles",
                                    "content": "A gradient boosting model was trained on 5184 samples.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        assert "gradient boosting model" in generate_abstract(paper)

    def test_experimental_template_headings_provide_methods(self):
        paper = {
            "sections": [
                {
                    "heading": "EXPERIMENTAL PROGRAM",
                    "content": "Six RC beams were tested under four-point bending using a displacement-controlled approach.",
                },
            ]
        }
        assert "four-point bending" in generate_abstract(paper)

    def test_no_duplicate_sentences(self):
        text = "This study proposes a framework for automated crack detection in concrete."
        paper = {
            "sections": [
                {"heading": "INTRODUCTION AND METHODOLOGY", "content": text},
                {"heading": "CONCLUSIONS", "content": text},
            ]
        }
        assert generate_abstract(paper).count("This study proposes") == 1


class TestImproveAbstractRegressions:
    def test_foundation_is_not_a_result(self):
        result = improve_abstract(
            "This study proposes a method for foundation design of piles.", {}
        )
        assert result["has_results"] is False

    def test_short_threshold_scales_with_journal_limit(self):
        # A 90-word abstract for a 150-word limit was called "short (aim for 150-150)".
        result = improve_abstract(" ".join(["word"] * 90) + ".", {}, max_words=150)
        assert not any("short" in s for s in result["suggestions"])
        result = improve_abstract("Too short.", {}, max_words=150)
        assert any("Aim for 90-150 words" in s for s in result["suggestions"])

    def test_improved_adds_missing_results_from_body(self):
        abstract = "This study proposes a method for structural analysis."
        result = improve_abstract(abstract, _make_paper())
        assert result["improved"].startswith(abstract)
        assert "0.9999" in result["improved"]

    def test_improved_trims_overlong_abstract_at_sentence_boundary(self):
        abstract = " ".join(
            f"This study shows model result number {i} with an effective accuracy of 0.9{i}."
            for i in range(10)
        )
        result = improve_abstract(abstract, _make_paper(), max_words=50)
        assert len(result["improved"].split()) <= 50
        assert result["improved"].endswith(".")

    def test_improved_is_none_when_nothing_to_change(self):
        abstract = (
            "This study investigates ML models for wind pressure prediction. "
            "Three models (RF, GBR, DNN) were compared using 10-fold cross-validation. "
            "Results show RF achieved R-squared of 0.999 and RMSE of 0.012. "
            "The framework provides an effective and practical surrogate for wind tunnel testing."
        )
        assert improve_abstract(abstract, _make_paper())["improved"] is None

    def test_none_abstract(self):
        result = improve_abstract(None, _make_paper())
        assert result["word_count"] == 0
        assert result["score"] < 50
