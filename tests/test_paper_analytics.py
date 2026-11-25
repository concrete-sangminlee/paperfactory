import pytest

from utils.paper_analytics import (
    _count_syllables,
    _split_sentences,
    _tokenize,
    analyze_paper,
    split_sentences,
)


def _make_paper():
    return {
        "abstract": "This study investigates machine learning methods for structural engineering applications.",
        "sections": [
            {
                "heading": "INTRODUCTION",
                "content": " ".join(["The research addresses wind pressure prediction."] * 50),
            },
            {
                "heading": "METHODOLOGY",
                "content": " ".join(["The method uses random forest models."] * 50),
            },
            {
                "heading": "RESULTS",
                "content": " ".join(["Results show high prediction accuracy."] * 50),
            },
            {
                "heading": "CONCLUSIONS",
                "content": " ".join(["The study demonstrates effective prediction."] * 30),
            },
        ],
    }


class TestAnalyzePaper:
    def test_returns_all_sections(self):
        result = analyze_paper(_make_paper())
        assert "word_stats" in result
        assert "readability" in result
        assert "section_balance" in result
        assert "vocabulary" in result
        assert "summary" in result

    def test_word_count_positive(self):
        result = analyze_paper(_make_paper())
        assert result["word_stats"]["total_words"] > 100

    def test_readability_in_range(self):
        result = analyze_paper(_make_paper())
        assert -10 < result["readability"]["flesch_kincaid_grade"] < 30

    def test_section_balance_detected(self):
        result = analyze_paper(_make_paper())
        assert "section_words" in result["section_balance"]
        assert len(result["section_balance"]["section_words"]) == 4

    def test_vocabulary_has_top_words(self):
        result = analyze_paper(_make_paper())
        assert len(result["vocabulary"]["top_words"]) > 0

    def test_summary_is_string(self):
        result = analyze_paper(_make_paper())
        assert isinstance(result["summary"], str)
        assert "Total words" in result["summary"]


class TestSyllableCount:
    def test_simple_words(self):
        assert _count_syllables("cat") == 1
        assert _count_syllables("water") == 2
        assert _count_syllables("beautiful") == 3

    def test_engineering_terms(self):
        assert _count_syllables("engineering") >= 4
        assert _count_syllables("structure") >= 2


class TestVocabulary:
    def test_empty_paper(self):
        result = analyze_paper({"sections": []})
        assert result["word_stats"]["total_words"] == 0

    def test_unbalanced_sections(self):
        paper = {
            "sections": [
                {"heading": "INTRO", "content": " ".join(["word"] * 1000)},
                {"heading": "CONCLUSIONS", "content": "short."},
            ],
        }
        result = analyze_paper(paper)
        assert result["section_balance"]["is_balanced"] is False


class TestSentenceSplitting:
    def test_decimals_do_not_split(self):
        assert _split_sentences("The model achieved R2 of 0.95 on test data. It was good.") == [
            "The model achieved R2 of 0.95 on test data.",
            "It was good.",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "As shown in Fig. 3, the error is small.",
            "Bre et al. (2018) found similar results, e.g. for gable roofs.",
            "Weng et al. [4] used gradient boosting, i.e. an ensemble method.",
            "The load follows Eq. (2) and Eqs. 3-4 for ACI 318 vs. Eurocode 2.",
        ],
    )
    def test_abbreviations_do_not_split(self, text):
        assert split_sentences(text) == [text]

    def test_sentence_ends_are_kept(self):
        assert split_sentences("Is it valid? Yes! The load was 5. Specimen A. Done.") == [
            "Is it valid?",
            "Yes!",
            "The load was 5.",
            "Specimen A.",
            "Done.",
        ]

    def test_paragraph_break_ends_sentence(self):
        assert split_sentences("Heading without period\n\nBody text here.") == [
            "Heading without period",
            "Body text here.",
        ]

    def test_korean_sentences(self):
        assert len(split_sentences("본 연구는 강도를 예측한다. 결과를 분석하였다.")) == 2

    def test_sentence_count_in_analysis(self):
        paper = {
            "sections": [
                {"heading": "R", "content": "RF scored 0.95 (see Fig. 2). GBR scored 0.93."}
            ]
        }
        assert analyze_paper(paper)["word_stats"]["total_sentences"] == 2


class TestTokenize:
    def test_possessives_are_one_word(self):
        assert _tokenize("Young's modulus and Poisson\u2019s ratio") == [
            "young's",
            "modulus",
            "and",
            "poisson\u2019s",
            "ratio",
        ]

    def test_non_ascii_letters(self):
        assert _tokenize("Möller's naïve model") == ["möller's", "naïve", "model"]

    def test_korean_words_are_counted(self):
        assert _tokenize("콘크리트 압축강도를 예측하였다.") == [
            "콘크리트",
            "압축강도를",
            "예측하였다",
        ]

    def test_numbers_are_not_words(self):
        assert _tokenize("R2 of 0.95 and CO2 in 2024") == ["r2", "of", "and", "co2", "in"]


class TestSyllableCountRegressions:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("table", 2),
            ("people", 2),
            ("simple", 2),
            ("possible", 3),
            ("tables", 2),
            ("samples", 2),
            ("used", 1),
            ("trained", 1),
            ("compared", 2),
            ("tested", 2),
            ("loaded", 2),
            ("studied", 2),
            ("makes", 1),
            ("structures", 2),
            ("cases", 2),
            ("values", 2),
            ("agree", 2),
            ("whole", 1),
            ("naïve", 2),
            ("콘크리트", 4),
        ],
    )
    def test_common_words(self, word, expected):
        assert _count_syllables(word) == expected


class TestReadability:
    def test_flesch_kincaid_formula(self):
        # 6 words, 1 sentence, 6 syllables
        result = analyze_paper({"abstract": "The cat sat on the mat."})
        r = result["readability"]
        assert r["flesch_kincaid_grade"] == pytest.approx(
            0.39 * 6 / 1 + 11.8 * 6 / 6 - 15.59, abs=0.05
        )
        assert r["flesch_reading_ease"] == pytest.approx(
            206.835 - 1.015 * 6 / 1 - 84.6 * 6 / 6, abs=0.05
        )

    def test_empty_text_has_neutral_scores(self):
        r = analyze_paper({"sections": []})["readability"]
        assert r["flesch_kincaid_grade"] == 0.0
        assert r["academic_level"] == "insufficient text"

    def test_korean_text_is_not_graded(self):
        paper = {
            "sections": [{"heading": "서론", "content": "본 연구는 콘크리트 압축강도를 예측한다."}]
        }
        assert analyze_paper(paper)["readability"]["academic_level"].startswith("n/a")


class TestAnalyzePaperRegressions:
    def test_empty_paper_counts_are_zero(self):
        result = analyze_paper({"sections": []})
        assert result["word_stats"]["total_sentences"] == 0
        assert result["section_balance"]["total_body_words"] == 0
        assert set(result["vocabulary"]) == {
            "type_token_ratio",
            "unique_words",
            "total_words",
            "top_words",
        }

    def test_duplicate_and_missing_headings_keep_all_words(self):
        paper = {
            "sections": [
                {"content": "alpha beta gamma delta."},
                {"content": "epsilon zeta eta theta iota."},
                {"heading": "Results", "content": "one two."},
                {"heading": "Results", "content": "three."},
            ]
        }
        balance = analyze_paper(paper)["section_balance"]
        assert balance["section_words"] == {
            "Unknown": 4,
            "Unknown (2)": 5,
            "Results": 2,
            "Results (2)": 1,
        }
        assert balance["total_body_words"] == 12

    def test_none_values_do_not_crash(self):
        paper = {
            "abstract": None,
            "sections": [{"heading": None, "content": None, "subsections": None}, None],
        }
        result = analyze_paper(paper)
        assert result["word_stats"]["total_words"] == 0

    def test_nested_subsubsections_are_counted(self):
        paper = {
            "sections": [
                {
                    "heading": "M",
                    "content": "Intro text here.",
                    "subsections": [
                        {
                            "heading": "S",
                            "content": "Sub text.",
                            "subsections": [
                                {"heading": "SS", "content": "Deep nested words here."}
                            ],
                        }
                    ],
                }
            ]
        }
        result = analyze_paper(paper)
        assert result["word_stats"]["total_words"] == 9
        assert result["section_balance"]["section_words"] == {"M": 9}

    def test_korean_sections_are_balanced_by_real_counts(self):
        paper = {
            "sections": [
                {
                    "heading": "서론",
                    "content": "본 연구는 콘크리트 압축강도를 예측한다. 실험 결과를 분석하였다.",
                },
                {"heading": "결론", "content": "제안 모델은 정확하다. 향후 연구가 필요하다."},
            ]
        }
        result = analyze_paper(paper)
        assert result["section_balance"]["section_words"] == {"서론": 8, "결론": 6}
        assert any(w["word"] == "모델은" for w in result["vocabulary"]["top_words"])
