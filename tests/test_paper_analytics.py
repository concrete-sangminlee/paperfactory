import pytest
from utils.paper_analytics import analyze_paper, _count_syllables, _readability_scores, _tokenize


def _make_paper():
    return {
        "abstract": "This study investigates machine learning methods for structural engineering applications.",
        "sections": [
            {"heading": "INTRODUCTION", "content": " ".join(["The research addresses wind pressure prediction."] * 50)},
            {"heading": "METHODOLOGY", "content": " ".join(["The method uses random forest models."] * 50)},
            {"heading": "RESULTS", "content": " ".join(["Results show high prediction accuracy."] * 50)},
            {"heading": "CONCLUSIONS", "content": " ".join(["The study demonstrates effective prediction."] * 30)},
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
