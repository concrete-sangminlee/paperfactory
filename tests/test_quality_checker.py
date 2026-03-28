import pytest

from utils.quality_checker import _count_body_words, check_paper, load_guideline


def _make_paper(**overrides):
    """Create a minimal valid paper_content dict with overrides."""
    base = {
        "title": "ML-Based Wind Pressure Prediction Using TPU Database",
        "authors": "S.M. Lee, J.H. Kim",
        "abstract": " ".join(["word"] * 200),
        "keywords": "wind pressure; machine learning; TPU database; prediction",
        "sections": [
            {"heading": "INTRODUCTION", "content": " ".join(["text"] * 1200)},
            {"heading": "METHODOLOGY", "content": " ".join(["text"] * 1500)},
            {"heading": "RESULTS AND DISCUSSION", "content": " ".join(["text"] * 2500)},
            {"heading": "CONCLUSIONS", "content": " ".join(["text"] * 800)},
        ],
        "tables": [
            {"caption": "Table 1.", "headers": ["A"], "rows": [["1"]]},
            {"caption": "Table 2.", "headers": ["A"], "rows": [["1"]]},
            {"caption": "Table 3.", "headers": ["A"], "rows": [["1"]]},
        ],
        "references": [
            f"[{i}] Author{i}, Title, J. Eng. {i} (2024) 1-10. https://doi.org/10.1000/test{i}"
            for i in range(1, 20)
        ],
        "figure_captions": [f"Fig. {i}." for i in range(1, 8)],
        "highlights": ["Highlight 1", "Highlight 2", "Highlight 3"],
        "data_availability": "Data available on request.",
    }
    base.update(overrides)
    return base


class TestCheckPaperPass:
    def test_valid_paper_passes(self):
        paper = _make_paper()
        result = check_paper(paper, "jweia")
        assert result["passed"] is True
        assert result["score"] >= 90

    def test_returns_summary_string(self):
        paper = _make_paper()
        result = check_paper(paper, "jweia")
        assert "Quality Score:" in result["summary"]
        assert "PASS" in result["summary"]

    def test_returns_checks_list(self):
        paper = _make_paper()
        result = check_paper(paper, "jweia")
        assert len(result["checks"]) >= 10
        for c in result["checks"]:
            assert "name" in c
            assert "passed" in c
            assert "severity" in c


class TestCheckPaperFail:
    def test_short_abstract_fails(self):
        paper = _make_paper(abstract="Too short.")
        result = check_paper(paper, "jweia")
        abstract_check = _find_check(result, "abstract_word_count")
        assert abstract_check["passed"] is True  # 2 words < 250 limit, still passes

    def test_abstract_over_limit_fails(self):
        paper = _make_paper(abstract=" ".join(["word"] * 300))
        result = check_paper(paper, "jweia")
        abstract_check = _find_check(result, "abstract_word_count")
        assert abstract_check["passed"] is False

    def test_too_few_references_fails(self):
        paper = _make_paper(references=["[1] One ref (2024)."])
        result = check_paper(paper, "jweia")
        ref_check = _find_check(result, "reference_count")
        assert ref_check["passed"] is False
        assert result["passed"] is False

    def test_too_few_figures_fails(self):
        paper = _make_paper(figure_captions=["Fig. 1."])
        result = check_paper(paper, "jweia")
        fig_check = _find_check(result, "figure_count")
        assert fig_check["passed"] is False

    def test_missing_sections_fails(self):
        paper = _make_paper(
            sections=[
                {"heading": "METHODOLOGY", "content": " ".join(["text"] * 6000)},
            ]
        )
        result = check_paper(paper, "jweia")
        sec_check = _find_check(result, "required_sections")
        assert sec_check["passed"] is False
        assert "INTRODUCTION" in sec_check["missing"]

    def test_low_body_words_fails(self):
        paper = _make_paper(
            sections=[
                {"heading": "INTRODUCTION", "content": "Short intro."},
                {"heading": "CONCLUSIONS", "content": "Short conclusion."},
            ]
        )
        result = check_paper(paper, "jweia")
        body_check = _find_check(result, "body_word_count")
        assert body_check["passed"] is False

    def test_no_title_fails(self):
        paper = _make_paper(title="")
        result = check_paper(paper, "jweia")
        title_check = _find_check(result, "title")
        assert title_check["passed"] is False


class TestCheckPaperWarnings:
    def test_old_references_warning(self):
        old_refs = [f"[{i}] Author, Title, J. ({2010}) 1-10." for i in range(1, 20)]
        paper = _make_paper(references=old_refs)
        result = check_paper(paper, "jweia")
        recent_check = _find_check(result, "recent_references")
        assert recent_check["passed"] is False

    def test_no_highlights_warning_for_required_journal(self):
        paper = _make_paper(highlights=[])
        result = check_paper(paper, "jweia")
        hl_check = _find_check(result, "highlights")
        assert hl_check is not None
        assert hl_check["passed"] is False

    def test_no_highlights_check_for_non_required_journal(self):
        paper = _make_paper(highlights=[])
        result = check_paper(paper, "asce_jse")
        hl_check = _find_check(result, "highlights")
        assert hl_check is None  # ASCE doesn't require highlights


class TestCountBodyWords:
    def test_counts_sections(self):
        paper = {
            "sections": [
                {"heading": "A", "content": "one two three"},
                {"heading": "B", "content": "four five"},
            ]
        }
        assert _count_body_words(paper) == 5

    def test_counts_subsections(self):
        paper = {
            "sections": [
                {
                    "heading": "A",
                    "content": "one two",
                    "subsections": [
                        {"heading": "A.1", "content": "three four five"},
                    ],
                },
            ]
        }
        assert _count_body_words(paper) == 5


class TestLoadGuideline:
    def test_loads_valid_guideline(self):
        g = load_guideline("jweia")
        assert "journal_name" in g

    def test_raises_for_invalid_journal(self):
        with pytest.raises(FileNotFoundError):
            load_guideline("nonexistent_journal")


class TestAllJournals:
    """Verify check_paper works with all 10 supported journals."""

    @pytest.mark.parametrize(
        "journal_key",
        [
            "asce_jse",
            "aci_sj",
            "jweia",
            "jbe",
            "eng_structures",
            "eesd",
            "thin_walled",
            "cem_con_comp",
            "comput_struct",
            "autom_constr",
        ],
    )
    def test_valid_paper_passes_all_journals(self, journal_key):
        paper = _make_paper(abstract=" ".join(["word"] * 100))
        result = check_paper(paper, journal_key)
        assert result["passed"] is True
        assert result["score"] >= 70


def _find_check(result, name):
    for c in result["checks"]:
        if c["name"] == name:
            return c
    return None
