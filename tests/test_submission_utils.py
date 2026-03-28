import os
import pytest

from utils.submission_utils import submission_checklist, generate_cover_letter, reformat_paper


def _make_paper():
    return {
        "title": "ML Wind Pressure Prediction",
        "authors": "S.M. Lee, J.H. Kim",
        "abstract": " ".join(["word"] * 200),
        "keywords": "wind pressure; machine learning; prediction",
        "highlights": ["Highlight one here", "Highlight two here", "Highlight three here"],
        "sections": [
            {"heading": "INTRODUCTION", "content": "Intro."},
            {"heading": "CONCLUSIONS", "content": "Conclusions."},
        ],
        "tables": [{"caption": f"Table {i}.", "headers": ["A"], "rows": [["1"]]} for i in range(4)],
        "references": [
            f"[{i}] Author, Title, J. ({2024}) 1-10. https://doi.org/10.1000/test{i}"
            for i in range(1, 20)
        ],
        "figure_captions": [f"Fig. {i}." for i in range(1, 8)],
        "data_availability": "Data available on request.",
        "acknowledgments": "Thanks.",
    }


class TestSubmissionChecklist:
    def test_valid_paper_ready(self):
        paper = _make_paper()
        paper["graphical_abstract"] = True  # JWEIA requires graphical abstract
        result = submission_checklist(paper, "jweia", figures=["f"] * 7)
        assert result["ready"] is True

    def test_missing_highlights_not_ready(self):
        paper = _make_paper()
        paper["highlights"] = []
        result = submission_checklist(paper, "jweia")
        hl_items = [i for i in result["items"] if "Highlights" in i["description"] and "provided" in i["description"]]
        assert len(hl_items) > 0
        assert hl_items[0]["passed"] is False

    def test_no_highlights_check_for_non_required(self):
        paper = _make_paper()
        paper["highlights"] = []
        result = submission_checklist(paper, "asce_jse")
        hl_items = [i for i in result["items"] if "Highlights provided" in i["description"]]
        assert len(hl_items) == 0

    def test_returns_journal_name(self):
        result = submission_checklist(_make_paper(), "jweia")
        assert "Wind" in result["journal"]

    def test_all_items_have_description(self):
        result = submission_checklist(_make_paper(), "eng_structures")
        for item in result["items"]:
            assert "description" in item
            assert "passed" in item


class TestGenerateCoverLetter:
    def test_contains_title(self):
        letter = generate_cover_letter(_make_paper(), "jweia")
        assert "ML Wind Pressure Prediction" in letter

    def test_contains_journal_name(self):
        letter = generate_cover_letter(_make_paper(), "jweia")
        assert "Wind Engineering" in letter

    def test_contains_authors(self):
        letter = generate_cover_letter(_make_paper(), "jweia")
        assert "S.M. Lee" in letter

    def test_custom_editor(self):
        letter = generate_cover_letter(_make_paper(), "jweia", editor_name="Prof. Smith")
        assert "Prof. Smith" in letter

    def test_contains_date(self):
        letter = generate_cover_letter(_make_paper(), "jweia")
        assert "2026" in letter


class TestReformatPaper:
    def test_adds_metadata(self):
        paper = _make_paper()
        new = reformat_paper(paper, "jweia", "eng_structures")
        assert new["_reformatted"]["from"] == "jweia"
        assert new["_reformatted"]["to"] == "eng_structures"

    def test_warns_on_missing_highlights(self):
        paper = _make_paper()
        paper["highlights"] = []
        new = reformat_paper(paper, "asce_jse", "jweia")
        assert any("highlights" in w.lower() for w in new.get("_warnings", []))

    def test_warns_on_abstract_too_long(self):
        paper = _make_paper()
        paper["abstract"] = " ".join(["word"] * 300)  # Over 250 limit for jweia
        new = reformat_paper(paper, "eng_structures", "jweia")
        assert any("abstract" in w.lower() for w in new.get("_warnings", []))

    def test_preserves_content(self):
        paper = _make_paper()
        new = reformat_paper(paper, "jweia", "asce_jse")
        assert new["title"] == paper["title"]
        assert new["abstract"] == paper["abstract"]
