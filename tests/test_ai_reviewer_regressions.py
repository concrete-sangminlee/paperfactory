"""Regression tests for utils.ai_reviewer (bugs found in review)."""

import datetime

import pytest

from utils.ai_reviewer import _has_recent_year, review_paper

THIS_YEAR = datetime.date.today().year


def _paper(headings=("INTRODUCTION", "METHODOLOGY", "RESULTS AND DISCUSSION", "CONCLUSIONS"), **kw):
    texts = {
        0: "Wind loads are critical. " * 150
        + "However, gaps remain. This study proposes a novel approach.",
        1: "The proposed method uses cross-validation with random seed 42. " * 40,
        2: "Table 1 and Fig. 1 compare models; the proposed model outperforms the baseline "
        "with R² = 0.99. " * 20,
        3: "This study demonstrated accurate prediction. Limitations and future work remain. " * 20,
    }
    sections = [
        {"heading": h, "content": texts.get(i, "Discussion text. " * 50)}
        for i, h in enumerate(headings)
    ]
    base = {
        "title": "ML Wind Pressure Prediction",
        "abstract": "Results show that the proposed model achieved R2 of 0.99. " * 12,
        "sections": sections,
        "tables": [{}] * 3,
        "references": [
            f"[{i}] A. Author, Title, J. Eng. 1 ({THIS_YEAR - 1}) 1-2. doi:10.1000/{i}"
            for i in range(1, 25)
        ],
        "figure_captions": [f"Fig. {i}." for i in range(1, 8)],
        "highlights": ["One highlight", "Two highlight", "Three highlight"],
    }
    base.update(kw)
    return base


def _structure_comments(result):
    return [c["comment"] for c in result["comments"] if c["section"] == "Structure"]


class TestStructure:
    @pytest.mark.parametrize(
        "methods_heading",
        ["2. Materials and Methods", "METHODS", "Proposed Framework", "Experimental program"],
    )
    def test_methodology_synonyms_not_flagged(self, methods_heading):
        headings = (
            "1. Introduction",
            methods_heading,
            "3. Results and Discussion",
            "4. Conclusions",
        )
        result = review_paper(_paper(headings), "jweia")
        assert not any("METHODOLOGY" in c for c in _structure_comments(result))

    def test_korean_headings_recognised(self):
        result = review_paper(
            _paper(("1. 서론", "2. 연구 방법", "3. 결과 및 고찰", "4. 결론")), "ksce_jce"
        )
        assert _structure_comments(result) == []

    def test_journal_specific_section_is_minor(self):
        result = review_paper(
            _paper(("INTRODUCTION", "METHODOLOGY", "RESULTS", "CONCLUSIONS")), "jweia"
        )
        comments = [c for c in result["comments"] if c["section"] == "Structure"]
        assert len(comments) == 1
        assert comments[0]["severity"] == "minor" and "'Discussion'" in comments[0]["comment"]

    def test_unknown_journal_key_gives_generic_review(self):
        result = review_paper(_paper(), "not_a_journal")
        assert result["major_issues"] == 0


class TestRobustness:
    @pytest.mark.parametrize(
        "field", ["abstract", "references", "sections", "tables", "figure_captions"]
    )
    def test_none_fields(self, field):
        result = review_paper(_paper(**{field: None}), "jweia")
        assert result["total_issues"] >= 1

    def test_none_heading_and_subsections(self):
        paper = _paper()
        paper["sections"].append({"heading": None, "content": None, "subsections": None})
        assert review_paper(paper, "jweia")["decision"]


class TestReferencesAndMetrics:
    def test_asce_doi_digits_not_taken_as_year(self):
        refs = [
            f'Smith, J. {THIS_YEAR - 1}. "Title {i}." J. Struct. Eng., 149 (2), 04022{i:03d}. '
            f"https://doi.org/10.1061/(ASCE)ST.1943-541X.000{i:04d}."
            for i in range(1, 20)
        ]
        assert all(_has_recent_year(r) for r in refs)
        result = review_paper(_paper(references=refs), "asce_jse")
        assert not any(c["section"] == "References" for c in result["comments"])

    def test_superscript_r_squared_is_a_metric(self):
        result = review_paper(_paper(), "jweia")
        assert not any(
            "quantitative performance metrics" in c["comment"] for c in result["comments"]
        )


class TestDecision:
    def test_only_minor_issues_is_not_major_revision(self):
        # 0 major issues but several minor ones: previously "Major revision required".
        paper = _paper(abstract="Short abstract.", tables=[], figure_captions=[])
        result = review_paper(paper, "jweia")
        assert result["major_issues"] == 0 and result["minor_issues"] > 2
        assert result["decision"] == "Minor revision required"

    def test_highlights_over_journal_limit_flagged(self):
        result = review_paper(_paper(highlights=["x" * 90] * 3), "eng_structures")
        assert any(c["section"] == "Highlights" for c in result["comments"])
