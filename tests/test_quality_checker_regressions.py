"""Regression tests for utils.quality_checker (bugs found in review)."""

import datetime

import pytest

from utils.quality_checker import (
    _count_body_words,
    analyze_sections,
    check_paper,
    highlight_rules,
    load_guideline,
    parse_keywords,
    parse_range,
)

THIS_YEAR = datetime.date.today().year


def _paper(**overrides):
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
        "tables": [{"caption": f"Table {i}."} for i in range(3)],
        "references": [
            f"[{i}] A. Author, Title {i}, J. Eng. {i} ({THIS_YEAR - 1}) 1-10. "
            f"https://doi.org/10.1000/test{i}"
            for i in range(1, 20)
        ],
        "figure_captions": [f"Fig. {i}." for i in range(1, 8)],
        "highlights": ["Highlight one", "Highlight two", "Highlight three"],
        "data_availability": "Data available on request.",
    }
    base.update(overrides)
    return base


def _sections(*headings, words=1600):
    return [{"heading": h, "content": " ".join(["text"] * words)} for h in headings]


def _check(result, name):
    return next((c for c in result["checks"] if c["name"] == name), None)


class TestMissingOrOddlyTypedFields:
    @pytest.mark.parametrize(
        "field",
        [
            "abstract",
            "title",
            "keywords",
            "highlights",
            "sections",
            "references",
            "tables",
            "figure_captions",
            "data_availability",
        ],
    )
    def test_none_field_does_not_crash(self, field):
        result = check_paper(_paper(**{field: None}), "jweia")
        assert 0 <= result["score"] <= 100

    def test_section_with_none_heading_content_and_subsections(self):
        paper = _paper(sections=[{"heading": None, "content": None, "subsections": None}])
        result = check_paper(paper, "jweia")
        assert _check(result, "required_sections")["passed"] is False

    def test_keywords_as_list(self):
        result = check_paper(_paper(keywords=["wind", "machine learning", "CFD"]), "jweia")
        assert _check(result, "keywords")["value"] == 3

    def test_keywords_comma_separated_string(self):
        result = check_paper(_paper(keywords="wind, machine learning, CFD"), "jweia")
        assert _check(result, "keywords")["value"] == 3

    def test_highlights_newline_string_counted_as_items(self):
        hl = "- First highlight\n- Second highlight\n- Third highlight"
        check = _check(check_paper(_paper(highlights=hl), "jweia"), "highlights")
        assert check["value"] == 3
        assert check["passed"] is True

    def test_single_highlight_string_is_one_item(self):
        check = _check(check_paper(_paper(highlights="Only one highlight"), "jweia"), "highlights")
        assert check["value"] == 1
        assert check["passed"] is False

    def test_blank_reference_entries_not_counted(self):
        refs = _paper()["references"][:10] + ["", "   ", None] * 4
        check = _check(check_paper(_paper(references=refs), "jweia"), "reference_count")
        assert check["value"] == 10

    def test_sub_subsections_counted_in_body(self):
        paper = {
            "sections": [
                {
                    "heading": "A",
                    "content": "a b",
                    "subsections": [
                        {
                            "heading": "A.1",
                            "content": "c",
                            "subsections": [{"heading": "A.1.1", "content": "d e f g"}],
                        }
                    ],
                }
            ]
        }
        assert _count_body_words(paper) == 7


class TestRecentReferenceYears:
    def test_asce_doi_digits_are_not_the_year(self):
        # ASCE DOIs contain "1943-541X"; the old code took the *last* 4-digit number.
        refs = [
            f'Smith, J. {THIS_YEAR - 1}. "Title {i}." J. Struct. Eng., 149 (2), 04022{i:03d}. '
            f"https://doi.org/10.1061/(ASCE)ST.1943-541X.000{i:04d}."
            for i in range(1, 20)
        ]
        check = _check(check_paper(_paper(references=refs), "asce_jse"), "recent_references")
        assert check["passed"] is True
        assert check["value"] == "100%"

    def test_page_ranges_are_not_the_year(self):
        refs = [
            f"[{i}] A. Author, Title, Struct. Saf. 10 (2010) {THIS_YEAR - 2}-{THIS_YEAR + 5}."
            for i in range(1, 20)
        ]
        check = _check(check_paper(_paper(references=refs), "jweia"), "recent_references")
        assert check["passed"] is False
        assert check["value"] == "0%"


class TestRequiredSectionsUseGuideline:
    def test_core_check_still_critical(self):
        result = check_paper(_paper(sections=_sections("METHODOLOGY", words=6000)), "jweia")
        assert _check(result, "required_sections")["missing"] == ["INTRODUCTION", "CONCLUSION"]
        assert result["passed"] is False

    @pytest.mark.parametrize(
        "headings",
        [
            ("1. Introduction", "2. Methods", "3. Results and discussion", "4. Concluding remarks"),
            ("INTRODUCTION", "MATERIALS AND METHODS", "RESULTS & DISCUSSION", "CONCLUSION"),
            ("1 Introduction", "2 Proposed framework", "3 Results", "4 Discussion", "5 Summary"),
            ("I. INTRODUCTION", "II. METHODOLOGY", "III. RESULTS", "IV. DISCUSSION", "V. CONCLUSIONS"),
        ],
    )
    def test_synonyms_and_numbering_satisfy_jweia_structure(self, headings):
        result = check_paper(_paper(sections=_sections(*headings)), "jweia")
        assert _check(result, "required_sections")["passed"] is True
        assert _check(result, "journal_sections")["passed"] is True
        assert result["score"] == 100

    def test_korean_headings(self):
        paper = _paper(sections=_sections("1. 서론", "2. 연구 방법", "3. 결과 및 고찰", "4. 결론"))
        result = check_paper(paper, "ksce_jce")
        assert _check(result, "required_sections")["passed"] is True
        assert _check(result, "journal_sections")["passed"] is True

    def test_missing_discussion_is_a_warning_not_critical(self):
        paper = _paper(sections=_sections("INTRODUCTION", "METHODOLOGY", "RESULTS", "CONCLUSIONS"))
        result = check_paper(paper, "jweia")  # JWEIA lists Results and Discussion separately
        check = _check(result, "journal_sections")
        assert check["severity"] == "warning"
        assert check["passed"] is False
        assert check["missing"] == ["Discussion"]
        assert result["passed"] is True

    def test_combined_guideline_entry_satisfied_by_results(self):
        # KSCE lists a combined "Results and Discussion".
        paper = _paper(sections=_sections("INTRODUCTION", "METHODOLOGY", "RESULTS", "CONCLUSIONS"))
        assert _check(check_paper(paper, "ksce_jce"), "journal_sections")["passed"] is True

    def test_missing_methodology_reported(self):
        paper = _paper(sections=_sections("INTRODUCTION", "RESULTS AND DISCUSSION", "CONCLUSIONS"))
        check = _check(check_paper(paper, "jweia"), "journal_sections")
        assert check["missing"] == ["Methodology"]

    def test_unrecognised_heading_between_intro_and_results_inferred_as_methods(self):
        paper = _paper(
            sections=_sections(
                "Introduction", "TPU aerodynamic database", "Results and discussion", "Conclusions"
            )
        )
        check = _check(check_paper(paper, "jweia"), "journal_sections")
        assert check["passed"] is True
        assert check["inferred"] == {"Methodology": "TPU aerodynamic database"}

    def test_aci_research_significance_required(self):
        paper = _paper(abstract=" ".join(["w"] * 100))
        check = _check(check_paper(paper, "aci_sj"), "journal_sections")
        assert "Research Significance" in check["missing"]
        paper["sections"].insert(1, {"heading": "RESEARCH SIGNIFICANCE", "content": "x"})
        assert _check(check_paper(paper, "aci_sj"), "journal_sections")["passed"] is True

    def test_asce_main_body(self):
        paper = _paper(sections=_sections("INTRODUCTION", "CONCLUSION", words=3500))
        assert _check(check_paper(paper, "asce_jse"), "journal_sections")["missing"] == [
            "Main Body"
        ]
        paper = _paper(sections=_sections("INTRODUCTION", "WIND TUNNEL TESTS", "CONCLUSION"))
        assert _check(check_paper(paper, "asce_jse"), "journal_sections")["passed"] is True

    def test_optional_sections_do_not_fail(self):
        result = check_paper(_paper(), "eng_structures")  # lists Theory/Background
        check = _check(result, "journal_sections")
        assert check["passed"] is True
        assert "Theory/Background" in check["optional_missing"]

    def test_front_and_back_matter_use_fields_not_headings(self):
        a = analyze_sections(_paper(), load_guideline("jweia"))
        for name in ("Title Page", "Highlights", "Abstract", "Keywords", "References"):
            assert a["matter"][name] is True
            assert name not in a["missing"] and name not in a["optional_missing"]
        assert a["matter"]["Graphical Abstract"] is False  # reported, never a failure
        assert a["missing"] == []

    def test_back_matter_heading_is_not_a_body_section(self):
        paper = _paper(
            sections=_sections("INTRODUCTION", "CONCLUSION", words=3500)
            + [{"heading": "ACKNOWLEDGMENTS", "content": "Thanks."}]
        )
        a = analyze_sections(paper, load_guideline("asce_jse"))
        assert a["missing"] == ["Main Body"]
        assert a["matter"]["Acknowledgments"] is True

    @pytest.mark.parametrize("journal", ["jweia", "eng_structures"])
    def test_demo_structure_scores_100(self, journal):
        # Mirrors examples/demo_*/generate_paper.py (same headings and front matter).
        paper = _paper(
            keywords="a; b; c; d; e; f; g",
            highlights=[f"Highlight number {i}" for i in range(5)],
            acknowledgments="Thanks.",
        )
        result = check_paper(paper, journal, figures=[f"fig{i}.png" for i in range(8)])
        assert result["score"] == 100
        assert result["passed"] is True


class TestJournalAwareLimits:
    def test_keyword_range_from_guideline(self):
        check = _check(check_paper(_paper(keywords="wind; pressure"), "eesd"), "keywords")
        assert check["passed"] is False  # EESD requires 4-7
        assert check["limit"] == "4-7"
        assert _check(check_paper(_paper(keywords="a; b; c; d"), "eesd"), "keywords")["passed"]

    def test_highlight_count_upper_bound(self):
        paper = _paper(highlights=[f"Highlight {i}" for i in range(7)])
        assert _check(check_paper(paper, "jweia"), "highlights")["passed"] is False

    def test_highlight_char_limit_reported(self):
        paper = _paper(highlights=["x" * 86, "short one", "short two"])
        check = _check(check_paper(paper, "eng_structures"), "highlights")
        assert check["char_limit"] == 85
        assert check["over_char_limit"] == [1]

    def test_asce_abstract_minimum(self):
        check = _check(
            check_paper(_paper(abstract=" ".join(["w"] * 100)), "asce_jse"),
            "abstract_word_minimum",
        )
        assert check is not None and check["passed"] is False and check["limit"] == 150

    def test_asce_practical_applications(self):
        check = _check(check_paper(_paper(), "asce_jse"), "practical_applications")
        assert check["passed"] is False
        paper = _paper(practical_applications=" ".join(["w"] * 170))
        assert _check(check_paper(paper, "asce_jse"), "practical_applications")["passed"]

    def test_aci_title_length(self):
        check = _check(check_paper(_paper(title="T" * 80), "aci_sj"), "title_length")
        assert check["passed"] is False and check["limit"] == 70

    def test_data_availability_section_heading_counts(self):
        paper = _paper(data_availability=None)
        paper["sections"].append({"heading": "Data Availability Statement", "content": "On request."})
        assert _check(check_paper(paper, "jweia"), "data_availability")["passed"] is True


class TestHelpers:
    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("1-7", (1, 7)),
            ("up to 9", (None, 9)),
            ("3-10", (3, 10)),
            ("150-200 words for non-academic audience", (150, 200)),
            (None, (None, None)),
            (6, (None, 6)),
        ],
    )
    def test_parse_range(self, spec, expected):
        assert parse_range(spec) == expected

    def test_highlight_rules_from_note(self):
        assert highlight_rules(load_guideline("jweia")) == {"min": 3, "max": 5, "max_chars": 85}
        custom = {"manuscript_structure": {"highlights_note": "2-4 bullet points, max 100 characters"}}
        assert highlight_rules(custom) == {"min": 2, "max": 4, "max_chars": 100}

    def test_parse_keywords_strips_label(self):
        assert parse_keywords("Keywords: wind; CFD.") == ["wind", "CFD"]
