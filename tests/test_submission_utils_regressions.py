"""Regression tests for utils.submission_utils (bugs found in review)."""

import copy

import pytest

from utils.guidelines import UnknownJournalError
from utils.submission_utils import (
    _sentences,
    generate_cover_letter,
    reformat_paper,
    submission_checklist,
)

ELSEVIER_REFS = [
    "[1] F. Bre, J.M. Gimenez, V.D. Fachinotti, Prediction of wind pressure coefficients on "
    "building surfaces using artificial neural networks, Energy Build. 158 (2018) 1429-1441. "
    "https://doi.org/10.1016/j.enbuild.2017.11.045",
    "[2] Y. Weng, S.G. Paal, Machine learning-based wind pressure prediction of low-rise "
    "non-isolated buildings, Eng. Struct. 258 (2022) 114148. "
    "https://doi.org/10.1016/j.engstruct.2022.114148",
    "[3] A. Der Kiureghian, O. Ditlevsen, Aleatory or epistemic? Does it matter?, Struct. Saf. "
    "31 (2009) 105-112. https://doi.org/10.1016/j.strusafe.2008.06.020",
]


def _paper(**overrides):
    base = {
        "title": "ML Wind Pressure Prediction",
        "authors": "S.M. Lee, J.H. Kim",
        "abstract": " ".join(["word"] * 200),
        "keywords": "wind pressure; machine learning; prediction",
        "highlights": ["Highlight one here", "Highlight two here", "Highlight three here"],
        "sections": [
            {"heading": "INTRODUCTION", "content": "Intro."},
            {"heading": "METHODOLOGY", "content": "Methods."},
            {"heading": "RESULTS AND DISCUSSION", "content": "Results."},
            {"heading": "CONCLUSIONS", "content": "Conclusions."},
        ],
        "tables": [{"caption": f"Table {i}."} for i in range(4)],
        "references": [
            f"[{i}] A. Author, Title {i}, J. Eng. {i} (2024) 1-10. https://doi.org/10.1000/test{i}"
            for i in range(1, 20)
        ],
        "figure_captions": [f"Fig. {i}." for i in range(1, 8)],
        "data_availability": "Data available on request.",
        "acknowledgments": "Thanks.",
        "graphical_abstract": "ga.png",
    }
    base.update(overrides)
    return base


def _item(result, prefix):
    return next((i for i in result["items"] if i["description"].startswith(prefix)), None)


class TestChecklistRobustness:
    @pytest.mark.parametrize(
        "field",
        ["title", "authors", "abstract", "keywords", "highlights", "sections", "references"],
    )
    def test_none_field_does_not_crash(self, field):
        result = submission_checklist(_paper(**{field: None}), "jweia")
        assert result["ready"] is False

    def test_blank_lines_in_reference_string_not_counted(self):
        refs = "\n\n".join(
            f"[{i}] A. B, T{i}, J. 1 (2024) 1-10. https://doi.org/10.1/x{i}" for i in range(1, 9)
        )
        item = _item(
            submission_checklist(_paper(references=refs + "\n"), "jweia"), "References present"
        )
        assert item["passed"] is False  # 8 references, not 16 "lines"

    def test_page_range_is_not_a_doi(self):
        refs = [f"[{i}] A. Author, Title, J. Eng. 10 (2024) 1-10." for i in range(1, 20)]
        item = _item(
            submission_checklist(_paper(references=refs), "jweia"), "References include DOIs"
        )
        assert item["passed"] is False

    def test_highlights_as_newline_string(self):
        result = submission_checklist(_paper(highlights="First\nSecond\nThird"), "jweia")
        assert _item(result, "Highlights provided")["passed"] is True

    def test_highlight_character_limit_checks_each_highlight_not_each_character(self):
        result = submission_checklist(_paper(highlights="x" * 300), "jweia")
        assert _item(result, "Highlights within character limit")["passed"] is False

    def test_empty_abstract_is_not_within_limit(self):
        assert (
            _item(submission_checklist(_paper(abstract=""), "jweia"), "Abstract within")["passed"]
            is False
        )

    def test_empty_conflict_field_falls_back_to_acknowledgments(self):
        paper = _paper(conflict_of_interest="")
        assert _item(submission_checklist(paper, "jweia"), "Conflict of interest")["passed"]


class TestChecklistJournalAware:
    def test_asce_specific_items(self):
        result = submission_checklist(_paper(abstract=" ".join(["w"] * 120)), "asce_jse")
        assert _item(result, "Practical Applications")["passed"] is False
        assert _item(result, "Abstract meets journal minimum (150+ words)")["passed"] is False

    def test_reference_style_must_match_journal(self):
        item = _item(submission_checklist(_paper(), "asce_jse"), "Reference list style")
        assert item["passed"] is False  # numbered list submitted to an author-date journal
        assert _item(submission_checklist(_paper(), "jweia"), "Reference list style")["passed"]

    def test_duplicate_references_block_submission(self):
        refs = _paper()["references"]
        refs[5] = refs[4].replace("[5]", "[6]")
        result = submission_checklist(_paper(references=refs), "jweia", figures=["f"] * 7)
        assert _item(result, "No duplicate references")["passed"] is False
        assert result["ready"] is False

    def test_advisory_section_items_do_not_block_ready(self):
        paper = _paper(sections=[{"heading": "INTRODUCTION"}, {"heading": "CONCLUSIONS"}])
        result = submission_checklist(paper, "jweia", figures=["f"] * 7)
        advisory = [i for i in result["items"] if not i["required"]]
        assert any("Methodology" in i["description"] and not i["passed"] for i in advisory)
        assert result["ready"] is True

    def test_highlight_limit_from_guideline_note(self):
        paper = _paper(highlights=["x" * 86, "ok", "ok"])
        item = _item(submission_checklist(paper, "eng_structures"), "Highlights within")
        assert item["passed"] is False and "#1 (86)" in item["note"]


class TestCoverLetter:
    def test_decimals_and_abbreviations_not_split(self):
        abstract = (
            "Wind loads matter. We propose a hybrid model. The model achieved R2 = 0.95 on the "
            "test set. The RMSE was 0.12 kPa, i.e. 30% lower than the baseline (Fig. 3)."
        )
        letter = generate_cover_letter(_paper(abstract=abstract), "jweia")
        assert "R2 = 0.95 on the test set." in letter
        assert "i.e. 30% lower" in letter
        assert "0. 95" not in letter and "0. 12" not in letter

    def test_sentence_splitter(self):
        assert _sentences("A is 1.5 m. B et al. found X. C") == [
            "A is 1.5 m.",
            "B et al. found X.",
            "C",
        ]

    def test_keywords_and_authors_as_lists(self):
        letter = generate_cover_letter(
            _paper(keywords=["wind engineering", "CFD"], authors=["S.M. Lee", "J.H. Kim"]), "jweia"
        )
        assert "S.M. Lee, J.H. Kim" in letter
        assert "['" not in letter

    def test_scope_match_ignores_stopwords(self):
        letter = generate_cover_letter(
            _paper(keywords="theory of everything; quantum gravity"), "jweia"
        )
        assert "specifically in the areas of" not in letter

    def test_scope_match_keeps_keyword_case(self):
        letter = generate_cover_letter(_paper(keywords="TPU database; wind tunnel"), "jweia")
        assert "wind tunnel" in letter

    def test_no_graphical_abstract_claim_without_one(self):
        letter = generate_cover_letter(_paper(graphical_abstract=None), "jweia")
        assert "graphical abstract is provided" not in letter.lower()
        letter = generate_cover_letter(_paper(), "jweia")
        assert "A graphical abstract is provided." in letter

    def test_editor_and_abstract_none(self):
        letter = generate_cover_letter(_paper(abstract=None), "jweia", editor_name=None)
        assert letter.startswith("Dear Editor-in-Chief,")

    def test_alias_key_uses_full_journal_name(self):
        letter = generate_cover_letter(_paper(), "JCSR")
        assert "Journal of Constructional Steel Research" in letter


class TestReformatPaper:
    def test_input_not_mutated(self):
        paper = _paper(_warnings=["pre-existing"], highlights=[])
        snapshot = copy.deepcopy(paper)
        new = reformat_paper(paper, "asce_jse", "jweia")
        new["sections"][0]["heading"] = "CHANGED"
        assert paper == snapshot
        assert new["_warnings"][0] == "pre-existing" and len(new["_warnings"]) > 1

    def test_numbered_to_author_date_converts_and_sorts(self):
        paper = _paper(references=list(reversed(ELSEVIER_REFS)))
        new = reformat_paper(paper, "jweia", "asce_jse")
        refs = new["references"]
        assert [r.split(",")[0] for r in refs] == ["Bre", "Der Kiureghian", "Weng"]
        assert refs[0].startswith("Bre, F., Gimenez, J.M., Fachinotti, V.D. (2018).")
        assert new["_original_references"] == paper["references"]
        assert new["_reformatted"]["references_converted"] is True
        assert any("in-text citations" in w for w in new["_warnings"])

    def test_author_date_to_numbered(self):
        ad = reformat_paper(_paper(references=ELSEVIER_REFS), "jweia", "asce_jse")
        back = reformat_paper(ad, "asce_jse", "eng_structures")
        assert all(r.startswith(f"[{i}] ") for i, r in enumerate(back["references"], 1))
        assert "Energy Build. 158 (2018) 1429-1441" in back["references"][0]

    def test_same_family_not_converted(self):
        new = reformat_paper(_paper(references=ELSEVIER_REFS), "jweia", "eng_structures")
        assert new["references"] == ELSEVIER_REFS
        assert "_original_references" not in new

    def test_conversion_can_be_disabled(self):
        new = reformat_paper(
            _paper(references=ELSEVIER_REFS), "jweia", "asce_jse", convert_references=False
        )
        assert new["references"] == ELSEVIER_REFS
        assert any("conversion was skipped" in w for w in new["_warnings"])

    def test_unknown_source_journal_warns(self):
        new = reformat_paper(_paper(), "not_a_journal", "jweia")
        assert any("Unknown source journal" in w for w in new["_warnings"])

    def test_unknown_target_journal_raises(self):
        with pytest.raises(UnknownJournalError):
            reformat_paper(_paper(), "jweia", "not_a_journal")

    def test_target_specific_warnings(self):
        new = reformat_paper(_paper(graphical_abstract=None), "eng_structures", "thin_walled")
        assert any("graphical abstract" in w for w in new["_warnings"])
        new = reformat_paper(_paper(), "jweia", "asce_jse")
        assert any("Practical Applications" in w for w in new["_warnings"])
        new = reformat_paper(_paper(keywords="a; b"), "jweia", "eesd")
        assert any("keywords" in w for w in new["_warnings"])

    def test_warnings_use_journal_name(self):
        new = reformat_paper(_paper(highlights=[]), "asce_jse", "jweia")
        assert any(
            "Journal of Wind Engineering and Industrial Aerodynamics requires highlights" in w
            for w in new["_warnings"]
        )

    def test_alias_keys_resolved_in_metadata(self):
        new = reformat_paper(_paper(), "JWEIA", "jcsr")
        assert new["_reformatted"]["from"] == "jweia"
        assert new["_reformatted"]["to"] == "steel_comp_struct"
