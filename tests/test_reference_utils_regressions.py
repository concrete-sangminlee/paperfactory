"""Regression tests for utils.reference_utils (bugs found in review)."""

import pytest

from utils.guidelines import load_guideline
from utils.reference_utils import (
    check_duplicates,
    detect_citation_style,
    extract_doi,
    extract_year,
    has_doi,
    normalize_references,
    validate_references,
)

GUIDELINE = load_guideline("eng_structures")


def _issues(refs, kind):
    return [i for i in validate_references(refs, GUIDELINE) if kind in i["issue"]]


class TestExtractYear:
    @pytest.mark.parametrize(
        "ref, year",
        [
            (
                "[1] F. Bre, Title, Energy Build. 158 (2018) 1429-1441. "
                "https://doi.org/10.1016/j.enbuild.2017.11.045",
                2018,
            ),
            (
                'Smith, J. 2023. "Title." J. Struct. Eng., 149 (2), 04022231. '
                "https://doi.org/10.1061/(ASCE)ST.1943-541X.0003512.",
                2023,
            ),
            ('Smith, J. 2013. "Lessons from the 2011 Tohoku earthquake." J., 1 (1), 1-2.', 2013),
            ("[2] A. B, Lessons of 2011, Wiley, 2015.", 2015),
            ("[3] EN 1993-1-3, Eurocode 3: Design of Steel Structures, CEN, Brussels, 2006.", 2006),
            ("[4] A. Author, Title, Struct. Saf. 10 (2010) 2024-2031.", 2010),
            ("1. Boore DM, Atkinson GM. Title. Earthquake Spectra 2008; 24(1):99-138.", 2008),
            ("[1] Kim, S.; Lee, J. Deep Learning. Buildings 2024, 14, 1523.", 2024),
            ("[1] The 2011 Tohoku earthquake (2011) review, Eng. Struct. 50 (2013) 1-10.", 2013),
            (
                "김철수, 이영희 (2023). 철근콘크리트 보의 전단 거동. 대한토목학회논문집, 43(2), 123-130.",
                2023,
            ),
            (
                "[5] A. Author, Title, J. Eng. 50, 1-10. https://doi.org/10.1016/j.x.2022.114148",
                None,
            ),
            ("[6] A. Author, Title, J. Eng. 50, 120145.", None),
        ],
    )
    def test_publication_year(self, ref, year):
        assert extract_year(ref, current_year=2026) == year

    def test_future_years_ignored(self):
        assert extract_year("[1] A. Author, Title, J. (2090) 1-2.", current_year=2026) is None


class TestDoi:
    @pytest.mark.parametrize(
        "ref",
        [
            "x https://doi.org/10.1016/j.engstruct.2022.114148",
            "x doi:10.1016/j.engstruct.2022.114148",
            "x DOI: 10.1007/s12205-023-1234-5.",
            "x Eng. Struct. 258 (2022) 114148. 10.1016/j.engstruct.2022.114148",
        ],
    )
    def test_doi_forms(self, ref):
        assert has_doi(ref)

    def test_trailing_punctuation_and_brackets(self):
        ref = "(see https://doi.org/10.1061/(ASCE)ST.1943-541X.0003512)."
        assert extract_doi(ref) == "10.1061/(ASCE)ST.1943-541X.0003512"

    def test_page_range_is_not_a_doi(self):
        assert not has_doi("[1] A. Author, Title, J. Eng. 10 (2010) 1-10.")


class TestValidateReferences:
    def test_string_input_is_split_into_lines(self):
        refs = (
            "[1] A. Author, Title, J. 1 (2024) 1-10. https://doi.org/10.1000/x\n\n"
            "[2] B. Author, Title two, J. 2 (2023) 3-4. https://doi.org/10.1000/y\n"
        )
        assert validate_references(refs, GUIDELINE) == []

    def test_none_entries_skipped_and_indices_preserved(self):
        refs = [None, "", "[2] B. Author, No doi ref, J. Eng. 50 (2024) 1-10."]
        issues = validate_references(refs, GUIDELINE)
        assert [i["index"] for i in issues] == [2]

    def test_year_only_inside_doi_is_missing_year(self):
        refs = ["[1] A. Author, Title, J. Eng. 50, 1-10. https://doi.org/10.1016/j.x.2022.114148"]
        assert _issues(refs, "year")

    def test_article_number_is_not_a_year(self):
        assert _issues(["[1] A. Author, Some title, J. Struct. Eng. 50, 120145."], "year")

    def test_bare_doi_accepted(self):
        refs = [
            "[1] A. Author, Title, Eng. Struct. 258 (2022) 114148. 10.1016/j.engstruct.2022.114148"
        ]
        assert not _issues(refs, "DOI")

    def test_guideline_none(self):
        assert validate_references(["[1] A. Author, Title, J. (2020) 1-2."], None) == []


class TestCheckDuplicates:
    def test_string_input(self):
        refs = "[1] A. Author, T, J. (2020) 1-2.\n[2] B. Author, T, J. (2021) 3-4."
        assert check_duplicates(refs) == []

    def test_same_doi_different_formatting(self):
        refs = [
            "[1] F. Bre, J.M. Gimenez, Prediction, Energy Build. 158 (2018) 1429-1441. "
            "https://doi.org/10.1016/j.enbuild.2017.11.045",
            None,
            "[2] F. Bre, J. M. Gimenez, Prediction, Energy Build. 158 (2018) 1429–1441. "
            "doi:10.1016/J.ENBUILD.2017.11.045",
        ]
        dupes = check_duplicates(refs)
        assert len(dupes) == 1
        assert (dupes[0]["index_a"], dupes[0]["index_b"], dupes[0]["reason"]) == (0, 2, "same DOI")

    def test_punctuation_only_difference(self):
        refs = [
            "[1] A. Author, Title one, J. Eng. 50 (2024) 1-10.",
            "[2] A. Author, Title one, J. Eng. 50 (2024) 1-10",
        ]
        assert len(check_duplicates(refs)) == 1


class TestDetectCitationStyle:
    @pytest.mark.parametrize(
        "ref",
        [
            'van der Berg, J. 2020. "Title." J. Struct. Eng., 1 (1), 1-2.',
            'O\'Brien, E. J. 2019. "Bridge loads." J. Bridge Eng., 1 (1), 1-2.',
            "ACI Committee 318. 2019. Building code requirements. Farmington Hills, MI: ACI.",
            "Bre, F., & Gimenez, J. M. (2018). Prediction. Energy Build., 158, 1429-1441.",
        ],
    )
    def test_author_date(self, ref):
        assert detect_citation_style([ref]) == "author-date"

    def test_string_input(self):
        assert detect_citation_style(
            "[1] A. B, T, J. (2020) 1-2.\n[2] C. D, T, J. (2021) 3-4."
        ) == ("numbered")

    def test_majority_vote(self):
        refs = ["Odd first entry without a year"] + [
            f"[{i}] A. B, T, J. (2020) 1-2." for i in range(5)
        ]
        assert detect_citation_style(refs) == "numbered"


def test_normalize_references():
    assert normalize_references(None) == []
    assert normalize_references("a\n\n b \n") == ["a", "b"]
    assert normalize_references(["a", None, " ", "b"]) == ["a", "b"]
