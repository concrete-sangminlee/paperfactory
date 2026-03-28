import json
import os

from utils.reference_utils import check_duplicates, detect_citation_style, validate_references

GUIDELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "guidelines")


def _load_guideline(key):
    with open(os.path.join(GUIDELINES_DIR, f"{key}.json")) as f:
        return json.load(f)


class TestValidateReferences:
    def test_missing_doi_warning(self):
        refs = ["[1] A. Author, Some title, J. Struct. Eng. 50 (2024) 100-110."]
        guideline = _load_guideline("eng_structures")
        issues = validate_references(refs, guideline)
        assert any("DOI" in i["issue"] for i in issues)

    def test_valid_ref_no_issues(self):
        refs = [
            "[1] A. Author, Some title, J. Struct. Eng. 50 (2024) 100-110. https://doi.org/10.1000/test."
        ]
        guideline = _load_guideline("eng_structures")
        issues = validate_references(refs, guideline)
        doi_issues = [i for i in issues if "DOI" in i["issue"]]
        assert len(doi_issues) == 0

    def test_missing_year_warning(self):
        refs = ["[1] A. Author, Some title, J. Struct. Eng."]
        guideline = _load_guideline("eng_structures")
        issues = validate_references(refs, guideline)
        assert any("year" in i["issue"].lower() for i in issues)

    def test_returns_index(self):
        refs = [
            "[1] A. Author, Good ref, J. Eng. 50 (2024) 1-10. https://doi.org/10.1000/a.",
            "[2] B. Author, No doi ref, J. Eng. 50 (2024) 1-10.",
        ]
        guideline = _load_guideline("eng_structures")
        issues = validate_references(refs, guideline)
        doi_issues = [i for i in issues if "DOI" in i["issue"]]
        assert doi_issues[0]["index"] == 1


class TestCheckDuplicates:
    def test_detects_exact_duplicates(self):
        refs = [
            "[1] A. Author, Title one, J. Eng. 50 (2024) 1-10.",
            "[2] A. Author, Title one, J. Eng. 50 (2024) 1-10.",
        ]
        dupes = check_duplicates(refs)
        assert len(dupes) == 1

    def test_no_duplicates(self):
        refs = [
            "[1] A. Author, Title one, J. Eng. 50 (2024) 1-10.",
            "[2] B. Author, Title two, J. Eng. 51 (2025) 11-20.",
        ]
        dupes = check_duplicates(refs)
        assert len(dupes) == 0


class TestDetectCitationStyle:
    def test_numbered_style(self):
        refs = ["[1] A. Author, Title, J. Eng. 50 (2024) 1-10."]
        assert detect_citation_style(refs) == "numbered"

    def test_author_date_style(self):
        refs = ['Author, A. 2024. "Title." J. Eng., 50(1), 1-10.']
        assert detect_citation_style(refs) == "author-date"
