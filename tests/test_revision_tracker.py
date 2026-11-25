import json
import os

import pytest

from utils.revision_tracker import (
    address_comment,
    create_revision,
    generate_response_letter,
    track_changes,
)


def _make_comments():
    return [
        {
            "reviewer": "Reviewer 1",
            "comment": "Add more references.",
            "section": "Introduction",
            "type": "major",
        },
        {
            "reviewer": "Reviewer 1",
            "comment": "Fix typo in Eq. 3.",
            "section": "Methodology",
            "type": "minor",
        },
        {
            "reviewer": "Reviewer 2",
            "comment": "Clarify feature selection.",
            "section": "Methodology",
            "type": "major",
        },
        {
            "reviewer": "Reviewer 2",
            "comment": "Improve figure resolution.",
            "section": "Results",
            "type": "minor",
        },
    ]


def _make_paper():
    return {"title": "Test Paper on ML for Wind Engineering"}


class TestCreateRevision:
    def test_creates_record(self):
        rev = create_revision(_make_paper(), _make_comments())
        assert rev["status"] == "in_progress"
        assert len(rev["comments"]) == 4

    def test_counts_major_minor(self):
        rev = create_revision(_make_paper(), _make_comments())
        assert rev["summary"]["major"] == 2
        assert rev["summary"]["minor"] == 2
        assert rev["summary"]["addressed"] == 0

    def test_saves_to_file(self, tmp_path):
        rev = create_revision(_make_paper(), _make_comments(), output_dir=str(tmp_path))
        path = os.path.join(str(tmp_path), "revision_record.json")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["summary"]["total"] == 4
        assert loaded == rev  # the saved record is the returned record

    def test_empty_comments(self):
        rev = create_revision(_make_paper(), [])
        assert rev["summary"]["total"] == 0
        assert rev["status"] == "in_progress"


class TestAddressComment:
    def test_marks_addressed(self):
        rev = create_revision(_make_paper(), _make_comments())
        rev = address_comment(rev, 1, "Added 5 references.", "Section 1 updated.")
        assert rev["comments"][0]["status"] == "addressed"
        assert rev["summary"]["addressed"] == 1

    def test_completes_when_all_addressed(self):
        rev = create_revision(_make_paper(), _make_comments())
        for i in range(1, 5):
            rev = address_comment(rev, i, f"Response {i}", f"Change {i}")
        assert rev["status"] == "complete"
        assert rev["summary"]["addressed"] == 4

    def test_partial_completion(self):
        rev = create_revision(_make_paper(), _make_comments())
        rev = address_comment(rev, 1, "Done", "Fixed")
        rev = address_comment(rev, 3, "Done", "Fixed")
        assert rev["status"] == "in_progress"
        assert rev["summary"]["addressed"] == 2


class TestGenerateResponseLetter:
    def test_generates_text(self):
        rev = create_revision(_make_paper(), _make_comments())
        rev = address_comment(rev, 1, "Added refs.", "5 new refs in Section 1.")
        letter = generate_response_letter(rev)
        assert "RESPONSE TO REVIEWERS" in letter
        assert "Reviewer 1" in letter
        assert "Reviewer 2" in letter
        assert "Added refs." in letter
        assert "PENDING" in letter  # comment 2 not addressed

    def test_all_addressed(self):
        rev = create_revision(_make_paper(), _make_comments())
        for i in range(1, 5):
            rev = address_comment(rev, i, f"Response {i}", f"Change {i}")
        letter = generate_response_letter(rev)
        assert "PENDING" not in letter
        assert "4/4 comments addressed" in letter

    def test_grouped_by_reviewer(self):
        rev = create_revision(_make_paper(), _make_comments())
        letter = generate_response_letter(rev)
        r1_pos = letter.index("Reviewer 1")
        r2_pos = letter.index("Reviewer 2")
        assert r1_pos < r2_pos


class TestTrackChanges:
    def test_detects_additions(self):
        result = track_changes("hello world", "hello world foo bar")
        assert result["added_words"] > 0
        assert result["net_change"] > 0

    def test_detects_removals(self):
        result = track_changes("hello world foo bar", "hello world")
        assert result["removed_words"] > 0
        assert result["net_change"] < 0

    def test_no_changes(self):
        result = track_changes("hello world", "hello world")
        assert result["added_words"] == 0
        assert result["removed_words"] == 0


class TestRevisionRegressions:
    def test_type_is_case_insensitive(self):
        comments = [{"comment": "a", "type": "Major"}, {"comment": "b", "type": " MAJOR "}]
        rev = create_revision(_make_paper(), comments)
        assert rev["summary"]["major"] == 2
        assert [c["type"] for c in rev["comments"]] == ["major", "major"]

    def test_none_fields_get_defaults_and_letter_renders(self):
        comments = [{"comment": None, "type": None, "section": None, "reviewer": None}]
        rev = create_revision(_make_paper(), comments)
        entry = rev["comments"][0]
        assert (entry["reviewer"], entry["section"], entry["type"]) == (
            "Reviewer",
            "General",
            "minor",
        )
        assert "[MINOR] (Section: General)" in generate_response_letter(rev)

    def test_unknown_comment_id_raises(self):
        rev = create_revision(_make_paper(), _make_comments())
        with pytest.raises(KeyError, match="99"):
            address_comment(rev, 99, "Response", "Change")
        assert rev["summary"]["addressed"] == 0

    def test_numeric_string_comment_id(self):
        rev = create_revision(_make_paper(), _make_comments())
        rev = address_comment(rev, "2", "Fixed.", "Eq. 3 corrected.")
        assert rev["comments"][1]["status"] == "addressed"

    def test_multiline_response_is_indented(self):
        rev = create_revision(_make_paper(), _make_comments()[:1])
        rev = address_comment(rev, 1, "First line.\nSecond line.", "Added refs.")
        assert "  First line.\n  Second line." in generate_response_letter(rev)


class TestTrackChangesRegressions:
    def test_repeated_words_are_counted(self):
        # A set-based diff saw no change when existing words were repeated.
        result = track_changes("a b c", "a b c a b c")
        assert (result["added_words"], result["removed_words"], result["net_change"]) == (3, 0, 3)

    def test_removed_duplicates_are_counted(self):
        result = track_changes("the load the load", "the load")
        assert (result["added_words"], result["removed_words"], result["net_change"]) == (0, 2, -2)

    def test_replacement(self):
        result = track_changes("the beam failed in shear", "the beam failed in flexure")
        assert (result["added_words"], result["removed_words"], result["net_change"]) == (1, 1, 0)

    def test_sections_changed_for_paper_dicts(self):
        original = {
            "sections": [
                {"heading": "INTRODUCTION", "content": "Old intro."},
                {"heading": "METHODS", "content": "Same text."},
            ]
        }
        revised = {
            "sections": [
                {"heading": "INTRODUCTION", "content": "New intro with refs."},
                {"heading": "METHODS", "content": "Same text."},
                {"heading": "LIMITATIONS", "content": "Added section."},
            ]
        }
        result = track_changes(original, revised)
        assert result["sections_changed"] == ["INTRODUCTION", "LIMITATIONS"]
        assert result["added_words"] > 0

    def test_documented_keys_present(self):
        result = track_changes("x", "x")
        assert result["sections_changed"] == []
        assert result["similarity"] == 1.0
