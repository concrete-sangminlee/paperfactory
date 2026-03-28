import json
import os

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
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["summary"]["total"] == 4

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
