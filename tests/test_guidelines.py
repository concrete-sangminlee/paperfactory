"""Tests for the central journal guideline registry and the guideline JSON files."""

import json
import os

import pytest

from utils.guidelines import (
    DEFAULT_GUIDELINES_DIR,
    GUIDELINES_DIR_ENV,
    JOURNAL_ALIASES,
    UnknownJournalError,
    available_journal_keys,
    list_journals,
    load_guideline,
    resolve_journal_key,
    validate_guideline,
)

ALL_KEYS = sorted(f[:-5] for f in os.listdir(DEFAULT_GUIDELINES_DIR) if f.endswith(".json"))


class TestGuidelineFiles:
    def test_fifteen_journals_bundled(self):
        assert len(ALL_KEYS) == 15

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_schema_valid(self, key):
        assert validate_guideline(load_guideline(key)) == []

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_reference_examples_are_strings(self, key):
        examples = load_guideline(key)["references"]["examples"]
        assert all(isinstance(e, str) and e.strip() for e in examples)

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_core_sections_listed(self, key):
        sections = " ".join(load_guideline(key)["manuscript_structure"]["sections"]).lower()
        assert "introduction" in sections
        assert "conclusion" in sections

    def test_every_journal_documented_in_claude_md(self):
        root = os.path.dirname(DEFAULT_GUIDELINES_DIR)
        with open(os.path.join(root, "CLAUDE.md"), encoding="utf-8") as f:
            text = f.read()
        for key in ALL_KEYS:
            assert f"`{key}`" in text, f"{key} missing from CLAUDE.md journal table"


class TestResolveJournalKey:
    def test_canonical(self):
        assert resolve_journal_key("jweia") == "jweia"

    @pytest.mark.parametrize("raw", ["JWEIA", "  jweia ", "jweia.json", "Jweia.JSON"])
    def test_normalisation(self, raw):
        assert resolve_journal_key(raw) == "jweia"

    def test_hyphens_and_spaces(self):
        assert resolve_journal_key("eng-structures") == "eng_structures"
        assert resolve_journal_key("eng structures") == "eng_structures"

    @pytest.mark.parametrize("alias,target", sorted(JOURNAL_ALIASES.items()))
    def test_aliases_point_to_real_files(self, alias, target):
        assert target in ALL_KEYS
        assert resolve_journal_key(alias) == target

    def test_unknown_suggests_close_match(self):
        with pytest.raises(UnknownJournalError, match="Did you mean: jweia"):
            resolve_journal_key("jweai")

    @pytest.mark.parametrize("bad", ["", "   ", None, 42])
    def test_invalid_input(self, bad):
        with pytest.raises(UnknownJournalError):
            resolve_journal_key(bad)

    def test_error_is_backwards_compatible(self):
        with pytest.raises(FileNotFoundError):
            load_guideline("nonexistent_journal")
        with pytest.raises(ValueError):
            load_guideline("nonexistent_journal")

    def test_path_traversal_rejected(self):
        with pytest.raises(UnknownJournalError):
            load_guideline("../pyproject")


class TestLoadGuideline:
    def test_returns_independent_copies(self):
        a = load_guideline("jweia")
        a["journal_name"] = "mutated"
        a["manuscript_structure"]["sections"].append("Bogus")
        b = load_guideline("jweia")
        assert b["journal_name"] != "mutated"
        assert "Bogus" not in b["manuscript_structure"]["sections"]

    def test_env_override_directory(self, tmp_path, monkeypatch):
        custom = load_guideline("jweia")
        custom["journal_name"] = "Custom Journal"
        (tmp_path / "custom_j.json").write_text(json.dumps(custom), encoding="utf-8")
        monkeypatch.setenv(GUIDELINES_DIR_ENV, str(tmp_path))
        assert available_journal_keys() == ["custom_j"]
        assert load_guideline("custom_j")["journal_name"] == "Custom Journal"

    def test_cache_invalidated_on_file_change(self, tmp_path, monkeypatch):
        g = load_guideline("jweia")
        path = tmp_path / "j.json"
        path.write_text(json.dumps(g), encoding="utf-8")
        monkeypatch.setenv(GUIDELINES_DIR_ENV, str(tmp_path))
        assert load_guideline("j")["journal_name"] == g["journal_name"]
        g["journal_name"] = "Changed"
        path.write_text(json.dumps(g), encoding="utf-8")
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert load_guideline("j")["journal_name"] == "Changed"


class TestListJournals:
    def test_metadata(self):
        journals = list_journals()
        assert [j["key"] for j in journals] == ALL_KEYS
        jweia = next(j for j in journals if j["key"] == "jweia")
        assert jweia["abstract_word_limit"] == 250
        assert jweia["highlights_required"] is True
        assert jweia["publisher"] == "Elsevier"


class TestValidateGuideline:
    def test_not_a_dict(self):
        assert validate_guideline([]) == ["Guideline must be a JSON object"]

    def test_reports_missing_keys(self):
        issues = validate_guideline({"journal_name": "X"})
        assert any("publisher" in i for i in issues)
        assert any("manuscript_structure" in i for i in issues)

    def test_bad_types(self):
        g = load_guideline("jweia")
        g["manuscript_structure"]["sections"] = "Introduction"
        g["manuscript_structure"]["abstract_word_limit"] = -1
        g["references"]["examples"] = []
        issues = validate_guideline(g)
        assert len(issues) == 3


def test_empty_guidelines_dir_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv(GUIDELINES_DIR_ENV, str(tmp_path))
    with pytest.raises(UnknownJournalError, match="pip install -e"):
        load_guideline("jweia")
