import pytest

from utils.guidelines import UnknownJournalError, load_guideline
from utils.template_system import generate_skeleton, get_template, list_templates


class TestListTemplates:
    def test_returns_all(self):
        templates = list_templates()
        assert len(templates) >= 4

    def test_has_required_fields(self):
        for t in list_templates():
            assert "key" in t
            assert "name" in t
            assert "description" in t


class TestGetTemplate:
    def test_ml_comparison(self):
        t = get_template("ml_comparison")
        assert "ML" in t["name"]
        assert len(t["sections"]) >= 4

    def test_experimental(self):
        t = get_template("experimental")
        assert "Experimental" in t["name"]

    def test_numerical(self):
        t = get_template("numerical")
        assert "Numerical" in t["name"]

    def test_review(self):
        t = get_template("review")
        assert "Review" in t["name"]

    def test_invalid_raises(self):
        with pytest.raises(KeyError):
            get_template("nonexistent")

    def test_all_have_figures_plan(self):
        for key in ["ml_comparison", "experimental", "numerical", "review"]:
            t = get_template(key)
            assert len(t["figures_plan"]) >= 4

    def test_all_have_tables_plan(self):
        for key in ["ml_comparison", "experimental", "numerical", "review"]:
            t = get_template(key)
            assert len(t["tables_plan"]) >= 3


class TestGenerateSkeleton:
    def test_returns_paper_content(self):
        skeleton = generate_skeleton("ml_comparison", topic="Wind pressure ML")
        assert "Wind pressure ML" in skeleton["title"]
        assert len(skeleton["sections"]) >= 4
        assert len(skeleton["figure_captions"]) >= 4

    def test_sections_have_outlines(self):
        skeleton = generate_skeleton("experimental")
        intro = skeleton["sections"][0]
        assert "[OUTLINE]" in intro.get("content", "")

    def test_includes_metadata(self):
        skeleton = generate_skeleton("review", journal_key="jweia")
        assert skeleton["_template"] == "review"
        assert skeleton["_journal"] == "jweia"

    def test_tables_from_plan(self):
        skeleton = generate_skeleton("ml_comparison")
        assert len(skeleton["tables"]) >= 3
        assert "[PLAN]" in skeleton["tables"][0]["caption"]


class TestTemplateRegressions:
    def test_get_template_returns_independent_copy(self):
        t = get_template("review")
        t["figures_plan"].append("LEAK")
        t["sections"].clear()
        fresh = get_template("review")
        assert "LEAK" not in fresh["figures_plan"]
        assert len(fresh["sections"]) >= 4

    def test_template_key_is_case_insensitive(self):
        assert generate_skeleton("ML-Comparison")["_template"] == "ml_comparison"

    def test_invalid_journal_key_raises(self):
        with pytest.raises(UnknownJournalError):
            generate_skeleton("ml_comparison", journal_key="not_a_journal")
        # Backwards compatible with callers catching FileNotFoundError
        with pytest.raises(FileNotFoundError):
            generate_skeleton("ml_comparison", journal_key="nonexistent")

    def test_journal_key_is_normalized(self):
        assert generate_skeleton("ml_comparison", journal_key=" JWEIA ")["_journal"] == "jweia"

    @pytest.mark.parametrize("journal_key", ["aci_sj", "asce_jse", "jweia"])
    def test_abstract_placeholder_uses_journal_limit(self, journal_key):
        limit = load_guideline(journal_key)["manuscript_structure"]["abstract_word_limit"]
        skeleton = generate_skeleton("ml_comparison", journal_key=journal_key)
        assert f"max {limit} words" in skeleton["abstract"]

    def test_highlights_placeholder_follows_journal(self):
        for key in ("jweia", "aci_sj"):
            required = load_guideline(key)["manuscript_structure"].get("highlights_required")
            assert ("highlights" in generate_skeleton("numerical", journal_key=key)) == bool(
                required
            )

    def test_without_journal_defaults(self):
        skeleton = generate_skeleton("experimental")
        assert skeleton["_journal"] == ""
        assert "max 250 words" in skeleton["abstract"]

    def test_every_section_has_content_key(self):
        for template in list_templates():
            for sec in generate_skeleton(template["key"])["sections"]:
                assert isinstance(sec["content"], str)
                for sub in sec.get("subsections", []):
                    assert sub["content"].startswith("[OUTLINE]")
