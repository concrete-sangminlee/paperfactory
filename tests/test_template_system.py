import pytest

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
