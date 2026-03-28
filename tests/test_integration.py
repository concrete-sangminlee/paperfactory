"""End-to-end integration tests for the full PaperFactory pipeline."""

import os
import json
import pytest
import numpy as np

from pipeline.orchestrator import PaperPipeline, PipelineStep
from utils.quality_checker import check_paper
from utils.ai_reviewer import review_paper
from utils.submission_utils import submission_checklist, generate_cover_letter, reformat_paper
from utils.word_generator import generate_word
from utils.pdf_generator import generate_pdf
from utils.latex_generator import generate_latex
from utils.reference_utils import validate_references, check_duplicates
from utils.citation_converter import convert_style
from utils.template_system import generate_skeleton
from utils.data_sources import suggest_sources


def _make_full_paper():
    """Create a realistic paper_content dict for integration testing."""
    return {
        "title": "Integration Test: ML Wind Pressure Prediction",
        "authors": "A. Author, B. Author",
        "abstract": " ".join(["This study investigates ML models for wind pressure prediction."] * 20),
        "keywords": "wind pressure; machine learning; test",
        "highlights": ["Test highlight one", "Test highlight two", "Test highlight three"],
        "sections": [
            {"heading": "INTRODUCTION", "content": " ".join(["Introduction text."] * 500)},
            {"heading": "METHODOLOGY", "content": " ".join(["Method text."] * 500),
             "subsections": [
                 {"heading": "Data", "content": " ".join(["Data text."] * 300)},
                 {"heading": "Models", "content": " ".join(["Model text."] * 300)},
             ]},
            {"heading": "RESULTS AND DISCUSSION", "content": " ".join(["Results text with Table 1 and Fig. 1 comparison outperform R-squared."] * 300)},
            {"heading": "CONCLUSIONS", "content": " ".join(["Conclusion with limitations and future work."] * 200)},
        ],
        "tables": [
            {"caption": f"Table {i}.", "headers": ["A", "B"], "rows": [["1", "2"]]}
            for i in range(1, 5)
        ],
        "references": [
            f"[{i}] A. Author, Title of paper {i}, J. Eng. {i} (2024) 1-10. https://doi.org/10.1000/test{i}"
            for i in range(1, 21)
        ],
        "figure_captions": [f"Fig. {i}. Test figure." for i in range(1, 8)],
        "data_availability": "Data available on request.",
        "acknowledgments": "Thanks.",
        "graphical_abstract": True,
    }


class TestEndToEndPipeline:
    """Test the full pipeline from creation to output."""

    def test_pipeline_to_word(self, tmp_path):
        """Pipeline → paper_content → Word output → quality check."""
        # Step 1: Create pipeline
        pipeline = PaperPipeline("ML wind pressure", "jweia")
        pipeline.state.output_dir = str(tmp_path)

        # Step 2: Complete all steps
        pipeline.complete_step(references=["ref1", "ref2"])
        pipeline.complete_step(research_design={"title": "Test"})
        pipeline.complete_step(code_outputs={"done": True}, figure_paths=[])
        pipeline.complete_step(analysis_results={"r2": 0.99})

        paper = _make_full_paper()
        pipeline.complete_step(paper_content=paper)
        assert pipeline.is_complete

        # Step 3: Generate Word
        word_path = generate_word(paper, "jweia", output_dir=str(tmp_path))
        assert os.path.exists(word_path)
        assert os.path.getsize(word_path) > 1000

        # Step 4: Quality check
        result = check_paper(paper, "jweia", figures=["f"] * 7)
        assert result["passed"] is True

    def test_pipeline_to_pdf(self, tmp_path):
        paper = _make_full_paper()
        pdf_path = generate_pdf(paper, "eng_structures", output_dir=str(tmp_path))
        assert os.path.exists(pdf_path)
        assert pdf_path.endswith(".pdf")

    def test_pipeline_to_latex(self, tmp_path):
        paper = _make_full_paper()
        tex_path, bib_path = generate_latex(paper, "jweia", output_dir=str(tmp_path))
        assert os.path.exists(tex_path)
        assert os.path.exists(bib_path)

    def test_all_three_outputs(self, tmp_path):
        """Generate Word, PDF, and LaTeX from the same content."""
        paper = _make_full_paper()
        word = generate_word(paper, "jweia", output_dir=str(tmp_path))
        pdf = generate_pdf(paper, "jweia", output_dir=str(tmp_path))
        tex, bib = generate_latex(paper, "jweia", output_dir=str(tmp_path))
        assert all(os.path.exists(p) for p in [word, pdf, tex, bib])


class TestWorkflowIntegration:
    """Test the full submission workflow."""

    def test_quality_then_review_then_submit(self):
        paper = _make_full_paper()

        # Quality check
        quality = check_paper(paper, "jweia", figures=["f"] * 7)
        assert quality["passed"]

        # AI review
        review = review_paper(paper, "jweia", figures=["f"] * 7)
        assert review["total_issues"] >= 0

        # Submission checklist
        checklist = submission_checklist(paper, "jweia", figures=["f"] * 7)
        assert len(checklist["items"]) > 0

        # Cover letter
        letter = generate_cover_letter(paper, "jweia")
        assert "Integration Test" in letter

    def test_reformat_and_recheck(self):
        paper = _make_full_paper()

        # Reformat from JWEIA to Engineering Structures
        new_paper = reformat_paper(paper, "jweia", "eng_structures")
        assert new_paper["_reformatted"]["to"] == "eng_structures"

        # Check quality for new journal
        result = check_paper(new_paper, "eng_structures", figures=["f"] * 7)
        assert result["score"] > 0

    def test_citation_conversion_roundtrip(self):
        paper = _make_full_paper()
        refs = paper["references"]

        # Convert to author-date
        ad_refs = convert_style(refs, "author_date")
        assert len(ad_refs) == len(refs)

        # Convert back to numbered
        num_refs = convert_style(ad_refs, "numbered")
        assert len(num_refs) == len(refs)
        assert num_refs[0].startswith("[1]")


class TestTemplateIntegration:
    """Test template → skeleton → quality check flow."""

    def test_template_to_skeleton(self):
        skeleton = generate_skeleton("ml_comparison", topic="Wind pressure ML", journal_key="jweia")
        assert len(skeleton["sections"]) >= 4
        assert skeleton["_template"] == "ml_comparison"

    def test_data_source_suggestion(self):
        sources = suggest_sources("wind pressure prediction on low-rise buildings")
        assert len(sources) >= 1
        assert sources[0]["key"] in ("tpu", "nist_aero")


class TestCrossJournalCompatibility:
    """Test that paper_content works across all journals."""

    @pytest.mark.parametrize("journal_key", [
        "asce_jse", "jweia", "eng_structures", "eesd", "buildings_mdpi",
    ])
    def test_generate_all_formats(self, tmp_path, journal_key):
        paper = _make_full_paper()
        word = generate_word(paper, journal_key, output_dir=str(tmp_path))
        pdf = generate_pdf(paper, journal_key, output_dir=str(tmp_path))
        tex, bib = generate_latex(paper, journal_key, output_dir=str(tmp_path))
        assert all(os.path.exists(p) for p in [word, pdf, tex, bib])
