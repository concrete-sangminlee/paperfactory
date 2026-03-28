import os
import json
import pytest
from pipeline.orchestrator import PaperPipeline, PipelineState, PipelineStep


class TestPipelineState:
    def test_creates_with_defaults(self):
        state = PipelineState(topic="Test topic", journal_key="jweia")
        assert state.current_step == PipelineStep.LITERATURE_REVIEW
        assert state.run_id.startswith("run_")
        assert state.created_at != ""

    def test_save_and_load(self, tmp_path):
        state = PipelineState(topic="Test", journal_key="jweia")
        state.references = ["ref1", "ref2"]
        path = state.save(str(tmp_path / "state.json"))
        assert os.path.exists(path)

        loaded = PipelineState.load(path)
        assert loaded.topic == "Test"
        assert loaded.journal_key == "jweia"
        assert len(loaded.references) == 2


class TestPaperPipeline:
    def test_create_pipeline(self):
        p = PaperPipeline("Wind pressure ML", "jweia")
        assert p.state.current_step == PipelineStep.LITERATURE_REVIEW
        assert p.progress == 0.0
        assert not p.is_complete

    def test_show_status(self):
        p = PaperPipeline("Test", "jweia")
        status = p.show_status()
        assert "Test" in status
        assert "jweia" in status
        assert "CURRENT" in status

    def test_complete_step(self, tmp_path):
        p = PaperPipeline("Test", "jweia")
        p.state.output_dir = str(tmp_path)
        p.complete_step(references=["ref1", "ref2"])
        assert p.state.current_step == PipelineStep.RESEARCH_DESIGN
        assert len(p.state.references) == 2
        assert p.progress == 0.2

    def test_full_pipeline(self, tmp_path):
        p = PaperPipeline("Test", "jweia")
        p.state.output_dir = str(tmp_path)

        p.complete_step(references=["ref1"])
        assert p.state.current_step == PipelineStep.RESEARCH_DESIGN

        p.complete_step(research_design={"title": "T"})
        assert p.state.current_step == PipelineStep.CODE_EXECUTION

        p.complete_step(code_outputs={"data": "ok"}, figure_paths=["fig1.png"])
        assert p.state.current_step == PipelineStep.RESULT_ANALYSIS

        p.complete_step(analysis_results={"r2": 0.99})
        assert p.state.current_step == PipelineStep.PAPER_WRITING

        p.complete_step(paper_content={"title": "Final"})
        assert p.is_complete
        assert p.progress == 1.0

    def test_resume(self, tmp_path):
        p = PaperPipeline("Test", "jweia")
        p.state.output_dir = str(tmp_path)
        p.complete_step(references=["ref1"])
        state_path = os.path.join(str(tmp_path), "pipeline_state.json")

        p2 = PaperPipeline.resume(state_path)
        assert p2.state.current_step == PipelineStep.RESEARCH_DESIGN
        assert p2.state.topic == "Test"

    def test_step_notes(self, tmp_path):
        p = PaperPipeline("Test", "jweia")
        p.state.output_dir = str(tmp_path)
        p.complete_step(references=["ref1"], note="Found 15 papers")
        assert "literature_review" in p.state.step_notes

    def test_requires_topic_or_state(self):
        with pytest.raises(ValueError):
            PaperPipeline()
