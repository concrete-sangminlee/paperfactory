import json
import os

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


class TestPipelineRobustness:
    @pytest.fixture(autouse=True)
    def _runs_dir(self, tmp_path, monkeypatch):
        from pipeline.orchestrator import RUNS_DIR_ENV

        monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
        self.runs_dir = tmp_path

    def test_default_output_dir_uses_runs_dir(self):
        p = PaperPipeline("Topic", "jweia")
        assert p.state.output_dir == os.path.join(str(self.runs_dir), p.state.run_id)

    def test_journal_key_is_validated_and_normalised(self):
        assert PaperPipeline("Topic", "JWEIA").state.journal_key == "jweia"
        assert PaperPipeline("Topic", "jcsr").state.journal_key == "steel_comp_struct"
        with pytest.raises(ValueError):
            PaperPipeline("Topic", "not_a_journal")

    def test_blank_topic_rejected(self):
        with pytest.raises(ValueError):
            PaperPipeline("   ", "jweia")

    def test_run_ids_do_not_collide(self):
        a = PaperPipeline("A", "jweia")
        a.state.save()
        b = PaperPipeline("B", "jweia")
        assert a.state.run_id != b.state.run_id

    def test_complete_after_finish_raises_and_progress_capped(self):
        p = PaperPipeline("Topic", "jweia")
        for _ in range(5):
            p.complete_step()
        assert p.is_complete and p.progress == 1.0
        with pytest.raises(RuntimeError):
            p.complete_step()
        assert p.progress == 1.0

    def test_revert_clears_later_outputs(self):
        p = PaperPipeline("Topic", "jweia")
        p.complete_step(references=["r"])
        p.complete_step(research_design={"t": 1})
        p.complete_step(code_outputs={"x": 1}, figure_paths=["f.png"], note="v1")
        p.complete_step(analysis_results={"r2": 0.1})
        p.revert_to(PipelineStep.CODE_EXECUTION, note="R2 too low, retune model")
        assert p.state.current_step == PipelineStep.CODE_EXECUTION
        assert p.state.steps_completed == ["literature_review", "research_design"]
        assert p.state.code_outputs == {} and p.state.figure_paths == []
        assert p.state.analysis_results == {}
        assert p.state.references == ["r"]
        assert p.state.step_notes["code_execution"] == "R2 too low, retune model"
        assert p.progress == pytest.approx(0.4)
        resumed = PaperPipeline.resume(p.state.state_path)
        assert resumed.state.current_step == PipelineStep.CODE_EXECUTION

    def test_revert_forward_rejected(self):
        p = PaperPipeline("Topic", "jweia")
        with pytest.raises(ValueError):
            p.revert_to(PipelineStep.PAPER_WRITING)
        with pytest.raises(ValueError):
            p.revert_to(PipelineStep.COMPLETE)

    def test_load_ignores_unknown_fields(self, tmp_path):
        p = PaperPipeline("Topic", "jweia")
        path = p.state.save()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["field_from_future_version"] = 1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        assert PipelineState.load(path).topic == "Topic"

    def test_load_rejects_non_state_json(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text('{"title": "a paper"}', encoding="utf-8")
        with pytest.raises(ValueError):
            PipelineState.load(str(path))

    def test_save_is_atomic_and_unicode(self):
        p = PaperPipeline("풍압 예측 연구", "ksce_jce")
        path = p.state.save()
        assert not [f for f in os.listdir(os.path.dirname(path)) if f.startswith(".state-")]
        assert PipelineState.load(path).topic == "풍압 예측 연구"

    def test_find_state_and_list_runs(self):
        from pipeline.orchestrator import find_state, list_runs

        assert find_state() is None
        a = PaperPipeline("A", "jweia")
        a.state.save()
        b = PaperPipeline("B", "eesd")
        b.complete_step(references=["r"])
        os.utime(a.state.state_path, (0, 0))
        assert find_state() == b.state.state_path
        assert find_state(a.state.run_id) == a.state.state_path
        assert find_state("run_missing") is None
        runs = list_runs()
        assert [r["topic"] for r in runs] == ["B", "A"]
        assert runs[0]["progress"] == pytest.approx(0.2)

    def test_status_shows_progress_and_notes(self):
        p = PaperPipeline("Topic", "jweia")
        p.complete_step(references=["r"], note="Found 18 papers")
        status = p.show_status()
        assert "Progress: 20%" in status
        assert "Found 18 papers" in status
