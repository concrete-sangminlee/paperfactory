"""Pipeline orchestrator for PaperFactory.

Manages the 5-step paper generation pipeline with state persistence,
allowing pause/resume and step-by-step execution.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum


class PipelineStep(str, Enum):
    LITERATURE_REVIEW = "literature_review"
    RESEARCH_DESIGN = "research_design"
    CODE_EXECUTION = "code_execution"
    RESULT_ANALYSIS = "result_analysis"
    PAPER_WRITING = "paper_writing"
    COMPLETE = "complete"


STEP_ORDER = [
    PipelineStep.LITERATURE_REVIEW,
    PipelineStep.RESEARCH_DESIGN,
    PipelineStep.CODE_EXECUTION,
    PipelineStep.RESULT_ANALYSIS,
    PipelineStep.PAPER_WRITING,
    PipelineStep.COMPLETE,
]

STEP_DESCRIPTIONS = {
    PipelineStep.LITERATURE_REVIEW: "Search and collect 15+ real papers with DOIs",
    PipelineStep.RESEARCH_DESIGN: "Design hypothesis, methodology, figure/table plans",
    PipelineStep.CODE_EXECUTION: "Write and execute Python research code, generate figures",
    PipelineStep.RESULT_ANALYSIS: "Statistical analysis, comparison with prior work",
    PipelineStep.PAPER_WRITING: "Assemble manuscript, validate references, export",
}


@dataclass
class PipelineState:
    """Persistent state for a paper generation pipeline run."""

    topic: str
    journal_key: str
    current_step: PipelineStep = PipelineStep.LITERATURE_REVIEW
    created_at: str = ""
    updated_at: str = ""
    run_id: str = ""
    output_dir: str = ""

    # Step outputs
    references: list = field(default_factory=list)
    research_design: dict = field(default_factory=dict)
    code_outputs: dict = field(default_factory=dict)
    analysis_results: dict = field(default_factory=dict)
    paper_content: dict = field(default_factory=dict)
    figure_paths: list = field(default_factory=list)

    # Step status
    steps_completed: list = field(default_factory=list)
    step_notes: dict = field(default_factory=dict)

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.run_id:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not self.output_dir:
            self.output_dir = os.path.join("outputs", "papers", self.run_id)

    def save(self, path: str = None):
        """Save state to JSON file."""
        if path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, "pipeline_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "PipelineState":
        """Load state from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["current_step"] = PipelineStep(data["current_step"])
        return cls(**data)


class PaperPipeline:
    """Orchestrates the 5-step paper generation pipeline.

    Usage
    -----
    pipeline = PaperPipeline("ML wind pressure prediction", "jweia")
    pipeline.show_status()
    pipeline.advance()  # Move to next step
    pipeline.complete_step(references=[...])  # Mark current step done with outputs
    """

    def __init__(self, topic: str = None, journal_key: str = None, state: PipelineState = None):
        if state:
            self.state = state
        elif topic and journal_key:
            self.state = PipelineState(topic=topic, journal_key=journal_key)
        else:
            raise ValueError("Provide either (topic, journal_key) or state")

    @classmethod
    def resume(cls, state_path: str) -> "PaperPipeline":
        """Resume a pipeline from a saved state file."""
        state = PipelineState.load(state_path)
        return cls(state=state)

    def show_status(self) -> str:
        """Return a formatted status string."""
        lines = [
            f"Pipeline: {self.state.run_id}",
            f"Topic: {self.state.topic}",
            f"Journal: {self.state.journal_key}",
            f"Current Step: {self.state.current_step.value}",
            "",
            "Steps:",
        ]
        for step in STEP_ORDER[:-1]:
            status = (
                "done"
                if step.value in self.state.steps_completed
                else (">>> CURRENT" if step == self.state.current_step else "pending")
            )
            desc = STEP_DESCRIPTIONS.get(step, "")
            lines.append(f"  [{status:>12}] {step.value}: {desc}")

        return "\n".join(lines)

    def complete_step(self, **outputs):
        """Mark the current step as complete and store its outputs."""
        step = self.state.current_step
        self.state.steps_completed.append(step.value)
        self.state.updated_at = datetime.now().isoformat()

        # Store step-specific outputs
        if step == PipelineStep.LITERATURE_REVIEW:
            self.state.references = outputs.get("references", [])
        elif step == PipelineStep.RESEARCH_DESIGN:
            self.state.research_design = outputs.get("research_design", {})
        elif step == PipelineStep.CODE_EXECUTION:
            self.state.code_outputs = outputs.get("code_outputs", {})
            self.state.figure_paths = outputs.get("figure_paths", [])
        elif step == PipelineStep.RESULT_ANALYSIS:
            self.state.analysis_results = outputs.get("analysis_results", {})
        elif step == PipelineStep.PAPER_WRITING:
            self.state.paper_content = outputs.get("paper_content", {})

        if outputs.get("note"):
            self.state.step_notes[step.value] = outputs["note"]

        # Advance to next step
        self.advance()
        self.state.save()

    def advance(self):
        """Move to the next step in the pipeline."""
        current_idx = STEP_ORDER.index(self.state.current_step)
        if current_idx < len(STEP_ORDER) - 1:
            self.state.current_step = STEP_ORDER[current_idx + 1]

    @property
    def is_complete(self) -> bool:
        return self.state.current_step == PipelineStep.COMPLETE

    @property
    def progress(self) -> float:
        """Return progress as a fraction (0.0 to 1.0)."""
        return len(self.state.steps_completed) / 5.0

    @property
    def current_step_description(self) -> str:
        return STEP_DESCRIPTIONS.get(self.state.current_step, "Complete")
