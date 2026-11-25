"""Pipeline orchestrator for PaperFactory.

Manages the 5-step paper generation pipeline with state persistence,
allowing pause/resume, step-by-step execution and stepping back to an
earlier step (e.g. returning from Result Analysis to Code Execution when
results contradict the hypothesis).
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum

from utils.guidelines import PROJECT_ROOT, resolve_journal_key

RUNS_DIR_ENV = "PAPERFACTORY_RUNS_DIR"
DEFAULT_RUNS_DIR = os.path.join(PROJECT_ROOT, "outputs", "papers")
STATE_FILENAME = "pipeline_state.json"


def get_runs_dir() -> str:
    """Return the directory that holds pipeline runs (env override or project default)."""
    return os.environ.get(RUNS_DIR_ENV) or DEFAULT_RUNS_DIR


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

#: The five working steps (everything except the terminal COMPLETE marker).
WORK_STEPS = STEP_ORDER[:-1]

STEP_DESCRIPTIONS = {
    PipelineStep.LITERATURE_REVIEW: "Search and collect 15+ real papers with DOIs",
    PipelineStep.RESEARCH_DESIGN: "Design hypothesis, methodology, figure/table plans",
    PipelineStep.CODE_EXECUTION: "Write and execute Python research code, generate figures",
    PipelineStep.RESULT_ANALYSIS: "Statistical analysis, comparison with prior work",
    PipelineStep.PAPER_WRITING: "Assemble manuscript, validate references, export",
}

# State fields populated by each step; cleared when the pipeline is reverted past it.
_STEP_FIELDS = {
    PipelineStep.LITERATURE_REVIEW: {"references": list},
    PipelineStep.RESEARCH_DESIGN: {"research_design": dict},
    PipelineStep.CODE_EXECUTION: {"code_outputs": dict, "figure_paths": list},
    PipelineStep.RESULT_ANALYSIS: {"analysis_results": dict},
    PipelineStep.PAPER_WRITING: {"paper_content": dict},
}


def _new_run_id(runs_dir: str) -> str:
    base = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id, n = base, 2
    while os.path.exists(os.path.join(runs_dir, run_id)):
        run_id = f"{base}_{n}"
        n += 1
    return run_id


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
        if not isinstance(self.current_step, PipelineStep):
            self.current_step = PipelineStep(self.current_step)
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.run_id:
            self.run_id = _new_run_id(get_runs_dir())
        if not self.output_dir:
            self.output_dir = os.path.join(get_runs_dir(), self.run_id)

    @property
    def state_path(self) -> str:
        return os.path.join(self.output_dir, STATE_FILENAME)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["current_step"] = self.current_step.value
        return data

    def save(self, path: str = None) -> str:
        """Save state to JSON atomically (write to a temp file, then rename)."""
        if path is None:
            path = self.state_path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".state-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return path

    @classmethod
    def load(cls, path: str) -> PipelineState:
        """Load state from JSON, ignoring unknown keys from newer/older versions."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "topic" not in data or "journal_key" not in data:
            raise ValueError(f"Not a valid pipeline state file: {path}")
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        data["current_step"] = PipelineStep(data.get("current_step", "literature_review"))
        return cls(**data)


class PaperPipeline:
    """Orchestrates the 5-step paper generation pipeline.

    Usage
    -----
    pipeline = PaperPipeline("ML wind pressure prediction", "jweia")
    pipeline.show_status()
    pipeline.complete_step(references=[...])  # Mark current step done with outputs
    pipeline.revert_to(PipelineStep.CODE_EXECUTION)  # Step back when results need rework
    """

    def __init__(self, topic: str = None, journal_key: str = None, state: PipelineState = None):
        if state:
            self.state = state
        elif topic and journal_key:
            if not str(topic).strip():
                raise ValueError("Topic must not be empty")
            self.state = PipelineState(
                topic=str(topic).strip(), journal_key=resolve_journal_key(journal_key)
            )
        else:
            raise ValueError("Provide either (topic, journal_key) or state")

    @classmethod
    def resume(cls, state_path: str) -> PaperPipeline:
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
            f"Progress: {self.progress:.0%}",
            "",
            "Steps:",
        ]
        for step in WORK_STEPS:
            status = (
                "done"
                if step.value in self.state.steps_completed
                else (">>> CURRENT" if step == self.state.current_step else "pending")
            )
            desc = STEP_DESCRIPTIONS.get(step, "")
            lines.append(f"  [{status:>12}] {step.value}: {desc}")
            note = self.state.step_notes.get(step.value)
            if note:
                lines.append(f"  {'':>14} note: {note}")

        return "\n".join(lines)

    def complete_step(self, **outputs):
        """Mark the current step as complete, store its outputs and advance.

        Raises
        ------
        RuntimeError
            If the pipeline is already complete.
        """
        step = self.state.current_step
        if step == PipelineStep.COMPLETE:
            raise RuntimeError("Pipeline is already complete; use revert_to() to redo a step")

        if step.value not in self.state.steps_completed:
            self.state.steps_completed.append(step.value)
        self.state.updated_at = datetime.now().isoformat()

        for name, factory in _STEP_FIELDS[step].items():
            setattr(self.state, name, outputs.get(name, factory()))

        if outputs.get("note"):
            self.state.step_notes[step.value] = outputs["note"]

        self.advance()
        self.state.save()

    def advance(self):
        """Move to the next step in the pipeline."""
        current_idx = STEP_ORDER.index(self.state.current_step)
        if current_idx < len(STEP_ORDER) - 1:
            self.state.current_step = STEP_ORDER[current_idx + 1]

    def revert_to(self, step, note: str = None, save: bool = True):
        """Return to an earlier *step*, discarding its outputs and those of later steps.

        Useful when, for example, Step 4 analysis shows the Step 3 code must change.
        """
        step = PipelineStep(step)
        if step == PipelineStep.COMPLETE:
            raise ValueError("Cannot revert to COMPLETE")
        target_idx = STEP_ORDER.index(step)
        if target_idx > STEP_ORDER.index(self.state.current_step):
            raise ValueError(
                f"Cannot revert forward from {self.state.current_step.value} to {step.value}"
            )
        for later in WORK_STEPS[target_idx:]:
            if later.value in self.state.steps_completed:
                self.state.steps_completed.remove(later.value)
            self.state.step_notes.pop(later.value, None)
            for name, factory in _STEP_FIELDS[later].items():
                setattr(self.state, name, factory())
        self.state.current_step = step
        self.state.updated_at = datetime.now().isoformat()
        if note:
            self.state.step_notes[step.value] = note
        if save:
            self.state.save()

    @property
    def is_complete(self) -> bool:
        return self.state.current_step == PipelineStep.COMPLETE

    @property
    def progress(self) -> float:
        """Return progress as a fraction (0.0 to 1.0)."""
        done = {s.value for s in WORK_STEPS} & set(self.state.steps_completed)
        return len(done) / len(WORK_STEPS)

    @property
    def current_step_description(self) -> str:
        return STEP_DESCRIPTIONS.get(self.state.current_step, "Complete")


def find_state(run_id: str = None, runs_dir: str = None) -> str | None:
    """Return the state file for *run_id*, or the most recent run when omitted."""
    runs_dir = runs_dir or get_runs_dir()
    if run_id:
        path = os.path.join(runs_dir, run_id, STATE_FILENAME)
        return path if os.path.exists(path) else None
    if not os.path.isdir(runs_dir):
        return None
    candidates = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name, STATE_FILENAME)
        if name.startswith("run_") and os.path.exists(path):
            candidates.append((os.path.getmtime(path), name, path))
    if not candidates:
        return None
    return max(candidates)[2]


def list_runs(runs_dir: str = None) -> list[dict]:
    """Return a summary of every saved run, newest first."""
    runs_dir = runs_dir or get_runs_dir()
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name, STATE_FILENAME)
        if not os.path.exists(path):
            continue
        try:
            p = PaperPipeline.resume(path)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        runs.append(
            {
                "run_id": p.state.run_id,
                "topic": p.state.topic,
                "journal_key": p.state.journal_key,
                "current_step": p.state.current_step.value,
                "progress": p.progress,
                "updated_at": p.state.updated_at,
                "path": path,
                "_mtime": os.path.getmtime(path),
            }
        )
    # Timestamps can tie (coarse clocks on Windows); break ties the same way find_state does.
    runs.sort(key=lambda r: (r["updated_at"], r["_mtime"], r["run_id"]), reverse=True)
    for r in runs:
        del r["_mtime"]
    return runs
