import json
import logging
import os
from pipeline.literature_review import run as run_literature_review
from pipeline.research_design import run as run_research_design
from pipeline.code_execution import run as run_code_execution
from pipeline.result_analysis import run as run_result_analysis
from pipeline.paper_writing import run as run_paper_writing
from utils.word_generator import load_guideline

logger = logging.getLogger(__name__)

STEPS = [
    {"id": "literature_review", "name": "Literature Review", "name_ko": "문헌 조사"},
    {"id": "research_design", "name": "Research Design", "name_ko": "연구 설계"},
    {"id": "code_execution", "name": "Code Execution", "name_ko": "코드 실행"},
    {"id": "result_analysis", "name": "Result Analysis", "name_ko": "결과 분석"},
    {"id": "paper_writing", "name": "Paper Writing", "name_ko": "논문 작성"},
]

JOURNAL_KEYS = {
    "ASCE Journal of Structural Engineering": "asce_jse",
    "ACI Structural Journal": "aci_sj",
    "Journal of Wind Engineering and Industrial Aerodynamics": "jweia",
    "Journal of Building Engineering": "jbe",
    "Engineering Structures": "eng_structures",
}


class PipelineOrchestrator:
    def __init__(self, topic: str, journal_name: str):
        self.topic = topic
        self.journal_name = journal_name
        self.journal_key = JOURNAL_KEYS.get(journal_name, "asce_jse")
        self.guideline = load_guideline(self.journal_key)
        self.results = {}
        self.current_step = 0
        self.status = "idle"

    def get_steps(self):
        return STEPS

    def get_current_step(self):
        if self.current_step < len(STEPS):
            return STEPS[self.current_step]
        return None

    def get_progress(self):
        return {
            "current_step": self.current_step,
            "total_steps": len(STEPS),
            "percentage": int((self.current_step / len(STEPS)) * 100),
            "status": self.status,
        }

    def run_step(self, step_id: str = None, feedback: str = None, progress_callback=None):
        if step_id is None:
            step_id = STEPS[self.current_step]["id"]

        self.status = "running"

        if step_id == "literature_review":
            result = run_literature_review(
                self.topic, self.guideline, progress_callback=progress_callback
            )
            self.results["literature"] = result

        elif step_id == "research_design":
            literature = self.results.get("literature", {})
            if feedback:
                literature["user_feedback"] = feedback
            result = run_research_design(
                self.topic, literature, self.guideline,
                progress_callback=progress_callback
            )
            self.results["design"] = result

        elif step_id == "code_execution":
            design = self.results.get("design", {})
            literature = self.results.get("literature", {})
            result = run_code_execution(
                self.topic, design, literature, self.guideline,
                progress_callback=progress_callback
            )
            self.results["code"] = result

        elif step_id == "result_analysis":
            result = run_result_analysis(
                self.topic,
                self.results.get("design", {}),
                self.results.get("literature", {}),
                self.results.get("code", {}),
                self.guideline,
                progress_callback=progress_callback
            )
            self.results["analysis"] = result

        elif step_id == "paper_writing":
            result = run_paper_writing(
                self.topic,
                self.results.get("literature", {}),
                self.results.get("design", {}),
                self.results.get("code", {}),
                self.results.get("analysis", {}),
                self.journal_key,
                self.guideline,
                progress_callback=progress_callback
            )
            self.results["paper"] = result

        self.current_step += 1
        if self.current_step >= len(STEPS):
            self.status = "completed"
        else:
            self.status = "waiting_approval"

        return result

    def run_all(self, progress_callback=None):
        for step in STEPS:
            self.run_step(step["id"], progress_callback=progress_callback)
        return self.results

    def get_step_result_summary(self, step_id: str) -> str:
        if step_id == "literature_review":
            lit = self.results.get("literature", {}).get("review", {})
            papers_count = len(lit.get("key_papers", []))
            gaps = lit.get("research_gaps", [])
            return f"Found {papers_count} key papers. Identified {len(gaps)} research gaps:\n" + \
                   "\n".join(f"  - {g}" for g in gaps[:5])

        elif step_id == "research_design":
            des = self.results.get("design", {}).get("design", {})
            return f"Title: {des.get('research_title', 'N/A')}\n" \
                   f"Objective: {des.get('research_objective', 'N/A')}\n" \
                   f"Novelty: {des.get('novelty_statement', 'N/A')}"

        elif step_id == "code_execution":
            code = self.results.get("code", {})
            figs = len(code.get("figures", []))
            status = code.get("status", "unknown")
            return f"Status: {status}\nFigures generated: {figs}"

        elif step_id == "result_analysis":
            ana = self.results.get("analysis", {}).get("analysis", {})
            return ana.get("results_summary", "Analysis complete.")

        elif step_id == "paper_writing":
            paper = self.results.get("paper", {})
            return f"Paper saved to: {paper.get('output_path', 'N/A')}\n" \
                   f"Figures included: {paper.get('figures_included', 0)}"

        return "No summary available."

    def get_output_path(self) -> str:
        return self.results.get("paper", {}).get("output_path", "")
