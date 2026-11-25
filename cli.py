#!/usr/bin/env python3
"""PaperFactory CLI — command-line interface for the paper generation toolkit.

Examples::

    paperfactory new --topic "..." --journal jweia
    paperfactory status [--run-id RUN_ID]
    paperfactory runs
    paperfactory check --paper paper.json --journal jweia [--strict] [--json]
    paperfactory export --paper paper.json --journal jweia --format docx latex pdf
    paperfactory journals [--json]

Every command exits with status 0 on success, 1 on a user/input error and,
with ``--strict``, 1 when a quality gate fails — so the CLI is safe to use in
scripts and CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import PaperPipeline, find_state, list_runs  # noqa: E402
from utils.guidelines import (  # noqa: E402
    UnknownJournalError,
    list_journals,
    load_guideline,
    resolve_journal_key,
)

__version__ = "1.1.0"

EXIT_OK = 0
EXIT_ERROR = 1


class CLIError(Exception):
    """A user-facing error: printed to stderr without a traceback."""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _journal(key: str) -> str:
    try:
        return resolve_journal_key(key)
    except UnknownJournalError as e:
        raise CLIError(str(e)) from None


def _load_paper(path: str) -> dict:
    """Load paper_content from a JSON file, raising CLIError on any problem."""
    if not os.path.exists(path):
        raise CLIError(f"File not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            paper = json.load(f)
    except json.JSONDecodeError as e:
        raise CLIError(f"Invalid JSON in {path}: {e}") from None
    if not isinstance(paper, dict):
        raise CLIError(f"{path} must contain a JSON object (paper_content dict)")
    return paper


def _write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _split_figures(value: str | None) -> list | None:
    if not value:
        return None
    figures = [p.strip() for p in value.split(",") if p.strip()]
    missing = [p for p in figures if not os.path.exists(p)]
    if missing:
        print(f"Warning: figure file(s) not found: {', '.join(missing)}", file=sys.stderr)
    return figures


def _references(paper: dict) -> list:
    refs = paper.get("references", [])
    if isinstance(refs, str):
        refs = [r for r in refs.splitlines() if r.strip()]
    return list(refs)


# ── Pipeline commands ────────────────────────────────────────────────────────


def cmd_new(args) -> int:
    """Start a new paper generation pipeline."""
    try:
        pipeline = PaperPipeline(args.topic, args.journal)
    except ValueError as e:
        raise CLIError(str(e)) from None
    path = pipeline.state.save()
    print(pipeline.show_status())
    print(f"\nPipeline created: {pipeline.state.run_id}")
    print(f"State saved to: {path}")
    print("\nNext: Open Claude Code and provide the topic to start Step 1 (Literature Review).")
    return EXIT_OK


def _resume(run_id: str | None) -> PaperPipeline:
    state_path = find_state(run_id)
    if not state_path:
        if run_id:
            raise CLIError(f"Pipeline not found: {run_id}")
        raise CLIError("No pipeline found. Use 'paperfactory new' to create one.")
    try:
        return PaperPipeline.resume(state_path)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise CLIError(f"Corrupt pipeline state {state_path}: {e}") from None


def cmd_status(args) -> int:
    """Show pipeline status."""
    pipeline = _resume(args.run_id)
    if args.json:
        _print_json(pipeline.state.to_dict() | {"progress": pipeline.progress})
    else:
        print(pipeline.show_status())
    return EXIT_OK


def cmd_resume(args) -> int:
    """Resume a pipeline."""
    pipeline = _resume(args.run_id)
    print(pipeline.show_status())
    if pipeline.is_complete:
        print("\nPipeline is complete.")
    else:
        print(f"\nTo continue, open Claude Code and work on: {pipeline.current_step_description}")
    return EXIT_OK


def cmd_runs(args) -> int:
    """List saved pipeline runs."""
    runs = list_runs()
    if args.json:
        _print_json(runs)
        return EXIT_OK
    if not runs:
        print("No pipeline runs found.")
        return EXIT_OK
    for r in runs:
        print(
            f"  {r['run_id']:<24} {r['progress']:>4.0%}  {r['current_step']:<18} "
            f"{r['journal_key']:<16} {r['topic'][:60]}"
        )
    return EXIT_OK


# ── Paper commands ───────────────────────────────────────────────────────────


def cmd_check(args) -> int:
    """Run quality check on a paper."""
    from utils.quality_checker import check_paper

    paper = _load_paper(args.paper)
    result = check_paper(paper, _journal(args.journal), figures=_split_figures(args.figures))
    if args.json:
        _print_json(result)
    else:
        print(result["summary"])
    return EXIT_ERROR if args.strict and not result["passed"] else EXIT_OK


def cmd_review(args) -> int:
    """Run AI reviewer simulation."""
    from utils.ai_reviewer import review_paper

    paper = _load_paper(args.paper)
    result = review_paper(paper, _journal(args.journal))
    if args.json:
        _print_json(result)
    else:
        print(result["summary"])
    return EXIT_OK


def cmd_checklist(args) -> int:
    """Run the journal-specific submission checklist."""
    from utils.submission_utils import submission_checklist

    paper = _load_paper(args.paper)
    result = submission_checklist(
        paper, _journal(args.journal), figures=_split_figures(args.figures)
    )
    if args.json:
        _print_json(result)
    else:
        for item in result["items"]:
            mark = "PASS" if item["passed"] else "FAIL"
            note = f" — {item['note']}" if item.get("note") else ""
            print(f"  [{mark}] {item['description']}{note}")
        print(f"\nReady to submit: {'YES' if result['ready'] else 'NO'}")
    return EXIT_ERROR if args.strict and not result["ready"] else EXIT_OK


def cmd_cover_letter(args) -> int:
    """Generate a cover letter."""
    from utils.submission_utils import generate_cover_letter

    paper = _load_paper(args.paper)
    letter = generate_cover_letter(paper, _journal(args.journal), editor_name=args.editor)
    print(letter)
    if args.output:
        _write_text(args.output, letter)
        print(f"\nSaved to: {args.output}")
    return EXIT_OK


def cmd_export(args) -> int:
    """Export a paper to Word, LaTeX and/or PDF."""
    paper = _load_paper(args.paper)
    journal = _journal(args.journal)
    figures = _split_figures(args.figures)
    outputs = {}
    for fmt in dict.fromkeys(args.format):  # de-duplicate, keep order
        try:
            if fmt == "docx":
                from utils.word_generator import generate_word

                outputs["docx"] = generate_word(paper, journal, figures, output_dir=args.output_dir)
            elif fmt == "latex":
                from utils.latex_generator import generate_latex

                tex, bib = generate_latex(paper, journal, figures, output_dir=args.output_dir)
                outputs["tex"], outputs["bib"] = tex, bib
            elif fmt == "pdf":
                from utils.pdf_generator import generate_pdf

                outputs["pdf"] = generate_pdf(paper, journal, figures, output_dir=args.output_dir)
        except (KeyError, TypeError, ValueError, OSError) as e:
            raise CLIError(f"{fmt} export failed: {type(e).__name__}: {e}") from None
    if args.json:
        _print_json(outputs)
    else:
        for kind, path in outputs.items():
            print(f"  {kind:<5} {path}")
    return EXIT_OK


def cmd_reformat(args) -> int:
    """Reformat a paper for a different journal (e.g. after rejection)."""
    from utils.submission_utils import reformat_paper

    paper = _load_paper(args.paper)
    new_paper = reformat_paper(paper, _journal(args.from_journal), _journal(args.to_journal))
    text = json.dumps(new_paper, indent=2, ensure_ascii=False)
    if args.output:
        _write_text(args.output, text)
        print(f"Reformatted paper saved to: {args.output}")
    else:
        print(text)
    for warning in new_paper.get("_warnings", []):
        print(f"Warning: {warning}", file=sys.stderr)
    return EXIT_OK


def cmd_validate_refs(args) -> int:
    """Validate references against a journal's reference rules."""
    from utils.reference_utils import check_duplicates, validate_references

    paper = _load_paper(args.paper)
    refs = _references(paper)
    issues = validate_references(refs, load_guideline(_journal(args.journal)))
    duplicates = check_duplicates(refs)
    if args.json:
        _print_json({"count": len(refs), "issues": issues, "duplicates": duplicates})
    else:
        print(f"References: {len(refs)}")
        for issue in issues:
            print(f"  [{issue.get('index', '?')}] {issue.get('issue', issue)}")
        for dup in duplicates:
            print(f"  duplicate: #{dup['index_a']} and #{dup['index_b']}")
        if not issues and not duplicates:
            print("  No issues found.")
    has_problems = bool(issues or duplicates)
    return EXIT_ERROR if args.strict and has_problems else EXIT_OK


def cmd_analytics(args) -> int:
    """Analyze paper readability and statistics."""
    from utils.paper_analytics import analyze_paper

    paper = _load_paper(args.paper)
    result = analyze_paper(paper)
    if args.json:
        _print_json(result)
        return EXIT_OK
    print(result["summary"])
    if args.verbose:
        print("\nSection balance:")
        for sec, words in result["section_balance"]["section_words"].items():
            pct = result["section_balance"]["section_proportions"].get(sec, 0)
            print(f"  {sec:<30} {words:>5} words ({pct:.0%})")
        print("\nTop content words:")
        for w in result["vocabulary"]["top_words"][:10]:
            print(f"  {w['word']:<20} {w['count']}")
    return EXIT_OK


def cmd_abstract(args) -> int:
    """Generate or improve abstract."""
    from utils.abstract_generator import generate_abstract, improve_abstract

    paper = _load_paper(args.paper)
    if args.generate:
        abstract = generate_abstract(paper, max_words=args.max_words)
        print("Generated abstract:")
        print(abstract)
        return EXIT_OK
    existing = paper.get("abstract", "")
    if not existing:
        raise CLIError("No abstract found. Use --generate to create one.")
    result = improve_abstract(existing, paper, max_words=args.max_words)
    print(f"Abstract score: {result['score']}/100")
    print(f"Word count: {result['word_count']}")
    if result["suggestions"]:
        print("\nSuggestions:")
        for s in result["suggestions"]:
            print(f"  - {s}")
    return EXIT_OK


def cmd_import_bib(args) -> int:
    """Import references from BibTeX file."""
    from utils.bibtex_import import import_bib

    if not os.path.exists(args.bib):
        raise CLIError(f"File not found: {args.bib}")
    refs = import_bib(args.bib)
    print(f"Imported {len(refs)} references:")
    for ref in refs:
        print(f"  {ref}")
    if args.output:
        _write_text(args.output, json.dumps({"references": refs}, indent=2, ensure_ascii=False))
        print(f"\nSaved to: {args.output}")
    return EXIT_OK


def cmd_convert_refs(args) -> int:
    """Convert reference citation style."""
    from utils.citation_converter import convert_style

    paper = _load_paper(args.paper)
    for ref in convert_style(_references(paper), args.style):
        print(ref)
    return EXIT_OK


def cmd_templates(args) -> int:
    """List or generate from paper templates."""
    from utils.template_system import generate_skeleton, list_templates

    if args.generate:
        keys = {t["key"] for t in list_templates()}
        if args.generate not in keys:
            raise CLIError(
                f"Unknown template '{args.generate}'. Available: {', '.join(sorted(keys))}"
            )
        journal = _journal(args.journal) if args.journal else ""
        skeleton = generate_skeleton(args.generate, topic=args.topic or "", journal_key=journal)
        text = json.dumps(skeleton, indent=2, ensure_ascii=False)
        if args.output:
            _write_text(args.output, text)
            print(f"Skeleton saved to: {args.output}")
        else:
            print(text)
    else:
        print("Available templates:")
        for t in list_templates():
            print(f"  {t['key']:<20} {t['name']}")
            print(f"  {'':20} {t['description']}")
    return EXIT_OK


# ── Reference commands ───────────────────────────────────────────────────────


def cmd_sources(args) -> int:
    """Suggest data sources for a topic."""
    from utils.data_sources import suggest_sources

    sources = suggest_sources(args.topic)
    if args.json:
        _print_json(sources)
        return EXIT_OK
    if not sources:
        print("No matching data sources found.")
        return EXIT_OK
    for s in sources:
        print(f"\n[{s.get('key', '?')}] {s['name']} (relevance: {s['relevance_score']})")
        print(f"  URL: {s['url']}")
        print(f"  Access: {s['access']}")
        desc = s["description"]
        print(f"  {desc[:120]}{'...' if len(desc) > 120 else ''}")
    return EXIT_OK


def cmd_journals(args) -> int:
    """List supported journals."""
    journals = list_journals()
    if args.json:
        _print_json(journals)
        return EXIT_OK
    for j in journals:
        print(f"  {j['key']:<20} {j['name']} ({j['publisher']})")
    return EXIT_OK


# ── Parser ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperfactory",
        description="PaperFactory — AI Research Paper Agent for Civil Engineering",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name, func, help_text, paper=False, journal=False, json_flag=False):
        p = sub.add_parser(name, help=help_text, description=help_text)
        if paper:
            p.add_argument("--paper", "-p", required=True, help="paper_content JSON file")
        if journal:
            p.add_argument("--journal", "-j", required=True, help="Journal key (see 'journals')")
        if json_flag:
            p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
        p.set_defaults(func=func)
        return p

    # Pipeline
    p = add("new", cmd_new, "Start a new paper pipeline")
    p.add_argument("--topic", "-t", required=True, help="Research topic")
    p.add_argument("--journal", "-j", required=True, help="Target journal key")

    p = add("status", cmd_status, "Show pipeline status", json_flag=True)
    p.add_argument("--run-id", "-r", help="Specific run ID (default: most recent)")

    p = add("resume", cmd_resume, "Resume a pipeline")
    p.add_argument("--run-id", "-r", required=True, help="Run ID to resume")

    add("runs", cmd_runs, "List saved pipeline runs", json_flag=True)

    # Paper quality & submission
    p = add("check", cmd_check, "Quality check a paper", paper=True, journal=True, json_flag=True)
    p.add_argument("--figures", "-f", help="Comma-separated figure paths")
    p.add_argument("--strict", action="store_true", help="Exit 1 if any critical check fails")

    add("review", cmd_review, "AI reviewer simulation", paper=True, journal=True, json_flag=True)

    p = add(
        "checklist",
        cmd_checklist,
        "Submission readiness checklist",
        paper=True,
        journal=True,
        json_flag=True,
    )
    p.add_argument("--figures", "-f", help="Comma-separated figure paths")
    p.add_argument("--strict", action="store_true", help="Exit 1 if not ready to submit")

    p = add("cover-letter", cmd_cover_letter, "Generate cover letter", paper=True, journal=True)
    p.add_argument("--editor", "-e", default="Editor-in-Chief", help="Editor name")
    p.add_argument("--output", "-o", help="Output file path")

    p = add(
        "export",
        cmd_export,
        "Export to Word / LaTeX / PDF",
        paper=True,
        journal=True,
        json_flag=True,
    )
    p.add_argument(
        "--format",
        nargs="+",
        choices=["docx", "latex", "pdf"],
        default=["docx"],
        help="Output format(s) (default: docx)",
    )
    p.add_argument("--figures", "-f", help="Comma-separated figure paths")
    p.add_argument("--output-dir", "-o", help="Output directory (default: outputs/papers)")

    p = add("reformat", cmd_reformat, "Reformat a paper for another journal", paper=True)
    p.add_argument("--from", dest="from_journal", required=True, help="Current journal key")
    p.add_argument("--to", dest="to_journal", required=True, help="Target journal key")
    p.add_argument("--output", "-o", help="Write reformatted paper JSON here")

    p = add(
        "validate-refs",
        cmd_validate_refs,
        "Validate references against journal rules",
        paper=True,
        journal=True,
        json_flag=True,
    )
    p.add_argument("--strict", action="store_true", help="Exit 1 if any issue is found")

    # Writing aids
    p = add(
        "analytics", cmd_analytics, "Paper readability & statistics", paper=True, json_flag=True
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Show detailed breakdown")

    p = add("abstract", cmd_abstract, "Generate or improve abstract", paper=True)
    p.add_argument("--generate", "-g", action="store_true", help="Generate new abstract")
    p.add_argument("--max-words", "-m", type=int, default=250, help="Max words")

    p = add("import-bib", cmd_import_bib, "Import BibTeX references")
    p.add_argument("--bib", "-b", required=True, help="Path to .bib file")
    p.add_argument("--output", "-o", help="Save as JSON")

    p = add("convert-refs", cmd_convert_refs, "Convert citation style", paper=True)
    p.add_argument("--style", "-s", required=True, choices=["numbered", "author_date", "apa"])

    p = add("templates", cmd_templates, "List or generate from templates")
    p.add_argument("--generate", "-g", help="Template key to generate from")
    p.add_argument("--topic", "-t", help="Research topic for skeleton")
    p.add_argument("--journal", "-j", help="Target journal key")
    p.add_argument("--output", "-o", help="Output JSON path")

    # Reference data
    p = add("sources", cmd_sources, "Suggest data sources", json_flag=True)
    p.add_argument("--topic", "-t", required=True, help="Research topic")

    add("journals", cmd_journals, "List supported journals", json_flag=True)

    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    try:
        return args.func(args)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
