#!/usr/bin/env python3
"""PaperFactory CLI — Command-line interface for paper generation pipeline.

Usage:
    python cli.py new --topic "..." --journal jweia
    python cli.py status [--run-id RUN_ID]
    python cli.py resume --run-id RUN_ID
    python cli.py check --paper paper_content.json --journal jweia
    python cli.py review --paper paper_content.json --journal jweia
    python cli.py cover-letter --paper paper_content.json --journal jweia
    python cli.py sources --topic "wind pressure prediction"
    python cli.py journals
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import PaperPipeline
from utils.abstract_generator import generate_abstract, improve_abstract
from utils.ai_reviewer import review_paper
from utils.bibtex_import import import_bib
from utils.citation_converter import convert_style
from utils.data_sources import suggest_sources
from utils.paper_analytics import analyze_paper
from utils.quality_checker import check_paper
from utils.submission_utils import generate_cover_letter
from utils.template_system import generate_skeleton, list_templates


def cmd_new(args):
    """Start a new paper generation pipeline."""
    pipeline = PaperPipeline(args.topic, args.journal)
    os.makedirs(pipeline.state.output_dir, exist_ok=True)
    pipeline.state.save()
    print(pipeline.show_status())
    print(f"\nPipeline created: {pipeline.state.run_id}")
    print(f"State saved to: {pipeline.state.output_dir}/pipeline_state.json")
    print("\nNext: Open Claude Code and provide the topic to start Step 1 (Literature Review).")


def cmd_status(args):
    """Show pipeline status."""
    state_path = _find_state(args.run_id)
    if not state_path:
        print("No pipeline found. Use 'python cli.py new' to create one.")
        return
    pipeline = PaperPipeline.resume(state_path)
    print(pipeline.show_status())
    print(f"\nProgress: {pipeline.progress:.0%}")


def cmd_resume(args):
    """Resume a pipeline."""
    state_path = _find_state(args.run_id)
    if not state_path:
        print(f"Pipeline not found: {args.run_id}")
        return
    pipeline = PaperPipeline.resume(state_path)
    print(pipeline.show_status())
    print(f"\nTo continue, open Claude Code and work on: {pipeline.current_step_description}")


def cmd_check(args):
    """Run quality check on a paper."""
    paper = _load_paper(args.paper)
    if not paper:
        return
    figures = args.figures.split(",") if args.figures else None
    result = check_paper(paper, args.journal, figures=figures)
    print(result["summary"])


def cmd_review(args):
    """Run AI reviewer simulation."""
    paper = _load_paper(args.paper)
    if not paper:
        return
    result = review_paper(paper, args.journal)
    print(result["summary"])


def cmd_cover_letter(args):
    """Generate a cover letter."""
    paper = _load_paper(args.paper)
    if not paper:
        return
    letter = generate_cover_letter(paper, args.journal, editor_name=args.editor)
    print(letter)
    if args.output:
        with open(args.output, "w") as f:
            f.write(letter)
        print(f"\nSaved to: {args.output}")


def cmd_sources(args):
    """Suggest data sources for a topic."""
    sources = suggest_sources(args.topic)
    if not sources:
        print("No matching data sources found.")
        return
    for s in sources:
        print(f"\n[{s.get('key', '?')}] {s['name']} (relevance: {s['relevance_score']})")
        print(f"  URL: {s['url']}")
        print(f"  Access: {s['access']}")
        print(f"  {s['description'][:120]}...")


def cmd_journals(args):
    """List supported journals."""
    guidelines_dir = os.path.join(os.path.dirname(__file__), "guidelines")
    for f in sorted(os.listdir(guidelines_dir)):
        if f.endswith(".json"):
            with open(os.path.join(guidelines_dir, f)) as fh:
                g = json.load(fh)
            key = f.replace(".json", "")
            name = g.get("journal_name", key)
            publisher = g.get("publisher", "")
            print(f"  {key:<20} {name} ({publisher})")


def cmd_analytics(args):
    """Analyze paper readability and statistics."""
    paper = _load_paper(args.paper)
    if not paper:
        return
    result = analyze_paper(paper)
    print(result["summary"])
    if args.verbose:
        print("\nSection balance:")
        for sec, words in result["section_balance"]["section_words"].items():
            pct = result["section_balance"]["section_proportions"].get(sec, 0)
            print(f"  {sec:<30} {words:>5} words ({pct:.0%})")
        print("\nTop content words:")
        for w in result["vocabulary"]["top_words"][:10]:
            print(f"  {w['word']:<20} {w['count']}")


def cmd_abstract(args):
    """Generate or improve abstract."""
    paper = _load_paper(args.paper)
    if not paper:
        return
    if args.generate:
        abstract = generate_abstract(paper, max_words=args.max_words)
        print("Generated abstract:")
        print(abstract)
    else:
        existing = paper.get("abstract", "")
        if not existing:
            print("No abstract found. Use --generate to create one.")
            return
        result = improve_abstract(existing, paper, max_words=args.max_words)
        print(f"Abstract score: {result['score']}/100")
        print(f"Word count: {result['word_count']}")
        if result["suggestions"]:
            print("\nSuggestions:")
            for s in result["suggestions"]:
                print(f"  - {s}")


def cmd_import_bib(args):
    """Import references from BibTeX file."""
    refs = import_bib(args.bib)
    print(f"Imported {len(refs)} references:")
    for ref in refs:
        print(f"  {ref}")
    if args.output:
        paper = {"references": refs}
        with open(args.output, "w") as f:
            json.dump(paper, f, indent=2)
        print(f"\nSaved to: {args.output}")


def cmd_convert_refs(args):
    """Convert reference citation style."""
    paper = _load_paper(args.paper)
    if not paper:
        return
    refs = paper.get("references", [])
    converted = convert_style(refs, args.style)
    for ref in converted:
        print(ref)


def cmd_templates(args):
    """List or generate from paper templates."""
    if args.generate:
        skeleton = generate_skeleton(
            args.generate, topic=args.topic or "", journal_key=args.journal or ""
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(skeleton, f, indent=2, ensure_ascii=False)
            print(f"Skeleton saved to: {args.output}")
        else:
            print(json.dumps(skeleton, indent=2, ensure_ascii=False))
    else:
        print("Available templates:")
        for t in list_templates():
            print(f"  {t['key']:<20} {t['name']}")
            print(f"  {'':20} {t['description']}")


def _find_state(run_id=None):
    """Find pipeline state file."""
    if run_id:
        path = os.path.join("outputs", "papers", run_id, "pipeline_state.json")
        return path if os.path.exists(path) else None
    # Find most recent
    pattern = os.path.join("outputs", "papers", "run_*", "pipeline_state.json")
    states = sorted(glob.glob(pattern), reverse=True)
    return states[0] if states else None


def _load_paper(path):
    """Load paper_content from JSON file."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        prog="paperfactory",
        description="PaperFactory — AI Research Paper Agent for Civil Engineering",
    )
    subparsers = parser.add_subparsers(dest="command")

    # new
    p_new = subparsers.add_parser("new", help="Start a new paper pipeline")
    p_new.add_argument("--topic", "-t", required=True, help="Research topic")
    p_new.add_argument("--journal", "-j", required=True, help="Target journal key")

    # status
    p_status = subparsers.add_parser("status", help="Show pipeline status")
    p_status.add_argument("--run-id", "-r", help="Specific run ID")

    # resume
    p_resume = subparsers.add_parser("resume", help="Resume a pipeline")
    p_resume.add_argument("--run-id", "-r", required=True, help="Run ID to resume")

    # check
    p_check = subparsers.add_parser("check", help="Quality check a paper")
    p_check.add_argument("--paper", "-p", required=True, help="Paper content JSON file")
    p_check.add_argument("--journal", "-j", required=True, help="Journal key")
    p_check.add_argument("--figures", "-f", help="Comma-separated figure paths")

    # review
    p_review = subparsers.add_parser("review", help="AI reviewer simulation")
    p_review.add_argument("--paper", "-p", required=True, help="Paper content JSON file")
    p_review.add_argument("--journal", "-j", required=True, help="Journal key")

    # cover-letter
    p_cl = subparsers.add_parser("cover-letter", help="Generate cover letter")
    p_cl.add_argument("--paper", "-p", required=True, help="Paper content JSON file")
    p_cl.add_argument("--journal", "-j", required=True, help="Journal key")
    p_cl.add_argument("--editor", "-e", default="Editor-in-Chief", help="Editor name")
    p_cl.add_argument("--output", "-o", help="Output file path")

    # sources
    p_src = subparsers.add_parser("sources", help="Suggest data sources")
    p_src.add_argument("--topic", "-t", required=True, help="Research topic")

    # journals
    subparsers.add_parser("journals", help="List supported journals")

    # analytics
    p_analytics = subparsers.add_parser("analytics", help="Paper readability & statistics")
    p_analytics.add_argument("--paper", "-p", required=True, help="Paper content JSON")
    p_analytics.add_argument("--verbose", "-v", action="store_true", help="Show detailed breakdown")

    # abstract
    p_abstract = subparsers.add_parser("abstract", help="Generate or improve abstract")
    p_abstract.add_argument("--paper", "-p", required=True, help="Paper content JSON")
    p_abstract.add_argument("--generate", "-g", action="store_true", help="Generate new abstract")
    p_abstract.add_argument("--max-words", "-m", type=int, default=250, help="Max words")

    # import-bib
    p_bib = subparsers.add_parser("import-bib", help="Import BibTeX references")
    p_bib.add_argument("--bib", "-b", required=True, help="Path to .bib file")
    p_bib.add_argument("--output", "-o", help="Save as JSON")

    # convert-refs
    p_conv = subparsers.add_parser("convert-refs", help="Convert citation style")
    p_conv.add_argument("--paper", "-p", required=True, help="Paper content JSON")
    p_conv.add_argument("--style", "-s", required=True, choices=["numbered", "author_date", "apa"])

    # templates
    p_tmpl = subparsers.add_parser("templates", help="List or generate from templates")
    p_tmpl.add_argument("--generate", "-g", help="Template key to generate from")
    p_tmpl.add_argument("--topic", "-t", help="Research topic for skeleton")
    p_tmpl.add_argument("--journal", "-j", help="Target journal key")
    p_tmpl.add_argument("--output", "-o", help="Output JSON path")

    args = parser.parse_args()

    commands = {
        "new": cmd_new,
        "status": cmd_status,
        "resume": cmd_resume,
        "check": cmd_check,
        "review": cmd_review,
        "cover-letter": cmd_cover_letter,
        "sources": cmd_sources,
        "journals": cmd_journals,
        "analytics": cmd_analytics,
        "abstract": cmd_abstract,
        "import-bib": cmd_import_bib,
        "convert-refs": cmd_convert_refs,
        "templates": cmd_templates,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
