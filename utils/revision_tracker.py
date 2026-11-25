"""Track paper revisions and generate response-to-reviewers documents."""

import difflib
import json
import os
import textwrap
from datetime import datetime

DEFAULT_REVISIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "papers"
)


def create_revision(
    original_content: dict,
    reviewer_comments: list,
    output_dir: str = None,
) -> dict:
    """Create a revision tracking record.

    Parameters
    ----------
    original_content : dict
        The original paper_content dict.
    reviewer_comments : list[dict]
        List of reviewer comments, each with keys:
        - reviewer (str): e.g. "Reviewer 1"
        - comment (str): the reviewer's comment
        - section (str, optional): which section the comment refers to
        - type (str): "major" or "minor"

    Returns
    -------
    dict
        Revision record with tracking info.
    """
    revision = {
        "created_at": datetime.now().isoformat(),
        "original_title": original_content.get("title", ""),
        "status": "in_progress",
        "comments": [],
        "summary": {
            "total": len(reviewer_comments),
            "major": 0,
            "minor": 0,
            "addressed": 0,
        },
    }

    for i, comment in enumerate(reviewer_comments, 1):
        entry = {
            "id": i,
            "reviewer": str(comment.get("reviewer") or "Reviewer").strip() or "Reviewer",
            "comment": str(comment.get("comment") or ""),
            "section": str(comment.get("section") or "General").strip() or "General",
            # "Major", " MAJOR " and "major" are the same type; anything else is minor.
            "type": "major" if _normalize_type(comment.get("type")) == "major" else "minor",
            "status": "pending",
            "response": "",
            "changes_made": "",
        }
        revision["comments"].append(entry)
        if entry["type"] == "major":
            revision["summary"]["major"] += 1
        else:
            revision["summary"]["minor"] += 1

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "revision_record.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(revision, f, indent=2, ensure_ascii=False)

    return revision


def address_comment(
    revision: dict,
    comment_id: int,
    response: str,
    changes_made: str,
) -> dict:
    """Mark a reviewer comment as addressed.

    Parameters
    ----------
    revision : dict
        The revision record.
    comment_id : int
        ID of the comment to address.
    response : str
        Response to the reviewer.
    changes_made : str
        Description of changes made.

    Returns
    -------
    dict
        Updated revision record.

    Raises
    ------
    KeyError
        If no comment has ``comment_id`` (an int, or a string of digits).
    """
    if isinstance(comment_id, str) and comment_id.strip().isdigit():
        comment_id = int(comment_id)
    for comment in revision["comments"]:
        if comment["id"] == comment_id:
            comment["status"] = "addressed"
            comment["response"] = response
            comment["changes_made"] = changes_made
            break
    else:
        available = ", ".join(str(c["id"]) for c in revision["comments"]) or "(none)"
        raise KeyError(f"No reviewer comment with id {comment_id!r}. Available: {available}")

    summary = revision["summary"]
    summary["total"] = len(revision["comments"])
    summary["addressed"] = sum(1 for c in revision["comments"] if c["status"] == "addressed")
    if summary["addressed"] == summary["total"]:
        revision["status"] = "complete"

    return revision


def _normalize_type(value) -> str:
    return str(value or "minor").strip().lower()


def _indent(text: str) -> str:
    """Indent every line of a (possibly multi-line) response by two spaces."""
    return textwrap.indent(str(text or ""), "  ", lambda line: True)


def generate_response_letter(revision: dict) -> str:
    """Generate a response-to-reviewers letter as formatted text.

    Returns
    -------
    str
        Formatted response letter text.
    """
    lines = []
    lines.append("RESPONSE TO REVIEWERS")
    lines.append("=" * 60)
    lines.append(f"\nManuscript: {revision.get('original_title', '')}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    summary = revision["summary"]
    lines.append(f"Status: {summary['addressed']}/{summary['total']} comments addressed")
    lines.append("")

    lines.append("Dear Editor and Reviewers,")
    lines.append("")
    lines.append(
        "We thank the reviewers for their constructive comments and suggestions. "
        "We have carefully revised the manuscript to address all the comments. "
        "Below we provide point-by-point responses."
    )
    lines.append("")

    # Group by reviewer
    reviewers = {}
    for comment in revision["comments"]:
        r = comment["reviewer"]
        if r not in reviewers:
            reviewers[r] = []
        reviewers[r].append(comment)

    for reviewer, comments in reviewers.items():
        lines.append("-" * 60)
        lines.append(f"\n{reviewer}")
        lines.append("-" * 60)

        for comment in comments:
            lines.append(
                f"\nComment {comment['id']} [{_normalize_type(comment.get('type')).upper()}] "
                f"(Section: {comment.get('section') or 'General'}):"
            )
            lines.append(_indent(f'"{comment["comment"]}"'))
            lines.append("")

            if comment["status"] == "addressed":
                lines.append("  Response:")
                lines.append(_indent(comment["response"]))
                lines.append("\n  Changes made:")
                lines.append(_indent(comment["changes_made"]))
            else:
                lines.append("  [PENDING - Not yet addressed]")
            lines.append("")

    lines.append("=" * 60)
    lines.append(
        "We believe the revised manuscript addresses all the reviewers' concerns "
        "and is now suitable for publication."
    )
    lines.append("")
    lines.append("Sincerely,")
    lines.append("The Authors")

    return "\n".join(lines)


def track_changes(original_text, revised_text) -> dict:
    """Compare original and revised text to identify changes.

    The comparison is a word-level diff (case-insensitive), so repeated,
    duplicated or re-ordered words are counted correctly.

    Parameters
    ----------
    original_text, revised_text : str or dict
        Plain text, or two ``paper_content`` dicts; with dicts, the sections
        whose text changed are reported in ``sections_changed``.

    Returns
    -------
    dict with keys:
        - added_words (int): words inserted in the revision
        - removed_words (int): words deleted from the original
        - net_change (int): revised word count minus original word count
        - sections_changed (list): headings of changed sections (dict inputs only)
        - similarity (float): 0-1 word-sequence similarity ratio
    """
    sections_changed = []
    if isinstance(original_text, dict) or isinstance(revised_text, dict):
        orig_sections = _section_texts(original_text if isinstance(original_text, dict) else {})
        rev_sections = _section_texts(revised_text if isinstance(revised_text, dict) else {})
        for heading in list(orig_sections) + [h for h in rev_sections if h not in orig_sections]:
            if orig_sections.get(heading, "").split() != rev_sections.get(heading, "").split():
                sections_changed.append(heading)
        original_text = "\n".join(orig_sections.values())
        revised_text = "\n".join(rev_sections.values())

    orig_words = str(original_text or "").lower().split()
    rev_words = str(revised_text or "").lower().split()

    matcher = difflib.SequenceMatcher(None, orig_words, rev_words, autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1

    return {
        "added_words": added,
        "removed_words": removed,
        "net_change": len(rev_words) - len(orig_words),
        "sections_changed": sections_changed,
        "similarity": round(matcher.ratio(), 4) if (orig_words or rev_words) else 1.0,
    }


def _section_texts(paper_content: dict) -> dict:
    """Map each section heading to its full text (including subsections)."""
    texts = {}
    for i, sec in enumerate(paper_content.get("sections") or [], 1):
        if not isinstance(sec, dict):
            continue
        heading = str(sec.get("heading") or f"Section {i}")
        parts = [sec.get("content") or ""]
        for sub in sec.get("subsections") or []:
            if isinstance(sub, dict):
                parts.append(sub.get("content") or "")
                for subsub in sub.get("subsections") or []:
                    if isinstance(subsub, dict):
                        parts.append(subsub.get("content") or "")
        key, n = heading, 2
        while key in texts:
            key, n = f"{heading} ({n})", n + 1
        texts[key] = "\n".join(parts)
    return texts
