"""Track paper revisions and generate response-to-reviewers documents."""

import json
import os
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
            "reviewer": comment.get("reviewer", "Reviewer"),
            "comment": comment.get("comment", ""),
            "section": comment.get("section", "General"),
            "type": comment.get("type", "minor"),
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
    """
    for comment in revision["comments"]:
        if comment["id"] == comment_id:
            comment["status"] = "addressed"
            comment["response"] = response
            comment["changes_made"] = changes_made
            break

    revision["summary"]["addressed"] = sum(
        1 for c in revision["comments"] if c["status"] == "addressed"
    )
    if revision["summary"]["addressed"] == revision["summary"]["total"]:
        revision["status"] = "complete"

    return revision


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
    lines.append(
        f"Status: {revision['summary']['addressed']}/{revision['summary']['total']} comments addressed"
    )
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
                f"\nComment {comment['id']} [{comment['type'].upper()}] "
                f"(Section: {comment['section']}):"
            )
            lines.append(f'  "{comment["comment"]}"')
            lines.append("")

            if comment["status"] == "addressed":
                lines.append("  Response:")
                lines.append(f"  {comment['response']}")
                lines.append("\n  Changes made:")
                lines.append(f"  {comment['changes_made']}")
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


def track_changes(original_text: str, revised_text: str) -> dict:
    """Compare original and revised text to identify changes.

    Returns
    -------
    dict with keys:
        - added_words (int): approximate words added
        - removed_words (int): approximate words removed
        - sections_changed (list): sections with changes
    """
    orig_words = set(original_text.lower().split())
    rev_words = set(revised_text.lower().split())

    added = rev_words - orig_words
    removed = orig_words - rev_words

    return {
        "added_words": len(added),
        "removed_words": len(removed),
        "net_change": len(rev_words) - len(orig_words),
    }
