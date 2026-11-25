"""Reference format validation and checking for PaperFactory.

Besides the public checks (:func:`validate_references`, :func:`check_duplicates`,
:func:`detect_citation_style`) this module provides the low-level helpers the
other checkers share, so that every module agrees on what a "year" or a "DOI"
inside a reference string is:

* :func:`normalize_references` -- accept a list, a newline-separated string or
  ``None`` and return clean reference strings;
* :func:`extract_year` -- the publication year, ignoring digits inside DOIs,
  URLs, page ranges, article numbers and standard designations
  (``EN 1993-1-3``);
* :func:`extract_doi` / :func:`has_doi` -- DOI detection (``https://doi.org/``,
  ``doi:`` / ``DOI:`` labels and bare ``10.xxxx/...`` DOIs).
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterable

# ── Shared helpers ───────────────────────────────────────────────────────────

#: A DOI: "10." + registrant code + "/" + suffix (no whitespace).
DOI_PATTERN = r"10\.\d{4,9}/[^\s\"<>]+"
_DOI_RE = re.compile(DOI_PATTERN)
# A DOI with whatever label/URL precedes it, used to blank DOIs out of a string.
_DOI_SPAN_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|\bdoi\s*:?\s*)?" + DOI_PATTERN, re.IGNORECASE
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

# Leading list numbering: "[12]", "12.", "(12)", "12)" -- never a bare year.
_NUMBERING_RE = re.compile(r"^\s*(?:\[\s*\d{1,4}\s*\]|\(\s*\d{1,4}\s*\)|\d{1,4}\s*[.)])\s*")

# A four-digit year 1800-2099 that is not glued to other digits, letters,
# hyphens/dashes (page ranges, "EN 1993-1-3") or slashes.  Hangul may follow
# ("2023년").  An optional disambiguation letter ("2023a") is allowed.
_YEAR_RE = re.compile(
    r"(?<![A-Za-z0-9\-‐-―/])((?:18|19|20)\d{2})([a-z])?(?![A-Za-z0-9\-‐-―/]|\.\d)"
)


def normalize_references(references) -> list[str]:
    """Return *references* as a list of non-empty, stripped strings.

    Accepts a list/tuple (``None`` and blank entries are dropped), a single
    newline-separated string, or ``None``.
    """
    if references is None:
        return []
    if isinstance(references, str):
        return [line.strip() for line in references.splitlines() if line.strip()]
    if isinstance(references, dict):
        references = references.values()
    if not isinstance(references, Iterable):
        references = [references]
    return [str(r).strip() for r in references if r is not None and str(r).strip()]


def _indexed_references(references) -> list[tuple[int, str]]:
    """Like :func:`normalize_references` but keep each entry's original index."""
    if references is None:
        return []
    if isinstance(references, str):
        return list(enumerate(normalize_references(references)))
    out = []
    for i, ref in enumerate(references):
        if ref is None:
            continue
        text = str(ref).strip()
        if text:
            out.append((i, text))
    return out


def strip_numbering(ref: str) -> str:
    """Remove a leading list number such as ``[1]``, ``1.`` or ``(1)``."""
    return _NUMBERING_RE.sub("", ref or "", count=1).strip()


def extract_doi(ref: str) -> str | None:
    """Return the bare DOI (``10.xxxx/...``) found in *ref*, or ``None``."""
    match = _DOI_RE.search(ref or "")
    if not match:
        return None
    doi = match.group().rstrip(".,;:")
    # Drop closing brackets that belong to the surrounding text, not the DOI.
    for open_c, close_c in (("(", ")"), ("[", "]")):
        while doi.endswith(close_c) and doi.count(close_c) > doi.count(open_c):
            doi = doi[:-1].rstrip(".,;:")
    return doi


def has_doi(ref: str) -> bool:
    """True if *ref* contains a DOI (URL form, ``doi:`` label or bare DOI)."""
    return extract_doi(ref) is not None


def _remove_identifiers(ref: str) -> str:
    """Blank out DOIs and URLs so their digits are not mistaken for years."""
    text = _DOI_SPAN_RE.sub(" ", ref or "")
    return _URL_RE.sub(" ", text)


def extract_year(ref: str, current_year: int | None = None) -> int | None:
    """Return the publication year of a reference string, or ``None``.

    Heuristics, in order of preference:

    1. a parenthesised year -- ``(2018)`` -- preferring the one that follows
       a volume/issue (``158 (2018)``) over one inside a title;
    2. a year standing as its own field (followed by ``.``, ``,``, ``;``,
       ``:`` or the end): the first one when it directly follows the author
       block (``Beskos, D. E. 1987.``), otherwise the last one (Elsevier book
       style ``..., CRC Press, Boca Raton, 2018.``);
    3. any other year-like number.

    Digits inside DOIs/URLs, page ranges (``2001-2015``), article numbers and
    standard designations (``EN 1993-1-3``) are ignored, as are years later
    than ``current_year + 1``.
    """
    if not ref:
        return None
    if current_year is None:
        current_year = datetime.date.today().year
    text = _remove_identifiers(str(ref))
    text = strip_numbering(text)

    candidates = []
    for m in _YEAR_RE.finditer(text):
        year = int(m.group(1))
        if year > current_year + 1:
            continue
        before = text[: m.start()]
        after = text[m.end() :]
        candidates.append((year, before, after))
    if not candidates:
        return None

    parenthesised = [
        c for c in candidates if c[1].rstrip().endswith("(") and c[2].lstrip().startswith(")")
    ]
    if parenthesised:
        for year, before, _after in parenthesised:
            prefix = before.rstrip()[:-1].rstrip()
            if prefix and (prefix[-1].isdigit() or prefix[-1] == ")"):
                return year
        return parenthesised[0][0]

    standalone = [c for c in candidates if re.match(r"\s*(?:[.,;:]|$)", c[2])]
    if standalone:
        first_year, first_before, first_after = standalone[0]
        if first_after.startswith(".") and first_before.rstrip().endswith("."):
            return first_year  # author-date: "Name, I. 2019. Title ..."
        return standalone[-1][0]

    return candidates[0][0]


# ── Public checks ────────────────────────────────────────────────────────────


def validate_references(references: list[str], guideline: dict) -> list[dict]:
    """Validate references against journal guideline. Returns list of issues.

    *references* may be a list or a newline-separated string; ``None`` and
    blank entries are skipped.  Each issue is a dict with ``index`` (position
    in the given list), ``issue`` and ``reference``.
    """
    issues = []
    ref_config = (guideline or {}).get("references", {}) or {}
    doi_required = ref_config.get("doi_required", False)

    for i, ref_stripped in _indexed_references(references):
        if doi_required and not has_doi(ref_stripped):
            issues.append(
                {
                    "index": i,
                    "issue": "Missing DOI",
                    "reference": ref_stripped,
                }
            )

        if extract_year(ref_stripped) is None:
            issues.append(
                {
                    "index": i,
                    "issue": "Missing or unrecognized year",
                    "reference": ref_stripped,
                }
            )

        if len(ref_stripped) < 20:
            issues.append(
                {
                    "index": i,
                    "issue": "Reference appears incomplete (too short)",
                    "reference": ref_stripped,
                }
            )

    return issues


def _normalize_for_compare(ref: str) -> str:
    """Normalise a reference for duplicate detection (numbering, case, punctuation)."""
    clean = strip_numbering(ref).lower()
    clean = re.sub(r"[‐-―]", "-", clean)  # unify dashes
    clean = re.sub(r"[^\w\s]", " ", clean)  # punctuation is not significant
    return re.sub(r"\s+", " ", clean).strip()


def check_duplicates(references: list[str]) -> list[dict]:
    """Detect duplicate references. Returns list of duplicate pairs.

    Two entries are duplicates when they are identical after removing list
    numbering, case, punctuation and whitespace differences, or when they
    carry the same DOI.  Each result has ``index_a`` (first occurrence),
    ``index_b`` (the duplicate), ``reference`` and ``reason``.
    """
    duplicates = []
    seen_text: dict[str, int] = {}
    seen_doi: dict[str, int] = {}

    for i, ref in _indexed_references(references):
        norm = _normalize_for_compare(ref)
        if not norm:
            continue
        doi = extract_doi(ref)
        doi = doi.lower() if doi else None

        if norm in seen_text:
            first, reason = seen_text[norm], "identical text"
        elif doi and doi in seen_doi:
            first, reason = seen_doi[doi], "same DOI"
        else:
            first = None
        if first is not None:
            duplicates.append(
                {"index_a": first, "index_b": i, "reference": ref, "reason": reason}
            )
        seen_text.setdefault(norm, i)
        if doi:
            seen_doi.setdefault(doi, i)

    return duplicates


_AUTHOR_DATE_MARKER_RE = re.compile(
    r"\(\s*(?:18|19|20)\d{2}[a-z]?\s*\)\s*[.,:]|(?<![\d\-])(?:18|19|20)\d{2}[a-z]?\.(?:\s|$)"
)


def _classify_reference(ref: str) -> str:
    if _NUMBERING_RE.match(ref) and not re.match(r"^\s*(?:18|19|20)\d{2}\.", ref):
        return "numbered"
    head = _remove_identifiers(ref)
    if _AUTHOR_DATE_MARKER_RE.search(head):
        return "author-date"
    if re.match(r"^[A-Z][a-z]+.*\d{4}", ref):
        return "author-date"
    return "unknown"


def detect_citation_style(references: list[str]) -> str:
    """Detect whether references use numbered [1] or author-date style.

    Returns ``"numbered"``, ``"author-date"`` or ``"unknown"``.  The decision
    is a majority vote over all references (a string is split into lines).
    """
    refs = normalize_references(references)
    if not refs:
        return "unknown"
    votes = {"numbered": 0, "author-date": 0}
    for ref in refs:
        kind = _classify_reference(ref)
        if kind in votes:
            votes[kind] += 1
    if not any(votes.values()):
        return "unknown"
    if votes["numbered"] == votes["author-date"]:
        # Tie: fall back to the first classifiable entry (old behaviour).
        for ref in refs:
            kind = _classify_reference(ref)
            if kind != "unknown":
                return kind
    return max(votes, key=votes.get)
