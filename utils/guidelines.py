"""Journal guideline registry — the single source of truth for guideline JSON files.

Every module that needs journal formatting rules should go through
:func:`load_guideline` instead of opening ``guidelines/*.json`` directly.  The
registry adds:

* case-insensitive keys, an optional ``.json`` suffix and a small alias table
  (e.g. ``jcsr`` -> ``steel_comp_struct``);
* helpful errors that suggest the closest valid key;
* in-process caching (callers always receive an independent deep copy, so
  mutating the returned dict never corrupts the cache);
* schema validation (:func:`validate_guideline`) used by the test-suite;
* an override directory via the ``PAPERFACTORY_GUIDELINES_DIR`` environment
  variable, so custom journals can be added without touching the repository.
"""

from __future__ import annotations

import copy
import difflib
import json
import os
from functools import lru_cache

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GUIDELINES_DIR = os.path.join(PROJECT_ROOT, "guidelines")
GUIDELINES_DIR_ENV = "PAPERFACTORY_GUIDELINES_DIR"

# Backwards-compatible alias kept for modules that imported the constant.
GUIDELINES_DIR = DEFAULT_GUIDELINES_DIR

#: Alternative spellings accepted for a journal key.
JOURNAL_ALIASES = {
    "jcsr": "steel_comp_struct",
    "engstruct": "eng_structures",
    "cbm": "const_build_mat",
    "ccc": "cem_con_comp",
    "aic": "autom_constr",
    "buildings": "buildings_mdpi",
    "ksce": "ksce_jce",
}

#: Top-level keys every guideline file must define.
REQUIRED_KEYS = (
    "journal_name",
    "abbreviation",
    "publisher",
    "scope",
    "formatting",
    "manuscript_structure",
    "references",
    "figures_tables",
    "submission_notes",
)

#: Keys required inside ``manuscript_structure``.
REQUIRED_STRUCTURE_KEYS = ("sections", "abstract_word_limit")

#: Keys required inside ``references``.
REQUIRED_REFERENCE_KEYS = ("style", "examples")


class UnknownJournalError(FileNotFoundError, ValueError):
    """Raised when a journal key does not match any guideline file.

    Subclasses :class:`FileNotFoundError` for backwards compatibility with code
    that caught the error raised by the old per-module loaders, and
    :class:`ValueError` because an unknown key is, semantically, a bad value.
    """


def get_guidelines_dir() -> str:
    """Return the active guidelines directory (env override or bundled default)."""
    return os.environ.get(GUIDELINES_DIR_ENV) or DEFAULT_GUIDELINES_DIR


def _available_keys(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))


def available_journal_keys() -> list[str]:
    """Return the sorted list of canonical journal keys."""
    return _available_keys(get_guidelines_dir())


def resolve_journal_key(journal_key: str) -> str:
    """Normalise *journal_key* to a canonical guideline key.

    Accepts mixed case, surrounding whitespace, a trailing ``.json`` and the
    aliases in :data:`JOURNAL_ALIASES`.

    Raises
    ------
    UnknownJournalError
        If no guideline matches; the message lists close matches.
    """
    if not isinstance(journal_key, str) or not journal_key.strip():
        raise UnknownJournalError(f"Journal key must be a non-empty string, got {journal_key!r}")

    key = journal_key.strip().lower()
    if key.endswith(".json"):
        key = key[:-5]
    key = key.replace("-", "_").replace(" ", "_")
    key = JOURNAL_ALIASES.get(key, key)

    keys = available_journal_keys()
    if key in keys:
        return key
    if not keys:
        raise UnknownJournalError(
            f"No guideline files found in {get_guidelines_dir()}. Run PaperFactory from a "
            f"repository clone (pip install -e .) or set {GUIDELINES_DIR_ENV}."
        )

    suggestions = difflib.get_close_matches(key, keys + list(JOURNAL_ALIASES), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise UnknownJournalError(
        f"Guideline not found for journal '{journal_key}'.{hint} "
        f"Available: {', '.join(keys) or '(none)'}"
    )


@lru_cache(maxsize=64)
def _load_cached(directory: str, key: str, mtime_ns: int) -> dict:
    path = os.path.join(directory, f"{key}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_guideline(journal_key: str) -> dict:
    """Load the guideline for *journal_key* and return an independent copy."""
    key = resolve_journal_key(journal_key)
    directory = get_guidelines_dir()
    path = os.path.join(directory, f"{key}.json")
    data = _load_cached(directory, key, os.stat(path).st_mtime_ns)
    return copy.deepcopy(data)


def list_journals() -> list[dict]:
    """Return metadata for every available journal, sorted by key."""
    journals = []
    for key in available_journal_keys():
        g = load_guideline(key)
        ms = g.get("manuscript_structure", {})
        journals.append(
            {
                "key": key,
                "name": g.get("journal_name", key),
                "abbreviation": g.get("abbreviation", ""),
                "publisher": g.get("publisher", ""),
                "reference_style": g.get("references", {}).get("style", ""),
                "abstract_word_limit": ms.get("abstract_word_limit"),
                "highlights_required": bool(ms.get("highlights_required")),
                "graphical_abstract_required": bool(ms.get("graphical_abstract_required")),
            }
        )
    return journals


def validate_guideline(guideline: dict) -> list[str]:
    """Return a list of schema problems for a guideline dict (empty if valid)."""
    issues = []
    if not isinstance(guideline, dict):
        return ["Guideline must be a JSON object"]

    for key in REQUIRED_KEYS:
        if key not in guideline:
            issues.append(f"Missing top-level key: {key}")

    ms = guideline.get("manuscript_structure", {})
    if isinstance(ms, dict):
        for key in REQUIRED_STRUCTURE_KEYS:
            if key not in ms:
                issues.append(f"Missing manuscript_structure.{key}")
        sections = ms.get("sections")
        if sections is not None and (
            not isinstance(sections, list) or not all(isinstance(s, str) for s in sections)
        ):
            issues.append("manuscript_structure.sections must be a list of strings")
        limit = ms.get("abstract_word_limit")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            issues.append("manuscript_structure.abstract_word_limit must be a positive integer")
    else:
        issues.append("manuscript_structure must be an object")

    refs = guideline.get("references", {})
    if isinstance(refs, dict):
        for key in REQUIRED_REFERENCE_KEYS:
            if key not in refs:
                issues.append(f"Missing references.{key}")
        examples = refs.get("examples")
        if examples is not None and (not isinstance(examples, list) or not examples):
            issues.append("references.examples must be a non-empty list")
    else:
        issues.append("references must be an object")

    for key in ("formatting", "figures_tables", "submission_notes"):
        if key in guideline and not isinstance(guideline[key], dict):
            issues.append(f"{key} must be an object")

    return issues
