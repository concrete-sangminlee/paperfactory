"""Reference format validation and checking for PaperFactory."""

import re


def validate_references(references: list[str], guideline: dict) -> list[dict]:
    """Validate references against journal guideline. Returns list of issues."""
    issues = []
    ref_config = guideline.get("references", {})
    doi_required = ref_config.get("doi_required", False)

    for i, ref in enumerate(references):
        ref_stripped = ref.strip()
        if not ref_stripped:
            continue

        # Check DOI — match doi.org/... URLs or explicit "DOI:" / "doi:" label followed by 10.
        has_doi = bool(re.search(r'doi\.org/10\.|(?:^|[\s,;])DOI:?\s*10\.', ref_stripped, re.IGNORECASE))
        if doi_required and not has_doi:
            issues.append({
                "index": i,
                "issue": "Missing DOI",
                "reference": ref_stripped,
            })

        # Check year
        has_year = bool(re.search(r'(19|20)\d{2}', ref_stripped))
        if not has_year:
            issues.append({
                "index": i,
                "issue": "Missing or unrecognized year",
                "reference": ref_stripped,
            })

        # Check empty or too short
        if len(ref_stripped) < 20:
            issues.append({
                "index": i,
                "issue": "Reference appears incomplete (too short)",
                "reference": ref_stripped,
            })

    return issues


def check_duplicates(references: list[str]) -> list[dict]:
    """Detect duplicate references. Returns list of duplicate pairs."""
    duplicates = []
    normalized = []
    for ref in references:
        # Strip numbering prefix like [1], 1., etc.
        clean = re.sub(r'^\s*\[?\d+\]?\s*\.?\s*', '', ref.strip()).lower()
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean)
        normalized.append(clean)

    seen = {}
    for i, norm in enumerate(normalized):
        if not norm:
            continue
        if norm in seen:
            duplicates.append({
                "index_a": seen[norm],
                "index_b": i,
                "reference": references[i].strip(),
            })
        else:
            seen[norm] = i

    return duplicates


def detect_citation_style(references: list[str]) -> str:
    """Detect whether references use numbered [1] or author-date style."""
    if not references:
        return "unknown"
    first = references[0].strip()
    if re.match(r'^\[?\d+\]', first):
        return "numbered"
    if re.match(r'^[A-Z][a-z]+.*\d{4}', first):
        return "author-date"
    return "unknown"
