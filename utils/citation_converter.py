"""Convert references between citation styles (numbered, author-date, APA)."""

import re


def convert_style(references: list, target_style: str) -> list:
    """Convert a list of reference strings to the target citation style.

    Parameters
    ----------
    references : list[str]
        List of reference strings in any format.
    target_style : str
        "numbered" — [1] Author, Title, Journal Vol (Year) Pages.
        "author_date" — Author (Year). Title. Journal, Vol, Pages.
        "apa" — Author, A. B. (Year). Title. Journal, Vol(Issue), Pages. DOI

    Returns
    -------
    list[str]
        Converted references.
    """
    parsed = [_parse_ref(ref) for ref in references]
    converters = {
        "numbered": _to_numbered,
        "author_date": _to_author_date,
        "apa": _to_apa,
    }
    if target_style not in converters:
        raise ValueError(f"Unknown style '{target_style}'. Choose: {list(converters.keys())}")

    return [converters[target_style](p, i + 1) for i, p in enumerate(parsed)]


def _parse_ref(ref: str) -> dict:
    """Parse a reference string into components (best-effort)."""
    fields = {"raw": ref}

    # Remove leading number [1] or 1.
    clean = re.sub(r"^\s*\[?\d+\]?\s*\.?\s*", "", ref).strip()

    # DOI
    doi_match = re.search(r"(https?://doi\.org/\S+|10\.\d{4,}/[^\s,]+)", clean)
    if doi_match:
        fields["doi"] = doi_match.group().rstrip(".")
        if not fields["doi"].startswith("http"):
            fields["doi"] = "https://doi.org/" + fields["doi"]

    # Year
    year_match = re.search(r"\((\d{4})\)|\b((?:19|20)\d{2})\b", clean)
    if year_match:
        fields["year"] = year_match.group(1) or year_match.group(2)

    # Try to split author from rest
    # Pattern: "A.B. Name, C.D. Name, Title..."
    parts = clean.split(",")
    if len(parts) >= 3:
        author_parts = []
        rest_start = 0
        for j, part in enumerate(parts):
            if re.search(r"\b[A-Z]\.\s*[A-Z]?\.", part.strip()) or (
                j == 0 and re.search(r"\b[A-Z]\.", part.strip())
            ):
                author_parts.append(part.strip())
                rest_start = j + 1
            else:
                break
        if author_parts:
            fields["authors"] = ", ".join(author_parts)
            remaining = ", ".join(parts[rest_start:])
            # Title is usually the first substantial text segment
            title_match = re.match(r"\s*(.+?),\s*([A-Z])", remaining)
            if title_match:
                fields["title"] = title_match.group(1).strip()
            elif rest_start < len(parts):
                fields["title"] = parts[rest_start].strip()

    # Volume
    vol_match = re.search(r"\b(\d{1,4})\s*\(", clean)
    if vol_match:
        fields["volume"] = vol_match.group(1)

    # Pages
    pages_match = re.search(r"(\d+[-–]\d+|\d{5,})", clean)
    if pages_match:
        fields["pages"] = pages_match.group().replace("–", "-")

    # Journal (heuristic: text between title and volume)
    if "title" in fields and "volume" in fields:
        title_end = clean.find(fields["title"]) + len(fields["title"])
        vol_start = clean.find(fields["volume"], title_end)
        if title_end < vol_start:
            journal = clean[title_end:vol_start].strip().strip(",").strip()
            if len(journal) > 3:
                fields["journal"] = journal

    return fields


def _to_numbered(fields: dict, num: int) -> str:
    """Convert to numbered style: [1] Author, Title, Journal Vol (Year) Pages."""
    parts = [f"[{num}]"]
    if "authors" in fields:
        parts.append(fields["authors"] + ",")
    if "title" in fields:
        parts.append(fields["title"] + ",")
    if "journal" in fields:
        j = fields["journal"]
        if "volume" in fields:
            j += f" {fields['volume']}"
        if "year" in fields:
            j += f" ({fields['year']})"
        if "pages" in fields:
            j += f" {fields['pages']}"
        parts.append(j + ".")
    elif "year" in fields:
        parts.append(f"({fields['year']}).")
    if "doi" in fields:
        parts.append(fields["doi"])
    if len(parts) <= 2:
        return f"[{num}] {fields['raw']}"
    return " ".join(parts)


def _to_author_date(fields: dict, num: int) -> str:
    """Convert to author-date style: Author (Year). Title. Journal, Vol, Pages."""
    parts = []
    if "authors" in fields:
        parts.append(fields["authors"])
    if "year" in fields:
        parts.append(f"({fields['year']}).")
    if "title" in fields:
        parts.append(f"{fields['title']}.")
    if "journal" in fields:
        j = fields["journal"]
        if "volume" in fields:
            j += f", {fields['volume']}"
        if "pages" in fields:
            j += f", {fields['pages']}"
        parts.append(f"{j}.")
    if "doi" in fields:
        parts.append(fields["doi"])
    if not parts:
        return fields["raw"]
    return " ".join(parts)


def _to_apa(fields: dict, num: int) -> str:
    """Convert to APA style: Author, A. B. (Year). Title. Journal, Vol(Issue), Pages. DOI"""
    parts = []
    if "authors" in fields:
        parts.append(f"{fields['authors']}")
    if "year" in fields:
        parts.append(f"({fields['year']}).")
    else:
        parts.append("(n.d.).")
    if "title" in fields:
        parts.append(f"{fields['title']}.")
    if "journal" in fields:
        j = f"*{fields['journal']}*"
        if "volume" in fields:
            j += f", *{fields['volume']}*"
        if "pages" in fields:
            j += f", {fields['pages']}"
        parts.append(f"{j}.")
    if "doi" in fields:
        parts.append(fields["doi"])
    if not parts:
        return fields["raw"]
    return " ".join(parts)
