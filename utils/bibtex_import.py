"""Import references from BibTeX (.bib) files into paper_content format."""

import re


def import_bib(bib_path: str) -> list:
    """Parse a .bib file and return a list of formatted reference strings.

    Parameters
    ----------
    bib_path : str
        Path to .bib file.

    Returns
    -------
    list[str]
        Numbered reference strings suitable for paper_content["references"].
    """
    with open(bib_path, encoding="utf-8") as f:
        content = f.read()
    return parse_bib_string(content)


def parse_bib_string(bib_content: str) -> list:
    """Parse BibTeX content string into formatted references."""
    entries = _extract_entries(bib_content)
    refs = []
    for i, entry in enumerate(entries, 1):
        formatted = _format_entry(entry, i)
        if formatted:
            refs.append(formatted)
    return refs


def _extract_entries(content: str) -> list:
    """Extract individual BibTeX entries from content."""
    entries = []
    pattern = re.compile(r"@(\w+)\s*\{([^,]*),\s*(.*?)\n\}", re.DOTALL)

    for match in pattern.finditer(content):
        entry_type = match.group(1).lower()
        cite_key = match.group(2).strip()
        body = match.group(3)

        fields = _parse_fields(body)
        fields["_type"] = entry_type
        fields["_key"] = cite_key
        entries.append(fields)

    return entries


def _parse_fields(body: str) -> dict:
    """Parse BibTeX field = {value} pairs."""
    fields = {}
    pattern = re.compile(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)

    for match in pattern.finditer(body):
        key = match.group(1).lower()
        value = match.group(2).strip()
        fields[key] = value

    return fields


def _format_entry(entry: dict, num: int) -> str:
    """Format a parsed BibTeX entry as a numbered reference string."""
    parts = [f"[{num}]"]

    author = entry.get("author", "")
    if author:
        author = _format_authors(author)
        parts.append(author + ",")

    title = entry.get("title", "")
    if title:
        title = title.strip("{}")
        parts.append(title + ",")

    journal = entry.get("journal", entry.get("booktitle", ""))
    if journal:
        vol = entry.get("volume", "")
        year = entry.get("year", "")
        pages = entry.get("pages", "")

        j_str = journal
        if vol:
            j_str += f" {vol}"
        if year:
            j_str += f" ({year})"
        if pages:
            j_str += f" {pages}"
        parts.append(j_str + ".")

    elif entry.get("_type") == "book":
        publisher = entry.get("publisher", "")
        address = entry.get("address", "")
        year = entry.get("year", "")
        book_parts = []
        if publisher:
            book_parts.append(publisher)
        if address:
            book_parts.append(address)
        if year:
            book_parts.append(year)
        if book_parts:
            parts.append(", ".join(book_parts) + ".")

    doi = entry.get("doi", "")
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    if len(parts) <= 1:
        note = entry.get("note", "")
        if note:
            parts.append(note)

    return " ".join(parts) if len(parts) > 1 else ""


def _format_authors(author_str: str) -> str:
    """Format BibTeX author string (Last, First and Last, First → F. Last, F. Last)."""
    authors = re.split(r"\s+and\s+", author_str)
    formatted = []
    for author in authors:
        author = author.strip()
        if "," in author:
            parts = author.split(",", 1)
            last = parts[0].strip()
            first = parts[1].strip()
            initials = re.findall(r"[A-Z]", first)
            formatted.append(".".join(initials) + ". " + last if initials else last)
        else:
            formatted.append(author)
    return ", ".join(formatted)
