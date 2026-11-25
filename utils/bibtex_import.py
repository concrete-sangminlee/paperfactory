"""Import references from BibTeX (.bib) files into paper_content format.

The parser follows the BibTeX grammar closely enough for real-world files
exported by Zotero, Mendeley, JabRef, Google Scholar and publishers:

* entries delimited by braces or parentheses, closed anywhere on a line;
* field values in ``{...}`` (arbitrarily nested), ``"..."``, bare numbers or
  ``@string`` macros (month abbreviations are predefined), joined with ``#``;
* ``@comment``, ``@preamble`` and ``@string`` blocks are not references;
* LaTeX accents and escapes (``{\\'e}``, ``\\"{u}``, ``\\ss``, ``\\&``) are
  converted to Unicode and protective braces are removed;
* author lists in all three BibTeX name forms ("First von Last",
  "von Last, First", "von Last, Jr, First"), ``and others`` and
  brace-protected corporate authors.
"""

from __future__ import annotations

import re
import unicodedata

_MONTHS = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}

# Accent commands -> Unicode combining characters.
_ACCENTS = {
    "'": 0x0301,
    "`": 0x0300,
    "^": 0x0302,
    '"': 0x0308,
    "~": 0x0303,
    "=": 0x0304,
    ".": 0x0307,
    "u": 0x0306,
    "v": 0x030C,
    "H": 0x030B,
    "c": 0x0327,
    "k": 0x0328,
    "r": 0x030A,
    "d": 0x0323,
    "b": 0x0331,
}

# Letter-like commands (longest names first so "\oe" wins over "\o").
_SPECIAL_LETTERS = {
    "ss": 0x00DF,
    "aa": 0x00E5,
    "AA": 0x00C5,
    "ae": 0x00E6,
    "AE": 0x00C6,
    "oe": 0x0153,
    "OE": 0x0152,
    "o": 0x00F8,
    "O": 0x00D8,
    "l": 0x0142,
    "L": 0x0141,
    "i": 0x0131,
    "j": 0x0237,
}

_ENTRY_START_RE = re.compile(r"@\s*([A-Za-z]+)\s*([{(])")
_FIELD_NAME_RE = re.compile(r"[A-Za-z][\w\-:.+/]*")
_BARE_VALUE_RE = re.compile(r"[^\s,#{}\"()]+")
_EN_DASH, _EM_DASH = chr(0x2013), chr(0x2014)


def import_bib(bib_path: str) -> list:
    """Parse a .bib file and return a list of formatted reference strings.

    Parameters
    ----------
    bib_path : str
        Path to .bib file (UTF-8, with or without BOM; Latin-1 is accepted
        as a fallback).

    Returns
    -------
    list[str]
        Numbered reference strings suitable for paper_content["references"].
    """
    with open(bib_path, "rb") as f:
        data = f.read()
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = data.decode("latin-1")
    return parse_bib_string(content)


def parse_bib_string(bib_content: str) -> list:
    """Parse BibTeX content string into formatted references.

    Entries that carry no usable information are skipped and the remaining
    references are numbered consecutively from ``[1]``.
    """
    refs = []
    for entry in parse_bib_entries(bib_content):
        formatted = _format_entry(entry, len(refs) + 1)
        if formatted:
            refs.append(formatted)
    return refs


def parse_bib_entries(bib_content: str) -> list[dict]:
    """Return the raw entries of a BibTeX string as dicts.

    Field names are lower-cased; values have macros expanded but keep their
    LaTeX markup.  ``_type`` and ``_key`` hold the entry type and cite key.
    """
    return _extract_entries(bib_content or "")


# ── Tokenizer ────────────────────────────────────────────────────────────────


def _find_closing(text: str, start: int, opener: str) -> int | None:
    """Index of the delimiter closing an entry opened just before *start*."""
    depth = 0  # brace depth inside the entry
    paren = 1
    in_quote = False
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if opener == "{":
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    return i
                depth -= 1
        else:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(depth - 1, 0)
            elif ch == '"' and depth == 0:
                in_quote = not in_quote
            elif depth == 0 and not in_quote:
                if ch == "(":
                    paren += 1
                elif ch == ")":
                    paren -= 1
                    if paren == 0:
                        return i
        i += 1
    return None


def _read_braced(text: str, i: int) -> tuple[str, int]:
    """Read a ``{...}`` group starting at ``text[i] == '{'``; return (inner, next index)."""
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
        j += 1
    return text[i + 1 :], len(text)


def _read_quoted(text: str, i: int) -> tuple[str, int]:
    """Read a ``"..."`` value (braces may protect inner quotes)."""
    depth = 0
    j = i + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
        elif ch == '"' and depth == 0:
            return text[i + 1 : j], j + 1
        j += 1
    return text[i + 1 :], len(text)


def _parse_fields(body: str, macros: dict | None = None) -> dict:
    """Parse BibTeX ``name = value`` pairs (values may be concatenated with #)."""
    macros = macros if macros is not None else dict(_MONTHS)
    fields: dict = {}
    i, n = 0, len(body)
    while i < n:
        while i < n and (body[i].isspace() or body[i] == ","):
            i += 1
        m = _FIELD_NAME_RE.match(body, i)
        if not m:
            nxt = body.find(",", i + 1)
            if nxt < 0:
                break
            i = nxt
            continue
        name = m.group().lower()
        i = m.end()
        while i < n and body[i].isspace():
            i += 1
        if i >= n or body[i] != "=":
            nxt = body.find(",", i)
            if nxt < 0:
                break
            i = nxt
            continue
        i += 1
        parts = []
        while True:
            while i < n and body[i].isspace():
                i += 1
            if i >= n:
                break
            ch = body[i]
            if ch == "{":
                value, i = _read_braced(body, i)
                parts.append(value)
            elif ch == '"':
                value, i = _read_quoted(body, i)
                parts.append(value)
            else:
                vm = _BARE_VALUE_RE.match(body, i)
                if not vm:
                    break
                token = vm.group()
                i = vm.end()
                parts.append(token if token.isdigit() else macros.get(token.lower(), token))
            while i < n and body[i].isspace():
                i += 1
            if i < n and body[i] == "#":
                i += 1
                continue
            break
        fields[name] = "".join(parts).strip()
    return fields


def _extract_entries(content: str) -> list:
    """Extract individual BibTeX entries from content."""
    entries = []
    macros = dict(_MONTHS)
    pos = 0
    while True:
        m = _ENTRY_START_RE.search(content, pos)
        if not m:
            break
        entry_type = m.group(1).lower()
        start = m.end()
        end = _find_closing(content, start, m.group(2))
        body = content[start:] if end is None else content[start:end]
        pos = len(content) if end is None else end + 1

        if entry_type in ("comment", "preamble"):
            continue
        if entry_type == "string":
            macros.update({k.lower(): v for k, v in _parse_fields(body, macros).items()})
            continue

        cite_key, sep, rest = body.partition(",")
        if not sep or "=" in cite_key:
            cite_key, rest = "", body if "=" in body else ""
        fields = _parse_fields(rest, macros)
        fields["_type"] = entry_type
        fields["_key"] = cite_key.strip()
        entries.append(fields)

    return entries


# ── LaTeX → text ─────────────────────────────────────────────────────────────


def _combine(letter: str, accent: str) -> str:
    if letter in ("\\i", "i"):
        letter = "i"
    elif letter in ("\\j", "j"):
        letter = "j"
    return unicodedata.normalize("NFC", letter + chr(_ACCENTS[accent]))


def _latex_to_text(value: str) -> str:
    """Convert LaTeX markup in a BibTeX value to plain Unicode text."""
    if not value:
        return ""
    text = value
    # Formatting commands: keep their argument.
    fmt = re.compile(
        r"\\(?:textit|textbf|textrm|textsf|textsc|texttt|textup|emph|mathrm|mathit|"
        r"mbox|text|url|href\{[^{}]*\})\s*\{([^{}]*)\}"
    )
    for _ in range(5):
        new = fmt.sub(r"\1", text)
        if new == text:
            break
        text = new
    # Symbol accents: \'e, \'{e}, {\'e}, \"{\i}
    text = re.sub(
        r"\\([`'^\"~=.])\s*(?:\{\s*(\\[ij]|[A-Za-z])\s*\}|(\\[ij](?![A-Za-z])|[A-Za-z]))",
        lambda m: _combine(m.group(2) or m.group(3), m.group(1)),
        text,
    )
    # Letter accents need a brace or a space: \v{c}, \c c, \H{o}
    text = re.sub(
        r"\\([uvHckrdb])(?:\s*\{\s*(\\[ij]|[A-Za-z])\s*\}|\s+([A-Za-z]))",
        lambda m: _combine(m.group(2) or m.group(3), m.group(1)),
        text,
    )
    for name in sorted(_SPECIAL_LETTERS, key=len, reverse=True):
        text = re.sub(rf"\\{name}(?![A-Za-z])\s*", chr(_SPECIAL_LETTERS[name]), text)
    for escaped, plain in (
        ("\\&", "&"),
        ("\\%", "%"),
        ("\\$", "$"),
        ("\\#", "#"),
        ("\\_", "_"),
        ("\\{", "\x00"),
        ("\\}", "\x01"),
    ):
        text = text.replace(escaped, plain)
    text = text.replace("---", _EM_DASH).replace("--", _EN_DASH)
    text = re.sub(r"(?<!\\)~", " ", text)
    text = re.sub(r"\\[,;: ]", " ", text)
    text = text.replace("$", "")
    text = re.sub(r"\\([A-Za-z]+)\s*", r"\1", text)  # unknown commands: keep the name
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\x00", "{").replace("\x01", "}")
    return re.sub(r"\s+", " ", text).strip()


# ── Names ────────────────────────────────────────────────────────────────────


def _split_top(text: str, separator: str) -> list[str]:
    """Split *text* on *separator* (a regex) at brace depth 0."""
    parts, depth, last = [], 0, 0
    i = 0
    pattern = re.compile(separator, re.IGNORECASE)
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
        elif depth == 0:
            m = pattern.match(text, i)
            if m and m.end() > i:
                parts.append(text[last:i])
                last = i = m.end()
                continue
        i += 1
    parts.append(text[last:])
    return [p.strip() for p in parts]


def _initials(first: str) -> str:
    """'Juan Manuel' -> 'J.M.', 'Lee-Sun' -> 'L.-S.', 'Jack P.' -> 'J.P.'"""
    out = []
    for word in first.split():
        pieces = []
        for piece in word.split("-"):
            letters = [c for c in piece if c.isalpha()]
            if not letters:
                continue
            if re.fullmatch(r"(?:[^\W\d_]\.)+", piece):  # already initials: "J.M."
                pieces.append(piece.upper())
            else:
                pieces.append(letters[0].upper() + ".")
        if pieces:
            out.append("-".join(pieces))
    return "".join(out)


def _format_name(raw: str) -> str | None:
    """Format one BibTeX name as 'F.M. von Last Jr.'; ``None`` for 'others'."""
    name = raw.strip().strip(",")
    if not name:
        return ""
    if name.lower() == "others":
        return None
    if name.startswith("{") and name.endswith("}"):
        inner, end = _read_braced(name, 0)
        if end == len(name):  # whole name protected: corporate author
            return _latex_to_text(inner)

    parts = _split_top(name, r",")
    jr = ""
    if len(parts) == 1:
        words = _split_top(name, r"\s+")
        words = [w for w in words if w]
        if len(words) == 1:
            return _latex_to_text(words[0])
        last = words[-1]
        von_start = next(
            (k for k in range(1, len(words) - 1) if _latex_to_text(words[k])[:1].islower()),
            len(words) - 1,
        )
        first = " ".join(words[:von_start])
        family = " ".join(words[von_start:-1] + [last])
    else:
        family = parts[0]
        if len(parts) >= 3:
            jr, first = parts[1], ", ".join(parts[2:])
        else:
            first = parts[1]
    family = _latex_to_text(family)
    initials = _initials(_latex_to_text(first))
    formatted = f"{initials} {family}" if initials else family
    if jr:
        formatted += f" {_latex_to_text(jr)}"
    return formatted.strip()


def _format_authors(author_str: str) -> str:
    """Format BibTeX author string (Last, First and Last, First → F. Last, F. Last)."""
    formatted = []
    et_al = False
    for raw in _split_top(author_str or "", r"\s+and\s+"):
        name = _format_name(raw)
        if name is None:
            et_al = True
        elif name:
            formatted.append(name)
    text = ", ".join(formatted)
    if et_al:
        text = f"{text}, et al." if text else "et al."
    return text


# ── Formatting ───────────────────────────────────────────────────────────────


def _clean(entry: dict, *keys: str) -> str:
    for key in keys:
        value = entry.get(key)
        if value:
            return _latex_to_text(value)
    return ""


def _year(entry: dict) -> str:
    year = _clean(entry, "year")
    if not year:
        m = re.match(r"\s*(\d{4})", entry.get("date", "") or "")
        year = m.group(1) if m else ""
    return year


def _pages(entry: dict) -> str:
    pages = _clean(entry, "pages", "eid")
    return re.sub(rf"\s*[{_EN_DASH}{_EM_DASH}-]+\s*", "-", pages)


def _doi_url(entry: dict) -> str:
    doi = _clean(entry, "doi")
    if doi:
        doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.IGNORECASE)
        return f"https://doi.org/{doi.strip()}"
    return _clean(entry, "url")


def _format_entry(entry: dict, num: int) -> str:
    """Format a parsed BibTeX entry as a numbered reference string."""
    parts = [f"[{num}]"]
    entry_type = entry.get("_type", "")

    author = entry.get("author", "")
    if author:
        parts.append(_format_authors(author) + ",")
    elif entry.get("editor"):
        editors = _format_authors(entry["editor"])
        label = "(Eds.)" if "," in editors else "(Ed.)"
        parts.append(f"{editors} {label},")

    title = _clean(entry, "title")
    if title:
        parts.append(title.rstrip(".") + ",")

    year = _year(entry)
    pages = _pages(entry)
    journal = _clean(entry, "journal", "journaltitle")
    booktitle = _clean(entry, "booktitle")

    if journal:
        j_str = journal
        vol = _clean(entry, "volume")
        issue = _clean(entry, "number", "issue")
        if vol:
            j_str += f" {vol}"
        if issue:
            j_str += f" ({issue})"
        if year:
            j_str += f" ({year})"
        if pages:
            j_str += f" {pages}"
        parts.append(j_str + ".")
    elif booktitle:
        pieces = [f"in: {booktitle}"]
        pieces += [
            p for p in (_clean(entry, "publisher", "organization"), _clean(entry, "address"))
        ]
        pieces.append(year)
        if pages:
            pieces.append(f"pp. {pages}")
        parts.append(", ".join(p for p in pieces if p) + ".")
    else:
        pieces = []
        edition = _clean(entry, "edition")
        if edition:
            pieces.append(edition if edition.lower().endswith("ed.") else f"{edition} ed.")
        if entry_type in ("phdthesis", "mastersthesis"):
            pieces.append("PhD Thesis" if entry_type == "phdthesis" else "Master's Thesis")
        elif entry_type == "techreport" and _clean(entry, "number"):
            pieces.append(f"Report {_clean(entry, 'number')}")
        pieces.append(
            _clean(entry, "publisher", "institution", "school", "organization", "howpublished")
        )
        pieces.append(_clean(entry, "address"))
        pieces.append(year)
        pieces = [p for p in pieces if p]
        if pieces:
            parts.append(", ".join(pieces) + ".")

    link = _doi_url(entry)
    if link:
        parts.append(link)

    if len(parts) <= 1:
        note = _clean(entry, "note")
        if note:
            parts.append(note)

    if len(parts) <= 1:
        return ""
    text = " ".join(parts)
    return re.sub(r",(\s+https?://|$)", r".\1", text)
