"""Convert references between citation styles (numbered, author-date, APA).

The converter parses each reference string into fields (authors, year, title,
journal, volume, issue, pages, DOI, and any trailing publisher information)
and renders them in the target style.  It understands the reference layouts
used by the bundled journal guidelines:

* Elsevier numbered -- ``[1] F. Bre, J.M. Gimenez, Title, Energy Build. 158 (2018) 1429-1441.``
* author-date / APA / ASCE / KSCE -- ``Bre, F. (2018). Title. Energy Build., 158, 1429-1441.``,
  ``Beskos, D. E. 1987. "Title." Appl. Mech. Rev., 40 (1), 1-23.``
* ACI -- ``DiAmato, S., and Jenner, W., "Title," ACI Structural Journal, V. 103, No. 1, ...``
* Wiley / MDPI sentence style --
  ``Boore DM, Atkinson GM. Title. Earthquake Spectra 2008; 24(1):99-138.``

Conversion never drops information silently: a reference that cannot be
parsed confidently is passed through unchanged (only its list number is
replaced), so round-trips such as numbered -> author_date -> numbered are
lossless for well-formed input.
"""

from __future__ import annotations

import re

from utils.reference_utils import (
    _DOI_SPAN_RE,
    _URL_RE,
    extract_doi,
    extract_year,
    normalize_references,
    strip_numbering,
)

_STYLE_ALIASES = {
    "numbered": "numbered",
    "elsevier": "numbered",
    "vancouver": "numbered",
    "author_date": "author_date",
    "author-date": "author_date",
    "authordate": "author_date",
    "harvard": "author_date",
    "apa": "apa",
}

_PARTICLES = {
    "van",
    "von",
    "der",
    "den",
    "de",
    "del",
    "della",
    "di",
    "da",
    "du",
    "la",
    "le",
    "dos",
    "das",
    "ter",
    "ten",
    "bin",
    "ibn",
    "al",
    "el",
    "y",
    "zu",
}
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}
_ET_AL = {"et al", "et al.", "et. al.", "et.al.", "others"}
_JOURNAL_STOPWORDS = {"of", "and", "in", "the", "for", "on", "&", "de", "und", "a", "an", "to"}
_HANGUL_FIRST, _HANGUL_LAST = chr(0xAC00), chr(0xD7A3)
_OPEN_QUOTES = '"\u201c\u00ab'
_CLOSE_QUOTES = '"\u201d\u00bb'


def convert_style(references: list, target_style: str) -> list:
    """Convert a list of reference strings to the target citation style.

    Parameters
    ----------
    references : list[str]
        List of reference strings in any format (a newline-separated string
        is also accepted; blank entries are dropped).
    target_style : str
        "numbered" — [1] Author, Title, Journal Vol (Year) Pages.
        "author_date" — Author (Year). Title. Journal, Vol, Pages.
        "apa" — Author, A. B. (Year). Title. Journal, Vol(Issue), Pages. DOI
        ("author-date" is accepted as an alias of "author_date").

    Returns
    -------
    list[str]
        Converted references (same length as the cleaned input).
    """
    style = _STYLE_ALIASES.get(str(target_style).strip().lower())
    converters = {
        "numbered": _to_numbered,
        "author_date": _to_author_date,
        "apa": _to_apa,
    }
    if style is None:
        raise ValueError(f"Unknown style '{target_style}'. Choose: {list(converters.keys())}")

    parsed = [_parse_ref(ref) for ref in normalize_references(references)]
    return [converters[style](p, i + 1) for i, p in enumerate(parsed)]


def first_author_family(reference: str) -> str:
    """Return the first author's surname (for alphabetical sorting), or ''."""
    fields = _parse_ref(reference)
    authors = fields.get("authors_list") or []
    if authors:
        first = authors[0]
        return first.get("family") or first.get("literal") or ""
    return fields.get("clean", "")


# ── Name parsing ─────────────────────────────────────────────────────────────


def _is_hangul(text: str) -> bool:
    return any(_HANGUL_FIRST <= ch <= _HANGUL_LAST for ch in text)


def _is_initials(token: str) -> bool:
    """'F.', 'J.M.', 'J. M.', 'L.-S.', 'Th.'"""
    compact = token.replace(" ", "")
    groups = re.findall(r"([^\W\d_])([a-z]?)\.(-?)", compact)
    if not groups or "".join(a + b + "." + c for a, b, c in groups) != compact:
        return False
    return all(a.isupper() for a, _b, _c in groups)


def _compact_initials(token: str) -> str:
    return token.replace(" ", "")


def _is_surname(token: str) -> bool:
    words = token.split()
    if not 1 <= len(words) <= 4:
        return False
    capitalised = 0
    for word in words:
        if word.lower() in _PARTICLES:
            continue
        if not re.fullmatch(r"[^\W\d_][\w'\u2019.\-]*", word) or "_" in word:
            return False
        if not word[0].isupper():
            return False
        if word.endswith(".") and len(word) <= 3:  # an initial, not a surname
            return False
        capitalised += 1
    return capitalised >= 1


def _initials_first(token: str) -> dict | None:
    """'J.M. Gimenez' -> {'given': 'J.M.', 'family': 'Gimenez'}."""
    m = re.match(r"^((?:[^\W\d_][a-z]?\.\s?-?\s?)+)\s*(.+)$", token)
    if not m:
        return None
    given, family = m.group(1).strip(), m.group(2).strip()
    if _is_initials(given) and _is_surname(family):
        return {"given": _compact_initials(given), "family": family}
    return None


def _wiley_name(token: str) -> dict | None:
    """'Boore DM' -> {'given': 'D.M.', 'family': 'Boore'} (Vancouver/Wiley)."""
    m = re.match(r"^(.+?)\s+([A-Z]{1,4})$", token)
    if not m or not _is_surname(m.group(1)):
        return None
    given = "".join(f"{c}." for c in m.group(2))
    return {"given": given, "family": m.group(1), "_form": "wiley"}


def _parse_authors(block: str):
    """Parse an author block into ``(authors, et_al)`` or return ``None``.

    Handles initials-first ("F. Bre, J.M. Gimenez"), surname-first
    ("Bre, F., & Gimenez, J. M."), mixed ASCE ("Evans, G. M., and J. C.
    Furlong"), MDPI ("Kim, S.; Lee, J."), Wiley ("Boore DM, Atkinson GM"),
    "et al." and Korean names.
    """
    text = block.strip().strip(",;").strip()
    if not text or any(q in text for q in _OPEN_QUOTES + _CLOSE_QUOTES):
        return None
    last_word = re.split(r"[\s,;]+", text)[-1]
    if text.endswith(".") and not _is_initials(last_word):
        if last_word.lower().rstrip(".") not in {"al", "jr", "sr"}:
            text = text[:-1].rstrip()  # sentence period after the last surname
    text = re.sub(r"\s*,?\s*(?:&|\band\b)\s+", ", ", text)
    tokens = [t.strip() for t in re.split(r"[;,]", text) if t.strip()]
    authors: list[dict] = []
    et_al = False
    pending = None
    for tok in tokens:
        low = tok.lower()
        if low in _ET_AL:
            if pending is not None:
                return None
            et_al = True
            continue
        if low in _SUFFIXES and authors and pending is None:
            authors[-1]["suffix"] = tok
            continue
        if pending is not None:
            if _is_initials(tok):
                authors.append({"given": _compact_initials(tok), "family": pending})
                pending = None
                continue
            return None
        if _is_initials(tok):
            return None
        name = _initials_first(tok) or _wiley_name(tok)
        if name:
            authors.append(name)
        elif _is_hangul(tok) and len(tok.split()) <= 2 and len(tok) <= 12:
            authors.append({"literal": tok})
        elif _is_surname(tok):
            pending = tok
        else:
            return None
    if pending is not None:
        if authors or len(tokens) != 1:
            return None
        authors.append({"literal": pending})
    if not authors:
        return None
    return authors, et_al


def _corporate_author(block: str):
    """Accept a short corporate author ("ASCE", "ACI Committee 318") as a literal."""
    text = block.strip().strip(",;.").strip()
    if not text or "," in text or any(q in text for q in _OPEN_QUOTES + _CLOSE_QUOTES):
        return None
    if len(text.split()) > 8 or not text[0].isupper():
        return None
    return [{"literal": text}], False


def _fmt_name(author: dict, order: str) -> str:
    if "literal" in author:
        return author["literal"]
    given, family = author.get("given", ""), author["family"]
    suffix = f" {author['suffix']}" if author.get("suffix") else ""
    if order == "initials_first":
        return f"{given} {family}{suffix}".strip()
    if order == "apa":
        spaced = re.sub(r"\.(?=[^\s\-])", ". ", given).strip()
        return f"{family}, {spaced}{suffix}" if spaced else family
    return f"{family}, {given}{suffix}" if given else family


def _fmt_authors(fields: dict, order: str) -> str:
    authors = fields.get("authors_list") or []
    names = [_fmt_name(a, order) for a in authors]
    et_al = fields.get("et_al")
    if order == "apa" and len(names) > 1 and not et_al:
        text = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        text = ", ".join(names)
    if et_al:
        text += ", et al." if names else "et al."
    return text


# ── Reference parsing ────────────────────────────────────────────────────────


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on *sep* outside parentheses, brackets and quotes."""
    parts, depth, quote, buf = [], 0, False, []
    for ch in text:
        if ch in _OPEN_QUOTES and not quote:
            quote = True
        elif ch in _CLOSE_QUOTES and quote:
            quote = False
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0 and not quote:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _journal_like(name: str) -> bool:
    name = name.strip()
    if not name or len(name) > 120 or name.endswith(("?", "!")):
        return False
    if _is_hangul(name):
        return True
    words = name.split()
    if len(words) > 12:
        return False
    for word in words:
        bare = word.strip("().,:;-")
        if not bare or bare.lower() in _JOURNAL_STOPWORDS or not bare[0].isalpha():
            continue
        if not bare[0].isupper():
            return False
    return name[0].isupper() or name[0].isdigit()


def _parse_tail(tail: str, year: str | None) -> dict | None:
    """Parse "Journal, Vol(Issue), Pages" style tails (author-date, ACI, KSCE,
    Wiley "Journal 2008; 24(1):99-138" and MDPI "Journal 2024, 14, 1523")."""
    tail = tail.strip().strip(",;").strip()
    if not tail:
        return None
    m = re.match(
        r"^(?P<journal>.+?)\s+(?P<year>(?:18|19|20)\d{2})[a-z]?\s*;\s*(?P<vol>\w+)?\s*"
        r"(?:\((?P<issue>[^)]+)\))?\s*(?::\s*(?P<pages>[\w\-\u2013]+))?$",
        tail.rstrip("."),
    )
    if m and _journal_like(m.group("journal")):
        out = {"journal": m.group("journal").strip()}
        for key, group in (("volume", "vol"), ("issue", "issue"), ("pages", "pages")):
            if m.group(group):
                out[key] = m.group(group)
        out["_year"] = m.group("year")
        return out if ("volume" in out or "pages" in out) else None

    m = re.match(  # Springer: "Energy Build 158(2):1429-1441"
        r"^(?P<journal>.+?)\s+(?P<vol>\d+[A-Za-z]?)(?:\s*\((?P<issue>[^)]+)\))?\s*:\s*"
        r"(?P<pages>[\w\-\u2013]+)$",
        tail.rstrip("."),
    )
    if m and _journal_like(m.group("journal")):
        out = {"journal": m.group("journal").strip(), "volume": m.group("vol")}
        if m.group("issue"):
            out["issue"] = m.group("issue")
        out["pages"] = m.group("pages")
        return out

    tokens = _split_top_level(tail)
    if not tokens:
        return None
    if len(tokens) == 1:  # journal name only, e.g. an online-first article
        name = tokens[0]
        abbreviated = re.search(r"[A-Za-z]{1,12}\.(?:\s|$|-)", name)
        if _journal_like(name.rstrip(".")) and (
            abbreviated or re.search(r"\b(?:Journal|Proceedings|Transactions|Review)\b", name)
        ):
            return {"journal": name}
        return None
    journal = tokens[0]
    found_year = None
    if year and re.search(rf"\s{year}[a-z]?$", journal):  # MDPI: "Buildings 2024"
        journal = re.sub(rf"\s{year}[a-z]?$", "", journal).strip()
        found_year = year
    if not _journal_like(journal):
        return None
    out: dict = {"journal": journal}
    for tok in tokens[1:]:
        tok = tok.strip().rstrip(".")
        if re.fullmatch(r"(?:V|Vol|Volume)\.?\s*(\w+)", tok):
            out["volume"] = re.fullmatch(r"(?:V|Vol|Volume)\.?\s*(\w+)", tok).group(1)
        elif re.fullmatch(r"(?:No|Issue|Iss)\.?\s*(\w+)", tok):
            out["issue"] = re.fullmatch(r"(?:No|Issue|Iss)\.?\s*(\w+)", tok).group(1)
        elif re.fullmatch(r"(?:pp?|Art|Article)\.?\s*([\w\-\u2013]+)", tok):
            out["pages"] = re.fullmatch(r"(?:pp?|Art|Article)\.?\s*([\w\-\u2013]+)", tok).group(1)
        elif re.fullmatch(r"(\d+[A-Za-z]?)\s*\(([^)]+)\)", tok):
            vm = re.fullmatch(r"(\d+[A-Za-z]?)\s*\(([^)]+)\)", tok)
            out["volume"], out["issue"] = vm.group(1), vm.group(2)
        elif re.fullmatch(r"[A-Za-z]?\d+[\-\u2013][A-Za-z]?\d+", tok):
            out["pages"] = tok
        elif re.fullmatch(r"\d+[A-Za-z]?", tok) and "volume" not in out:
            out["volume"] = tok
        elif re.fullmatch(r"e?\d+", tok) and "pages" not in out:
            out["pages"] = tok
        elif year and year[:4] in tok and len(tok) <= 20:  # a date such as "Jan.-Feb. 2006"
            found_year = found_year or year
        else:
            return None
    if "volume" not in out and "pages" not in out:
        return None
    if found_year:
        out["_year"] = found_year
    return out


def _split_title(rest: str, year: str | None) -> tuple[str, dict]:
    """Split "Title. Journal, Vol, Pages" into a title and the parsed tail."""
    rest = rest.strip()
    if rest and rest[0] in _OPEN_QUOTES:
        close = min((i for i in (rest.find(q, 1) for q in _CLOSE_QUOTES) if i > 0), default=-1)
        if close > 0:
            title = rest[1:close].strip().rstrip(",.").strip()
            tail = rest[close + 1 :].strip().lstrip(".,;: ")
            parsed = _parse_tail(tail, year)
            return title, (parsed or ({"rest": tail.rstrip(".")} if tail.rstrip(".") else {}))
    for m in re.finditer(r"[.?!](?=\s+\S)", rest):
        right = rest[m.end() :].strip()
        parsed = _parse_tail(right, year)
        if parsed:
            title = rest[: m.end()].strip()
            return (title[:-1] if title.endswith(".") else title), parsed
    m = re.search(r"\.\s+", rest)
    if m:
        return rest[: m.start()].strip(), {"rest": rest[m.end() :].strip().rstrip(".")}
    if ", " in rest:
        title, _, tail = rest.partition(", ")
        return title.strip(), {"rest": tail.strip().rstrip(".")}
    return rest.rstrip("."), {}


def _parse_author_date(work: str, year: str | None) -> dict | None:
    """Authors (Year). Title. ... -- APA, author_date, ASCE, KSCE, Springer."""
    if not year:
        return None
    y = re.escape(year[:4])
    for m in re.finditer(rf"\(\s*{y}([a-z])?\s*\)|(?<![\w\-]){y}([a-z])?(?![\w\-])", work):
        block = work[: m.start()].strip().rstrip(",")
        after = work[m.end() :]
        parenthesised = m.group(0).startswith("(")
        if after.strip() and not re.match(r"\s*[.,:]", after) and not parenthesised:
            continue
        names = _parse_authors(block) or _corporate_author(block)
        if not names:
            continue
        title, tail = _split_title(after.lstrip(" .,:"), year)
        if not title:
            continue
        out = {"authors": block, "authors_list": names[0], "et_al": names[1], "title": title}
        out.update(tail)
        suffix = m.group(1) or m.group(2)
        if suffix:
            out["year"] = year[:4] + suffix
        out["_layout"] = "author_date"
        return out
    return None


def _parse_quoted(work: str, year: str | None) -> dict | None:
    """Authors, "Title," Journal, V. x, No. y, Date, pp. z. -- ACI style."""
    m = re.search(r"[\"\u201c](.+?)[\"\u201d]", work)
    if not m:
        return None
    block = work[: m.start()].strip().rstrip(",")
    names = _parse_authors(block) or _corporate_author(block)
    if not names:
        return None
    title = m.group(1).strip().rstrip(",.").strip()
    tail = work[m.end() :].strip().lstrip(".,;: ")
    out = {"authors": block, "authors_list": names[0], "et_al": names[1], "title": title}
    tail_fields = _parse_tail(tail, year)
    out.update(tail_fields or ({"rest": tail.rstrip(".")} if tail.rstrip(".") else {}))
    out["_layout"] = "quoted"
    return out


_ELSEVIER_JOURNAL_RE = re.compile(
    r"^(?P<journal>.*?[^\d\s(])\s*(?:\s(?P<vol>\d+[A-Za-z]?))?\s*(?:\((?P<issue>[^()]+)\)\s*)?"
    r"\(\s*(?P<year>(?:18|19|20)\d{2})[a-z]?\s*\)\s*(?P<pages>.*)$"
)


def _parse_elsevier(work: str, year: str | None) -> dict | None:
    """A. Author, B. Author, Title, Journal Vol (Issue) (Year) Pages."""
    tokens = _split_top_level(work)
    authors: list[dict] = []
    et_al = False
    i = 0
    while i < len(tokens):
        parsed = _parse_authors(tokens[i])
        if not parsed or any(
            a.get("_form") == "wiley" or ("literal" in a and not _is_hangul(a["literal"]))
            for a in parsed[0]
        ):
            if tokens[i].lower() in _ET_AL and authors:
                et_al = True
                i += 1
                continue
            break
        authors.extend(parsed[0])
        et_al = et_al or parsed[1]
        i += 1
    if not authors or i >= len(tokens):
        return None
    out: dict = {
        "authors": ", ".join(tokens[:i]),
        "authors_list": authors,
        "et_al": et_al,
        "_layout": "elsevier",
    }
    for j in range(i + 1, len(tokens)):
        m = _ELSEVIER_JOURNAL_RE.match(tokens[j])
        if m and _journal_like(m.group("journal")):
            out["title"] = ", ".join(tokens[i:j])
            out["journal"] = m.group("journal").strip()
            if m.group("vol"):
                out["volume"] = m.group("vol")
            if m.group("issue"):
                out["issue"] = m.group("issue").strip()
            pages = m.group("pages").strip().rstrip(".").strip()
            if pages:
                out["pages"] = pages
            out["_year"] = m.group("year")
            if j + 1 < len(tokens):
                out["extra"] = ", ".join(tokens[j + 1 :]).rstrip(".")
            return out
    end = i + 1
    for k in range(i + 1, len(tokens)):
        if re.match(r"(?:\w+ ed\.|\d+(?:st|nd|rd|th) ed\.|in:|vol\.)", tokens[k], re.IGNORECASE):
            end = k
            break
    out["title"] = ", ".join(tokens[i:end]).rstrip(".")
    if end < len(tokens):
        out["rest"] = ", ".join(tokens[end:]).rstrip(".")
    return out


def _parse_sentence(work: str, year: str | None) -> dict | None:
    """Boore DM, Atkinson GM. Title. Journal 2008; 24(1):99-138. (Wiley/MDPI)"""
    best = None
    for m in re.finditer(r"\.(?=\s+\S)", work):
        block = work[: m.end()]
        names = _parse_authors(block.rstrip("."))
        if not names:
            names = _parse_authors(block)
        if names:
            best = (m.end(), names)
    if not best:
        return None
    end, names = best
    title, tail = _split_title(work[end:].strip(), year)
    if not title:
        return None
    out = {
        "authors": work[:end].strip(),
        "authors_list": names[0],
        "et_al": names[1],
        "title": title,
    }
    out.update(tail)
    out["_layout"] = "sentence"
    return out


def _parse_ref(ref: str) -> dict:
    """Parse a reference string into components (best-effort).

    Always returns ``raw`` and ``clean`` (the reference without its list
    number); ``_parsed`` tells whether authors and title were identified.
    Other keys, when found: ``authors`` (original author text),
    ``authors_list``, ``et_al``, ``year``, ``title``, ``journal``, ``volume``,
    ``issue``, ``pages``, ``doi`` (as an https://doi.org/ URL), ``url``,
    ``rest`` (publisher/report details of non-journal items) and ``extra``.
    """
    raw = "" if ref is None else str(ref)
    clean = strip_numbering(raw)
    fields: dict = {"raw": raw, "clean": clean}

    doi = extract_doi(clean)
    if doi:
        fields["doi"] = "https://doi.org/" + doi
    work = _DOI_SPAN_RE.sub(" ", clean)
    url = _URL_RE.search(work)
    if url:
        fields["url"] = url.group().rstrip(".,;")
    work = _URL_RE.sub(" ", work)
    work = re.sub(r"\b(?:DOI|doi|Available at|Retrieved from)\s*:?\s*$", "", work.strip())
    work = work.replace("*", "")
    work = re.sub(r"\s+", " ", work).strip().strip(",;").strip()

    year_int = extract_year(clean)
    year = str(year_int) if year_int else None
    if year:
        fields["year"] = year

    parsed = None
    for parser in (_parse_author_date, _parse_quoted, _parse_elsevier, _parse_sentence):
        parsed = parser(work, year)
        if parsed and parsed.get("title"):
            break
        parsed = None

    if parsed:
        found_year = parsed.pop("_year", None)
        if found_year and "year" not in fields:
            fields["year"] = found_year
        if "pages" in parsed:
            parsed["pages"] = parsed["pages"].replace("\u2013", "-")
        fields.update(parsed)
        fields["_parsed"] = True
    else:
        fields["_parsed"] = False
    return fields


# ── Rendering ────────────────────────────────────────────────────────────────


def _rest_without_year(fields: dict) -> str:
    rest = fields.get("rest", "")
    year = fields.get("year")
    if not rest or not year:
        return rest
    tokens = [t for t in _split_top_level(rest) if t.strip(" .") != year]
    return ", ".join(tokens)


def _end(text: str) -> str:
    """Terminate a sentence-like field with a period unless already punctuated."""
    text = text.strip()
    return text if text.rstrip("*").endswith((".", "?", "!")) else text + "."


def _link(fields: dict) -> str:
    return fields.get("doi") or fields.get("url") or ""


def _to_numbered(fields: dict, num: int) -> str:
    """Convert to numbered style: [1] Author, Title, Journal Vol (Year) Pages."""
    if not fields.get("_parsed"):
        return f"[{num}] {fields.get('clean') or fields.get('raw', '')}".rstrip()
    parts = [_fmt_authors(fields, "initials_first"), fields["title"]]
    year = fields.get("year")
    if fields.get("journal"):
        j = fields["journal"]
        if fields.get("volume"):
            j += f" {fields['volume']}"
        if fields.get("issue"):
            j += f" ({fields['issue']})"
        if year:
            j += f" ({year})"
        if fields.get("pages"):
            j += f" {fields['pages']}"
        parts.append(j)
    elif fields.get("rest"):
        rest = fields["rest"]
        parts.append(rest if not year or year in rest else f"{rest}, {year}")
    elif year:
        parts.append(year)
    if fields.get("extra"):
        parts.append(fields["extra"])
    text = f"[{num}] " + _end(", ".join(p for p in parts if p))
    link = _link(fields)
    return f"{text} {link}" if link else text


def _to_author_date(fields: dict, num: int) -> str:
    """Convert to author-date style: Author (Year). Title. Journal, Vol, Pages."""
    if not fields.get("_parsed"):
        return fields.get("clean") or fields.get("raw", "")
    parts = [_fmt_authors(fields, "family_first")]
    if fields.get("year"):
        parts[0] += f" ({fields['year']})."
    else:
        parts[0] = _end(parts[0])
    parts.append(_end(fields["title"]))
    if fields.get("journal"):
        j = fields["journal"]
        if fields.get("volume"):
            j += f", {fields['volume']}"
            if fields.get("issue"):
                j += f"({fields['issue']})"
        elif fields.get("issue"):
            j += f", ({fields['issue']})"
        if fields.get("pages"):
            j += f", {fields['pages']}"
        parts.append(_end(j))
    elif _rest_without_year(fields):
        parts.append(_end(_rest_without_year(fields)))
    if fields.get("extra"):
        parts.append(_end(fields["extra"]))
    link = _link(fields)
    if link:
        parts.append(link)
    return " ".join(parts)


def _to_apa(fields: dict, num: int) -> str:
    """Convert to APA style: Author, A. B. (Year). Title. Journal, Vol(Issue), Pages. DOI"""
    if not fields.get("_parsed"):
        return fields.get("clean") or fields.get("raw", "")
    parts = [_fmt_authors(fields, "apa")]
    parts.append(f"({fields['year']})." if fields.get("year") else "(n.d.).")
    parts.append(_end(fields["title"]))
    if fields.get("journal"):
        j = f"*{fields['journal']}*"
        if fields.get("volume"):
            j += f", *{fields['volume']}*"
            if fields.get("issue"):
                j += f"({fields['issue']})"
        if fields.get("pages"):
            j += f", {fields['pages']}"
        parts.append(_end(j))
    elif _rest_without_year(fields):
        parts.append(_end(_rest_without_year(fields)))
    if fields.get("extra"):
        parts.append(_end(fields["extra"]))
    link = _link(fields)
    if link:
        parts.append(link)
    return " ".join(p for p in parts if p)
