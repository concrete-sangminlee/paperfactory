"""Paper analytics: readability, word statistics, section balance, and writing quality metrics."""

import re
import unicodedata
from collections import Counter

# Words are runs of letters/digits (optionally joined by an apostrophe, so that
# "Young's" is one word, not "young" + "s") that contain at least one letter.
_WORD_RE = re.compile(r"\w+(?:['\u2019]\w+)*")

# Candidate sentence ends: terminal punctuation, optional closing quotes or
# brackets, then whitespace or end of text.  A decimal point ("0.95") is never a
# candidate because it is not followed by whitespace.
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s|$)")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")

# Abbreviations whose period never ends a sentence ("e.g. Fig. 3", "vs. RF").
_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "cf",
        "vs",
        "viz",
        "approx",
        "resp",
        "ca",
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
    }
)
# Abbreviations that precede a number or bracket ("Fig. 3", "Eq. (2)", "et al. [4]").
_REFERENCE_ABBREVIATIONS = frozenset(
    {
        "fig",
        "figs",
        "eq",
        "eqs",
        "eqn",
        "eqns",
        "ref",
        "refs",
        "tab",
        "no",
        "nos",
        "sec",
        "secs",
        "ch",
        "vol",
        "pp",
        "al",
    }
)
_REFERENCE_FOLLOWER_RE = re.compile(r"[(\[]|[A-Z]?\d")

_HANGUL_SYLLABLE_RE = re.compile("[\uac00-\ud7a3]")
_VOWELS = "aeiouy"


def analyze_paper(paper_content: dict) -> dict:
    """Comprehensive analysis of paper content.

    Returns dict with: word_stats, readability, section_balance, vocabulary, summary.
    """
    sections = paper_content.get("sections") or []
    abstract = paper_content.get("abstract") or ""
    all_text = _collect_all_text(sections)
    full_text = abstract + "\n\n" + all_text

    words = _tokenize(full_text)
    sentences = _split_sentences(full_text)

    word_stats = _word_statistics(words, sentences)
    readability = _readability_scores(words, sentences)
    section_balance = _section_balance(sections)
    vocabulary = _vocabulary_analysis(words)

    summary_lines = [
        f"Total words: {word_stats['total_words']}",
        f"Total sentences: {word_stats['total_sentences']}",
        f"Avg words/sentence: {word_stats['avg_words_per_sentence']:.1f}",
        f"Readability (Flesch-Kincaid Grade): {readability['flesch_kincaid_grade']:.1f}",
        f"Vocabulary richness (TTR): {vocabulary['type_token_ratio']:.3f}",
        f"Section balance: {'Good' if section_balance['is_balanced'] else 'Unbalanced'}",
    ]

    return {
        "word_stats": word_stats,
        "readability": readability,
        "section_balance": section_balance,
        "vocabulary": vocabulary,
        "summary": "\n".join(summary_lines),
    }


def _iter_sections(sections):
    """Yield section dicts, skipping anything that is not a dict (e.g. None)."""
    for sec in sections or []:
        if isinstance(sec, dict):
            yield sec


def _section_text(section: dict) -> str:
    """Return the text of a section and all nested (sub)subsections."""
    parts = [section.get("content") or ""]
    for sub in _iter_sections(section.get("subsections")):
        parts.append(_section_text(sub))
    return "\n\n".join(p for p in parts if p)


def _collect_all_text(sections: list) -> str:
    """Collect all text from sections and (nested) subsections."""
    return "\n\n".join(t for t in (_section_text(s) for s in _iter_sections(sections)) if t)


def _tokenize(text: str) -> list:
    """Split text into lower-cased words (Unicode-aware; numbers are not words)."""
    return [w for w in _WORD_RE.findall((text or "").lower()) if any(ch.isalpha() for ch in w)]


def split_sentences(text: str) -> list:
    """Split text into sentences.

    Unlike a naive split on ``.``, this keeps decimals ("0.95"), common
    abbreviations ("e.g.", "i.e.", "vs.") and reference abbreviations followed
    by a number or bracket ("Fig. 3", "Eq. (2)", "Bre et al. (2018)") inside
    their sentence.  Blank lines (paragraph breaks) always end a sentence.
    Works for any script that separates sentences with ``. `` (e.g. Korean).
    """
    sentences = []
    for paragraph in _PARAGRAPH_BREAK_RE.split(text or ""):
        start = 0
        for match in _SENTENCE_END_RE.finditer(paragraph):
            if not _is_sentence_boundary(paragraph, match):
                continue
            sentence = paragraph[start : match.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def _is_sentence_boundary(text: str, match: re.Match) -> bool:
    punct = match.group()
    if "!" in punct or "?" in punct:
        return True
    following = text[match.end() :].lstrip()
    if not following:
        return True
    if following[0].islower():
        # "e.g. the", "approx. two": a new sentence does not start in lower case.
        return False
    preceding = text[: match.start()].split()
    if not preceding:
        return True
    token = preceding[-1].lstrip("([{\"'\u201c\u2018").lower()
    if token in _ABBREVIATIONS:
        return False
    if token in _REFERENCE_ABBREVIATIONS and _REFERENCE_FOLLOWER_RE.match(following):
        return False
    return True


def _split_sentences(text: str) -> list:
    """Split text into sentences that contain at least one word."""
    return [s for s in split_sentences(text) if _tokenize(s)]


def _count_syllables(word: str) -> int:
    """Estimate syllable count for a word.

    English words use a vowel-group heuristic with the usual corrections for a
    silent final "e", "-ed" and "-es" (but not "-le", "-ee", "-ted", "-ches"...).
    Each Hangul syllable block counts as one syllable.
    """
    if not word:
        return 0
    hangul = len(_HANGUL_SYLLABLE_RE.findall(word))
    if hangul:
        return hangul
    # Fold accents ("café" -> "cafe") and drop non-letters; a diaeresis marks a
    # vowel that starts a new syllable ("naïve" = na-ïve).
    letters, new_group = [], set()
    for ch in unicodedata.normalize("NFKD", word.lower()):
        if ch == "̈" and letters:
            new_group.add(len(letters) - 1)
        elif "a" <= ch <= "z":
            letters.append(ch)
    word = "".join(letters)
    if len(word) <= 3:
        return 1
    count = 0
    prev_vowel = False
    for i, char in enumerate(word):
        is_vowel = char in _VOWELS
        if is_vowel and (not prev_vowel or i in new_group):
            count += 1
        prev_vowel = is_vowel
    if count > 1:
        if word.endswith("e") and not word.endswith("ee"):
            # Silent final "e" ("structure"), but "-le" after a consonant is
            # syllabic ("table", "simple", "possible").
            if not (word.endswith("le") and word[-3] not in _VOWELS):
                count -= 1
        elif word.endswith("ed") and word[-3] not in _VOWELS + "td":
            # "trained", "used", "compared" (but "tested", "loaded", "studied").
            count -= 1
        elif word.endswith("es") and word[-3] not in _VOWELS + "sxzcgh":
            # "makes", "structures" (but "cases", "boxes", "values", "bridges")
            # and "-les" after a consonant is syllabic ("tables", "samples").
            if not (word.endswith("les") and len(word) > 4 and word[-4] not in _VOWELS):
                count -= 1
    return max(count, 1)


def _is_latin_word(word: str) -> bool:
    return any("a" <= ch <= "z" for ch in unicodedata.normalize("NFKD", word))


def _word_statistics(words: list, sentences: list) -> dict:
    """Compute basic word statistics."""
    n_words = len(words)
    n_sentences = len(sentences)
    word_lengths = [len(w) for w in words]

    return {
        "total_words": n_words,
        "total_sentences": n_sentences,
        "avg_words_per_sentence": n_words / max(n_sentences, 1),
        "avg_word_length": sum(word_lengths) / max(n_words, 1),
        "max_word_length": max(word_lengths) if word_lengths else 0,
        "long_words_pct": sum(1 for w in words if len(w) > 8) / max(n_words, 1) * 100,
    }


def _readability_scores(words: list, sentences: list) -> dict:
    """Compute readability scores (Flesch Reading Ease, Flesch-Kincaid, ARI)."""
    if not words:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "automated_readability_index": 0.0,
            "academic_level": "insufficient text",
        }

    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    n_syllables = sum(_count_syllables(w) for w in words)

    # Flesch Reading Ease
    fre = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)

    # Flesch-Kincaid Grade Level
    fkgl = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59

    # Automated Readability Index
    n_chars = sum(len(w) for w in words)
    ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sentences) - 21.43

    if sum(1 for w in words if _is_latin_word(w)) * 2 < n_words:
        # The formulas are calibrated for English only.
        level = "n/a (non-English text)"
    elif 12 <= fkgl <= 18:
        # Academic target: grade 12-18 is typical for research papers
        level = "appropriate"
    else:
        level = "too simple" if fkgl < 12 else "very complex"

    return {
        "flesch_reading_ease": round(fre, 1),
        "flesch_kincaid_grade": round(fkgl, 1),
        "automated_readability_index": round(ari, 1),
        "academic_level": level,
    }


def _section_balance(sections: list) -> dict:
    """Analyze section word count balance.

    Duplicate or missing headings are disambiguated ("Unknown", "Unknown (2)")
    so that no section's words are silently dropped.
    """
    section_words = {}
    for sec in _iter_sections(sections):
        base = str(sec.get("heading") or "").strip() or "Unknown"
        name, n = base, 2
        while name in section_words:
            name, n = f"{base} ({n})", n + 1
        section_words[name] = len(_tokenize(_section_text(sec)))

    total = sum(section_words.values())
    proportions = {k: v / max(total, 1) for k, v in section_words.items()}

    # Check balance: no section should be > 55% or < 3% of total
    is_balanced = all(0.03 <= p <= 0.55 for p in proportions.values()) if proportions else True

    return {
        "section_words": section_words,
        "section_proportions": {k: round(v, 3) for k, v in proportions.items()},
        "total_body_words": total,
        "is_balanced": is_balanced,
        "largest_section": max(section_words, key=section_words.get) if section_words else "",
        "smallest_section": min(section_words, key=section_words.get) if section_words else "",
    }


def _vocabulary_analysis(words: list) -> dict:
    """Analyze vocabulary diversity and common terms."""
    if not words:
        return {"type_token_ratio": 0, "unique_words": 0, "total_words": 0, "top_words": []}

    unique = set(words)
    freq = Counter(words)

    # Filter out common stopwords for top words
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "which",
        "who",
        "whom",
        "what",
        "where",
        "when",
        "why",
        "how",
        "if",
        "because",
        "although",
        "while",
        "since",
        "until",
        "unless",
        "also",
        "however",
        "therefore",
    }

    # Short ASCII tokens are mostly noise ("rf", "cv"); short Hangul words are not.
    content_words = {
        w: c for w, c in freq.items() if w not in stopwords and (len(w) > 2 or not w.isascii())
    }
    top_words = sorted(content_words.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "type_token_ratio": len(unique) / max(len(words), 1),
        "unique_words": len(unique),
        "total_words": len(words),
        "top_words": [{"word": w, "count": c} for w, c in top_words],
    }
