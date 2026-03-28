"""Paper analytics: readability, word statistics, section balance, and writing quality metrics."""

import re
from collections import Counter


def analyze_paper(paper_content: dict) -> dict:
    """Comprehensive analysis of paper content.

    Returns dict with: word_stats, readability, section_balance, vocabulary, summary.
    """
    sections = paper_content.get("sections", [])
    abstract = paper_content.get("abstract", "")
    all_text = _collect_all_text(sections)
    full_text = abstract + " " + all_text

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


def _collect_all_text(sections: list) -> str:
    """Collect all text from sections and subsections."""
    parts = []
    for sec in sections:
        parts.append(sec.get("content", ""))
        for sub in sec.get("subsections", []):
            parts.append(sub.get("content", ""))
    return " ".join(parts)


def _tokenize(text: str) -> list:
    """Split text into words."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def _split_sentences(text: str) -> list:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _count_syllables(word: str) -> int:
    """Estimate syllable count for English word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _word_statistics(words: list, sentences: list) -> dict:
    """Compute basic word statistics."""
    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    word_lengths = [len(w) for w in words]

    return {
        "total_words": n_words,
        "total_sentences": n_sentences,
        "avg_words_per_sentence": n_words / n_sentences,
        "avg_word_length": sum(word_lengths) / max(n_words, 1),
        "max_word_length": max(word_lengths) if word_lengths else 0,
        "long_words_pct": sum(1 for w in words if len(w) > 8) / max(n_words, 1) * 100,
    }


def _readability_scores(words: list, sentences: list) -> dict:
    """Compute readability scores."""
    n_words = max(len(words), 1)
    n_sentences = max(len(sentences), 1)
    n_syllables = sum(_count_syllables(w) for w in words)

    # Flesch Reading Ease
    fre = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)

    # Flesch-Kincaid Grade Level
    fkgl = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59

    # Automated Readability Index
    n_chars = sum(len(w) for w in words)
    ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sentences) - 21.43

    # Academic target: grade 12-16 is typical for research papers
    level = "appropriate" if 12 <= fkgl <= 18 else "too simple" if fkgl < 12 else "very complex"

    return {
        "flesch_reading_ease": round(fre, 1),
        "flesch_kincaid_grade": round(fkgl, 1),
        "automated_readability_index": round(ari, 1),
        "academic_level": level,
    }


def _section_balance(sections: list) -> dict:
    """Analyze section word count balance."""
    section_words = {}
    for sec in sections:
        name = sec.get("heading", "Unknown")
        count = len(_tokenize(sec.get("content", "")))
        for sub in sec.get("subsections", []):
            count += len(_tokenize(sub.get("content", "")))
        section_words[name] = count

    total = max(sum(section_words.values()), 1)
    proportions = {k: v / total for k, v in section_words.items()}

    # Check balance: no section should be > 50% or < 5% of total
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
        return {"type_token_ratio": 0, "unique_words": 0, "top_words": []}

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

    content_words = {w: c for w, c in freq.items() if w not in stopwords and len(w) > 2}
    top_words = sorted(content_words.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "type_token_ratio": len(unique) / max(len(words), 1),
        "unique_words": len(unique),
        "total_words": len(words),
        "top_words": [{"word": w, "count": c} for w, c in top_words],
    }
