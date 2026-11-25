"""Regression tests for bugs fixed in utils/latex_generator.py.

No TeX distribution is assumed, so the tests check the generated source for
constructs that are known to break or mis-render under pdfLaTeX/BibTeX, plus
structural validity (balanced braces and environments).
"""

import os
import re

import pytest

from utils import latex_generator, word_generator
from utils.guidelines import UnknownJournalError, available_journal_keys
from utils.latex_generator import (
    _bibtex_authors,
    _escape_latex,
    _get_document_class,
    _parse_reference,
    _strip_caption_label,
    generate_latex,
)


@pytest.fixture(autouse=True)
def _no_writes_to_outputs(tmp_path, monkeypatch):
    """Redirect the default output directory so no test can write into outputs/."""
    for module in (latex_generator, word_generator):
        monkeypatch.setattr(module, "DEFAULT_PAPERS_DIR", str(tmp_path / "default_papers"))


BASE = {
    "title": "Regression Paper",
    "authors": "A. Author",
    "abstract": "Abstract text.",
    "keywords": "a; b",
    "sections": [{"heading": "Introduction", "content": "Body text."}],
    "references": ["[1] A. Author, Title of paper, J. Eng. 50 (2024) 1-10."],
}


def _gen(tmp_path, content=None, journal="eng_structures", **kw):
    tex, bib = generate_latex(content or BASE, journal, output_dir=str(tmp_path), **kw)
    with open(tex, encoding="utf-8") as f:
        tex_src = f.read()
    with open(bib, encoding="utf-8") as f:
        bib_src = f.read()
    return tex, bib, tex_src, bib_src


def assert_tex_balanced(src):
    """Braces balance (ignoring escaped ones) and \\begin/\\end environments nest."""
    depth, i = 0, 0
    while i < len(src):
        ch = src[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "%":
            i = src.find("\n", i)
            if i < 0:
                break
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            assert depth >= 0, f"unbalanced '}}' near: {src[max(0, i - 40) : i + 1]!r}"
        i += 1
    assert depth == 0
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", src):
        if kind == "begin":
            stack.append(env)
        else:
            assert stack and stack.pop() == env, f"mismatched \\end{{{env}}}"
    assert not stack


def assert_bib_balanced(src):
    """BibTeX counts every brace, escaped or not."""
    for entry in re.split(r"\n(?=@)", src.strip()):
        assert entry.count("{") == entry.count("}"), entry


# ---------------------------------------------------------------------------
# Journal handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["struct_safety", "const_build_mat", "steel_comp_struct", "jcsr"])
def test_all_elsevier_journals_use_elsarticle(key):
    # These used to fall back to the plain article class.
    assert _get_document_class(key) == "elsarticle"


def test_unknown_journal_raises(tmp_path):
    with pytest.raises(UnknownJournalError):
        generate_latex(BASE, "no_such_journal", output_dir=str(tmp_path))
    assert _get_document_class("no_such_journal") == "article"


@pytest.mark.parametrize("key", ["aci_sj", "eesd", "ksce_jce", "buildings_mdpi"])
def test_numbered_journals_keep_citation_order(tmp_path, key):
    # "plain" sorts alphabetically, breaking "[n] in order of appearance".
    _, _, tex, _ = _gen(tmp_path, journal=key)
    assert r"\bibliographystyle{unsrt}" in tex


def test_us_letter_journal_uses_letterpaper_and_guideline_spacing(tmp_path):
    _, _, tex, _ = _gen(tmp_path, journal="aci_sj")
    assert r"\documentclass[12pt,letterpaper]{article}" in tex
    assert r"\doublespacing" in tex and r"\usepackage[margin=1.00in]{geometry}" in tex
    _, _, mdpi, _ = _gen(tmp_path / "m", journal="buildings_mdpi")
    assert r"\documentclass[10pt,a4paper]{article}" in mdpi and r"\doublespacing" not in mdpi


def test_line_numbers_follow_guideline(tmp_path):
    _, _, asce, _ = _gen(tmp_path, journal="asce_jse")
    assert r"\linenumbers" in asce  # ASCE requires line numbers (was elsarticle-only)
    _, _, ksce, _ = _gen(tmp_path / "k", journal="ksce_jce")
    assert r"\linenumbers" not in ksce


def test_unnumbered_journals_disable_section_numbers(tmp_path):
    _, _, asce, _ = _gen(tmp_path, journal="asce_jse")
    assert r"\setcounter{secnumdepth}{0}" in asce
    _, _, els, _ = _gen(tmp_path / "e")
    assert "secnumdepth" not in els


def test_manual_heading_numbers_are_stripped(tmp_path):
    content = {**BASE, "sections": [{"heading": "1. Introduction", "content": "x"}]}
    _, _, tex, _ = _gen(tmp_path, content)
    assert r"\section{Introduction}" in tex


def test_back_matter_headings_are_unnumbered(tmp_path):
    content = {**BASE, "sections": [{"heading": "Acknowledgements", "content": "x"}]}
    _, _, tex, _ = _gen(tmp_path, content)
    assert r"\section*{Acknowledgements}" in tex


# ---------------------------------------------------------------------------
# Content that used to be silently dropped
# ---------------------------------------------------------------------------


RICH = {
    **BASE,
    "highlights": ["HL-ONE"],
    "practical_applications": "PRACTICAL-TEXT",
    "data_availability": "DA-TEXT",
    "acknowledgments": "ACK-TEXT",
    "notation": [{"symbol": "$f_c$", "definition": "NOTATION-DEF"}],
    "tables": [{"caption": "Table 1. TABLE-CAP.", "headers": ["A", "B"], "rows": [["1", "2"]]}],
    "sections": [
        {
            "heading": "Intro",
            "content": "x",
            "subsections": [
                {"heading": "Sub", "subsections": [{"heading": "SUBSUB-H", "content": "z"}]}
            ],
        }
    ],
}


def test_elsevier_paper_keeps_all_content(tmp_path):
    _, _, tex, _ = _gen(tmp_path, RICH)
    assert "\\begin{highlights}\n\\item HL-ONE\n\\end{highlights}" in tex
    assert "\\section*{Data Availability Statement}\nDA-TEXT" in tex
    assert "\\section*{Acknowledgements}\nACK-TEXT" in tex
    assert r"\subsubsection{SUBSUB-H}" in tex
    assert "$f_c$ & NOTATION-DEF" in tex
    assert "PRACTICAL-TEXT" not in tex  # ASCE-only block
    assert "\\caption{TABLE-CAP.}" in tex and "A & B \\\\" in tex and r"\toprule" in tex
    assert_tex_balanced(tex)


def test_asce_paper_keeps_practical_applications_but_not_highlights(tmp_path):
    _, _, tex, _ = _gen(tmp_path, RICH, journal="asce_jse")
    assert "\\section*{Practical Applications}\nPRACTICAL-TEXT" in tex
    assert "HL-ONE" not in tex
    assert tex.index("Data Availability") < tex.index("Acknowledgments") < tex.index("Notation")
    assert_tex_balanced(tex)


def test_references_are_cited_so_bibtex_prints_them(tmp_path):
    _, bib, tex, _ = _gen(tmp_path)
    stem = os.path.splitext(os.path.basename(bib))[0]
    assert f"\\nocite{{*}}\n\\bibliography{{{stem}}}" in tex


def test_no_bibliography_without_references(tmp_path):
    # An empty thebibliography environment is a LaTeX error.
    _, bib, tex, bib_src = _gen(tmp_path, {**BASE, "references": []})
    assert r"\bibliography" not in tex and r"\bibliographystyle" not in tex
    assert os.path.exists(bib) and bib_src == ""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_figure_caption_is_not_double_numbered(tmp_path):
    content = {**BASE, "figure_captions": ["Fig. 1. Model comparison."]}
    _, _, tex, _ = _gen(tmp_path, content, figures=["fig_1.png"])
    assert r"\caption{Model comparison.}" in tex


def test_default_figure_caption_is_derived_from_file_name(tmp_path):
    _, _, tex, _ = _gen(tmp_path, figures=["fig_2_load_path.png"])
    assert r"\caption{Load path}" in tex and r"\caption{Figure 2}" not in tex


def test_figures_in_other_directory_are_found_via_graphicspath(tmp_path):
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    fig = fig_dir / "fig_1.png"
    fig.write_bytes(b"")
    _, _, tex, _ = _gen(tmp_path / "papers", figures=[str(fig)])
    assert r"\graphicspath{{../figures/}}" in tex
    assert "keepaspectratio]{fig_1.png}" in tex


def test_surplus_captions_and_none_figures_are_kept(tmp_path):
    content = {**BASE, "figure_captions": ["Fig. 1. A.", "Fig. 2. B."]}
    _, _, tex, _ = _gen(tmp_path, content, figures=[None])
    assert r"\caption{A.}" in tex and r"\caption{B.}" in tex
    assert r"\includegraphics" not in tex
    assert_tex_balanced(tex)


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("p < 0.05", r"p \textless{} 0.05"),
        ("a > b | c", r"a \textgreater{} b \textbar{} c"),
        ("$$E = mc^2$$", r"\[E = mc^2\]"),
        ("σ ≤ 0.5 μm", r"\ensuremath{\sigma} \ensuremath{\leq} 0.5 \ensuremath{\mu}m"),
        ("Δ=10⁻³ and x₁", r"\ensuremath{\Delta}=10\textsuperscript{-}³ and x\textsubscript{1}"),
        ("R² = 0.99 ± 0.01 °C", "R² = 0.99 ± 0.01 °C"),
        ("keep $\\sigma_y$ math", "keep $\\sigma_y$ math"),
        (None, ""),
        (42, "42"),
    ],
)
def test_escape_latex(text, expected):
    assert _escape_latex(text) == expected


def test_korean_paper_loads_kotex(tmp_path):
    content = {**BASE, "title": "철근콘크리트 보의 전단강도", "abstract": "한국어 초록"}
    tex_path, _, tex, _ = _gen(tmp_path, content, journal="ksce_jce")
    assert r"\usepackage{kotex}" in tex
    assert "철근콘크리트 보의 전단강도" in tex
    # The title stays Korean inside the document, but the .tex/.bib names must be ASCII:
    # BibTeX cannot open a \bibdata name made of kotex \unihangulchar macros (found by
    # compiling with TeX Live in CI).
    assert os.path.basename(tex_path) == "paper_ksce_jce.tex"
    _, _, en, _ = _gen(tmp_path / "en", journal="ksce_jce")
    assert "kotex" not in en


def test_blank_lines_in_single_line_contexts_are_collapsed(tmp_path):
    # A paragraph break inside \caption, \section or a tabular cell is a LaTeX error.
    content = {
        **BASE,
        "sections": [{"heading": "Intro\n\nduction", "content": "x"}],
        "tables": [{"caption": "Cap\n\ntion", "headers": ["A\n\nB"], "rows": [["1\n\n2"]]}],
    }
    _, _, tex, _ = _gen(tmp_path, content)
    assert r"\section{Intro duction}" in tex
    assert r"\caption{Cap tion}" in tex
    assert "A B \\\\" in tex and "1 2 \\\\" in tex


# ---------------------------------------------------------------------------
# Robustness of inputs and output paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"keywords": ["a", "b"]},
        {"authors": ["A. Author", "B. Author"]},
        {"title": None},
        {"sections": None},
        {"sections": [{"heading": "H", "subsections": None}]},
        {"references": ["[1] ok", None]},
        {"references": "[1] first\n\n[2] second"},
        {"tables": [{"caption": "T", "headers": [], "rows": []}]},
    ],
)
def test_plausible_inputs_do_not_crash(tmp_path, override):
    _, _, tex, bib = _gen(tmp_path, {**BASE, **override})
    assert_tex_balanced(tex)
    assert_bib_balanced(bib)


def test_keyword_list_is_split_into_elsarticle_keywords(tmp_path):
    _, _, tex, _ = _gen(tmp_path, {**BASE, "keywords": ["wind", "ML_model"]})
    assert "wind \\sep ML\\_model" in tex


@pytest.mark.parametrize("title", ["???", "", None])
def test_empty_title_does_not_create_hidden_bib_file(tmp_path, title):
    tex, bib, tex_src, _ = _gen(tmp_path, {**BASE, "title": title})
    assert os.path.basename(bib) == "paper_eng_structures.bib"
    assert os.path.basename(tex) == "paper_eng_structures.tex"
    assert r"\bibliography{paper_eng_structures}" in tex_src


def test_each_journal_export_has_its_own_bib(tmp_path):
    # The .bib used to be named after the title only, so exporting the same paper
    # for a second journal overwrote the first export's bibliography.
    tex1, bib1, src1, _ = _gen(tmp_path, journal="jweia")
    tex2, bib2, src2, _ = _gen(tmp_path, journal="asce_jse")
    assert bib1 != bib2 and os.path.exists(bib1) and os.path.exists(bib2)
    assert os.path.basename(bib1)[:-4] in src1 and os.path.basename(bib2)[:-4] in src2


def test_relative_output_dir_returns_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tex, bib = generate_latex(BASE, "jweia", output_dir="rel")
    assert os.path.isabs(tex) and os.path.isabs(bib)


# ---------------------------------------------------------------------------
# References -> BibTeX
# ---------------------------------------------------------------------------


def test_bib_entries_keep_every_reference_verbatim_or_structured(tmp_path):
    refs = [
        "[1] A.B. Smith, C.D. Jones, Seismic response of RC frames, Eng. Struct. 50 (2020) 100-120. https://doi.org/10.1016/j.engstruct.2020.110555",
        '[2] Beskos, D. E. 1987. "Boundary element methods in dynamic analysis." Appl. Mech. Rev., 40 (1), 1-23. https://doi.org/10.1115/1.3149529.',
        "[3] Park, R. and Paulay, T. (1975). Reinforced Concrete Structures, John Wiley & Sons, New York.",
        "",
        "[5] Some 100% {unbalanced reference",
    ]
    _, _, _, bib = _gen(tmp_path, {**BASE, "references": refs})
    entries = re.split(r"\n(?=@)", bib.strip())
    assert [e.split("{", 1)[0] for e in entries] == ["@article", "@article", "@misc", "@misc"]
    # Keys are consecutive even though one reference line was blank.
    assert re.findall(r"@\w+\{(ref\d+),", bib) == ["ref1", "ref2", "ref3", "ref4"]
    assert "author = {A.B. Smith and C.D. Jones}" in entries[0]
    assert "title = {{Seismic response of RC frames}}" in entries[0]
    assert "pages = {100--120}" in entries[0]
    assert "doi = {10.1016/j.engstruct.2020.110555}" in entries[0]
    assert "author = {Beskos, D. E.}" in entries[1]
    assert "journal = {Appl. Mech. Rev.}" in entries[1] and "number = {1}" in entries[1]
    # Previously parsed as title={Park} with everything else discarded.
    assert "John Wiley \\& Sons" in entries[2] and "note = {Park, R. and Paulay, T." in entries[2]
    assert "100\\% \\textbraceleft{}unbalanced" in entries[3]
    assert_bib_balanced(bib)


@pytest.mark.parametrize(
    "ref,expected",
    [
        (
            "A. Author, B. Author, Title of paper, J. Eng. 1 (2024) 1-10",
            {"author": "A. Author and B. Author", "title": "Title of paper", "journal": "J. Eng."},
        ),
        (
            'Kim, S., Lee, J., and Park, H. (2023). "ML-based damage prediction." KSCE Journal of Civil Engineering, Vol. 27, No. 5, pp. 2145-2158, DOI: 10.1007/s12205-023-1234-5.',
            {
                "author": "Kim, S. and Lee, J. and Park, H.",
                "journal": "KSCE Journal of Civil Engineering",
                "volume": "27",
                "number": "5",
                "pages": "2145-2158",
                "doi": "10.1007/s12205-023-1234-5",
            },
        ),
        (
            "J.-H. Kim, A. van der Berg et al., Fatigue of welds, Thin-Walled Struct. 180 (2022) 109876",
            {"author": "J.-H. Kim and A. van der Berg and others", "pages": "109876"},
        ),
        ("ACI Committee 318 (2019). Building Code Requirements.", {"year": "2019"}),
    ],
)
def test_parse_reference(ref, expected):
    fields = _parse_reference(ref)
    for key, value in expected.items():
        assert fields.get(key) == value
    if "author" not in expected:
        assert "author" not in fields and "title" not in fields


def test_bibtex_authors():
    assert _bibtex_authors("Evans, G. M., and J. C. Furlong") == "Evans, G. M. and J. C. Furlong"
    assert _bibtex_authors("Park, R. and Paulay, T.") == "Park, R. and Paulay, T."
    assert _bibtex_authors("A. Smith & B. Jones") == "A. Smith and B. Jones"


@pytest.mark.parametrize(
    "caption,kind,expected",
    [
        ("Fig. 1. Model comparison.", "figure", "Model comparison."),
        ("Figure 12: Load path", "figure", "Load path"),
        ("Fig. 3\u2014Crack pattern", "figure", "Crack pattern"),
        ("Fig. 3D view of the model", "figure", "Fig. 3D view of the model"),
        ("Table 2 Results.", "table", "Results."),
        ("Table 1. Results.", "figure", "Table 1. Results."),
    ],
)
def test_strip_caption_label(caption, kind, expected):
    assert _strip_caption_label(caption, kind) == expected


@pytest.mark.parametrize("journal_key", available_journal_keys())
def test_full_paper_is_structurally_valid_for_every_journal(tmp_path, journal_key):
    content = {
        **RICH,
        "title": "Wind & Seismic: 100% {robust} σ-test",
        "abstract": "σ ≤ 0.5 and p < 0.05 with $\\alpha$.\n\n$$a+b$$",
        "references": [
            "[1] A.B. Smith, Title, J. Eng. 1 (2020) 1-2.",
            "[2] Plain {reference} with & and %",
        ],
        "figure_captions": ["Fig. 1. A & B."],
    }
    tex_path, bib_path, tex, bib = _gen(tmp_path, content, journal=journal_key, figures=["f.png"])
    assert tex_path.endswith(f"_{journal_key}.tex") and bib_path.endswith(f"_{journal_key}.bib")
    assert tex.startswith("\\documentclass[") and tex.rstrip().endswith(r"\end{document}")
    assert_tex_balanced(tex)
    assert_bib_balanced(bib)


def test_default_output_dir_is_module_constant(tmp_path):
    # Patching latex_generator.DEFAULT_PAPERS_DIR must redirect the output.
    tex, bib = generate_latex(BASE, "jweia")
    assert os.path.dirname(tex) == os.path.dirname(bib) == str(tmp_path / "default_papers")


@pytest.mark.parametrize(
    "title,stem",
    [
        ("Évaluation sismique des ponts", "Evaluation_sismique_des_ponts_eng_structures"),
        ("철근콘크리트 보 and RC beams", "and_RC_beams_eng_structures"),
        ("Shear: V/(b·d) ≤ 0.5√f'c", "Shear_Vbd_05fc_eng_structures"),
    ],
)
def test_tex_and_bib_names_are_ascii(tmp_path, title, stem):
    tex_path, bib_path, tex, _ = _gen(tmp_path, {**BASE, "title": title}, journal="eng_structures")
    assert os.path.basename(tex_path) == f"{stem}.tex"
    assert os.path.basename(bib_path) == f"{stem}.bib"
    assert f"\\bibliography{{{stem}}}" in tex


def test_ascelike_does_not_load_amssymb(tmp_path):
    # ascelike v3 loads newtxmath, which already defines the AMS symbols; loading
    # amssymb again aborts pdflatex ("Command \\Bbbk already defined"), found by
    # compiling with TeX Live in CI.
    _, _, tex, _ = _gen(tmp_path, journal="asce_jse")
    assert r"\documentclass[Journal]{ascelike}" in tex
    assert "amssymb" not in tex
    assert r"\usepackage{amsmath}" in tex
    _, _, elsevier, _ = _gen(tmp_path / "e", journal="eng_structures")
    assert r"\usepackage{amsmath,amssymb}" in elsevier
