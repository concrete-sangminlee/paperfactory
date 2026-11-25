"""Unit tests for the shared paper_content / guideline helpers in utils.word_generator."""

import pytest

from utils.guidelines import load_guideline
from utils.word_generator import (
    _all_text,
    _as_items,
    _as_sections,
    _as_text,
    _default_caption,
    _describe_figure,
    _document_layout,
    _figure_entries,
    _heading_rules,
    _HeadingNumberer,
    _highlight_items,
    _is_unnumbered_heading,
    _journal_features,
    _line_spacing_value,
    _notation_entries,
    _parse_margin,
    _primary_font,
    _safe_stem,
    _strip_heading_number,
    _table_parts,
)


class TestNormalisation:
    def test_as_text(self):
        assert _as_text(None) == ""
        assert _as_text(["A", None, " ", {"name": "B"}]) == "A, B"
        assert _as_text(["x", "y"], "; ") == "x; y"
        assert _as_text(3.5) == "3.5"
        assert _as_text({"first": "A", "last": "B"}) == "A, B"

    def test_as_text_strips_xml_invalid_characters(self):
        assert _as_text("a\x0cb\x00c\x07d\te\nf") == "a bcd\te\nf"

    def test_as_items(self):
        assert _as_items("[1] a\n\n[2] b\n") == ["[1] a", "[2] b"]
        assert _as_items(["a", None, "", "  b "]) == ["a", "b"]
        assert _as_items(None) == []
        assert _as_items("single", split_lines=False) == ["single"]

    def test_highlight_items_strip_bullets(self):
        assert _highlight_items("• One\n- Two\n* Three") == ["One", "Two", "Three"]
        assert _highlight_items("-3 dB loss") == ["-3 dB loss"]

    def test_as_sections(self):
        assert _as_sections(None) == []
        assert _as_sections({"heading": "H"}) == [{"heading": "H"}]
        assert _as_sections(["Loose text", None]) == [{"heading": "", "content": "Loose text"}]

    def test_all_text_walks_nested_values(self):
        text = _all_text({"a": ["x", {"b": "철근"}], "c": None, "d": 2})
        assert "x" in text and "철근" in text and "2" in text

    @pytest.mark.parametrize(
        "value,expected",
        [
            ([{"symbol": "f_c", "definition": "strength"}], [("f_c", "strength")]),
            ({"symbol": "E", "definition": "modulus"}, [("E", "modulus")]),
            ({"E": "modulus", "nu": "Poisson"}, [("E", "modulus"), ("nu", "Poisson")]),
            ([("E", "modulus"), ["G"], "h", None, {"symbol": None}], [("E", "modulus"), ("G", ""), ("h", "")]),
        ],
    )
    def test_notation_entries(self, value, expected):
        assert _notation_entries(value) == expected

    def test_table_parts_pads_ragged_rows(self):
        caption, headers, rows = _table_parts(
            {"caption": None, "headers": ["A", "B"], "rows": [["1"], ["1", 2, 3], {"x": 4}]}
        )
        assert caption == ""
        assert headers == ["A", "B", ""]
        assert rows == [["1", "", ""], ["1", "2", "3"], ["4", "", ""]]
        assert _table_parts("not a dict") == ("", [], [])

    def test_figure_entries_pairs_longest(self):
        assert _figure_entries("a.png", None) == [("a.png", "")]
        assert _figure_entries(["a.png", None], ["A", "B", "C"]) == [
            ("a.png", "A"),
            (None, "B"),
            (None, "C"),
        ]
        assert _figure_entries(None, None) == []


class TestCaptions:
    @pytest.mark.parametrize(
        "style,expected",
        [
            ("Fig. {n}. Description.", "Fig. 3. Load path."),
            ("Fig. {n}. Caption text in sentence case.", "Fig. 3. Load path."),
            ("Fig. {n}—Description", "Fig. 3—Load path"),
            ("Figure {n}. Description.", "Figure 3. Load path."),
            ("Table {n} Description.", "Table 3 Load path."),
            ("Fig. {n}: {desc}", "Fig. 3: Load path"),
            (None, "Fig. 3. Load path"),
            ("no number placeholder", "Fig. 3. Load path"),
        ],
    )
    def test_default_caption(self, style, expected):
        assert _default_caption(style, 3, "Load path") == expected

    def test_default_caption_without_description(self):
        assert _default_caption("Fig. {n}—Description", 2) == "Fig. 2"
        assert _default_caption("Table {n}. Description.", 1) == "Table 1."

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("outputs/figures/fig_1_model_comparison.png", "Model comparison"),
            ("Figure-2a-load_path.tif", "Load path"),
            ("test_fig.png", "Test fig"),
            ("figurehead.png", "Figurehead"),
            ("fig3.png", ""),
            (None, ""),
        ],
    )
    def test_describe_figure(self, path, expected):
        assert _describe_figure(path) == expected


class TestGuidelineInterpretation:
    def test_primary_font(self):
        assert _primary_font("Times New Roman or Arial") == "Times New Roman"
        assert _primary_font("Arial, Helvetica") == "Arial"
        assert _primary_font(None) == "Times New Roman"

    @pytest.mark.parametrize(
        "value,expected",
        [("single", 1.0), ("1.5", 1.5), (1.5, 1.5), ("one and a half", 1.5), ("double", 2.0), (2, 2.0), (None, 2.0), ("weird", 2.0)],
    )
    def test_line_spacing_value(self, value, expected):
        assert _line_spacing_value(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [("1 inch all sides", 1.0), ("25 mm all sides", 25 / 25.4), ("2 cm", 2 / 2.54), ("0.75 in", 0.75), ("standard", 1.0), (None, 1.0)],
    )
    def test_parse_margin(self, value, expected):
        assert _parse_margin(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "key,numbered,upper",
        [
            ("asce_jse", False, True),
            ("aci_sj", False, True),
            ("eesd", True, True),
            ("eng_structures", True, False),
            ("ksce_jce", True, False),
            ("buildings_mdpi", True, False),
        ],
    )
    def test_heading_rules(self, key, numbered, upper):
        rules = _heading_rules(load_guideline(key))
        assert (rules["numbered"], rules["upper"]) == (numbered, upper)

    def test_heading_rules_default_without_heading_format(self):
        rules = _heading_rules({"manuscript_structure": {}})
        assert rules["upper"] and not rules["numbered"]

    def test_heading_numberer(self):
        n = _HeadingNumberer(True)
        assert n.label(1, "Intro") == "1. Intro"
        assert n.label(2, "Data") == "1.1. Data"
        assert n.label(3, "Deep") == "1.1.1. Deep"
        assert n.label(2, "Model") == "1.2. Model"
        assert n.label(1, "Results") == "2. Results"
        assert n.label(2, "Fit") == "2.1. Fit"
        assert n.label(1, "Appendix A") == "Appendix A"
        assert n.label(2, "Derivation") == "Derivation"
        assert _HeadingNumberer(False).label(1, "Intro") == "Intro"
        manual = _HeadingNumberer(True, [{"heading": "Intro", "subsections": [{"heading": "2.1 X"}]}])
        assert manual.label(1, "Intro") == "Intro"

    def test_heading_number_helpers(self):
        assert _strip_heading_number("2.3 Method") == "Method"
        assert _strip_heading_number("1. Intro") == "Intro"
        assert _strip_heading_number("2024 Turkey earthquake") == "2024 Turkey earthquake"
        assert _is_unnumbered_heading("ACKNOWLEDGEMENTS")
        assert _is_unnumbered_heading("Appendix B. Derivation")
        assert not _is_unnumbered_heading("Introduction")

    @pytest.mark.parametrize(
        "key,order,front",
        [
            ("asce_jse", ["data_availability", "acknowledgments", "notation", "references", "tables", "figures"], False),
            ("aci_sj", ["acknowledgments", "data_availability", "references", "notation", "tables", "figures"], False),
            ("jbe", ["notation", "data_availability", "acknowledgments", "references", "figures", "tables"], False),
            ("eng_structures", ["acknowledgments", "data_availability", "references", "tables", "figures"], True),
            ("buildings_mdpi", ["tables", "figures", "notation", "data_availability", "acknowledgments", "references"], False),
        ],
    )
    def test_document_layout(self, key, order, front):
        layout = _document_layout(load_guideline(key))
        assert layout["order"] == order
        assert layout["front_notation"] is front

    def test_document_layout_labels(self):
        assert _document_layout(load_guideline("ksce_jce"))["labels"]["acknowledgments"] == (
            "Acknowledgements"
        )
        assert _document_layout(load_guideline("jweia"))["labels"]["notation"] == "Nomenclature"
        assert _document_layout({})["labels"]["references"] == "References"

    def test_journal_features(self):
        assert _journal_features(load_guideline("asce_jse"), "asce_jse")["practical_applications"]
        assert not _journal_features(load_guideline("asce_jse"), "asce_jse")["highlights"]
        assert _journal_features(load_guideline("jweia"), "jweia")["highlights"]
        assert not _journal_features(load_guideline("eesd"), "eesd")["highlights"]


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Wind: Pressure & ML!", "Wind_Pressure_ML"),
        ("철근콘크리트 보", "철근콘크리트_보"),
        ("-leading dash", "leading_dash"),
        ("???", "paper"),
        (None, "paper"),
        ("x" * 80, "x" * 50),
    ],
)
def test_safe_stem(title, expected):
    assert _safe_stem(title) == expected
