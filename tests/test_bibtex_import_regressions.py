"""Regression tests for utils.bibtex_import (bugs found in review)."""

import pytest

from utils.bibtex_import import _format_authors, import_bib, parse_bib_entries, parse_bib_string


class TestEntryParsing:
    def test_closing_brace_on_same_line(self):
        bib = (
            "@article{a, author={Bre, Facundo}, title={First paper}, journal={Energy Build.},"
            " volume={158}, year={2018}}\n"
            "@article{b, author={Weng, Yifan}, title={Second paper}, journal={Eng. Struct.},"
            " volume={258}, year={2022}}\n"
        )
        assert parse_bib_string(bib) == [
            "[1] F. Bre, First paper, Energy Build. 158 (2018).",
            "[2] Y. Weng, Second paper, Eng. Struct. 258 (2022).",
        ]

    def test_quoted_and_bare_values(self):
        bib = '@article{c,\n author = "Kim, Sung",\n title = "Quoted title",\n journal = "J. X",\n volume = 234,\n year = 2024,\n month = jan\n}\n'
        assert parse_bib_string(bib) == ["[1] S. Kim, Quoted title, J. X 234 (2024)."]

    def test_string_macros_and_comments(self):
        bib = (
            "@string{jweia = {J. Wind Eng. Ind. Aerodyn.}}\n"
            "@comment{This is a comment, with a comma}\n"
            '@preamble{"\\newcommand{\\noop}[1]{}"}\n'
            "@article{d,\n author = {Lee, Ji},\n title = {Macro journal},\n journal = jweia,\n year = {2023}\n}\n"
        )
        assert parse_bib_string(bib) == [
            "[1] J. Lee, Macro journal, J. Wind Eng. Ind. Aerodyn. (2023)."
        ]

    def test_concatenation(self):
        entry = parse_bib_entries('@misc{e, note = "Part " # "two", month = jan # "~1"}')[0]
        assert entry["note"] == "Part two"
        assert entry["month"] == "January~1"

    def test_nested_braces(self):
        bib = "@article{e,\n author = {Park, Hyun},\n title = {{Seismic {RC} {Frame} design}},\n journal = {Eng. Struct.},\n year = {2020},\n pages = {1--10}\n}\n"
        assert parse_bib_string(bib) == [
            "[1] H. Park, Seismic RC Frame design, Eng. Struct. (2020) 1-10."
        ]

    def test_parenthesised_entry(self):
        bib = "@ARTICLE(h,\n AUTHOR = {Doe, Jane},\n TITLE = {Paren entry},\n JOURNAL = {J. X},\n YEAR = {2021}\n)\n"
        assert parse_bib_string(bib) == ["[1] J. Doe, Paren entry, J. X (2021)."]

    def test_numbering_has_no_gaps(self):
        bib = "@misc{empty,}\n@article{j, author={Doe, Jane}, title={Second}, journal={J. X}, year={2021}}"
        assert parse_bib_string(bib) == ["[1] J. Doe, Second, J. X (2021)."]


class TestFieldFormatting:
    def test_latex_accents(self):
        bib = (
            "@article{f,\n author = {M{\\\"u}ller, J{\\\"o}rg and Garc{\\'\\i}a, Jos{\\'e}},\n"
            ' title = {{\\"U}ber die Stabilit{\\"a}t {\\&} {\\v{S}}koda},\n journal = {Stahlbau},\n'
            " year = {2019},\n pages = {100--110}\n}\n"
        )
        assert parse_bib_string(bib) == [
            "[1] J. Müller, J. García, Über die Stabilität & Škoda, Stahlbau (2019) 100-110."
        ]

    def test_techreport_keeps_institution_and_year(self):
        bib = "@techreport{g,\n author = {Smith, John},\n title = {Wind tunnel testing report},\n institution = {NIST},\n number = {TN-1},\n year = {2019}\n}\n"
        assert parse_bib_string(bib) == [
            "[1] J. Smith, Wind tunnel testing report, Report TN-1, NIST, 2019."
        ]

    def test_inproceedings_elsevier_style(self):
        bib = (
            "@inproceedings{k, author = {Kang, S. C. and Chi, H. L.}, title = {{iWalk}: Agent navigation},"
            " booktitle = {Proc. ISARC}, address = {Seoul, Korea}, year = 2011, pages = {586--591}}"
        )
        assert parse_bib_string(bib) == [
            "[1] S.C. Kang, H.L. Chi, iWalk: Agent navigation, in: Proc. ISARC, Seoul, Korea, 2011, pp. 586-591."
        ]

    @pytest.mark.parametrize(
        "doi", ["doi:10.1000/xyz", "https://doi.org/10.1000/xyz", "http://dx.doi.org/10.1000/xyz"]
    )
    def test_doi_prefixes_normalised(self, doi):
        bib = f"@article{{i, author={{Doe, Jane}}, title={{T}}, journal={{J}}, year={{2021}}, doi={{{doi}}}}}"
        assert parse_bib_string(bib)[0].endswith(" https://doi.org/10.1000/xyz")

    def test_issue_number_included(self):
        bib = "@article{l, author={Li, Victor C.}, title={ECC}, journal={ACI Mater. J.}, volume={98}, number={6}, year={2001}, pages={483--492}}"
        assert parse_bib_string(bib) == ["[1] V.C. Li, ECC, ACI Mater. J. 98 (6) (2001) 483-492."]


class TestFormatAuthors:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Juan Manuel Gimenez and Jan van der Berg", "J.M. Gimenez, J. van der Berg"),
            ("van der Berg, Jan and de la Cruz, Maria", "J. van der Berg, M. de la Cruz"),
            ("An, Lee-Sun", "L.-S. An"),
            ("King, Jr., Martin Luther", "M.L. King Jr."),
            ("Doe, Jane and others", "J. Doe, et al."),
            ("{ACI Committee 318}", "ACI Committee 318"),
            ("{Smith and Sons Ltd.} and Doe, Jane", "Smith and Sons Ltd., J. Doe"),
            ("doe, jane", "J. doe"),
            ("Smith, DeAndre", "D. Smith"),
            ("Doe, Jane AND Roe, Rick", "J. Doe, R. Roe"),
        ],
    )
    def test_name_forms(self, raw, expected):
        assert _format_authors(raw) == expected


class TestImportBib:
    def test_utf8_bom(self, tmp_path):
        path = tmp_path / "bom.bib"
        path.write_bytes(
            b"\xef\xbb\xbf@article{a, author={M\xc3\xbcller, J}, title={T}, journal={J}, year={2020}}"
        )
        assert import_bib(str(path)) == ["[1] J. Müller, T, J (2020)."]

    def test_latin1_fallback(self, tmp_path):
        path = tmp_path / "latin1.bib"
        path.write_bytes(b"@article{a, author={M\xfcller, J}, title={T}, journal={J}, year={2020}}")
        assert import_bib(str(path)) == ["[1] J. Müller, T, J (2020)."]
