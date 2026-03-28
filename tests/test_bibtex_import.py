import os
import pytest
from utils.bibtex_import import parse_bib_string, import_bib, _format_authors


SAMPLE_BIB = """
@article{bre2018prediction,
  author = {Bre, Facundo and Gimenez, Juan Manuel and Fachinotti, V{\\'i}ctor Daniel},
  title = {Prediction of wind pressure coefficients on building surfaces using artificial neural networks},
  journal = {Energy and Buildings},
  volume = {158},
  pages = {1429--1441},
  year = {2018},
  doi = {10.1016/j.enbuild.2017.11.045}
}

@article{weng2022ml,
  author = {Weng, Yifan and Paal, Stephanie German},
  title = {Machine learning-based wind pressure prediction of low-rise non-isolated buildings},
  journal = {Engineering Structures},
  volume = {258},
  pages = {114148},
  year = {2022},
  doi = {10.1016/j.engstruct.2022.114148}
}

@book{moehle2015seismic,
  author = {Moehle, Jack P.},
  title = {Seismic Design of Reinforced Concrete Buildings},
  publisher = {McGraw-Hill},
  address = {New York},
  year = {2015}
}
"""


class TestParseBibString:
    def test_parses_articles(self):
        refs = parse_bib_string(SAMPLE_BIB)
        assert len(refs) == 3

    def test_first_ref_numbered(self):
        refs = parse_bib_string(SAMPLE_BIB)
        assert refs[0].startswith("[1]")

    def test_contains_author(self):
        refs = parse_bib_string(SAMPLE_BIB)
        assert "Bre" in refs[0]

    def test_contains_doi(self):
        refs = parse_bib_string(SAMPLE_BIB)
        assert "10.1016" in refs[0]

    def test_contains_year(self):
        refs = parse_bib_string(SAMPLE_BIB)
        assert "2018" in refs[0]

    def test_book_entry(self):
        refs = parse_bib_string(SAMPLE_BIB)
        book_ref = refs[2]
        assert "McGraw-Hill" in book_ref
        assert "2015" in book_ref

    def test_empty_bib(self):
        refs = parse_bib_string("")
        assert refs == []


class TestImportBib:
    def test_import_from_file(self, tmp_path):
        bib_path = tmp_path / "test.bib"
        bib_path.write_text(SAMPLE_BIB)
        refs = import_bib(str(bib_path))
        assert len(refs) == 3


class TestFormatAuthors:
    def test_last_first_format(self):
        result = _format_authors("Bre, Facundo and Gimenez, Juan Manuel")
        assert "F. Bre" in result
        assert "J.M. Gimenez" in result

    def test_single_author(self):
        result = _format_authors("Moehle, Jack P.")
        assert "J.P. Moehle" in result
