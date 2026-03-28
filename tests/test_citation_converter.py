import pytest

from utils.citation_converter import _parse_ref, convert_style

SAMPLE_REFS = [
    "[1] F. Bre, J.M. Gimenez, Prediction of wind pressure coefficients, Energy Build. 158 (2018) 1429-1441. https://doi.org/10.1016/j.enbuild.2017.11.045",
    "[2] Y. Weng, S.G. Paal, ML-based wind pressure prediction, Eng. Struct. 258 (2022) 114148.",
]


class TestParseRef:
    def test_extracts_year(self):
        p = _parse_ref(SAMPLE_REFS[0])
        assert p["year"] == "2018"

    def test_extracts_doi(self):
        p = _parse_ref(SAMPLE_REFS[0])
        assert "10.1016" in p["doi"]

    def test_extracts_authors(self):
        p = _parse_ref(SAMPLE_REFS[0])
        assert "Bre" in p.get("authors", "")

    def test_extracts_title(self):
        p = _parse_ref(SAMPLE_REFS[0])
        assert "wind pressure" in p.get("title", "").lower()


class TestConvertStyle:
    def test_numbered(self):
        result = convert_style(SAMPLE_REFS, "numbered")
        assert len(result) == 2
        assert result[0].startswith("[1]")
        assert result[1].startswith("[2]")

    def test_author_date(self):
        result = convert_style(SAMPLE_REFS, "author_date")
        assert "(2018)" in result[0]
        assert "(2022)" in result[1]

    def test_apa(self):
        result = convert_style(SAMPLE_REFS, "apa")
        assert "(2018)" in result[0]

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError):
            convert_style(SAMPLE_REFS, "chicago")

    def test_preserves_count(self):
        refs = SAMPLE_REFS * 5
        for style in ["numbered", "author_date", "apa"]:
            result = convert_style(refs, style)
            assert len(result) == len(refs)
