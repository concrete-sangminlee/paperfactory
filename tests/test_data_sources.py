import pytest

from utils.data_sources import get_source, list_sources, suggest_sources


class TestListSources:
    def test_returns_all(self):
        sources = list_sources()
        assert len(sources) >= 7

    def test_filter_by_wind(self):
        sources = list_sources("wind_engineering")
        assert len(sources) >= 2
        assert all("wind_engineering" in s["fields"] for s in sources)

    def test_filter_by_earthquake(self):
        sources = list_sources("earthquake_engineering")
        assert len(sources) >= 3


class TestGetSource:
    def test_valid_key(self):
        s = get_source("tpu")
        assert s["name"] == "TPU Aerodynamic Database"
        assert "url" in s

    def test_invalid_key_raises(self):
        with pytest.raises(KeyError):
            get_source("nonexistent")

    def test_all_sources_have_required_fields(self):
        for key in ["tpu", "peer_nga", "cesmd", "nist_aero", "designsafe", "knet", "cosmos"]:
            s = get_source(key)
            assert "name" in s
            assert "url" in s
            assert "description" in s
            assert "access" in s


class TestSuggestSources:
    def test_wind_topic(self):
        sources = suggest_sources("wind pressure prediction on low-rise buildings")
        assert len(sources) >= 1
        assert sources[0]["key"] in ("tpu", "nist_aero")

    def test_earthquake_topic(self):
        sources = suggest_sources(
            "seismic damage assessment of RC frames using ground motion records"
        )
        assert len(sources) >= 2

    def test_no_match(self):
        sources = suggest_sources("quantum computing optimization")
        assert len(sources) == 0

    def test_sorted_by_relevance(self):
        sources = suggest_sources("wind pressure aerodynamic database low-rise roof")
        if len(sources) >= 2:
            assert sources[0]["relevance_score"] >= sources[1]["relevance_score"]
