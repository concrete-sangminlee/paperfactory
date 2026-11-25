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


class TestSuggestSourcesRegressions:
    @pytest.mark.parametrize(
        "topic",
        [
            "deterministic analysis of window glass",  # "nist" in deterministic, "wind" in window
            "throughput of output data",  # "tpu" in output/throughput
            "brainstorm session about shakedown analysis",  # "storm", "shake"
            "globally optimal winding layout",  # "global" -> globally, "wind" -> winding
        ],
    )
    def test_no_substring_false_positives(self, topic):
        assert suggest_sources(topic) == []

    def test_hyphen_and_plural_variants_match(self):
        keys = {s["key"]: s for s in suggest_sources("strong-motion records of earthquakes")}
        assert "strong motion" in keys["knet"]["matched_keywords"]
        assert keys["cosmos"]["relevance_score"] == 2
        tpu = next(
            s for s in suggest_sources("low rise buildings under winds") if s["key"] == "tpu"
        )
        assert set(tpu["matched_keywords"]) == {"wind", "low-rise"}

    def test_response_spectra(self):
        assert [s["key"] for s in suggest_sources("response spectra matching")] == ["peer_nga"]

    def test_empty_or_none_topic(self):
        assert suggest_sources("") == []
        assert suggest_sources(None) == []

    def test_relevance_score_counts_matched_keywords(self):
        for s in suggest_sources("wind pressure on low-rise roofs"):
            assert s["relevance_score"] == len(s["matched_keywords"])


class TestRegistryIsolation:
    def test_get_source_returns_copy(self):
        get_source("tpu")["fields"].append("LEAK")
        assert "LEAK" not in get_source("tpu")["fields"]

    def test_list_sources_returns_copies_with_keys(self):
        sources = list_sources()
        sources[0]["name"] = "LEAK"
        sources[0]["data_types"].clear()
        fresh = list_sources()
        assert fresh[0]["name"] != "LEAK" and fresh[0]["data_types"]
        assert all("key" in s for s in fresh)
        assert get_source(fresh[0]["key"])["name"] == fresh[0]["name"]

    def test_suggest_sources_returns_deep_copies(self):
        suggest_sources("wind")[0]["data_types"].append("LEAK")
        assert all("LEAK" not in s["data_types"] for s in list_sources())
