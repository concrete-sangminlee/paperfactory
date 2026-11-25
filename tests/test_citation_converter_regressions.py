"""Regression tests for utils.citation_converter (bugs found in review)."""

import pytest

from utils.citation_converter import _parse_ref, convert_style, first_author_family

ROUND_TRIP_REFS = [
    "[1] F. Bre, J.M. Gimenez, V.D. Fachinotti, Prediction of wind pressure coefficients on "
    "building surfaces using artificial neural networks, Energy Build. 158 (2018) 1429-1441. "
    "https://doi.org/10.1016/j.enbuild.2017.11.045",
    "[2] S. Kim, J. Lee, Convolutional neural network for seismic damage classification of RC "
    "structures, Eng. Struct. 289 (2024) 116234.",
    "[3] Y. Xie, J. Zhang, R. DesRoches, Damage detection using deep learning in seismic response "
    "data, Earthquake Eng. Struct. Dyn. 48 (14) (2019) 1736-1752. https://doi.org/10.1002/eqe.3228",
    "[4] L.-S. An, S. Jung, Data-driven prediction of wind pressure on low-rise buildings, Build. "
    "Environ. 265 (2024) 112022. https://doi.org/10.1016/j.buildenv.2024.112022",
    "[5] J.D. Holmes, Wind Loading of Structures, third ed., CRC Press, Boca Raton, 2018.",
    "[6] F.M. Nav, S.F. Mirfakhar, R. Snaiki, A hybrid machine learning framework for wind "
    "pressure prediction, Comput.-Aided Civ. Infrastruct. Eng. (2025). "
    "https://doi.org/10.1111/mice.13488",
    "[7] A. Der Kiureghian, O. Ditlevsen, Aleatory or epistemic? Does it matter?, Struct. Saf. 31 "
    "(2009) 105-112.",
    "[8] J.L. Provis, J.S.J. van Deventer, Alkali Activated Materials: State-of-the-Art Report, "
    "RILEM TC 224-AAM, Springer, Dordrecht, 2014.",
    "[9] P.K. Mehta, P.J.M. Monteiro, Concrete: Microstructure, Properties, and Materials, fourth "
    "ed., McGraw-Hill, New York, 2014.",
]


def _strip_number(ref):
    return ref.split("] ", 1)[1]


class TestRoundTrips:
    @pytest.mark.parametrize("ref", ROUND_TRIP_REFS)
    @pytest.mark.parametrize("via", ["author_date", "apa"])
    def test_numbered_round_trip_is_lossless(self, ref, via):
        middle = convert_style([ref], via)
        back = convert_style(middle, "numbered")
        assert _strip_number(back[0]) == _strip_number(ref)

    def test_pages_survive_author_date_round_trip(self):
        back = convert_style(convert_style(ROUND_TRIP_REFS[:1], "author_date"), "numbered")
        assert "1429-1441" in back[0]


class TestParsing:
    def test_two_single_initial_authors(self):
        # Engineering Structures' own guideline example used to put "J. Lee" in the title.
        p = _parse_ref(ROUND_TRIP_REFS[1])
        assert [a["family"] for a in p["authors_list"]] == ["Kim", "Lee"]
        assert p["title"].startswith("Convolutional neural network")
        assert p["journal"] == "Eng. Struct."

    def test_asce_author_date_to_numbered_keeps_everything(self):
        asce = (
            'Beskos, D. E. 1987. "Boundary element methods in dynamic analysis." Appl. Mech. Rev., '
            "40 (1), 1-23. https://doi.org/10.1115/1.3149529."
        )
        assert convert_style([asce], "numbered") == [
            "[1] D.E. Beskos, Boundary element methods in dynamic analysis, Appl. Mech. Rev. 40 (1) "
            "(1987) 1-23. https://doi.org/10.1115/1.3149529"
        ]

    def test_asce_multi_author_with_and(self):
        p = _parse_ref(
            "Evans, G. M., and J. C. Furlong. 2003. Environmental biotechnology. Chichester, UK: Wiley."
        )
        assert [a["family"] for a in p["authors_list"]] == ["Evans", "Furlong"]
        assert p["rest"] == "Chichester, UK: Wiley"

    def test_wiley_style(self):
        p = _parse_ref(
            "1. Boore DM, Atkinson GM. Ground-motion prediction equations. Earthquake Spectra 2008; "
            "24(1):99-138."
        )
        assert [a["given"] for a in p["authors_list"]] == ["D.M.", "G.M."]
        assert (p["journal"], p["volume"], p["issue"], p["pages"]) == (
            "Earthquake Spectra",
            "24",
            "1",
            "99-138",
        )

    def test_aci_quoted_title(self):
        p = _parse_ref(
            '1. DiAmato, S., and Jenner, W., "Mechanisms of Shear Resistance," ACI Structural '
            "Journal, V. 103, No. 1, Jan.-Feb. 2006, pp. 67-74."
        )
        assert p["title"] == "Mechanisms of Shear Resistance"
        assert (p["volume"], p["issue"], p["pages"], p["year"]) == ("103", "1", "67-74", "2006")

    def test_standard_year_not_taken_from_designation(self):
        p = _parse_ref(
            "[3] EN 1993-1-3, Eurocode 3: Design of Steel Structures, CEN, Brussels, 2006."
        )
        assert p["year"] == "2006"

    def test_particles_and_apostrophes(self):
        ref = 'van der Berg, J., and O\'Brien, E. J. 2020. "Bridge loads." J. Bridge Eng., 25 (3), 04020001.'
        assert convert_style([ref], "numbered")[0].startswith("[1] J. van der Berg, E.J. O'Brien,")


class TestRendering:
    def test_unparseable_reference_not_double_numbered(self):
        ref = "[3] EN 1993-1-3, Eurocode 3: Design of Steel Structures, CEN, Brussels, 2006."
        assert convert_style([ref], "numbered") == [
            "[1] EN 1993-1-3, Eurocode 3: Design of Steel Structures, CEN, Brussels, 2006."
        ]
        assert convert_style([ref], "author_date")[0].startswith("EN 1993-1-3")

    def test_apa_author_format(self):
        out = convert_style(ROUND_TRIP_REFS[:1], "apa")[0]
        assert out.startswith("Bre, F., Gimenez, J. M., & Fachinotti, V. D. (2018).")
        assert "*Energy Build.*, *158*, 1429-1441." in out

    def test_hyphenated_initials_preserved(self):
        assert convert_style([ROUND_TRIP_REFS[3]], "author_date")[0].startswith(
            "An, L.-S., Jung, S."
        )

    def test_string_input_and_blank_entries(self):
        out = convert_style(ROUND_TRIP_REFS[1] + "\n\n" + ROUND_TRIP_REFS[4] + "\n", "numbered")
        assert [r[:3] for r in out] == ["[1]", "[2]"]
        assert convert_style(["", None], "numbered") == []

    def test_style_alias(self):
        assert convert_style(ROUND_TRIP_REFS[:1], "author-date") == convert_style(
            ROUND_TRIP_REFS[:1], "author_date"
        )

    def test_first_author_family(self):
        assert first_author_family(ROUND_TRIP_REFS[6]) == "Der Kiureghian"
