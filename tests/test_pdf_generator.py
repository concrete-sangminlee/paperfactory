import os

from utils.pdf_generator import generate_pdf


def _make_paper():
    return {
        "title": "Test Paper for PDF Generation",
        "authors": "A. Author, B. Author",
        "abstract": "This is a test abstract for PDF generation.",
        "keywords": "test; pdf; generation",
        "highlights": ["Highlight one", "Highlight two"],
        "sections": [
            {"heading": "INTRODUCTION", "content": "Introduction text here."},
            {
                "heading": "METHODOLOGY",
                "content": "Method text.",
                "subsections": [
                    {"heading": "Data collection", "content": "Data text."},
                ],
            },
            {"heading": "CONCLUSIONS", "content": "Conclusion text."},
        ],
        "tables": [
            {"caption": "Table 1. Test.", "headers": ["A", "B"], "rows": [["1", "2"]]},
        ],
        "references": [
            "[1] A. Author, Test paper, J. Test 1 (2024) 1-10.",
            "[2] B. Author, Another paper, J. Test 2 (2023) 11-20.",
        ],
        "figure_captions": ["Fig. 1. Test figure."],
        "data_availability": "Data available on request.",
        "acknowledgments": "Thanks to everyone.",
    }


class TestGeneratePdf:
    def test_creates_pdf_file(self, tmp_path):
        paper = _make_paper()
        path = generate_pdf(paper, "jweia", output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".pdf")

    def test_pdf_has_content(self, tmp_path):
        paper = _make_paper()
        path = generate_pdf(paper, "jweia", output_dir=str(tmp_path))
        assert os.path.getsize(path) > 1000

    def test_works_with_different_journals(self, tmp_path):
        paper = _make_paper()
        for jk in ["jweia", "asce_jse", "eng_structures"]:
            path = generate_pdf(paper, jk, output_dir=str(tmp_path))
            assert os.path.exists(path)

    def test_with_figures(self, tmp_path):
        paper = _make_paper()
        # Create a dummy figure
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        fig_path = os.path.join(str(tmp_path), "test_fig.png")
        fig.savefig(fig_path)
        plt.close()

        path = generate_pdf(paper, "jweia", figures=[fig_path], output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000

    def test_empty_sections(self, tmp_path):
        paper = _make_paper()
        paper["sections"] = []
        path = generate_pdf(paper, "jweia", output_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_special_characters_in_text(self, tmp_path):
        paper = _make_paper()
        paper["abstract"] = "Test with special chars: <angle> & ampersand > more"
        path = generate_pdf(paper, "jweia", output_dir=str(tmp_path))
        assert os.path.exists(path)
