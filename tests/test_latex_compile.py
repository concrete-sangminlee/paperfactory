"""Compile the LaTeX export for every journal with a real TeX toolchain.

These tests need ``pdflatex`` and ``bibtex`` on PATH. They are skipped when TeX
is not installed, except when ``PAPERFACTORY_REQUIRE_TEX=1`` is set (the
dedicated CI job does this), in which case a missing toolchain is a failure.
"""

import json
import os
import re
import shutil
import subprocess

import matplotlib.pyplot as plt
import pytest

from utils.guidelines import available_journal_keys
from utils.latex_generator import generate_latex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_TEX = bool(shutil.which("pdflatex") and shutil.which("bibtex"))
REQUIRE_TEX = os.environ.get("PAPERFACTORY_REQUIRE_TEX") == "1"

if not HAVE_TEX and not REQUIRE_TEX:
    pytest.skip("pdflatex/bibtex not installed", allow_module_level=True)


def _run(cmd, cwd):
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=180
    )


def _compile(tex_path):
    workdir = os.path.dirname(tex_path)
    stem = os.path.splitext(os.path.basename(tex_path))[0]
    pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"]
    first = _run(pdflatex, workdir)
    assert first.returncode == 0, first.stdout[-3000:]
    with open(os.path.join(workdir, stem + ".aux"), encoding="utf-8", errors="replace") as f:
        aux = f.read()
    if r"\bibdata" in aux:
        bib = _run(["bibtex", stem], workdir)
        assert bib.returncode == 0, bib.stdout[-3000:]
    for _ in range(2):
        again = _run(pdflatex, workdir)
        assert again.returncode == 0, again.stdout[-3000:]
    pdf = os.path.join(workdir, stem + ".pdf")
    assert os.path.getsize(pdf) > 1000
    with open(os.path.join(workdir, stem + ".log"), encoding="utf-8", errors="replace") as f:
        return f.read()


@pytest.fixture(scope="module")
def figure_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("figs") / "fig_1_model_comparison.png"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot([0, 1, 2], [0, 1, 4], marker="o")
    ax.set_xlabel("Wind speed (m/s)")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


@pytest.fixture(scope="module")
def paper():
    with open(os.path.join(ROOT, "examples", "sample_paper.json"), encoding="utf-8") as f:
        p = json.load(f)
    p["sections"][0].setdefault("subsections", []).append(
        {
            "heading": "Special characters",
            "content": "Costs of 5% & $10 under 50_000 N loads; σ ≤ 0.5 μm; "
            "inline math $f_c' = 30$ MPa; Fig. 1 shows the trend [1].",
        }
    )
    p["practical_applications"] = "Engineers can use the model for preliminary design."
    p["notation"] = [{"symbol": "C_p", "definition": "pressure coefficient"}]
    return p


@pytest.mark.parametrize("journal", available_journal_keys())
def test_every_journal_compiles(tmp_path, journal, paper, figure_path):
    tex, bib = generate_latex(paper, journal, [figure_path], output_dir=str(tmp_path))
    log = _compile(tex)
    assert "! " not in log  # no TeX errors
    if paper.get("references"):
        assert not re.search(r"Citation `[^']+' .*undefined", log), "undefined citations"


def test_korean_paper_compiles(tmp_path, figure_path):
    paper = {
        "title": "풍하중을 받는 고층 건물의 머신러닝 기반 응답 예측",
        "authors": "홍길동, 김철수",
        "abstract": "본 연구는 고층 건물의 풍응답을 예측하는 머신러닝 모델을 제안한다.",
        "keywords": "풍하중; 머신러닝; 고층 건물",
        "sections": [
            {"heading": "서론", "content": "풍하중은 고층 건물 설계에서 중요하다 [1]."},
            {"heading": "결론", "content": "제안 모델은 R2 = 0.95를 달성하였다."},
        ],
        "references": [
            "[1] J.D. Holmes, Wind Loading of Structures, third ed., CRC Press, Boca Raton, 2018."
        ],
        "figure_captions": ["Fig. 1. 풍속에 따른 응답."],
    }
    tex, _ = generate_latex(paper, "ksce_jce", [figure_path], output_dir=str(tmp_path))
    log = _compile(tex)
    assert "! " not in log
