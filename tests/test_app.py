"""Smoke tests for the Streamlit web UI (app.py) using Streamlit's AppTest harness."""

import json
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
with open(os.path.join(ROOT, "examples", "sample_paper.json"), encoding="utf-8") as _f:
    SAMPLE_JSON = _f.read()


@pytest.fixture
def at():
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def _by_key(elements, key):
    return next(e for e in elements if e.key == key)


def _markdown(app):
    return "\n".join(m.value for m in app.markdown)


def test_initial_render(at):
    assert at.title[0].value.startswith("📝 PaperFactory")
    assert len(at.tabs) >= 7
    # every journal selectbox lists all 15 bundled guidelines
    assert len(_by_key(at.selectbox, "qc_journal").options) == 15


def test_quality_check_runs(at):
    _by_key(at.text_area, "qc_text").input(SAMPLE_JSON)
    _by_key(at.button, "qc_run").click()
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Quality Score" for m in at.metric)


def test_quality_check_invalid_json(at):
    _by_key(at.text_area, "qc_text").input("{broken")
    _by_key(at.button, "qc_run").click()
    at.run()
    assert not at.exception
    assert any("Invalid JSON" in e.value for e in at.error)


def test_quality_check_without_input_warns(at):
    _by_key(at.button, "qc_run").click()
    at.run()
    assert not at.exception
    assert any("Provide paper_content" in w.value for w in at.warning)


def test_ai_review_runs(at):
    _by_key(at.text_area, "review_text").input(SAMPLE_JSON)
    _by_key(at.button, "review_run").click()
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Decision" for m in at.metric)


def test_submission_checklist_runs(at):
    _by_key(at.text_area, "sub_text").input(SAMPLE_JSON)
    _by_key(at.button, "sub_run").click()
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Ready to Submit" for m in at.metric)


def test_cover_letter_runs(at):
    _by_key(at.text_area, "cl_text").input(SAMPLE_JSON)
    _by_key(at.button, "cl_run").click()
    at.run()
    assert not at.exception, at.exception
    letter = _by_key(at.text_area, "cl_out").value
    assert json.loads(SAMPLE_JSON)["title"] in letter


def test_export_word(at):
    _by_key(at.text_area, "export_text").input(SAMPLE_JSON)
    _by_key(at.button, "export_run").click()
    at.run()
    assert not at.exception, at.exception
    assert any("Generated" in s.value for s in at.success)


def test_topic_suggests_sources(at):
    at.text_area[0].input("wind pressure prediction on low-rise buildings")
    at.run()
    assert not at.exception
    assert "TPU" in _markdown(at)
