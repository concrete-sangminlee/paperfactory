import os
import sys

import matplotlib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Headless backend for every test (CI runners have no display).
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _isolate_runs_dir(tmp_path_factory, monkeypatch):
    """Keep pipeline runs created by tests out of the repository's outputs/ folder."""
    monkeypatch.setenv("PAPERFACTORY_RUNS_DIR", str(tmp_path_factory.mktemp("runs")))
