# Contributing to PaperFactory

Thank you for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/concrete-sangminlee/paperfactory.git paperfactory
cd paperfactory
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,ui]"         # editable install: keeps guidelines/ importable
pre-commit install                 # ruff on commit
pre-commit install --hook-type pre-push   # fast test run before push
```

## Development

### Running Tests

```bash
pytest                              # full suite
pytest --cov                        # with coverage report
pytest tests/test_cli.py -k export  # a subset
```

Tests never write into `outputs/`: `tests/conftest.py` redirects pipeline runs to a
temporary directory (`PAPERFACTORY_RUNS_DIR`) and forces the headless `Agg`
matplotlib backend. Write file outputs to pytest's `tmp_path`.

### Code Style

We use [ruff](https://docs.astral.sh/ruff/) (pinned to the version in
`pyproject.toml` / `.pre-commit-config.yaml`) for linting and formatting:

```bash
ruff check .
ruff format .
```

CI fails on any lint error or unformatted file.

### Continuous Integration

`.github/workflows/tests.yml` runs on every pull request:

| Job | What it checks |
|-----|----------------|
| Lint | `ruff check`, `ruff format --check`, JSON validity of guidelines/examples |
| Test | pytest + coverage on Ubuntu/macOS × Python 3.10–3.13, Windows × 3.12; CLI smoke test |
| Build | sdist/wheel build and Docker image build |

## Adding a Journal

1. Copy an existing file in `guidelines/` (e.g. `jweia.json`) to `guidelines/<key>.json`.
2. Fill every top-level section: `journal_name`, `abbreviation`, `publisher`, `scope`,
   `formatting`, `manuscript_structure` (incl. `sections`, `abstract_word_limit`),
   `references` (incl. `style`, `examples`), `figures_tables`, `submission_notes`.
3. Add the journal to the table in `CLAUDE.md` and `README.md`.
4. Run `pytest tests/test_guidelines.py` — it validates the schema of every guideline
   and checks the journal is documented.

Load guidelines only through `utils.guidelines.load_guideline()`; it resolves aliases,
caches files and raises a helpful `UnknownJournalError` for typos. Custom journals can
also live outside the repository via the `PAPERFACTORY_GUIDELINES_DIR` environment variable.

## Contributing Guidelines

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feat/my-feature`
3. **Write tests** for new functionality (bug fixes need a regression test)
4. **Run** `ruff check . && ruff format --check . && pytest`
5. **Commit** with a clear message and add an entry to `CHANGELOG.md`
6. **Push** and open a Pull Request

## What to Contribute

- **New journal guidelines** — see *Adding a Journal* above
- **New paper templates** — Add templates in `utils/template_system.py`
- **Bug fixes** — Check the Issues tab
- **Data source connectors** — Add to `utils/data_sources.py`
- **Test coverage** — More edge case tests

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
