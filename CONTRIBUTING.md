# Contributing to PaperFactory

Thank you for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/concrete-sangminlee/paperfactory.git paperfactory
cd paperfactory
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

### Pre-commit Hooks

```bash
pre-commit install
```

## Contributing Guidelines

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feat/my-feature`
3. **Write tests** for new functionality
4. **Run tests** to ensure everything passes: `pytest tests/ -v`
5. **Commit** with a clear message
6. **Push** and open a Pull Request

## What to Contribute

- **New journal guidelines** — Add a JSON file in `guidelines/`
- **New paper templates** — Add templates in `utils/template_system.py`
- **Bug fixes** — Check the Issues tab
- **Data source connectors** — Add to `utils/data_sources.py`
- **Test coverage** — More edge case tests

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
