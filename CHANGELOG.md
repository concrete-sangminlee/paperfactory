# Changelog

## [1.1.0] - 2026-11-25

Repository-wide hardening pass: every module was reviewed, each bug was reproduced
before being fixed, and each fix has a regression test (test suite: 245 → 1,000+ tests,
run on Python 3.10–3.13 × Ubuntu/macOS plus Windows).

### Fixed — application and packaging
- **Web UI crashed** on the Quality Check, AI Review and Submission tabs (`json` was never imported in `app.py`).
- **`pip install .` failed**: `pyproject.toml` named a non-existent build backend; `cli.py` was not shipped, so the `paperfactory` console script could not import.
- **CLI** returned exit code 0 on errors and printed tracebacks for unknown journals or bad JSON; `review` ignored `--figures`.
- **Pipeline**: `complete_step()` after completion pushed progress above 100%; runs started in the same second overwrote each other; unknown journal keys were accepted; state paths depended on the working directory.

### Fixed — document generators (Word / LaTeX / PDF)
- Guideline rules that were ignored: heading numbering, line spacing, line/page numbers, justification, paper size, back-matter order and spelling, figure/table placement.
- LaTeX: three Elsevier journals and the `jcsr` alias got the plain `article` class; references were never cited, so BibTeX printed nothing; tables, highlights and acknowledgments were dropped; `&`, `%`, `<`, `>`, Greek letters and Korean text broke compilation; one `.bib` was shared between journals.
- PDF: the guideline was never loaded; Greek/math symbols and Hangul were mangled; wide tables overflowed the page; figures lost their aspect ratio.
- Crashes on `None` values, list-valued keywords/authors, ragged tables and missing figures; captions dropped or double-numbered.
- The competing-interest declaration required by seven journals is now rendered.

### Fixed — quality, submission and reference tools
- `check_paper` only checked for Introduction/Conclusion; it now maps the journal's required sections (with synonyms, numbering and Korean headings) and reports front/back matter separately.
- Year detection picked digits out of ASCE DOIs and page ranges (2025 references counted as 0% recent).
- Cover letters falsely claimed a graphical abstract was provided and garbled decimals; `reformat_paper` mutated its input and never converted the reference style.
- `ai_reviewer` flagged "Materials and Methods" as a missing methodology and returned "Major revision" for papers with only minor issues.
- Citation conversion lost authors/titles for ASCE references and double-numbered fall-backs; BibTeX import dropped `@string` macros, single-line and parenthesised entries, accents and "Jr." names.

### Fixed — analysis helpers
- `research_advisor` tested normality on pooled data, so normal groups with different means got Mann-Whitney (and a flaky CI test); now per-group with Welch / repeated-measures options.
- `abstract_generator` cut abstracts at decimal points or mid-sentence; `paper_analytics` counted Korean text as 0 words and split sentences on "e.g." and "0.95"; `table_utils` printed `\u00b1` instead of "±"; `data_sources` matched "wind" inside "window"; `save_figure` leaked figures; `equation_utils` treated `\$` as an equation delimiter.

### Fixed — demos
- Both demo manuscripts stated results that contradicted their own computed output (e.g. "RF achieved the highest accuracy" while reporting a higher DNN accuracy). Every comparative claim is now derived from the computed results, with confidence intervals and a McNemar significance test in the seismic demo.

### Added
- `utils/guidelines.py`: single cached guideline registry with case-insensitive keys, aliases (`jcsr`, `cbm`, …), "Did you mean …?" errors, schema validation and a `PAPERFACTORY_GUIDELINES_DIR` override.
- Pipeline `revert_to()` (e.g. Step 4 → Step 3), atomic state saves, `find_state()` / `list_runs()`, `PAPERFACTORY_RUNS_DIR`.
- CLI: 18 commands (new: `runs`, `checklist`, `export`, `reformat`, `validate-refs`), `--json`, `--strict`, `--version`.
- Web UI: JSON upload, Export (Word/PDF download) and Journals tabs.
- CI: ruff lint/format gate, coverage, CLI smoke test, sdist/wheel and Docker builds, and a TeX Live job that compiles the LaTeX export for all 15 journals plus a Korean paper.
- Docker image runs as a non-root user with a health check; `requirements-dev.txt`; pre-commit on ruff 0.16.9 with a pre-push test hook.

### Changed
- Unknown journal keys now raise `UnknownJournalError` (a `FileNotFoundError`/`ValueError` subclass) in every module instead of being silently accepted.
- `save_figure()` closes the figure by default (`close=False` keeps it open).
- The LaTeX `.bib` file is named per journal (`<stem>_<journal>.bib`).
- Submission checklist items carry a `required` flag; new required items (duplicate references, reference style, ASCE practical applications, keyword range, title length) can turn `ready` to `False` for papers that passed before.

## [1.0.0] - 2026-03-28

### Added
- **12 utility modules**: word_generator, latex_generator, pdf_generator, figure_utils, table_utils, reference_utils, quality_checker, submission_utils, ai_reviewer, research_advisor, revision_tracker, data_sources, citation_converter, template_system
- **15 journal guidelines**: ASCE JSE, ACI SJ, JWEIA, JBE, Engineering Structures, EESD, Thin-Walled, CCC, C&S, Automation in Construction, Structural Safety, Construction & Building Materials, KSCE JCE, Buildings (MDPI), Journal of Constructional Steel Research (key `steel_comp_struct`)
- **Pipeline orchestrator**: 5-step pipeline with state persistence and resume
- **CLI tool**: 8 commands (new, status, resume, check, review, cover-letter, sources, journals)
- **Web UI**: Streamlit app with 5 tabs
- **PaperBanana integration**: Nano Banana Pro for methodology diagrams
- **2 complete demos**: TPU wind pressure (JWEIA), Seismic damage (Eng. Structures)
- **4 paper templates**: ML comparison, experimental, numerical, review
- **CI/CD**: GitHub Actions with cross-platform testing
- **Docker support**: Dockerfile for containerized deployment
- **163+ tests** with full integration test suite

### Infrastructure
- pyproject.toml for pip installation
- Pre-commit hooks with ruff
- CONTRIBUTING.md guide
- Example paper JSON for CLI testing

## [0.1.0] - 2026-03-23

### Added
- Initial prototype with Streamlit UI
- First paper generation (wind pressure topic)
- Basic word_generator and figure_utils
- 5 journal guidelines (ASCE, ACI, JWEIA, JBE, Engineering Structures)
