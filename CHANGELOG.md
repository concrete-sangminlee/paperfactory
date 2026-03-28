# Changelog

## [1.0.0] - 2026-03-28

### Added
- **12 utility modules**: word_generator, latex_generator, pdf_generator, figure_utils, table_utils, reference_utils, quality_checker, submission_utils, ai_reviewer, research_advisor, revision_tracker, data_sources, citation_converter, template_system
- **15 journal guidelines**: ASCE JSE, ACI SJ, JWEIA, JBE, Engineering Structures, EESD, Thin-Walled, CCC, C&S, Automation in Construction, Structural Safety, Construction & Building Materials, KSCE JCE, Buildings (MDPI), Steel & Composite Structures
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
