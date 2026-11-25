"""PaperFactory utilities — public API facade.

Import from the individual modules (``from utils.word_generator import
generate_word``) or from this package (``from utils import generate_word``).
"""

from utils.abstract_generator import generate_abstract, improve_abstract
from utils.ai_reviewer import review_paper
from utils.bibtex_import import import_bib, parse_bib_string
from utils.citation_converter import convert_style
from utils.data_sources import get_source, list_sources, suggest_sources
from utils.equation_utils import (
    equations_to_text,
    extract_equations,
    number_equations,
    render_equation_image,
    save_equation_image,
)
from utils.figure_utils import get_colors, get_figsize, save_figure, setup_style
from utils.guidelines import (
    UnknownJournalError,
    available_journal_keys,
    list_journals,
    load_guideline,
    resolve_journal_key,
    validate_guideline,
)
from utils.latex_generator import generate_latex
from utils.paper_analytics import analyze_paper
from utils.pdf_generator import generate_pdf
from utils.quality_checker import check_paper
from utils.reference_utils import check_duplicates, detect_citation_style, validate_references
from utils.research_advisor import recommend_figure_type, recommend_statistical_tests
from utils.revision_tracker import (
    address_comment,
    create_revision,
    generate_response_letter,
    track_changes,
)
from utils.submission_utils import generate_cover_letter, reformat_paper, submission_checklist
from utils.table_utils import create_table_figure, format_mean_std, format_number, save_table_csv
from utils.template_system import generate_skeleton, get_template, list_templates
from utils.word_generator import generate_word

__all__ = [
    "UnknownJournalError",
    "address_comment",
    "analyze_paper",
    "available_journal_keys",
    "check_duplicates",
    "check_paper",
    "convert_style",
    "create_revision",
    "create_table_figure",
    "detect_citation_style",
    "equations_to_text",
    "extract_equations",
    "format_mean_std",
    "format_number",
    "generate_abstract",
    "generate_cover_letter",
    "generate_latex",
    "generate_pdf",
    "generate_response_letter",
    "generate_skeleton",
    "generate_word",
    "get_colors",
    "get_figsize",
    "get_source",
    "get_template",
    "import_bib",
    "improve_abstract",
    "list_journals",
    "list_sources",
    "list_templates",
    "load_guideline",
    "number_equations",
    "parse_bib_string",
    "recommend_figure_type",
    "recommend_statistical_tests",
    "reformat_paper",
    "render_equation_image",
    "resolve_journal_key",
    "review_paper",
    "save_equation_image",
    "save_figure",
    "save_table_csv",
    "setup_style",
    "submission_checklist",
    "suggest_sources",
    "track_changes",
    "validate_guideline",
    "validate_references",
]
