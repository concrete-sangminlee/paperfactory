from utils.abstract_generator import generate_abstract, improve_abstract
from utils.ai_reviewer import review_paper
from utils.bibtex_import import import_bib, parse_bib_string
from utils.citation_converter import convert_style
from utils.data_sources import get_source, list_sources, suggest_sources
from utils.figure_utils import get_colors, get_figsize, save_figure, setup_style
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
from utils.equation_utils import (
    equations_to_text,
    extract_equations,
    number_equations,
    render_equation_image,
    save_equation_image,
)
from utils.word_generator import generate_word, load_guideline
