from utils.word_generator import generate_word, load_guideline
from utils.latex_generator import generate_latex
from utils.figure_utils import setup_style, save_figure, get_colors, get_figsize
from utils.reference_utils import validate_references, check_duplicates, detect_citation_style
from utils.quality_checker import check_paper
from utils.table_utils import create_table_figure, save_table_csv, format_number, format_mean_std
from utils.pdf_generator import generate_pdf
from utils.revision_tracker import create_revision, address_comment, generate_response_letter, track_changes
from utils.submission_utils import submission_checklist, generate_cover_letter, reformat_paper
from utils.research_advisor import recommend_statistical_tests, recommend_figure_type
from utils.ai_reviewer import review_paper
from utils.data_sources import list_sources, get_source, suggest_sources
from utils.citation_converter import convert_style
from utils.template_system import list_templates, get_template, generate_skeleton
