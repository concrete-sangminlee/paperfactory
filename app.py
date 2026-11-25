"""PaperFactory Web UI — Streamlit-based interface.

Run with: streamlit run app.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    print("Then: streamlit run app.py")
    sys.exit(1)

from pipeline.orchestrator import STEP_DESCRIPTIONS, WORK_STEPS, list_runs  # noqa: E402
from utils.ai_reviewer import review_paper  # noqa: E402
from utils.data_sources import list_sources, suggest_sources  # noqa: E402
from utils.guidelines import list_journals  # noqa: E402
from utils.quality_checker import check_paper  # noqa: E402
from utils.submission_utils import generate_cover_letter, submission_checklist  # noqa: E402

st.set_page_config(page_title="PaperFactory", page_icon="📝", layout="wide")

JOURNALS = list_journals()
JOURNAL_KEYS = [j["key"] for j in JOURNALS]
JOURNAL_LABELS = {j["key"]: f"{j['key']} — {j['name']}" for j in JOURNALS}


def journal_select(label: str, key: str) -> str:
    """Selectbox over every bundled journal guideline."""
    default = JOURNAL_KEYS.index("jweia") if "jweia" in JOURNAL_KEYS else 0
    return st.selectbox(label, JOURNAL_KEYS, index=default, key=key, format_func=JOURNAL_LABELS.get)


def paper_input(key: str, height: int = 250):
    """Paper JSON from an uploaded file or a text area. Returns dict or None."""
    uploaded = st.file_uploader("Upload paper_content JSON", type=["json"], key=f"{key}_file")
    text = st.text_area(
        "…or paste paper_content JSON",
        height=height,
        key=f"{key}_text",
        placeholder='{"title": "...", "abstract": "...", "sections": [...]}',
    )
    raw = uploaded.getvalue().decode("utf-8") if uploaded is not None else text
    if not raw or not raw.strip():
        return None
    try:
        paper = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return None
    if not isinstance(paper, dict):
        st.error("paper_content must be a JSON object")
        return None
    return paper


st.title("📝 PaperFactory")
st.markdown("**AI Research Paper Agent for Civil Engineering**")

tabs = st.tabs(
    ["Pipeline", "Quality Check", "AI Review", "Data Sources", "Submission", "Export", "Journals"]
)

# ── Tab 1: Pipeline ──────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Paper Generation Pipeline")
    st.markdown(
        "Open Claude Code in this repository and give it a **topic** and **target journal**; "
        "it runs the five steps below with your approval after each one."
    )
    for i, step in enumerate(WORK_STEPS, 1):
        st.markdown(f"{i}. **{step.value.replace('_', ' ').title()}** — {STEP_DESCRIPTIONS[step]}")

    topic = st.text_area(
        "Research Topic",
        placeholder="e.g., ML-based wind pressure prediction on low-rise buildings",
    )
    if topic.strip():
        sources = suggest_sources(topic)
        if sources:
            st.subheader("Recommended Data Sources")
            for s in sources[:3]:
                st.markdown(f"- **[{s['name']}]({s['url']})** — {s['description'][:100]}...")
        else:
            st.info("No matching public data source; synthetic data may be needed.")

    runs = list_runs()
    st.subheader("Saved Runs")
    if runs:
        for r in runs:
            st.markdown(
                f"- `{r['run_id']}` · **{r['topic']}** · {r['journal_key']} · "
                f"{r['current_step']} ({r['progress']:.0%})"
            )
            st.progress(r["progress"])
    else:
        st.caption("No saved runs yet. Create one with `paperfactory new`.")

# ── Tab 2: Quality Check ────────────────────────────────────────────────────
with tabs[1]:
    st.header("Paper Quality Check")
    paper = paper_input("qc", height=300)
    qc_journal = journal_select("Journal for quality check", key="qc_journal")

    if st.button("Run Quality Check", key="qc_run"):
        if paper is None:
            st.warning("Provide paper_content JSON first.")
        else:
            result = check_paper(paper, qc_journal)
            col1, col2 = st.columns(2)
            col1.metric("Quality Score", f"{result['score']}/100")
            col2.metric("Status", "PASS ✅" if result["passed"] else "FAIL ❌")
            for check in result["checks"]:
                icon = "✅" if check["passed"] else "❌"
                st.markdown(f"{icon} **[{check['severity'].upper()}]** {check['message']}")

# ── Tab 3: AI Review ────────────────────────────────────────────────────────
with tabs[2]:
    st.header("AI Reviewer Simulation")
    st.markdown("Pre-submission peer review simulation.")
    paper = paper_input("review", height=300)
    review_journal = journal_select("Journal", key="review_journal")

    if st.button("Run AI Review", key="review_run"):
        if paper is None:
            st.warning("Provide paper_content JSON first.")
        else:
            result = review_paper(paper, review_journal)
            st.metric("Decision", result["decision"])
            col1, col2 = st.columns(2)
            col1.metric("Major Issues", result["major_issues"])
            col2.metric("Minor Issues", result["minor_issues"])
            for c in result["comments"]:
                severity_color = "🔴" if c["severity"] == "major" else "🟡"
                st.markdown(f"{severity_color} **[{c['section']}]** {c['comment']}")

# ── Tab 4: Data Sources ─────────────────────────────────────────────────────
with tabs[3]:
    st.header("Research Data Sources")
    all_sources = list_sources()
    fields = sorted({f for s in all_sources for f in s.get("fields", [])})
    field_filter = st.selectbox("Filter by field", ["All", *fields])
    sources = all_sources if field_filter == "All" else list_sources(field_filter)

    for s in sources:
        with st.expander(f"📊 {s['name']} ({s['access']})"):
            st.markdown(f"**Organization:** {s.get('organization', 'N/A')}")
            st.markdown(f"**URL:** [{s['url']}]({s['url']})")
            st.markdown(f"**Description:** {s['description']}")
            st.markdown(f"**Data types:** {', '.join(s.get('data_types', []))}")
            st.markdown(f"**Format:** {s.get('format', 'N/A')}")

# ── Tab 5: Submission ────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Submission Preparation")
    sub_tab1, sub_tab2 = st.tabs(["Checklist", "Cover Letter"])

    with sub_tab1:
        paper = paper_input("sub", height=200)
        sub_journal = journal_select("Journal", key="sub_journal")
        if st.button("Check Readiness", key="sub_run"):
            if paper is None:
                st.warning("Provide paper_content JSON first.")
            else:
                result = submission_checklist(paper, sub_journal)
                st.metric("Ready to Submit", "YES ✅" if result["ready"] else "NO ❌")
                for item in result["items"]:
                    icon = "✅" if item["passed"] else "❌"
                    note = f" — {item['note']}" if item.get("note") else ""
                    st.markdown(f"{icon} {item['description']}{note}")

    with sub_tab2:
        paper = paper_input("cl", height=200)
        cl_journal = journal_select("Journal", key="cl_journal")
        editor = st.text_input("Editor Name", "Editor-in-Chief")
        if st.button("Generate Cover Letter", key="cl_run"):
            if paper is None:
                st.warning("Provide paper_content JSON first.")
            else:
                letter = generate_cover_letter(paper, cl_journal, editor)
                st.text_area("Cover Letter", letter, height=400, key="cl_out")
                st.download_button("Download cover letter", letter, file_name="cover_letter.txt")

# ── Tab 6: Export ────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Export Manuscript")
    paper = paper_input("export", height=200)
    export_journal = journal_select("Journal", key="export_journal")
    fmt = st.radio("Format", ["Word (.docx)", "PDF (.pdf)"], horizontal=True, key="export_fmt")
    if st.button("Generate", key="export_run"):
        if paper is None:
            st.warning("Provide paper_content JSON first.")
        else:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    if fmt.startswith("Word"):
                        from utils.word_generator import generate_word

                        path = generate_word(paper, export_journal, None, output_dir=tmp)
                        mime = (
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        )
                    else:
                        from utils.pdf_generator import generate_pdf

                        path = generate_pdf(paper, export_journal, None, output_dir=tmp)
                        mime = "application/pdf"
                    with open(path, "rb") as f:
                        data = f.read()
                st.success(f"Generated {os.path.basename(path)}")
                st.download_button("Download", data, file_name=os.path.basename(path), mime=mime)
            except (KeyError, TypeError, ValueError, OSError) as e:
                st.error(f"Export failed: {type(e).__name__}: {e}")

# ── Tab 7: Journals ──────────────────────────────────────────────────────────
with tabs[6]:
    st.header("Supported Journals")
    st.dataframe(
        [
            {
                "Key": j["key"],
                "Journal": j["name"],
                "Publisher": j["publisher"],
                "Reference style": j["reference_style"],
                "Abstract limit": j["abstract_word_limit"],
                "Highlights": "required" if j["highlights_required"] else "",
                "Graphical abstract": "required" if j["graphical_abstract_required"] else "",
            }
            for j in JOURNALS
        ],
        hide_index=True,
    )
