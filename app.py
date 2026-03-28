"""PaperFactory Web UI — Streamlit-based interface.

Run with: streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    print("Then: streamlit run app.py")
    sys.exit(1)

from pipeline.orchestrator import PaperPipeline
from utils.ai_reviewer import review_paper
from utils.data_sources import list_sources, suggest_sources
from utils.quality_checker import check_paper
from utils.submission_utils import generate_cover_letter, submission_checklist

st.set_page_config(page_title="PaperFactory", page_icon="📝", layout="wide")

st.title("📝 PaperFactory")
st.markdown("**AI Research Paper Agent for Civil Engineering**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pipeline", "Quality Check", "AI Review", "Data Sources", "Submission"
])

# ── Tab 1: Pipeline Status ───────────────────────────────────────────────────
with tab1:
    st.header("Paper Generation Pipeline")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_area("Research Topic", placeholder="e.g., ML-based wind pressure prediction on low-rise buildings")
    with col2:
        journals = [
            "jweia", "eng_structures", "asce_jse", "aci_sj", "jbe",
            "eesd", "thin_walled", "cem_con_comp", "comput_struct", "autom_constr",
            "struct_safety", "const_build_mat", "ksce_jce", "buildings_mdpi", "steel_comp_struct",
        ]
        journal = st.selectbox("Target Journal", journals)

    if topic:
        pipeline = PaperPipeline(topic, journal)
        st.code(pipeline.show_status())
        st.progress(pipeline.progress)

        # Suggest data sources
        sources = suggest_sources(topic)
        if sources:
            st.subheader("Recommended Data Sources")
            for s in sources[:3]:
                st.markdown(f"- **[{s['name']}]({s['url']})** — {s['description'][:100]}...")

# ── Tab 2: Quality Check ────────────────────────────────────────────────────
with tab2:
    st.header("Paper Quality Check")
    st.markdown("Upload or paste paper content as JSON to run quality checks.")

    json_input = st.text_area("Paper Content (JSON)", height=300,
                              placeholder='{"title": "...", "abstract": "...", "sections": [...]}')
    qc_journal = st.selectbox("Journal for quality check", journals, key="qc_journal")

    if st.button("Run Quality Check") and json_input:
        try:
            paper = json.loads(json_input)
            result = check_paper(paper, qc_journal)
            st.metric("Quality Score", f"{result['score']}/100")
            st.metric("Status", "PASS ✅" if result["passed"] else "FAIL ❌")
            for check in result["checks"]:
                icon = "✅" if check["passed"] else "❌"
                st.markdown(f"{icon} **[{check['severity'].upper()}]** {check['message']}")
        except json.JSONDecodeError:
            st.error("Invalid JSON input")

# ── Tab 3: AI Review ────────────────────────────────────────────────────────
with tab3:
    st.header("AI Reviewer Simulation")
    st.markdown("Pre-submission peer review simulation.")

    review_json = st.text_area("Paper Content (JSON)", height=300, key="review_json",
                               placeholder='{"title": "...", "sections": [...]}')
    review_journal = st.selectbox("Journal", journals, key="review_journal")

    if st.button("Run AI Review") and review_json:
        try:
            paper = json.loads(review_json)
            result = review_paper(paper, review_journal)
            st.metric("Decision", result["decision"])
            col1, col2 = st.columns(2)
            col1.metric("Major Issues", result["major_issues"])
            col2.metric("Minor Issues", result["minor_issues"])
            for c in result["comments"]:
                severity_color = "🔴" if c["severity"] == "major" else "🟡"
                st.markdown(f"{severity_color} **[{c['section']}]** {c['comment']}")
        except json.JSONDecodeError:
            st.error("Invalid JSON input")

# ── Tab 4: Data Sources ─────────────────────────────────────────────────────
with tab4:
    st.header("Research Data Sources")

    field_filter = st.selectbox("Filter by field", [
        "All", "wind_engineering", "earthquake_engineering",
        "structural_health_monitoring", "coastal_engineering",
    ])

    if field_filter == "All":
        sources = list_sources()
    else:
        sources = list_sources(field_filter)

    for s in sources:
        with st.expander(f"📊 {s['name']} ({s['access']})"):
            st.markdown(f"**Organization:** {s['organization']}")
            st.markdown(f"**URL:** [{s['url']}]({s['url']})")
            st.markdown(f"**Description:** {s['description']}")
            st.markdown(f"**Data types:** {', '.join(s.get('data_types', []))}")
            st.markdown(f"**Format:** {s.get('format', 'N/A')}")

# ── Tab 5: Submission ────────────────────────────────────────────────────────
with tab5:
    st.header("Submission Preparation")

    sub_tab1, sub_tab2 = st.tabs(["Checklist", "Cover Letter"])

    with sub_tab1:
        st.markdown("Paste paper JSON to check submission readiness.")
        sub_json = st.text_area("Paper Content (JSON)", height=200, key="sub_json")
        sub_journal = st.selectbox("Journal", journals, key="sub_journal")
        if st.button("Check Readiness") and sub_json:
            try:
                paper = json.loads(sub_json)
                result = submission_checklist(paper, sub_journal)
                st.metric("Ready to Submit", "YES ✅" if result["ready"] else "NO ❌")
                for item in result["items"]:
                    icon = "✅" if item["passed"] else "❌"
                    note = f" — {item['note']}" if item.get("note") else ""
                    st.markdown(f"{icon} {item['description']}{note}")
            except json.JSONDecodeError:
                st.error("Invalid JSON")

    with sub_tab2:
        cl_json = st.text_area("Paper Content (JSON)", height=200, key="cl_json")
        cl_journal = st.selectbox("Journal", journals, key="cl_journal")
        editor = st.text_input("Editor Name", "Editor-in-Chief")
        if st.button("Generate Cover Letter") and cl_json:
            try:
                paper = json.loads(cl_json)
                letter = generate_cover_letter(paper, cl_journal, editor)
                st.text_area("Cover Letter", letter, height=400)
            except json.JSONDecodeError:
                st.error("Invalid JSON")

# Need json import at top
