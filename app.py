import streamlit as st
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import PipelineOrchestrator, STEPS, JOURNAL_KEYS

st.set_page_config(
    page_title="PaperFactory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
        color: #1a1a2e;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ddd;
    }
    .step-active {
        border-left-color: #ff6b35;
        background: #fff5f0;
    }
    .step-done {
        border-left-color: #28a745;
        background: #f0fff4;
    }
    .step-waiting {
        border-left-color: #ffc107;
        background: #fffef0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">PaperFactory</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Academic Paper Generator for Structural Engineering</p>', unsafe_allow_html=True)

# Initialize session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "pipeline_started" not in st.session_state:
    st.session_state.pipeline_started = False
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False
if "step_status" not in st.session_state:
    st.session_state.step_status = {}
if "progress_messages" not in st.session_state:
    st.session_state.progress_messages = []
if "current_running" not in st.session_state:
    st.session_state.current_running = False

# Sidebar
with st.sidebar:
    st.header("Settings")

    topic = st.text_area(
        "Research Topic",
        placeholder="e.g., Deep learning-based seismic damage detection in reinforced concrete structures",
        height=100,
    )

    journal = st.selectbox(
        "Target Journal",
        options=list(JOURNAL_KEYS.keys()),
        index=0,
    )

    st.divider()

    auto_mode = st.checkbox(
        "Full Auto Mode",
        value=False,
        help="Skip all approval gates and run the entire pipeline automatically",
    )
    st.session_state.auto_mode = auto_mode

    st.divider()

    start_btn = st.button(
        "Start Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not topic or st.session_state.current_running,
    )

    if start_btn and topic:
        st.session_state.orchestrator = PipelineOrchestrator(topic, journal)
        st.session_state.pipeline_started = True
        st.session_state.step_status = {s["id"]: "pending" for s in STEPS}
        st.session_state.progress_messages = []
        st.session_state.current_running = False

    st.divider()
    st.caption("PaperFactory v1.0")
    st.caption("Powered by Claude Code")


def progress_callback(msg: str):
    st.session_state.progress_messages.append(msg)


# Main content
if not st.session_state.pipeline_started:
    # Landing page
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### Step 1-2
        **Literature Review & Research Design**

        AI conducts comprehensive literature survey and designs research methodology.
        """)
    with col2:
        st.markdown("""
        ### Step 3-4
        **Code Execution & Analysis**

        Generates and runs research code, then analyzes results with scholarly rigor.
        """)
    with col3:
        st.markdown("""
        ### Step 5
        **Paper Writing**

        Produces a journal-formatted Word document ready for submission.
        """)

    st.divider()

    st.markdown("#### Supported Journals")
    journal_cols = st.columns(5)
    journal_names = list(JOURNAL_KEYS.keys())
    for i, col in enumerate(journal_cols):
        with col:
            st.info(journal_names[i])

else:
    orch = st.session_state.orchestrator

    # Progress sidebar
    st.markdown("### Pipeline Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step cards
    step_cols = st.columns(5)
    for i, step in enumerate(STEPS):
        with step_cols[i]:
            step_st = st.session_state.step_status.get(step["id"], "pending")
            if step_st == "completed":
                icon = "✅"
            elif step_st == "running":
                icon = "🔄"
            elif step_st == "waiting":
                icon = "⏸️"
            else:
                icon = "⏳"
            st.markdown(f"**{icon} Step {i+1}**")
            st.caption(step["name_ko"])

    st.divider()

    # Update progress bar
    progress_pct = orch.get_progress()["percentage"]
    progress_bar.progress(progress_pct / 100)
    status_text.text(f"Progress: {progress_pct}%")

    # Show progress messages
    if st.session_state.progress_messages:
        with st.expander("Progress Log", expanded=False):
            for msg in st.session_state.progress_messages:
                st.text(f"  → {msg}")

    # Run pipeline logic
    current_step = orch.get_current_step()

    if current_step and not st.session_state.current_running:
        step_id = current_step["id"]
        step_idx = orch.current_step

        # Check if we need approval for previous step
        prev_step_id = STEPS[step_idx - 1]["id"] if step_idx > 0 else None
        prev_status = st.session_state.step_status.get(prev_step_id, "pending") if prev_step_id else "completed"

        if prev_status == "waiting" and not st.session_state.auto_mode:
            # Show approval gate
            st.markdown(f"### Step {step_idx} Results: {STEPS[step_idx-1]['name_ko']}")
            summary = orch.get_step_result_summary(prev_step_id)
            st.info(summary)

            col_approve, col_revise = st.columns(2)
            with col_approve:
                if st.button("Approve & Continue ✅", type="primary", use_container_width=True):
                    st.session_state.step_status[prev_step_id] = "completed"
                    st.rerun()
            with col_revise:
                feedback = st.text_area("Revision feedback (optional):")
                if st.button("Revise 🔄", use_container_width=True):
                    st.session_state.step_status[prev_step_id] = "pending"
                    orch.current_step -= 1
                    st.rerun()

        elif prev_status in ("completed", "pending") or st.session_state.auto_mode:
            # Run current step
            if prev_status == "waiting" and st.session_state.auto_mode:
                st.session_state.step_status[prev_step_id] = "completed"

            st.markdown(f"### Running: Step {step_idx + 1} — {current_step['name_ko']}")
            st.session_state.step_status[step_id] = "running"

            with st.spinner(f"Running {current_step['name']}..."):
                st.session_state.current_running = True
                try:
                    result = orch.run_step(step_id, progress_callback=progress_callback)

                    if st.session_state.auto_mode:
                        st.session_state.step_status[step_id] = "completed"
                    else:
                        st.session_state.step_status[step_id] = "waiting"

                    st.session_state.current_running = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in {current_step['name']}: {str(e)}")
                    st.session_state.current_running = False
                    st.session_state.step_status[step_id] = "error"

    elif orch.status == "completed" or orch.current_step >= len(STEPS):
        # Pipeline complete
        last_step = STEPS[-1]["id"]
        last_status = st.session_state.step_status.get(last_step, "pending")

        if last_status == "waiting" and not st.session_state.auto_mode:
            st.markdown(f"### Step {len(STEPS)} Results: {STEPS[-1]['name_ko']}")
            summary = orch.get_step_result_summary(last_step)
            st.info(summary)
            if st.button("Approve Final ✅", type="primary"):
                st.session_state.step_status[last_step] = "completed"
                st.rerun()
        else:
            st.session_state.step_status[last_step] = "completed"
            st.success("Pipeline completed successfully!")

            output_path = orch.get_output_path()
            if output_path and os.path.exists(output_path):
                st.markdown("### Download Your Paper")

                with open(output_path, "rb") as f:
                    file_bytes = f.read()

                st.download_button(
                    label="Download Word Document 📄",
                    data=file_bytes,
                    file_name=os.path.basename(output_path),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary",
                    use_container_width=True,
                )

                st.markdown(f"**File:** `{output_path}`")
                st.markdown(f"**Figures included:** {orch.results.get('paper', {}).get('figures_included', 0)}")

                # Show paper content preview
                paper_content = orch.results.get("paper", {}).get("paper_content", {})
                if paper_content:
                    with st.expander("Paper Preview"):
                        st.markdown(f"**Title:** {paper_content.get('title', '')}")
                        st.markdown(f"**Abstract:** {paper_content.get('abstract', '')}")
                        for sec in paper_content.get("sections", []):
                            st.markdown(f"**{sec.get('heading', '')}**")
                            st.markdown(sec.get("content", "")[:500] + "...")

            # Show all figures
            code_results = orch.results.get("code", {})
            figures = code_results.get("figures", [])
            if figures:
                with st.expander("Generated Figures"):
                    for fig_path in figures:
                        if os.path.exists(fig_path):
                            st.image(fig_path, caption=os.path.basename(fig_path))

            # Reset button
            if st.button("Start New Paper 🔄"):
                st.session_state.pipeline_started = False
                st.session_state.orchestrator = None
                st.session_state.step_status = {}
                st.session_state.progress_messages = []
                st.rerun()
