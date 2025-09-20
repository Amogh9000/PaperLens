import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import base64
from PIL import Image, ImageDraw
import altair as alt
import json
from mock_data import SUBJECTS, make_single, generate_batch
import engine
import os
import hashlib
import random
import db

# Page configuration
st.set_page_config(
    page_title="PaperLens - OMR Evaluation System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    with st.expander("Admin"):
        if st.button("Reset DB (Drop & Recreate)"):
            try:
                db.reset_database()
                st.session_state.has_real_data = False
                st.session_state.batch_df = None
                st.session_state.batch_results = []
                st.success("Database reset.")
            except Exception as e:
                st.error(f"Reset failed: {e}")
        cols = st.text_input("Columns to drop (comma-separated)", "")
        if st.button("Drop Columns"):
            try:
                targets = [c.strip() for c in cols.split(",") if c.strip()]
                db.migrate_drop_columns(targets)
                st.success(f"Dropped: {targets}")
            except Exception as e:
                st.error(f"Migration failed: {e}")

# Base CSS (light theme) for better styling
st.markdown("""
<style>
    :root {
        --bg: #F7FAFC;
        --card: #FFFFFF;
        --muted: #E2E8F0;
        --accent: #3182CE;
        --accent-600: #2B6CB0;
        --success: #38A169;
        --warning: #D69E2E;
        --danger: #E53E3E;
        --text: #2D3748;
        --subtext: #718096;
    }
    .topbar {
        background: var(--card);
        border: 1px solid var(--muted);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.75rem 0 1rem 0;
        box-shadow: none;
    }
    .brand { font-weight: 800; color: var(--text); letter-spacing: 0.3px; font-size: 1.5rem; }
    .subtitle { color: var(--subtext); font-size: 1rem; font-weight: 500; }
    .card {
        background: var(--card);
        border: 1px solid var(--muted);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: none;
        margin-bottom: 1rem;
    }
    .score-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 45%, #0ea5e9 100%);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid transparent;
        text-align: center;
        margin: 1rem 0;
        box-shadow: none;
    }
    .score-card h1 { margin: 0; font-size: 3rem; color: #ffffff; }
    .score-card h3 { margin: 0.25rem 0 0.5rem; color: rgba(255,255,255,0.8); }
    .subject-card {
        background: var(--card);
        border: 1px solid var(--muted);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: none;
    }
    .progress-bg { width: 100%; height: 8px; background: #EDF2F7; border-radius: 999px; overflow: hidden; margin-top: 8px; }
    .progress-fill { height: 100%; border-radius: 999px; transition: width 300ms ease; background: linear-gradient(90deg, #34d399 0%, #3b82f6 100%); }
    .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1rem; }
    .stat-card { background: var(--card); border: 1px solid var(--muted); border-radius: 12px; padding: 1rem; box-shadow: none; }
    .stat-label { color: var(--subtext); font-size: 0.85rem; }
    .stat-value { color: var(--text); font-size: 1.5rem; font-weight: 700; }
    .stat-delta { font-size: 0.8rem; color: var(--subtext); }
    .upload-section {
        border: 2px dashed var(--muted);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0 0;
        background: #FAFBFC;
    }
    .success-message {
        background: #F0FFF4;
        color: #22543D;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #C6F6D5;
        margin: 1rem 0;
    }
    body, [data-testid="stAppViewContainer"] { background: var(--bg); }
    [data-testid="stSidebar"] { background: var(--card); }
    [data-testid="stSidebar"] * { color: var(--text); }
    h1, h2, h3, h4, h5, h6, p, span, label, div, .stText, .stMarkdown { color: var(--text) !important; }
    .stCaption, .st-emotion-cache-10trblm, .st-emotion-cache-ur2l3v { color: var(--subtext) !important; }
    .block-container { max-width: 1200px; padding-top: 1rem; }
    [role="tablist"] { border-bottom: 1px solid var(--muted); margin-bottom: 0.5rem; }
    [role="tab"] { color: var(--subtext); }
    [role="tab"][aria-selected="true"] { color: var(--text); border-bottom: 2px solid var(--accent); }
    div[data-baseweb="select"] { background: var(--card); border: 1px solid var(--muted); border-radius: 10px; color: var(--text); }
    div[data-baseweb="select"] * { color: var(--text); }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--card) !important;
        border: 1px dashed var(--muted) !important;
        border-radius: 12px !important;
        color: var(--subtext) !important;
    }
    [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] p { color: var(--subtext) !important; }
    [data-testid="stFileUploader"] [data-testid="stWidgetLabelHelp"] ~ div { background: transparent !important; }
    [data-testid="stFileUploader"] button { background: var(--card) !important; border: 1px solid var(--muted) !important; color: var(--text) !important; border-radius: 10px !important; }
    [data-testid="stFileUploader"] button:hover { background: #F7FAFC !important; }
    .stButton > button {
        background: var(--accent);
        color: #fff;
        border-radius: 10px;
        border: 1px solid var(--accent-600);
        padding: 0.6rem 1rem;
    }
    .stButton > button:hover { background: var(--accent-600); }
    .stAlert { background: var(--card); border: 1px solid var(--muted); border-left: 4px solid var(--accent); color: var(--text); }
    @media (max-width: 1024px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 640px) { .stat-grid { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

# Dummy dashboard seed (only used in Demo Mode)
DEMO_DASHBOARD_DATA = {
    "total_students": 156,
    "average_score": 78.4,
    "grade_distribution": {
        "A+ (90-100)": 23,
        "A (80-89)": 45,
        "B (70-79)": 52,
        "C (60-69)": 28,
        "D (50-59)": 8
    }
}

def create_dummy_annotated_image(seed: int = 42, header_text: str = "AUTOMATED OMR EVALUATION SHEET", student_id: str = "-"):
    """Create a dummy OMR sheet with annotations (deterministic with seed)."""
    rng = random.Random(seed)
    img = Image.new('RGB', (600, 800), 'white')
    draw = ImageDraw.Draw(img)
    # Header
    draw.rectangle([50, 50, 550, 100], outline='black', width=2)
    draw.text((60, 65), header_text, fill='black')
    # Student info
    draw.rectangle([50, 120, 550, 180], outline='black', width=1)
    draw.text((60, 135), f"Student ID: {student_id}", fill='black')
    draw.text((60, 155), "Total Score: --/100", fill='black')
    # Sections
    y_start = 200
    subjects = ["Mathematics", "English", "Science", "Logical Reasoning", "General Knowledge"]
    for i, subject in enumerate(subjects):
        draw.rectangle([50, y_start + i*100, 550, y_start + i*100 + 30], fill='lightgray', outline='black')
        draw.text((60, y_start + i*100 + 8), f"{subject}", fill='black')
        for j in range(5):
            x_pos = 70 + j * 90
            y_pos = y_start + i*100 + 45
            draw.text((x_pos, y_pos), f"Q{j+1}", fill='black')
            correct_idx = rng.randint(0, 3)
            for k, option in enumerate(['A', 'B', 'C', 'D']):
                bubble_x = x_pos + 10 + k * 15
                bubble_y = y_pos + 15
                if k == correct_idx:
                    color, fill_color = 'green', 'lightgreen'
                    draw.ellipse([bubble_x-5, bubble_y-5, bubble_x+5, bubble_y+5], fill=fill_color, outline=color, width=2)
                else:
                    draw.ellipse([bubble_x-5, bubble_y-5, bubble_x+5, bubble_y+5], outline='gray', width=1)
    return img

def get_cached_annotated_image(result: dict):
    """Return a cached deterministic annotated image for the given result."""
    if 'annotated_cache' not in st.session_state:
        st.session_state.annotated_cache = {}
    sid = result.get('student_id', 'UNKNOWN')
    key = hashlib.md5(sid.encode()).hexdigest()
    if key not in st.session_state.annotated_cache:
        seed = int(key[:8], 16)
        st.session_state.annotated_cache[key] = create_dummy_annotated_image(seed=seed, student_id=sid)
    return st.session_state.annotated_cache[key]

def init_state():
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = None
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'answer_key' not in st.session_state:
        st.session_state.answer_key = None
    if 'current_overlay' not in st.session_state:
        st.session_state.current_overlay = None
    if 'key_set' not in st.session_state:
        st.session_state.key_set = 'A'
    if 'has_real_data' not in st.session_state:
        st.session_state.has_real_data = False  # becomes True after a successful DB save

def load_default_key():
    if st.session_state.answer_key is None:
        key_file = f"set_{st.session_state.key_set}.json"
        key_path = os.path.join(os.path.dirname(__file__), 'maps', key_file)
        try:
            with open(key_path, 'r', encoding='utf-8') as f:
                st.session_state.answer_key = json.load(f)
        except Exception:
            st.session_state.answer_key = {
                "subjects": SUBJECTS,
                "answer_key": {s: ["A"]*20 for s in SUBJECTS}
            }

def get_grade(percentage):
    if percentage >= 90:
        return "A+"
    elif percentage >= 80:
        return "A"
    elif percentage >= 70:
        return "B"
    elif percentage >= 60:
        return "C"
    elif percentage >= 50:
        return "D"
    else:
        return "F"

def main():
    init_state()
    load_default_key()

    # Header
    st.markdown("""
    <div class="topbar">
        <div class="brand">PaperLens</div>
        <div class="subtitle">Intelligent OMR Evaluation</div>
    </div>
    """, unsafe_allow_html=True)

    ctrl1, ctrl2, _ = st.columns([1,1,3])
    with ctrl1:
        st.session_state.demo_mode = st.toggle("Demo Mode", value=st.session_state.demo_mode, help="View sample results and dashboard without uploading a file")
    with ctrl2:
        st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode, help="Switch between light and dark themes")

    # Robust dark mode CSS override
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        :root {
            --bg: #0b1221;
            --card: #0f172a;
            --muted: #334155;
            --accent: #60A5FA;
            --accent-600: #3B82F6;
            --success: #22C55E; --warning: #F59E0B; --danger: #EF4444;
            --text: #E5E7EB; --subtext: #94A3B8;
        }
        body, [data-testid="stAppViewContainer"], .block-container {
            background: var(--bg) !important;
            color: var(--text) !important;
        }
        h1, h2, h3, h4, h5, h6, p, span, div, label, li, code, kbd, pre {
            color: var(--text) !important;
        }
        .stCaption, .st-emotion-cache-10trblm, .st-emotion-cache-ur2l3v {
            color: var(--subtext) !important;
        }
        .card, .subject-card, .stat-card, .topbar {
            background: var(--card) !important;
            border-color: var(--muted) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebar"] * {
            background: var(--card) !important;
            color: var(--text) !important;
        }
        [role="tablist"] { border-bottom: 1px solid var(--muted) !important; }
        [role="tab"] { color: var(--subtext) !important; }
        [role="tab"][aria-selected="true"] { color: var(--text) !important; border-bottom: 2px solid var(--accent) !important; }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--card) !important;
            border: 1px dashed var(--muted) !important;
            color: var(--subtext) !important;
        }
        [data-testid="stFileUploader"] * { color: var(--text) !important; }
        input, textarea, select {
            background: var(--card) !important;
            color: var(--text) !important;
            border-color: var(--muted) !important;
        }
        .stDataFrame, .stDataFrame div, .stDataFrame th, .stDataFrame td {
            color: var(--text) !important;
            background: var(--card) !important;
            border-color: var(--muted) !important;
        }
        .stAlert {
            background: var(--card) !important;
            border: 1px solid var(--muted) !important;
            border-left: 4px solid var(--accent) !important;
            color: var(--text) !important;
        }
        .stButton > button {
            background: var(--accent) !important;
            color: #fff !important;
            border-radius: 10px !important;
            border: 1px solid var(--accent-600) !important;
        }
        .stButton > button:hover { background: var(--accent-600) !important; }
        </style>
        """, unsafe_allow_html=True)

    tab_upload, tab_results, tab_review, tab_dash = st.tabs(["Upload", "Results", "Review", "Dashboard"])

    with tab_upload:
        upload_tab()
    with tab_results:
        results_tab()
    with tab_review:
        review_tab()
    with tab_dash:
        dashboard_tab()

def upload_tab():
    st.header("Upload OMR Sheet")

    # Key Set selector
    col_sel, _ = st.columns([1,3])
    with col_sel:
        new_set = st.selectbox("Answer Key Set", options=["A","B"], index=0 if st.session_state.key_set=='A' else 1, help="Choose the sheet set to score against")
        if new_set != st.session_state.key_set:
            st.session_state.key_set = new_set
            st.session_state.answer_key = None
            load_default_key()

    st.markdown("""
    <div class="card">
      <div class="upload-section">
        <p>Drag and drop OMR image (PNG/JPG) or PDF, or click to browse.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an OMR sheet file",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Supported formats: PNG, JPG, JPEG, PDF. Max size: 10MB"
    )

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**File**")
            st.write(uploaded_file.name)
            st.caption(f"{uploaded_file.size/1024:.1f} KB • {uploaded_file.type}")
        with col2:
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, caption="Preview", use_column_width=True)

    st.markdown("""
    <div class="card" style="margin-top: .5rem"> 
      <h3 style="margin:0 0 .5rem 0">Start Evaluation</h3>
    </div>
    """, unsafe_allow_html=True)

    run_col, demo_col2 = st.columns([2, 1])
    with run_col:
        if st.button("Run Evaluation", type="primary"):
            evaluate_omr_sheet()
    with demo_col2:
        if st.session_state.demo_mode and st.button("Load Demo Result"):
            st.session_state.current_result = make_single("DEMO001")
            st.session_state.current_overlay = None
            st.session_state.has_real_data = False
            st.success("Loaded demo result.")

def evaluate_omr_sheet():
    """Simulate OMR evaluation process"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        ("Loading image...", 0.2),
        ("Detecting OMR structure...", 0.4),
        ("Identifying answer bubbles...", 0.6),
        ("AI-powered evaluation...", 0.8),
        ("Generating results...", 1.0)
    ]

    with st.spinner("Evaluating OMR sheet..."):
        for step, progress in steps:
            status_text.text(step)
            progress_bar.progress(progress)
            time.sleep(1)

    status_text.text("Evaluation completed!")

    # Demo vs real engine path
    if st.session_state.demo_mode or not st.session_state.uploaded_file:
        st.session_state.current_result = make_single("DEMO-UPLOAD")
        st.session_state.current_overlay = None
        # In demo flow, do not mark as real data
        st.session_state.has_real_data = False
    else:
        try:
            file_bytes = st.session_state.uploaded_file.getvalue()
            res = engine.evaluate_omr(file_bytes, st.session_state.answer_key)
            st.session_state.current_result = {
                "student_id": res.student_id,
                "subject_scores": res.subject_scores,
                "total_score": res.total_score,
                "evaluation_time": res.evaluation_time,
                "confidence_score": res.confidence_score,
            }
            st.session_state.current_overlay = res.annotated_image
            # Persist to DB
            try:
                db_id = db.save_result(
                    student_id=res.student_id,
                    subject_scores=res.subject_scores,
                    total_score=res.total_score,
                    evaluation_time_str=res.evaluation_time,
                    confidence_score=res.confidence_score,
                    key_set=st.session_state.key_set,
                    source_file=getattr(st.session_state.uploaded_file, 'name', None),
                )
                st.session_state.has_real_data = True
                st.toast(f"Saved to database (ID: {db_id})", icon="✅")
            except Exception as db_err:
                st.toast(f"DB save failed: {db_err}", icon="⚠️")
        except Exception as e:
            st.error(f"Engine failed: {e}")
            st.session_state.current_result = None
            st.session_state.current_overlay = None

    st.success("Evaluation completed. Open the Results or Review tab.")

def results_tab():
    st.header("Results")
    if not st.session_state.current_result:
        st.info("No results yet. Upload a file and run evaluation, or enable Demo Mode.")
        return

    # Main score display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <h1>{st.session_state.current_result['total_score']}/100</h1>
            <h3>Overall Score</h3>
            <div class="progress-bg"><div class="progress-fill" style="width: {st.session_state.current_result['total_score']}%; background: var(--accent);"></div></div>
            <p>Confidence: {st.session_state.current_result.get('confidence_score', 0)}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Subject-wise scores
    st.subheader("Subject-wise Performance")
    cols = st.columns(len(st.session_state.current_result['subject_scores']))
    for i, (subject, score) in enumerate(st.session_state.current_result['subject_scores'].items()):
        with cols[i]:
            percentage = (score / 20) * 100
            progress_color = 'var(--success)' if percentage >= 80 else ('var(--warning)' if percentage >= 60 else 'var(--danger)')
            st.markdown(f"""
            <div class="subject-card" style="border-left: 4px solid {progress_color};">
                <h4 style="margin:0;">{subject}</h4>
                <div style="display:flex;align-items:baseline;gap:8px;margin-top:4px;">
                    <h2 style="margin:0;">{score}/20</h2>
                    <span style="color: var(--subtext);">{percentage:.1f}%</span>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {percentage}%; background: {progress_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Detailed breakdown table
    st.subheader("Detailed Breakdown")
    df = pd.DataFrame([
        {"Subject": s, "Score": f"{sc}/20", "Percentage": f"{(sc/20)*100:.1f}%", "Grade": get_grade((sc/20)*100)}
        for s, sc in st.session_state.current_result['subject_scores'].items()
    ])
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Export buttons
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        csv_buf = io.StringIO()
        pd.DataFrame([{**{"Student_ID": st.session_state.current_result['student_id']}, **st.session_state.current_result['subject_scores'], "Total_Score": st.session_state.current_result['total_score']}]).to_csv(csv_buf, index=False)
        st.download_button("Download CSV", data=csv_buf.getvalue(), file_name=f"results_{st.session_state.current_result['student_id']}.csv", mime="text/csv", use_container_width=True)
    with exp_c2:
        xbuf = io.BytesIO()
        pd.DataFrame([{**{"Student_ID": st.session_state.current_result['student_id']}, **st.session_state.current_result['subject_scores'], "Total_Score": st.session_state.current_result['total_score']}]).to_excel(xbuf, index=False, engine='openpyxl')
        st.download_button("Download Excel", data=xbuf.getvalue(), file_name=f"results_{st.session_state.current_result['student_id']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

def review_tab():
    st.header("Review Annotated Sheet")
    if not st.session_state.current_result:
        st.info("No results to review. Run an evaluation first or use Demo Mode.")
        return

    c1, c2 = st.columns([3,1])
    with c1:
        w = st.slider("Zoom width", min_value=400, max_value=900, value=720)
        if st.session_state.current_overlay is not None:
            st.image(st.session_state.current_overlay, caption="Engine-generated evaluation overlay", width=w)
        else:
            annotated_img = get_cached_annotated_image(st.session_state.current_result)
            st.image(annotated_img, caption="System-generated evaluation overlay", width=w)
    with c2:
        st.markdown("**Legend**")
        st.markdown("- Green: Correct\n- Red: Incorrect\n- Gray: Not Marked\n- Yellow: Flagged")

    # Downloads
    st.markdown("---")
    st.subheader("Download Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_buf = io.StringIO()
        pd.DataFrame([{**{"Student_ID": st.session_state.current_result['student_id']}, **st.session_state.current_result['subject_scores'], "Total_Score": st.session_state.current_result['total_score']}]).to_csv(csv_buf, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"omr_results_{st.session_state.current_result['student_id']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        xbuf = io.BytesIO()
        pd.DataFrame([{**{"Student_ID": st.session_state.current_result['student_id']}, **st.session_state.current_result['subject_scores'], "Total_Score": st.session_state.current_result['total_score']}]).to_excel(xbuf, index=False, engine='openpyxl')
        st.download_button(
            label="Download Excel",
            data=xbuf.getvalue(),
            file_name=f"omr_results_{st.session_state.current_result['student_id']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col3:
        st.button("Generate PDF Report", use_container_width=True, help="PDF generation will be available in the next version")

def dashboard_tab():
    """Dashboard with aggregate statistics based on Demo Mode or DB state"""
    st.header("Evaluation Dashboard")

    using_demo = st.session_state.demo_mode
    using_real = (not using_demo) and st.session_state.has_real_data

    if using_demo:
        # Prepare demo batch only in demo mode
        if st.session_state.batch_df is None or len(st.session_state.batch_df) == 0:
            st.session_state.batch_df, st.session_state.batch_results = generate_batch(25)
        df = st.session_state.batch_df.copy()

        total_students = len(df.index)
        # Heuristic: subject columns numeric and not meta fields
        subject_cols = [c for c in df.columns if c not in ['Student_ID','Total','Created_At'] and pd.api.types.is_numeric_dtype(df[c])]
        if 'Total' in df.columns and pd.api.types.is_numeric_dtype(df['Total']):
            avg_total_pct = float(np.mean(df['Total']))
        else:
            # Sum across subject columns and normalize to percentage if needed
            per_row = df[subject_cols].sum(axis=1)
            # If scores already out of 20 each for 5 subjects, total is out of 100
            avg_total_pct = float(per_row.mean())

        # Grade distribution
        def letter_grade(p):
            if p >= 90: return "A+ (90-100)"
            if p >= 80: return "A (80-89)"
            if p >= 70: return "B (70-79)"
            if p >= 60: return "C (60-69)"
            if p >= 50: return "D (50-59)"
            return "F (<50)"
        grade_counts = {}
        totals = df['Total'] if 'Total' in df.columns and pd.api.types.is_numeric_dtype(df['Total']) else df[subject_cols].sum(axis=1)
        for p in totals:
            g = letter_grade(float(p))
            grade_counts[g] = grade_counts.get(g, 0) + 1

        # Subject averages (out of 20)
        subject_avgs = {}
        for c in subject_cols:
            subject_avgs[c] = float(df[c].mean())

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-label">Total Students</div>
            <div class="stat-value">{total_students}</div>
            <div class="stat-delta">Demo Mode</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Average Score</div>
            <div class="stat-value">{avg_total_pct:.1f}%</div>
            <div class="stat-delta">Demo Mode</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Pass Rate</div>
            <div class="stat-value">{(100.0 * (totals >= 50).sum() / max(total_students,1)):.1f}%</div>
            <div class="stat-delta">Demo Mode</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Avg Processing Time</div>
            <div class="stat-value">2.3 sec</div>
            <div class="stat-delta">Simulated</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Subject-wise Average Scores")
            subject_df = pd.DataFrame([{"Subject": s, "Average Score": v} for s, v in subject_avgs.items()])
            chart = (
                alt.Chart(subject_df)
                .mark_bar(color='#3182CE')
                .encode(
                    x=alt.X('Subject:N', sort='-y', title='Subject'),
                    y=alt.Y('Average Score:Q', scale=alt.Scale(domain=[0, 20]), title='Avg Score (out of 20)'),
                    tooltip=['Subject', 'Average Score']
                )
                .properties(height=300, background='transparent')
                .configure_view(stroke=None)
                .configure_axis(grid=True, gridColor='#E2E8F0', domain=False)
            )
            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.subheader("Grade Distribution")
            grade_df = pd.DataFrame([{"Grade": g, "Count": c} for g, c in grade_counts.items()])
            pie_chart = (
                alt.Chart(grade_df)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta('Count:Q'),
                    color=alt.Color('Grade:N', scale=alt.Scale(scheme='category10'), legend=None),
                    tooltip=['Grade', 'Count']
                )
                .properties(height=300, background='transparent')
                .configure_view(stroke=None)
            )
            st.altair_chart(pie_chart, use_container_width=True)

        st.subheader("Recent Evaluations (Demo)")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif using_real:
        # Query aggregates from DB; if none, show empty state
        try:
            agg = db.get_aggregates()  # implement: returns dict with total_students, averages, grade_dist, recent_df
        except Exception as e:
            st.warning(f"Failed to load dashboard from database: {e}")
            agg = None

        if not agg or agg.get("total_students", 0) == 0:
            st.info("No real data yet. Upload and evaluate sheets to populate the dashboard.")
            return

        total_students = agg.get("total_students", 0)
        avg_total_pct = float(agg.get("average_total", 0.0))
        pass_rate = float(agg.get("pass_rate", 0.0))
        subject_avgs = agg.get("subject_averages", {})  # dict subject->avg out of 20
        grade_counts = agg.get("grade_distribution", {})  # dict grade->count
        recent_df = agg.get("recent_df", pd.DataFrame())

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-label">Total Students</div>
            <div class="stat-value">{total_students}</div>
            <div class="stat-delta">Real Data</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Average Score</div>
            <div class="stat-value">{avg_total_pct:.1f}%</div>
            <div class="stat-delta">Real Data</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Pass Rate</div>
            <div class="stat-value">{pass_rate:.1f}%</div>
            <div class="stat-delta">Real Data</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Avg Processing Time</div>
            <div class="stat-value">—</div>
            <div class="stat-delta">From engine logs</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Subject-wise Average Scores")
            subject_df = pd.DataFrame([{"Subject": s, "Average Score": v} for s, v in subject_avgs.items()])
            chart = (
                alt.Chart(subject_df)
                .mark_bar(color='#3182CE')
                .encode(
                    x=alt.X('Subject:N', sort='-y', title='Subject'),
                    y=alt.Y('Average Score:Q', scale=alt.Scale(domain=[0, 20]), title='Avg Score (out of 20)'),
                    tooltip=['Subject', 'Average Score']
                )
                .properties(height=300, background='transparent')
                .configure_view(stroke=None)
                .configure_axis(grid=True, gridColor='#E2E8F0', domain=False)
            )
            st.altair_chart(chart, use_container_width=True)
        with col2:
            st.subheader("Grade Distribution")
            grade_df = pd.DataFrame([{"Grade": g, "Count": c} for g, c in grade_counts.items()])
            pie_chart = (
                alt.Chart(grade_df)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta('Count:Q'),
                    color=alt.Color('Grade:N', scale=alt.Scale(scheme='category10'), legend=None),
                    tooltip=['Grade', 'Count']
                )
                .properties(height=300, background='transparent')
                .configure_view(stroke=None)
            )
            st.altair_chart(pie_chart, use_container_width=True)

        st.subheader("Recent Evaluations")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if isinstance(recent_df, pd.DataFrame) and not recent_df.empty:
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.caption("No recent records available.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Not demo and no real data
        st.info("No data available. Enable Demo Mode to preview, or upload and evaluate sheets to populate real data.")

def help_page():
    st.header("Help & Information")
    tab1, tab2, tab3 = st.tabs(["User Guide", "Technical Info", "FAQ"])
    with tab1:
        st.markdown("""
        ## How to Use the OMR Evaluation System
        1. Upload OMR sheet image or PDF in the Upload tab.
        2. Click Run Evaluation and wait for processing.
        3. Review results and annotated overlay.
        4. Use Dashboard for aggregate insights.
        """)
    with tab2:
        st.markdown("""
        ## Technical Specifications
        - Image Resolution: ≥300 DPI recommended
        - Formats: PNG, JPG, JPEG, PDF
        - Processing Time: 2–5s per sheet
        """)
    with tab3:
        st.markdown("""
        ## FAQ
        - Rotated/skewed sheets are corrected automatically.
        - Batch processing is planned for next versions.
        """)

if __name__ == "__main__":
    main()
