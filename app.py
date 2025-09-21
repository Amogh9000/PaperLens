import streamlit as st
import pandas as pd
import numpy as np
import cv2
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
    # Flags to safely clear uploaders before widget instantiation
    if 'clear_omr_upl' not in st.session_state:
        st.session_state.clear_omr_upl = False
    if 'clear_ak_upl' not in st.session_state:
        st.session_state.clear_ak_upl = False
    if 'answer_key_data' not in st.session_state:
        st.session_state.answer_key_data = None  # full dict from uploaded JSON/Excel
    if 'answer_key_set' not in st.session_state:
        st.session_state.answer_key_set = None  # set name like set_1 / set_2
    if 'local_answer_key_data' not in st.session_state:
        st.session_state.local_answer_key_data = None  # discovered from disk
    if 'use_local_keys' not in st.session_state:
        st.session_state.use_local_keys = True
    if 'local_key_sources' not in st.session_state:
        st.session_state.local_key_sources = {}  # e.g., {"A": "answer_key_set1.xlsx", "B": "answer_key_set2.xlsx"}
    if 'uploaded_key_source' not in st.session_state:
        st.session_state.uploaded_key_source = None  # filename of uploaded key
    if 'current_overlay' not in st.session_state:
        st.session_state.current_overlay = None
    if 'key_set' not in st.session_state:
        st.session_state.key_set = 'A'
    if 'has_real_data' not in st.session_state:
        st.session_state.has_real_data = False  # becomes True after a successful DB save
    if 'sample_image' not in st.session_state:
        st.session_state.sample_image = None  # path to Img1/2/20
    if 'detailed_results' not in st.session_state:
        st.session_state.detailed_results = None

def load_default_key():
    # Keep old fallback for mock mode; real flow uses uploaded key
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

def _normalize_key_sets(data: dict) -> dict:
    """Return a dict of sets with normalized names and safe access aliases.
    Accepts structures like {"set_1": {...}, "set_2": {...}} or {"A": {...}, "B": {...}}.
    Produces keys for both styles so UI can choose A/B but backend can still resolve.
    """
    if not isinstance(data, dict):
        return {}
    out = {}
    # If it's an answer-key root with sets inside
    for k, v in data.items():
        lk = str(k).strip().lower()
        if lk in ("a", "set_a", "set 1", "set_1", "1", "set1"):
            out["A"] = v; out["set_1"] = v
        elif lk in ("b", "set_b", "set 2", "set_2", "2", "set2"):
            out["B"] = v; out["set_2"] = v
        else:
            # keep original
            out[k] = v
    # If we ended up with only one set but not aliased, keep as-is
    return out

def _discover_local_answer_keys() -> dict:
    """Look for answer keys in the project directory.
    Supports:
      - answer_key.json (with sets)
      - answer_key_set1.xlsx and/or answer_key_set2.xlsx (each sheet with columns question/answer)
    Returns a normalized dict with keys like A/B and set_1/set_2.
    """
    base = os.path.dirname(__file__)
    # JSON first
    json_path = os.path.join(base, 'answer_key.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            norm = _normalize_key_sets(data)
            # Record sources
            try:
                st.session_state.local_key_sources = {}
                if 'A' in norm or 'set_1' in norm:
                    st.session_state.local_key_sources['A'] = f"answer_key.json -> {'A' if 'A' in norm else 'set_1'}"
                if 'B' in norm or 'set_2' in norm:
                    st.session_state.local_key_sources['B'] = f"answer_key.json -> {'B' if 'B' in norm else 'set_2'}"
            except Exception:
                pass
            return norm
        except Exception:
            pass
    # Excel split files
    def _load_excel_to_map(xlsx_path: str) -> dict:
        try:
            xls = pd.ExcelFile(xlsx_path)
            # take first sheet
            sheet = xls.sheet_names[0]
            df = xls.parse(sheet)
            cols = {c.lower(): c for c in df.columns}
            qcol = cols.get('question') or cols.get('q') or df.columns[0]
            acol = cols.get('answer') or cols.get('ans') or df.columns[1]
            m = {}
            for _, row in df.iterrows():
                if pd.notna(row[qcol]) and pd.notna(row[acol]):
                    q = str(int(row[qcol]))
                    a = str(row[acol]).strip().lower()
                    m[q] = a
            return m
        except Exception:
            return {}
    set_map: dict = {}
    set1 = os.path.join(base, 'answer_key_set1.xlsx')
    set2 = os.path.join(base, 'answer_key_set2.xlsx')
    if os.path.exists(set1):
        m1 = _load_excel_to_map(set1)
        if m1:
            set_map['A'] = m1; set_map['set_1'] = m1
            try:
                st.session_state.local_key_sources['A'] = 'answer_key_set1.xlsx'
            except Exception:
                pass
    if os.path.exists(set2):
        m2 = _load_excel_to_map(set2)
        if m2:
            set_map['B'] = m2; set_map['set_2'] = m2
            try:
                st.session_state.local_key_sources['B'] = 'answer_key_set2.xlsx'
            except Exception:
                pass
    return set_map

def _load_answer_key_from_upload(file) -> dict:
    """Load answer key from JSON or Excel into a dict with set names.
    JSON format: {"set_1": {"1": "a", ...}, "set_2": {...}}
    Excel format: expects columns like [question, answer] for each sheet or a single sheet; two files for set1/set2 also supported.
    """
    import io
    name = getattr(file, 'name', 'uploaded')
    if name.lower().endswith('.json'):
        return json.loads(file.getvalue().decode('utf-8'))
    # Excel: read with pandas
    xls = pd.ExcelFile(file)
    data = {}
    # Prefer sheets named set_1/set_2; else first two sheets
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # find columns heuristically
        cols = {c.lower(): c for c in df.columns}
        qcol = cols.get('question') or cols.get('q') or df.columns[0]
        acol = cols.get('answer') or cols.get('ans') or df.columns[1]
        mapping = {}
        for _, row in df.iterrows():
            q = str(int(row[qcol])) if pd.notna(row[qcol]) else None
            a = str(row[acol]).strip().lower() if pd.notna(row[acol]) else ''
            if q:
                mapping[q] = a
        set_name = sheet if sheet else 'set_1'
        data[set_name] = mapping
    return data

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
    # Discover local keys once per session
    if st.session_state.local_answer_key_data is None:
        st.session_state.local_answer_key_data = _discover_local_answer_keys()

    # Header
    st.markdown("""
    <div class="topbar">
        <div class="brand">PaperLens</div>
        <div class="subtitle">Intelligent OMR Evaluation</div>
    </div>
    """, unsafe_allow_html=True)

    ctrl1, _ = st.columns([1,5])
    with ctrl1:
        st.session_state.demo_mode = st.toggle("Demo Mode", value=st.session_state.demo_mode, help="View sample results and dashboard without uploading a file")


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

    # Key Set selector (drives grading set for current key source)
    col_sel, _ = st.columns([1,3])
    with col_sel:
        # Always show A/B; we’ll warn if the chosen set isn’t available
        default_idx = 0 if (st.session_state.answer_key_set or "A") == "A" else 1
        new_set = st.selectbox("Answer Key Set", options=["A","B"], index=default_idx, help="Choose the sheet set to score against")
        st.session_state.answer_key_set = new_set

    # Minimal uploader (browse only look & feel)
    st.markdown("""
    <style>
    /* Tweak the uploader to look like a simple button and remove dashed box */
    [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] button {
        background: var(--accent) !important;
        color: #fff !important;
        border: 1px solid var(--accent-600) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1rem !important;
    }
    [data-testid="stFileUploader"] div div div { display: none !important; } /* hide helper text */
    </style>
    """, unsafe_allow_html=True)

    # If last run asked to clear the OMR uploader, do it BEFORE creating the widget
    if st.session_state.clear_omr_upl:
        if 'omr_upl' in st.session_state:
            del st.session_state['omr_upl']
        st.session_state.clear_omr_upl = False

    uploaded_file = st.file_uploader(
        "Choose an OMR sheet file",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Supported formats: PNG, JPG, JPEG, PDF. Max size: 10MB",
        key="omr_upl"
    )
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.has_real_data = True
        c1, c2 = st.columns([4,1])
        with c1:
            st.caption(f"File: {uploaded_file.name}")
        with c2:
            if st.button("✖ Remove", key="remove_omr_file"):
                # Set flag to clear uploader key on next run
                st.session_state.clear_omr_upl = True
                st.session_state.uploaded_file = None
                st.session_state.current_result = None
                st.session_state.review_images = None
                try:
                    st.rerun()
                except Exception:
                    import streamlit as _st
                    _st.experimental_rerun()
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

    # Answer key source
    ak_src1, ak_src2 = st.columns([2,2])
    with ak_src1:
        st.session_state.use_local_keys = st.toggle("Use local answer keys from project", value=bool(st.session_state.local_answer_key_data))
        if st.session_state.use_local_keys and st.session_state.local_answer_key_data:
            st.session_state.answer_key_data = st.session_state.local_answer_key_data
            keys = st.session_state.answer_key_data.keys()
            has_A = ("A" in keys) or ("set_1" in keys)
            has_B = ("B" in keys) or ("set_2" in keys)
            found = ", ".join([s for s, ok in (("A",has_A),("B",has_B)) if ok]) or "None"
            st.caption(f"Local keys found: {found}")
        else:
            st.info("Local keys disabled or not found. Use Upload on the right.")
    with ak_src2:
        st.markdown("Upload Answer Key (JSON or Excel)")
        # If last run asked to clear the Answer Key uploader, do it BEFORE creating the widget
        if st.session_state.clear_ak_upl and 'ak_upl' in st.session_state:
            del st.session_state['ak_upl']
            st.session_state.clear_ak_upl = False
        ak_file = st.file_uploader("Answer Key", type=["json","xlsx","xls"], key="ak_upl")
        if ak_file is not None:
            try:
                st.session_state.answer_key_data = _normalize_key_sets(_load_answer_key_from_upload(ak_file))
                # Update selected set to A if available, else B/None
                keys = st.session_state.answer_key_data.keys()
                if ("A" in keys) or ("set_1" in keys):
                    st.session_state.answer_key_set = "A"
                elif ("B" in keys) or ("set_2" in keys):
                    st.session_state.answer_key_set = "B"
                else:
                    st.session_state.answer_key_set = None
                st.session_state.uploaded_key_source = getattr(ak_file, 'name', 'uploaded')
                # Show name and provide remove button
                c1, c2 = st.columns([4,1])
                with c1:
                    st.success("Answer key loaded.")
                    st.caption(f"File: {st.session_state.uploaded_key_source}")
                with c2:
                    if st.button("✖ Remove", key="remove_key_file"):
                        # Set flag to clear uploader key on next run
                        st.session_state.clear_ak_upl = True
                        st.session_state.answer_key_data = None
                        st.session_state.uploaded_key_source = None
                        try:
                            st.rerun()
                        except Exception:
                            import streamlit as _st
                            _st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to load answer key: {e}")

    st.markdown("""
    <div class="card" style="margin-top: .5rem"> 
      <h3 style="margin:0 0 .5rem 0">Run</h3>
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

    # Show which answer key file is in use and warn if missing
    if st.session_state.answer_key_data and st.session_state.answer_key_set:
        src_hint = None
        if st.session_state.use_local_keys and st.session_state.local_key_sources:
            src_hint = st.session_state.local_key_sources.get(st.session_state.answer_key_set)
        elif st.session_state.uploaded_key_source:
            src_hint = f"Uploaded: {st.session_state.uploaded_key_source}"
        if src_hint:
            st.caption(f"Using key set {st.session_state.answer_key_set} from {src_hint}")
        # Warn if selected set not present in keys
        keys = st.session_state.answer_key_data.keys()
        sel = st.session_state.answer_key_set
        has_sel = (sel in keys) or (sel == 'A' and 'set_1' in keys) or (sel == 'B' and 'set_2' in keys)
        if not has_sel:
            st.warning(f"Key set '{sel}' not found in the current answer key source. Add it to answer_key.json (as '{'set_1' if sel=='A' else 'set_2'}' or '{sel}') or upload a file that contains it.")

def evaluate_omr_sheet():
    """Run OMR evaluation using the new backend if answer key is provided; otherwise fallback to demo."""
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
    using_uploaded = st.session_state.uploaded_file is not None
    if st.session_state.demo_mode or not using_uploaded or not st.session_state.answer_key_data or not st.session_state.answer_key_set:
        st.session_state.current_result = make_single("DEMO-UPLOAD")
        st.session_state.current_overlay = None
        # In demo flow, do not mark as real data
        st.session_state.has_real_data = False
    else:
        try:
            # Load image from upload only (support image or PDF)
            up = st.session_state.uploaded_file
            data = up.getvalue()
            name = getattr(up, 'name', '').lower()
            img = None
            if name.endswith('.pdf'):
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(stream=data, filetype='pdf')
                    if doc.page_count == 0:
                        raise ValueError('Empty PDF')
                    page = doc.load_page(0)
                    zoom = 200/72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    npimg = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img = cv2.cvtColor(npimg, cv2.COLOR_RGBA2BGR)
                    elif pix.n == 3:
                        img = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
                    else:
                        img = cv2.cvtColor(npimg, cv2.COLOR_GRAY2BGR)
                except Exception as e:
                    raise ValueError(f"Failed to render PDF: {e}")
            else:
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError("Failed to read image (unsupported format or corrupted file)")

            # Use new backend: OMRGrader
            key = st.session_state.answer_key_data
            # Resolve set name strictly; error if missing
            sel = st.session_state.answer_key_set
            candidates = [sel, sel.upper(), sel.lower()]
            if sel in ("A","B"):
                candidates.extend([f"set_{1 if sel=='A' else 2}"])
            elif str(sel).startswith("set_"):
                if sel.endswith("1"): candidates.append("A")
                if sel.endswith("2"): candidates.append("B")
            set_name = next((c for c in candidates if c in key), None)
            if not set_name:
                raise ValueError(f"Selected key set '{sel}' not found. Provide it in answer_key.json or upload a key containing '{sel}' (or its alias).")
            grader = engine.OMRGrader(key)
            results = grader.grade(img, set_name=set_name, debug=False)

            # Build review visuals (original, flattened, answer area, detected bubbles)
            import tempfile
            review = {}
            def _safe_encode_png(arr):
                try:
                    if arr is None or getattr(arr, 'size', 0) == 0:
                        return None
                    ok, b = cv2.imencode('.png', arr)
                    return b.tobytes() if ok else None
                except Exception:
                    return None
            # Original
            review['original'] = _safe_encode_png(img)
            # Flattened (engine expects path) - use delete=False on Windows and close before reading
            tmp_path = None
            try:
                tf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                tf.write(cv2.imencode('.png', img)[1].tobytes())
                tf.flush()
                tmp_path = tf.name
                tf.close()
                flat = engine.flatten_image(tmp_path, show_steps=False)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            if flat is None or flat.size == 0:
                flat = img.copy()
            review['flattened'] = _safe_encode_png(flat)
            # Answer area + bubbles
            thresh, _zone = grader.detect_bubble_area(flat, show_steps=False)
            # Recreate cleaned image crop to draw on
            h, w = flat.shape[:2]
            start_y = grader.detect_subject_headings(flat, thresh, show_steps=False)
            top, bottom = max(0, min(h-2, int(start_y + 50))), max(2, min(h, int(h - 50)))
            if bottom <= top:
                top = 0; bottom = h
            cleaned = flat[top:bottom, :].copy()
            bubbles = grader.detect_bubbles(thresh, cleaned, show_steps=False)
            dbg = cleaned.copy()
            try:
                cv2.drawContours(dbg, bubbles, -1, (0, 0, 255), 2)
            except Exception:
                pass
            review['answer_area'] = _safe_encode_png(cleaned)
            review['bubbles'] = _safe_encode_png(dbg)
            st.session_state.review_images = review

            # Prepare session result
            st.session_state.current_result = {
                "student_id": f"UI-{hashlib.md5((st.session_state.uploaded_file.name if st.session_state.uploaded_file else os.path.basename(st.session_state.sample_image)).encode()).hexdigest()[:6].upper()}",
                "subject_scores": results.get("subject_scores", {}),
                "total_score": int((results.get("total_score", 0) / max(results.get("total_questions", 1),1)) * 100),
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence_score": round(results.get("accuracy", 0.0), 2),
            }
            st.session_state.detailed_results = results.get("detailed_results", [])
            st.session_state.current_overlay = None
            # Persist to DB
            try:
                db_id = db.save_result(
                    student_id=st.session_state.current_result["student_id"],
                    subject_scores=st.session_state.current_result["subject_scores"],
                    total_score=st.session_state.current_result["total_score"],
                    evaluation_time_str=st.session_state.current_result["evaluation_time"],
                    confidence_score=st.session_state.current_result["confidence_score"],
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
            st.session_state.detailed_results = None

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

    # Per-question side-by-side (if available)
    if st.session_state.get('detailed_results'):
        st.subheader("Per-question Details")
        det_df = pd.DataFrame(st.session_state.detailed_results)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(det_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download CSV/JSON of detailed results
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            csv_buf = io.StringIO()
            det_df.to_csv(csv_buf, index=False)
            st.download_button("Download Per-question CSV", data=csv_buf.getvalue(), file_name="omr_side_by_side.csv", mime="text/csv", use_container_width=True)
        with cdl2:
            st.download_button("Download Per-question JSON", data=json.dumps(st.session_state.detailed_results, indent=2), file_name="omr_results.json", mime="application/json", use_container_width=True)

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
        w = st.slider("Zoom width", min_value=400, max_value=1000, value=800)
        imgs = st.session_state.get('review_images') or {}
        tabs = st.tabs(["Original", "Flattened", "Answer Area", "Detected Bubbles"])
        with tabs[0]:
            if imgs.get('original'): st.image(imgs['original'], caption="Original Upload", width=w)
        with tabs[1]:
            if imgs.get('flattened'): st.image(imgs['flattened'], caption="Flattened / Deskewed", width=w)
        with tabs[2]:
            if imgs.get('answer_area'): st.image(imgs['answer_area'], caption="Cropped Answer Area", width=w)
        with tabs[3]:
            if imgs.get('bubbles'): st.image(imgs['bubbles'], caption="Detected Bubbles (debug)", width=w)
    with c2:
        st.markdown("**Legend**")
        st.markdown("- Red contours: Detected bubbles\n- Use tabs to switch between steps")

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
