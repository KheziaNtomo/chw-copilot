"""CHW Copilot — Agentic Surveillance Demo

MedGemma-powered syndromic surveillance support tool for
Community Health Workers and District Health Officers.

Six-agent pipeline: Extract → Ground → Verify → Tag → Checklist → Validate
Powered by MedGemma 1.5 + Strawberry (Pythea) hallucination detection.
"""
import sys
from pathlib import Path

import streamlit as st

# Add app directory to path for imports
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="CHW Copilot — Agentic Surveillance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Authentication ───────────────────────────────────────────
def check_password():
    """Returns `True` if the user had a correct password."""
    # Check if password is set in secrets (for cloud deployment)
    # If no secrets file or no PASSWORD key, assume local/unprotected mode
    try:
        if "PASSWORD" not in st.secrets:
            return True
    except (FileNotFoundError, KeyError):
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "🔒 Enter Competition Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again.
        st.text_input(
            "🔒 Enter Competition Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# ── Load Custom CSS ──────────────────────────────────────────
css_path = app_dir / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Google Fonts ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1rem 0;'>
        <div style='font-size:1.2rem;font-weight:700;color:#ffffff;letter-spacing:-0.01em;'>CHW Copilot</div>
        <div style='font-size:0.75rem;color:#8da87d;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.25rem;'>Syndromic Surveillance</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    role = st.radio(
        "View",
        ["CHW Field View", "District Dashboard"],
        index=0,
    )

    st.markdown("---")

    # Pipeline info
    st.markdown("### Pipeline Agents")
    agents = [
        ("Encounter Extractor",    "MedGemma 4b"),
        ("Evidence Grounder",       "Deterministic"),
        ("Syndrome Tagger",         "Keyword rules"),
        ("Sub-syndrome Classifier", "Rule-based"),
        ("ICCM Recommendations",   "Rule-based"),
        ("Schema Validator",        "Deterministic"),
    ]
    for name, engine in agents:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:0.2rem 0;border-bottom:1px solid rgba(255,255,255,0.06);margin:0.1rem 0;">'
            f'<span style="color:#c8d8b8;font-size:0.82rem;">{name}</span>'
            f'<span style="color:#5a7a4a;font-size:0.72rem;font-weight:600;">{engine}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="color:#5a7a4a;font-size:0.72rem;line-height:1.6;margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.07)">'
        'Not for clinical diagnosis.<br>'
        'Surveillance support tool only.<br>'
        'All outputs require human verification.<br><br>'
        'Privacy-by-design · Offline-first<br>'
        'Powered by MedGemma 4b'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Main Content ─────────────────────────────────────────────
if "CHW" in role:
    from chw_view import render_chw_view
    render_chw_view()
else:
    from district_view import render_district_view
    render_district_view()
