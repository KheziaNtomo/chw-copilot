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
    st.markdown("# 🏥 CHW Copilot")
    st.markdown("**Agentic Syndromic Surveillance**")
    st.markdown("---")

    role = st.radio(
        "Select Role",
        ["👩‍⚕️ Community Health Worker", "📊 District Officer"],
        index=0,
    )

    st.markdown("---")

    # Pipeline info
    st.markdown("### ⚙️ Pipeline Agents")
    agents = [
        ("🔬", "Encounter Extractor", "MedGemma 1.5"),
        ("✅", "Evidence Grounder", "Deterministic"),
        ("🍓", "Hallucination Detector", "Strawberry"),
        ("🏷️", "Syndrome Tagger", "MedGemma 1.5"),
        ("📋", "Checklist Generator", "MedGemma 1.5"),
        ("📐", "Schema Validator", "Deterministic"),
    ]
    for icon, name, engine in agents:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;">'
            f'<span>{icon}</span>'
            f'<span style="color:#f1f5f9;font-size:0.85rem">{name}</span>'
            f'<span style="color:#64748b;font-size:0.7rem;margin-left:auto">{engine}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="color:#64748b;font-size:0.75rem;line-height:1.4;">'
        '⚠️ <strong>Not for clinical diagnosis.</strong><br>'
        'Surveillance support tool only.<br>'
        'All outputs require human verification.<br><br>'
        '🔒 Privacy-by-design · Offline-first<br>'
        'Powered by MedGemma 1.5 (HAI-DEF)'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Main Content ─────────────────────────────────────────────
if "👩‍⚕️" in role:
    from chw_view import render_chw_view
    render_chw_view()
else:
    from district_view import render_district_view
    render_district_view()
