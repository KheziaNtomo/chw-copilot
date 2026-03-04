"""CHW Copilot — Agentic Surveillance Demo

MedGemma-powered syndromic surveillance support tool for
Community Health Workers and District Health Officers.

Six-agent pipeline: Extract → Ground → Verify → Tag → Checklist → Validate
Powered by Gemini API with medical system prompts.
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
    page_icon="+",
    layout="wide",
    initial_sidebar_state="collapsed",
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

import os

try:
    from src.models import try_load_model, is_model_available, get_load_error
    if not is_model_available():
        try_load_model()
except Exception:
    pass

# ── Model Status ─────────────────────────────────────────────
try:
    model_active = is_model_available()
except Exception:
    model_active = False

# ── Sidebar — Minimal ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0;'>
        <div style='font-size:1.3rem;font-weight:700;color:#E3D6C5;letter-spacing:-0.01em;'>CHW Copilot</div>
        <div style='font-size:0.7rem;color:#8D957E;text-transform:uppercase;letter-spacing:0.12em;margin-top:0.35rem;'>
            Syndromic Surveillance
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Simple demo mode indicator
    st.markdown(
        '<div style="background:rgba(151,67,21,0.08);border:1px solid rgba(151,67,21,0.15);'
        'border-radius:10px;padding:0.6rem 0.85rem;margin:0.5rem 0 1rem 0;">'
        '<span style="color:#974315;font-size:0.8rem;font-weight:600;">Demo Mode</span><br>'
        '<span style="color:#6a6255;font-size:0.7rem;">'
        'Pre-computed pipeline results</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div style="color:#8D957E;font-size:0.7rem;line-height:1.6;padding-top:0.5rem;">'
        'Not for clinical diagnosis.<br>'
        'Surveillance support tool only.<br>'
        'All outputs require human verification.<br>'
        'Syndromic surveillance data is synthetic — for demonstration purposes only.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Main Content — 3 Tabs ────────────────────────────────────
tab_about, tab_chw, tab_district = st.tabs([
    "About",
    "Field Notes",
    "District Dashboard",
])

# ── Tab 1: About the Project ────────────────────────────────
with tab_about:
    # Title
    st.markdown("""
    <div style="padding:1.5rem 0 0.5rem 0;">
        <h1 style="font-size:2rem;margin-bottom:0.15rem;">Community Health Worker (CHW) Copilot</h1>
        <p style="color:#6a6255;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;">
            Agentic Syndromic Surveillance
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Editorial statement + rationale
    col_statement, col_rationale = st.columns([3, 2])

    with col_statement:
        st.markdown("""
        <div style="padding:1rem 0 2rem 0;">
            <p style="font-size:1.5rem;font-weight:300;color:#3a3225;line-height:1.45;letter-spacing:-0.01em;">
                From field notes to early warnings,<br>
                turning CHW observations into<br>
                <span style="color:#974315;font-weight:500;">evidence-grounded surveillance.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_rationale:
        st.markdown("""
        <div style="padding:1.25rem 0 2rem 0;">
            <p style="color:#3a3225;font-size:0.88rem;line-height:1.75;">
                Community Health Worker programmes generate thousands of
                narrative field notes each week. In Nigeria alone, over one
                million CHWs have been deployed, while Uganda's 18,000+
                Community Health Extension Workers saw a 40% improvement in
                reporting rates when digitised (UNICEF, 2024). Processing
                these notes manually into structured surveillance data is
                slow and error-prone. CHW Copilot accelerates this workflow
                by using an agentic AI pipeline to automatically extract,
                verify, and aggregate clinical observations into actionable
                syndromic signals, enabling earlier outbreak detection at the
                community level.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Divider
    st.markdown(
        '<div style="border-top:1px solid rgba(151,67,21,0.10);margin:0 0 2rem 0;"></div>',
        unsafe_allow_html=True,
    )

    # ── Pipeline — 3 clear stages ──────────────────────────────
    st.markdown(
        '<p style="color:#6a6255;font-size:0.72rem;text-transform:uppercase;'
        'letter-spacing:0.12em;font-weight:600;margin-bottom:0.75rem;">How It Works</p>',
        unsafe_allow_html=True,
    )

    stage1, arr1, stage2, arr2, stage3, arr3, stage4 = st.columns([3, 0.4, 4, 0.4, 3, 0.4, 3])

    with stage1:
        st.markdown(
            '<div style="background:rgba(240,237,228,0.7);border:1px solid rgba(151,67,21,0.12);'
            'border-radius:12px;padding:1.1rem;height:100%;">'
            '<div style="color:#974315;font-size:0.7rem;text-transform:uppercase;'
            'letter-spacing:0.1em;font-weight:700;margin-bottom:0.6rem;">① Extract</div>'
            '<div style="color:#1a1510;font-size:0.9rem;font-weight:600;margin-bottom:0.4rem;">'
            'Encounter Extractor</div>'
            '<div style="color:#6a6255;font-size:0.78rem;line-height:1.5;">'
            'Transforms free-text CHW field notes into structured, '
            'schema-validated JSON clinical encounters.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with arr1:
        st.markdown(
            '<div style="text-align:center;padding-top:3rem;">'
            '<span style="color:#974315;font-size:20px;">→</span></div>',
            unsafe_allow_html=True,
        )

    with stage2:
        st.markdown(
            '<div style="background:rgba(240,237,228,0.7);border:1px solid rgba(151,67,21,0.12);'
            'border-radius:12px;padding:1.1rem;height:100%;">'
            '<div style="color:#f59e0b;font-size:0.7rem;text-transform:uppercase;'
            'letter-spacing:0.1em;font-weight:700;margin-bottom:0.6rem;">② Verify</div>'
            '<div style="color:#1a1510;font-size:0.82rem;font-weight:600;">Evidence Grounder</div>'
            '<div style="color:#6a6255;font-size:0.75rem;margin-bottom:0.35rem;">'
            'Links every claim to a verbatim quote</div>'
            '<div style="color:#1a1510;font-size:0.82rem;font-weight:600;">Hallucination Detector</div>'
            '<div style="color:#6a6255;font-size:0.75rem;">'
            'Flags contradictions via self-consistency</div>'
            '<div style="margin-top:0.5rem;border-top:1px dashed rgba(151,67,21,0.12);'
            'padding-top:0.4rem;">'
            '<span style="color:#974315;font-size:0.68rem;font-weight:600;letter-spacing:0.5px;">'
            'MULTI-LAYERED VERIFICATION</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with arr2:
        st.markdown(
            '<div style="text-align:center;padding-top:3rem;">'
            '<span style="color:#974315;font-size:20px;">→</span></div>',
            unsafe_allow_html=True,
        )

    with stage3:
        st.markdown(
            '<div style="background:rgba(240,237,228,0.7);border:1px solid rgba(151,67,21,0.12);'
            'border-radius:12px;padding:1.1rem;height:100%;">'
            '<div style="color:#14b8a6;font-size:0.7rem;text-transform:uppercase;'
            'letter-spacing:0.1em;font-weight:700;margin-bottom:0.6rem;">③ Classify</div>'
            '<div style="color:#1a1510;font-size:0.82rem;font-weight:600;">Syndrome Tagger</div>'
            '<div style="color:#6a6255;font-size:0.75rem;margin-bottom:0.35rem;">'
            'Respiratory fever / diarrhoea / other</div>'
            '<div style="color:#1a1510;font-size:0.82rem;font-weight:600;">Checklist &amp; Validator</div>'
            '<div style="color:#6a6255;font-size:0.75rem;">'
            'Follow-up questions + JSON Schema check</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with arr3:
        st.markdown(
            '<div style="text-align:center;padding-top:3rem;">'
            '<span style="color:#974315;font-size:20px;">→</span></div>',
            unsafe_allow_html=True,
        )

    with stage4:
        st.markdown(
            '<div style="background:rgba(240,237,228,0.7);border:1px solid rgba(151,67,21,0.12);'
            'border-radius:12px;padding:1.1rem;height:100%;">'
            '<div style="color:#a78bfa;font-size:0.7rem;text-transform:uppercase;'
            'letter-spacing:0.1em;font-weight:700;margin-bottom:0.6rem;">④ Output</div>'
            '<div style="color:#1a1510;font-size:0.82rem;font-weight:600;">Structured Extraction</div>'
            '<div style="color:#6a6255;font-size:0.75rem;margin-bottom:0.35rem;">'
            'Evidence-grounded JSON encounters</div>'
            '<div style="color:#1a1510;font-size:0.82rem;font-weight:600;">Syndromic Surveillance</div>'
            '<div style="color:#6a6255;font-size:0.75rem;">'
            'Aggregated signals for outbreak detection</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Current focus — footnote
    st.markdown(
        '<p style="color:#4a3f35;font-size:0.72rem;line-height:1.5;margin-top:1rem;">'
        '<em>Currently targeting acute respiratory febrile illness and acute diarrhoeal disease. '
        'Additional syndrome categories can be added as the system evolves.</em></p>',
        unsafe_allow_html=True,
    )

    # Divider
    st.markdown(
        '<div style="border-top:1px solid rgba(151,67,21,0.10);margin:0.5rem 0 1.5rem 0;"></div>',
        unsafe_allow_html=True,
    )

    # Impact metrics — centred
    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            '<div style="padding:0.5rem 0;text-align:center;">'
            '<div style="font-size:2.5rem;font-weight:300;color:#974315;'
            'letter-spacing:-0.03em;line-height:1;">6</div>'
            '<div style="color:#6a6255;font-size:0.75rem;margin-top:0.5rem;'
            'line-height:1.4;text-transform:uppercase;letter-spacing:0.05em;'
            'font-weight:500;">Pipeline<br>agents</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            '<div style="padding:0.5rem 0;text-align:center;">'
            '<div style="font-size:2.5rem;font-weight:300;color:#974315;'
            'letter-spacing:-0.03em;line-height:1;">60+</div>'
            '<div style="color:#6a6255;font-size:0.75rem;margin-top:0.5rem;'
            'line-height:1.4;text-transform:uppercase;letter-spacing:0.05em;'
            'font-weight:500;">Synthetic notes<br>tested</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Disclaimer + references
    st.markdown("""
    <div style="border-top:1px solid rgba(151,67,21,0.10);margin:2rem 0 1rem 0;"></div>
    <p style="color:#4a3f35;font-size:0.75rem;line-height:1.6;text-align:center;">
        Not for clinical diagnosis. Surveillance support tool only.
        All outputs require human verification. Powered by MedGemma via Hugging Face.
    </p>
    <div style="margin-top:1rem;">
        <p style="color:#4a3f35;font-size:0.66rem;font-weight:600;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:0.5rem;">References</p>
        <ol style="color:#3a5a3a;font-size:0.66rem;line-height:1.7;margin:0;padding-left:1.2rem;">
            <li>Agarwal, S. et al. (2015). Evidence on feasibility and effective use of mHealth strategies
            by frontline health workers in developing countries: systematic review.
            <em>Tropical Medicine &amp; International Health</em>, 20(8), 1003–1014.</li>
            <li>UNICEF Uganda (2024). Electronic Community Health Information System: 18,454 CHEWs/VHTs
            across 21 districts achieved 40% improvement in DHIS2 reporting rates.</li>
            <li>Africa CDC &amp; African Union (2024). Community health landscape survey:
            over 1 million CHWs deployed across Africa towards 2030 target of 2 million.</li>
            <li>WHO (2018). WHO guideline on health policy and system support to optimise
            community health worker programmes. Geneva: World Health Organization.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 2: Field Notes (CHW View) ────────────────────────────
with tab_chw:
    from chw_view import render_chw_view
    render_chw_view()


# ── Tab 3: District Dashboard ────────────────────────────────
with tab_district:
    from district_view import render_district_view
    render_district_view()
