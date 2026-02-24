"""CHW View — Field worker interface for note processing.

Features:
- Note entry (text area) + voice note upload
- Side-by-side view: raw note with evidence ↔ structured encounter
- Evidence grounding visual (grounded ✓ / downgraded ⚠ / flagged 🚨)
- Strawberry hallucination detection results
- Syndrome tag badge with trigger quotes
- Follow-up checklist with priority colors
- Pipeline trace (collapsible agent-by-agent view)
"""
import streamlit as st
import re
from typing import Dict, Any, Optional


def highlight_evidence_in_note(note_text: str, encounter: Dict, hallucination_check: Optional[Dict] = None) -> str:
    """Highlight evidence quotes in the original note text.

    Green for grounded claims, red for flagged hallucinations.
    """
    highlighted = note_text
    flagged_quotes = set()

    # Collect flagged claim quotes
    if hallucination_check and hallucination_check.get("flagged"):
        for fc in hallucination_check.get("flagged_claims", []):
            # Find the corresponding evidence quote
            claim = fc.get("claim", "")
            for sym_data in encounter.get("symptoms", {}).values():
                if isinstance(sym_data, dict) and sym_data.get("evidence_quote"):
                    if sym_data["evidence_quote"].lower() in claim.lower() or claim.lower() in sym_data["evidence_quote"].lower():
                        flagged_quotes.add(sym_data["evidence_quote"])

    # Collect all evidence quotes
    all_quotes = []
    for section in ["symptoms", "other_symptoms"]:
        for key, val in encounter.get(section, {}).items():
            if isinstance(val, dict) and val.get("evidence_quote"):
                quote = val["evidence_quote"]
                if quote:
                    is_flagged = quote in flagged_quotes
                    all_quotes.append((quote, is_flagged))

    for flag in encounter.get("red_flags", []):
        if isinstance(flag, dict) and flag.get("evidence_quote"):
            all_quotes.append((flag["evidence_quote"], False))

    # Sort by length (longest first) to avoid partial replacements
    all_quotes.sort(key=lambda x: len(x[0]), reverse=True)

    for quote, is_flagged in all_quotes:
        if quote.lower() in highlighted.lower():
            idx = highlighted.lower().find(quote.lower())
            original = highlighted[idx:idx + len(quote)]
            css_class = "flagged" if is_flagged else ""
            highlighted = (
                highlighted[:idx]
                + f'<mark class="{css_class}">{original}</mark>'
                + highlighted[idx + len(quote):]
            )

    return highlighted


def render_symptom_card(name: str, data: Dict, budget_gaps: Dict = None) -> None:
    """Render a single symptom with evidence status."""
    value = data.get("value", "unknown")
    quote = data.get("evidence_quote", "")

    # Determine status
    if value == "yes":
        claim_key = f"Patient has {name.replace('_', ' ')}"
        matching_gap = None
        if budget_gaps:
            for k, v in budget_gaps.items():
                if name.replace("_", " ") in k.lower():
                    matching_gap = v
                    break

        if matching_gap is not None and matching_gap > 2:
            icon = "🚨"
            css = "evidence-flagged"
            status_text = f"FLAGGED (gap: {matching_gap:.1f} bits)"
        elif quote:
            icon = "✅"
            css = "evidence-grounded"
            status_text = "Grounded"
        else:
            icon = "⚠️"
            css = "evidence-downgraded"
            status_text = "Downgraded — no evidence"
    elif value == "no":
        icon = "❌"
        css = "evidence-grounded"
        status_text = "Absent"
    else:
        icon = "❓"
        css = ""
        status_text = "Unknown"

    with st.container():
        col_name, col_status = st.columns([3, 2])
        with col_name:
            st.markdown(f"{icon} **{name.replace('_', ' ').title()}**: `{value}`")
        with col_status:
            if css:
                st.markdown(f"<small style='opacity:0.7'>{status_text}</small>", unsafe_allow_html=True)

        if quote and value == "yes":
            st.markdown(
                f'<div class="{css}"><span class="evidence-quote">"…{quote}…"</span></div>',
                unsafe_allow_html=True,
            )


def render_pipeline_trace(agent_trace: list) -> None:
    """Render the pipeline agent trace."""
    total_time = sum(step.get("duration_s", 0) for step in agent_trace)

    st.markdown(f"**Total pipeline time: {total_time:.2f}s** | {len(agent_trace)} agents")

    for step in agent_trace:
        fallback = step.get("fallback_used", False)
        is_flagged = "flagged" in step.get("output_summary", "").lower() and "0 flagged" not in step.get("output_summary", "")
        border_color = "#ef4444" if is_flagged else ("#f59e0b" if fallback else "#14b8a6")
        bg = "rgba(239,68,68,0.08)" if is_flagged else ("rgba(245,158,11,0.08)" if fallback else "rgba(20,184,166,0.06)")

        status_badge = ""
        if is_flagged:
            status_badge = '<span style="background:rgba(239,68,68,0.15);color:#ef4444;padding:2px 8px;border-radius:10px;font-size:0.75em;font-weight:600;">⚠ FLAGGED</span>'
        elif fallback:
            status_badge = '<span style="background:rgba(245,158,11,0.15);color:#f59e0b;padding:2px 8px;border-radius:10px;font-size:0.75em;font-weight:600;">↩ FALLBACK</span>'

        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:0.6rem 1rem;background:{bg};border-radius:8px;
                    margin:0.35rem 0;border-left:3px solid {border_color};">
            <div>
                <strong style="color:#f1f5f9">{step['name']}</strong> {status_badge}
                <br><small style="color:#94a3b8">{step.get('output_summary', '')}</small>
            </div>
            <div style="text-align:right;min-width:70px;">
                <span style="color:#14b8a6;font-weight:600;">{step['duration_s']:.3f}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_chw_view():
    """Main CHW view renderer."""
    from demo_data import DEMO_NOTES, DEMO_RESULTS, FAILURE_MODE

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin:0;font-size:2rem;font-weight:700;color:#1e2a1e;">CHW Field View</h1>
        <p style="color:#8a9a7a;margin:0.25rem 0 0 0;font-size:0.95rem;font-weight:400;">
            Enter a field note or select a demo case to process through the pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Section ────────────────────────────────────────
    tab_text, tab_demo = st.tabs(["Text Note", "Demo Cases"])

    selected_result = None
    selected_note = None

    with tab_text:
        note_input = st.text_area(
            "Field Note",
            height=110,
            placeholder="e.g. Child 3yo M fever 3 days cough bad rash on chest no diarrhea mother says not eating gave ORS referred health center",
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Process", type="primary", use_container_width=True):
                if note_input.strip():
                    st.session_state.custom_note = note_input
                    st.session_state.show_custom = True
                    st.session_state.pop("custom_result", None)  # clear previous result

        if st.session_state.get("show_custom") and st.session_state.get("custom_note"):
            # Run the pipeline — use MedGemma if available, else deterministic
            if "custom_result" not in st.session_state:
                import sys
                from pathlib import Path
                # Add project root to path for src imports
                project_root = str(Path(__file__).parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                try:
                    from src.pipeline import process_encounter
                    from src.models import is_model_available

                    if is_model_available():
                        # Live MedGemma mode
                        with st.spinner("Running 6-agent pipeline with MedGemma..."):
                            st.session_state.custom_result = process_encounter(
                                st.session_state.custom_note,
                                encounter_id="custom_001",
                                location_id="custom",
                                week_id=0,
                                extractor="medgemma",
                                use_model_tagger=True,
                                use_model_checklist=True,
                                run_hallucination_check=True,
                            )
                        st.session_state.custom_mode = "live"
                    else:
                        # Deterministic fallback mode
                        with st.spinner("Running 6-agent pipeline (deterministic mode)..."):
                            st.session_state.custom_result = process_encounter(
                                st.session_state.custom_note,
                                encounter_id="custom_001",
                                location_id="custom",
                                week_id=0,
                                extractor="stub",
                                use_model_tagger=False,
                                use_model_checklist=False,
                                run_hallucination_check=False,
                            )
                        st.session_state.custom_mode = "deterministic"
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.session_state.custom_result = None

            if st.session_state.get("custom_result"):
                if st.session_state.get("custom_mode") == "live":
                    st.success("Pipeline complete — powered by MedGemma 1.5 4B-IT")
                else:
                    st.info("Pipeline complete — running in demo mode (deterministic fallbacks). Deploy to HF Spaces with GPU for live MedGemma extraction.")
                selected_result = st.session_state.custom_result
                selected_note = st.session_state.custom_note

    with tab_demo:
        # Split demo buttons into two rows for readability
        row_size = 4
        for row_start in range(0, len(DEMO_NOTES), row_size):
            row_notes = DEMO_NOTES[row_start:row_start + row_size]
            cols = st.columns(row_size)
            for j, note in enumerate(row_notes):
                idx = row_start + j
                with cols[j]:
                    if st.button(note['title'], key=f"demo_{idx}", use_container_width=True):
                        selected_result = DEMO_RESULTS[idx]
                        selected_note = note["note_text"]
                        st.session_state.selected_demo = idx

        # Failure mode button on its own line
        fail_col, _ = st.columns([1, 3])
        with fail_col:
            if st.button("Failure Mode", key="demo_fail", use_container_width=True):
                st.session_state.show_failure = True

        # Check session state for persistent selection
        if st.session_state.get("selected_demo") is not None and selected_result is None:
            i = st.session_state.selected_demo
            selected_result = DEMO_RESULTS[i]
            selected_note = DEMO_NOTES[i]["note_text"]

    # ── Failure Mode View ────────────────────────────────────
    if st.session_state.get("show_failure"):
        st.markdown("---")
        st.markdown(f"### {FAILURE_MODE['title']}")
        st.warning(FAILURE_MODE["description"])

        col_note, col_result = st.columns(2)
        with col_note:
            st.markdown("**Raw Note:**")
            highlighted = highlight_evidence_in_note(
                FAILURE_MODE["note_text"],
                FAILURE_MODE["encounter"],
                FAILURE_MODE["hallucination_check"],
            )
            st.markdown(
                f'<div class="glass-card note-text" style="line-height:1.8">{highlighted}</div>',
                unsafe_allow_html=True,
            )

        with col_result:
            st.markdown("**Strawberry Detection:**")
            hc = FAILURE_MODE["hallucination_check"]
            for fc in hc.get("flagged_claims", []):
                st.markdown(
                    f'<div class="evidence-flagged">'
                    f'🚨 <strong>{fc["claim"]}</strong><br>'
                    f'<small>Budget gap: {fc["budget_gap"]:.1f} bits — {fc["reason"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**Budget Gaps (bits):**")
            for claim, gap in hc.get("budget_gaps", {}).items():
                color = "#ef4444" if gap > 2 else ("#f59e0b" if gap > 0 else "#22c55e")
                label = "FLAGGED" if gap > 2 else ("Suspicious" if gap > 0 else "Supported")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;">'
                    f'<span style="color:#94a3b8">{claim}</span>'
                    f'<span style="color:{color};font-weight:600">{gap:+.1f} — {label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("🔍 Pipeline Trace", expanded=False):
            render_pipeline_trace(FAILURE_MODE["agent_trace"])
        return

    # ── Results Display ──────────────────────────────────────
    if selected_result is None:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3rem;">'
            '<h3 style="color:#94a3b8">Select a demo case or enter a note to begin</h3>'
            '<p style="color:#64748b">The agentic pipeline will process the note through 6 specialized agents</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    encounter = selected_result["encounter"]
    syndrome = selected_result["syndrome_tag"]
    recommendations = selected_result.get("recommendations", [])
    hallucination = selected_result.get("hallucination_check", {})
    budget_gaps = hallucination.get("budget_gaps", {})

    st.markdown("---")

    # ── Side-by-side: Note ↔ Encounter ───────────────────────
    col_note, col_encounter = st.columns(2)

    with col_note:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Field Note</p>',
            unsafe_allow_html=True,
        )
        highlighted = highlight_evidence_in_note(selected_note, encounter, hallucination)
        st.markdown(
            f'<div class="note-text" style="line-height:1.85;font-size:0.95em">{highlighted}</div>',
            unsafe_allow_html=True,
        )

    with col_encounter:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Structured Encounter</p>',
            unsafe_allow_html=True,
        )
        # Patient info
        patient = encounter.get("patient", {})
        age = patient.get("age_years", "?")
        sex = patient.get("sex", "?")
        severity = encounter.get("severity", "?")
        st.markdown(
            f'<div class="glass-card">'
            f'<strong>Patient:</strong> {age}y {sex} &nbsp;|&nbsp; '
            f'<strong>Severity:</strong> {severity} &nbsp;|&nbsp; '
            f'<strong>Onset:</strong> {encounter.get("onset_days", "?")} day(s) ago'
            f'  <span style="color:#64748b;font-size:0.8rem">(est. onset week {encounter.get("estimated_onset_week", "?")})  </span>'
            f'&nbsp;|&nbsp; <strong>Referral:</strong> {"Yes ✓" if encounter.get("referral") else "No"}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Symptoms
        st.markdown("**Symptoms:**")
        for name, data in encounter.get("symptoms", {}).items():
            if isinstance(data, dict):
                render_symptom_card(name, data, budget_gaps)

        if encounter.get("other_symptoms"):
            st.markdown("**Other Symptoms:**")
            for name, data in encounter.get("other_symptoms", {}).items():
                if isinstance(data, dict):
                    render_symptom_card(name, data, budget_gaps)

        # Red flags
        if encounter.get("red_flags"):
            st.markdown("**🚩 Red Flags:**")
            for flag in encounter["red_flags"]:
                if isinstance(flag, dict):
                    st.markdown(
                        f'<div class="evidence-grounded">'
                        f'🚩 <strong>{flag.get("flag", "").replace("_", " ").title()}</strong>: '
                        f'<span class="evidence-quote">"…{flag.get("evidence_quote", "")}…"</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Syndrome Tag ─────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_syn, col_check = st.columns(2)

    with col_syn:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Syndrome Classification</p>',
            unsafe_allow_html=True,
        )
        tag = syndrome.get("syndrome_tag", "?")
        conf = syndrome.get("confidence", "?")
        conf_color = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(conf, "#94a3b8")

        st.markdown(
            f'<div class="glass-card">'
            f'<span style="font-size:1.5rem;font-weight:700;color:#14b8a6">{tag.replace("_", " ").upper()}</span>'
            f'&nbsp;&nbsp;<span style="background:rgba(0,0,0,0.3);color:{conf_color};'
            f'padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">{conf.upper()}</span>'
            f'<br><br><span style="color:#94a3b8">{syndrome.get("reasoning", "")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        sub = syndrome.get("sub_syndrome")
        if sub:
            sub_colors = {
                "pneumonia-like": "#f59e0b",
                "malaria-like":   "#a78bfa",
                "TB-like":        "#f87171",
                "upper-respiratory": "#60a5fa",
                "lower-respiratory": "#f97316",
                "measles-like":     "#ec4899",
            }
            sc = sub_colors.get(sub, "#94a3b8")
            st.markdown(
                f'<div style="margin-top:0.5rem;">'  
                f'<span style="background:rgba(0,0,0,0.3);color:{sc};'
                f'padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;">'
                f'🔍 {sub.upper()}</span></div>',
                unsafe_allow_html=True,
            )

        if syndrome.get("trigger_quotes"):
            st.markdown("**Trigger Quotes:**")
            for tq in syndrome["trigger_quotes"]:
                st.markdown(f'<span class="evidence-quote">"…{tq}…"</span>', unsafe_allow_html=True)

    # ── ICCM Recommendations ─────────────────────────────────
    with col_check:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">ICCM Recommendations</p>',
            unsafe_allow_html=True,
        )
        if recommendations:
            for rec in recommendations:
                is_urgent = rec.startswith("🚨")
                bg = "rgba(239,68,68,0.15)" if is_urgent else "rgba(255,255,255,0.05)"
                border = "#ef4444" if is_urgent else "#334155"
                st.markdown(
                    f'<div style="background:{bg};border-left:3px solid {border};'
                    f'padding:0.5rem 0.75rem;margin-bottom:0.4rem;border-radius:4px;'
                    f'font-size:0.9rem;line-height:1.5;">'
                    f'{rec}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recommendations generated.")

    # ── Pipeline Trace ───────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Follow-up Checklist ───────────────────────────────────
    checklist = selected_result.get("checklist", [])
    if checklist:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Follow-up Checklist (WHO ICCM Protocol)</p>',
            unsafe_allow_html=True,
        )
        for i, question in enumerate(checklist, 1):
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03);border-left:3px solid #5a7a4a;'
                f'padding:0.5rem 0.75rem;margin-bottom:0.35rem;border-radius:4px;'
                f'font-size:0.88rem;line-height:1.5;color:#c8d8b8;">'
                f'<strong style="color:#8da87d;">{i}.</strong> {question}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.expander("Pipeline Trace — Agent-by-Agent Execution", expanded=False):
        render_pipeline_trace(selected_result.get("agent_trace", []))
