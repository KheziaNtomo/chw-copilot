"""CHW View -- Field worker interface for note processing.

Layout:
1. Demo selector: Adult/Child tabs -> disease dropdown -> case buttons
2. Field Note with evidence highlighting
3. Syndrome Classification badge (under the note)
4. Three boxes: Symptoms Extracted | Symptoms Unknown | Red Flags
5. Actions Recommended (pre-ticked for done, un-ticked for ICCM remaining)
6. Pipeline trace (collapsible)
"""
import streamlit as st
import re
from typing import Dict, Any, Optional


def _strip_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    return re.sub(r'[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F600-\U0001F64F'
                  r'\U0001F680-\U0001F6FF\U00002702-\U000027B0\U0000FE0F'
                  r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F'
                  r'\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+', '', text).strip()


def highlight_evidence_in_note(note_text: str, encounter: Dict, hallucination_check: Optional[Dict] = None) -> str:
    """Highlight evidence quotes in the original note text."""
    highlighted = note_text
    flagged_quotes = set()

    if hallucination_check and hallucination_check.get("flagged"):
        for fc in hallucination_check.get("flagged_claims", []):
            claim = fc.get("claim", "")
            for sym_data in encounter.get("symptoms", {}).values():
                if isinstance(sym_data, dict) and sym_data.get("evidence_quote"):
                    if sym_data["evidence_quote"].lower() in claim.lower() or claim.lower() in sym_data["evidence_quote"].lower():
                        flagged_quotes.add(sym_data["evidence_quote"])

    all_quotes = []
    for section in ["symptoms", "other_symptoms"]:
        for key, val in encounter.get(section, {}).items():
            if isinstance(val, dict) and val.get("evidence_quote"):
                quote = val["evidence_quote"]
                if quote:
                    all_quotes.append((quote, quote in flagged_quotes))

    for flag in encounter.get("red_flags", []):
        if isinstance(flag, dict) and flag.get("evidence_quote"):
            all_quotes.append((flag["evidence_quote"], False))

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


def render_pipeline_trace(agent_trace: list) -> None:
    """Render the pipeline agent trace."""
    total_time = sum(step.get("duration_s", 0) for step in agent_trace)
    st.markdown(f"**Total pipeline time: {total_time:.2f}s** | {len(agent_trace)} agents")

    for step in agent_trace:
        fallback = step.get("fallback_used", False)
        is_flagged = "flagged" in step.get("output_summary", "").lower() and "0 flagged" not in step.get("output_summary", "")
        border_color = "#c0392b" if is_flagged else ("#b5770d" if fallback else "#8D957E")
        bg = "rgba(192,57,43,0.08)" if is_flagged else ("rgba(181,119,13,0.08)" if fallback else "rgba(141,149,126,0.06)")

        status_badge = ""
        if is_flagged:
            status_badge = '<span style="background:rgba(192,57,43,0.15);color:#c0392b;padding:2px 8px;border-radius:10px;font-size:0.75em;font-weight:600;">FLAGGED</span>'
        elif fallback:
            status_badge = '<span style="background:rgba(181,119,13,0.15);color:#b5770d;padding:2px 8px;border-radius:10px;font-size:0.75em;font-weight:600;">FALLBACK</span>'

        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:0.6rem 1rem;background:{bg};border-radius:8px;
                    margin:0.35rem 0;border-left:3px solid {border_color};">
            <div>
                <strong style="color:#1a1510">{step['name']}</strong> {status_badge}
                <br><small style="color:#6a6255">{step.get('output_summary', '')}</small>
            </div>
            <div style="text-align:right;min-width:70px;">
                <span style="color:#974315;font-weight:600;">{step['duration_s']:.3f}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_chw_view():
    """Main CHW view renderer."""
    from demo_data import DEMO_NOTES, DEMO_RESULTS, FAILURE_MODE

    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin:0;font-size:2rem;font-weight:700;">CHW Field View</h1>
        <p style="color:#6a6255;margin:0.25rem 0 0 0;font-size:0.95rem;font-weight:400;">
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
                    st.session_state.pop("custom_result", None)

        if st.session_state.get("show_custom") and st.session_state.get("custom_note"):
            if "custom_result" not in st.session_state:
                import sys
                from pathlib import Path
                project_root = str(Path(__file__).parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                try:
                    from src.pipeline import process_encounter
                    from src.models import is_model_available

                    if is_model_available():
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
                    st.success("Pipeline complete -- powered by MedGemma 4B-IT")
                else:
                    st.info("Pipeline complete -- running in demo mode (deterministic fallbacks).")
                selected_result = st.session_state.custom_result
                selected_note = st.session_state.custom_note

    with tab_demo:
        # ── Two-level selector: Adult/Paediatric tabs -> Disease dropdown -> Cases ──

        # Categorise demo notes by age
        paediatric = []
        adult = []
        for i, note in enumerate(DEMO_NOTES):
            if i >= len(DEMO_RESULTS):
                break
            result = DEMO_RESULTS[i]
            age = result["encounter"].get("patient", {}).get("age_years", 0)
            age_months = result["encounter"].get("patient", {}).get("age_months")
            if (age is not None and age < 18) or age_months is not None:
                paediatric.append((i, note, result))
            else:
                adult.append((i, note, result))

        age_tab_paed, age_tab_adult = st.tabs(["Paediatric", "Adult"])

        def _render_disease_selector(cases, prefix):
            """Render disease dropdown then case buttons."""
            # Group by syndrome
            categories = {}
            for idx, note, result in cases:
                syn = result.get("syndrome_tag", {}).get("syndrome_tag", "other")
                if "respiratory" in syn:
                    cat = "Respiratory"
                elif "diarrhea" in syn or "diarrhoea" in syn:
                    cat = "Diarrhoeal"
                else:
                    cat = "Other"
                categories.setdefault(cat, []).append((idx, note))

            available = [c for c in ["Respiratory", "Diarrhoeal", "Other"] if c in categories]
            if not available:
                st.info("No cases in this category.")
                return

            disease = st.selectbox(
                "Disease type",
                available,
                key=f"disease_{prefix}",
            )

            if disease and disease in categories:
                cat_cases = categories[disease]
                cols = st.columns(min(len(cat_cases), 4))
                for j, (idx, note) in enumerate(cat_cases):
                    with cols[j % len(cols)]:
                        if st.button(note["title"], key=f"demo_{prefix}_{idx}", use_container_width=True):
                            st.session_state.selected_demo = idx
                            st.session_state.pop("show_failure", None)

        with age_tab_paed:
            _render_disease_selector(paediatric, "paed")

        with age_tab_adult:
            _render_disease_selector(adult, "adult")

        # Failure mode button
        st.markdown("---")
        fail_col, _ = st.columns([1, 3])
        with fail_col:
            if st.button("Failure Mode (Hallucination)", key="demo_fail", use_container_width=True):
                st.session_state.show_failure = True
                st.session_state.pop("selected_demo", None)

        # Persistent selection from session
        if st.session_state.get("selected_demo") is not None and selected_result is None:
            i = st.session_state.selected_demo
            if i < len(DEMO_RESULTS):
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
            st.markdown("**Hallucination Detection:**")
            hc = FAILURE_MODE["hallucination_check"]
            for fc in hc.get("flagged_claims", []):
                st.markdown(
                    f'<div class="evidence-flagged">'
                    f'<strong>{fc["claim"]}</strong><br>'
                    f'<small>Confidence gap: {fc["budget_gap"]:.1f} bits -- {fc["reason"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**Confidence Scores (bits):**")
            for claim, gap in hc.get("budget_gaps", {}).items():
                color = "#c0392b" if gap > 2 else ("#b5770d" if gap > 0 else "#5a7a4a")
                label = "FLAGGED" if gap > 2 else ("Suspicious" if gap > 0 else "Supported")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;">'
                    f'<span style="color:#6a6255">{claim}</span>'
                    f'<span style="color:{color};font-weight:600">{gap:+.1f} -- {label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Pipeline Trace", expanded=False):
            render_pipeline_trace(FAILURE_MODE["agent_trace"])
        return

    # ── Results Display ──────────────────────────────────────
    if selected_result is None:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3rem;">'
            '<h3 style="color:#6a6255">Select a demo case or enter a note to begin</h3>'
            '<p style="color:#8D957E">The agentic pipeline will process the note through 6 specialised agents</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    encounter = selected_result["encounter"]
    syndrome = selected_result["syndrome_tag"]
    recommendations = selected_result.get("recommendations", [])
    checklist = selected_result.get("checklist", [])
    hallucination = selected_result.get("hallucination_check", {})
    budget_gaps = hallucination.get("budget_gaps", {})

    st.markdown("---")

    # ================================================================
    # ROW 1: Field Note + Syndrome Classification
    # ================================================================
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'font-weight:700;color:#6a6255;margin-bottom:0.5rem;">Field Note</p>',
        unsafe_allow_html=True,
    )
    highlighted = highlight_evidence_in_note(selected_note, encounter, hallucination)
    st.markdown(
        f'<div class="note-text" style="line-height:1.85;font-size:0.95em">{highlighted}</div>',
        unsafe_allow_html=True,
    )

    # Syndrome badge directly under the note
    tag = syndrome.get("syndrome_tag", "?")
    conf = syndrome.get("confidence", "?")
    conf_color = {"high": "#5a7a4a", "medium": "#b5770d", "low": "#c0392b"}.get(conf, "#6a6255")
    sub = syndrome.get("sub_syndrome", "")
    sub_badge = ""
    if sub:
        sub_badge = (
            f'&nbsp;&nbsp;<span style="background:rgba(151,67,21,0.12);color:#974315;'
            f'padding:3px 10px;border-radius:16px;font-size:0.75rem;font-weight:600;">'
            f'{sub.upper()}</span>'
        )

    st.markdown(
        f'<div style="margin:0.75rem 0 0.25rem 0;">'
        f'<span style="font-size:1.2rem;font-weight:700;color:#974315">{tag.replace("_", " ").upper()}</span>'
        f'&nbsp;&nbsp;<span style="background:rgba(151,67,21,0.12);color:{conf_color};'
        f'padding:3px 10px;border-radius:16px;font-size:0.75rem;font-weight:600;">{conf.upper()}</span>'
        f'{sub_badge}'
        f'</div>'
        f'<div style="color:#6a6255;font-size:0.85rem;margin-bottom:0.5rem;">{syndrome.get("reasoning", "")}</div>',
        unsafe_allow_html=True,
    )

    # ================================================================
    # ROW 2: Three boxes — Symptoms Extracted | Symptoms Unknown | Red Flags
    # ================================================================
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Categorise symptoms
    symptoms_yes = []  # extracted / grounded
    symptoms_unknown = []  # unknown or absent
    red_flags = []

    for name, data in encounter.get("symptoms", {}).items():
        if isinstance(data, dict):
            value = data.get("value", "unknown")
            quote = data.get("evidence_quote", "")
            display_name = name.replace("_", " ").title()
            if value == "yes":
                # Check grounding
                matching_gap = None
                if budget_gaps:
                    for k, v in budget_gaps.items():
                        if name.replace("_", " ") in k.lower():
                            matching_gap = v
                            break
                grounding = "Grounded" if quote else "No evidence"
                if matching_gap is not None and matching_gap > 2:
                    grounding = f"Flagged (gap: {matching_gap:.1f})"
                symptoms_yes.append((display_name, quote, grounding))
            else:
                symptoms_unknown.append((display_name, value))

    for name, data in encounter.get("other_symptoms", {}).items():
        if isinstance(data, dict):
            value = data.get("value", "unknown")
            display_name = name.replace("_", " ").title()
            symptoms_unknown.append((display_name, value))

    for flag in encounter.get("red_flags", []):
        if isinstance(flag, dict):
            red_flags.append((
                flag.get("flag", "").replace("_", " ").title(),
                flag.get("evidence_quote", ""),
            ))

    col_extracted, col_unknown, col_flags = st.columns(3)

    with col_extracted:
        st.markdown(
            '<p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#5a7a4a;margin-bottom:0.4rem;">Symptoms Extracted</p>',
            unsafe_allow_html=True,
        )
        if symptoms_yes:
            for name, quote, grounding in symptoms_yes:
                quote_html = f'<br><span style="color:#6a6255;font-size:0.8rem;font-style:italic;">"{quote}"</span>' if quote else ""
                grounding_color = "#5a7a4a" if "Grounded" in grounding else "#c0392b"
                st.markdown(
                    f'<div style="padding:0.35rem 0;border-bottom:1px solid rgba(0,0,0,0.05);">'
                    f'<span style="color:#5a7a4a;font-weight:700;">&#10003;</span> '
                    f'<strong>{name}</strong> '
                    f'<span style="color:{grounding_color};font-size:0.75rem;">({grounding})</span>'
                    f'{quote_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span style="color:#6a6255">None extracted</span>', unsafe_allow_html=True)

    with col_unknown:
        st.markdown(
            '<p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#b5770d;margin-bottom:0.4rem;">Symptoms Unknown / Absent</p>',
            unsafe_allow_html=True,
        )
        if symptoms_unknown:
            for name, value in symptoms_unknown:
                icon = '<span style="color:#c0392b;">&#10007;</span>' if value == "no" else '<span style="color:#b5770d;">?</span>'
                label = "absent" if value == "no" else "unknown"
                st.markdown(
                    f'<div style="padding:0.35rem 0;border-bottom:1px solid rgba(0,0,0,0.05);">'
                    f'{icon} <strong>{name}</strong> '
                    f'<span style="color:#6a6255;font-size:0.8rem;">({label})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span style="color:#6a6255">None</span>', unsafe_allow_html=True)

    with col_flags:
        st.markdown(
            '<p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#c0392b;margin-bottom:0.4rem;">Red Flags</p>',
            unsafe_allow_html=True,
        )
        if red_flags:
            for name, quote in red_flags:
                quote_html = f'<br><span style="color:#6a6255;font-size:0.8rem;font-style:italic;">"{quote}"</span>' if quote else ""
                st.markdown(
                    f'<div style="padding:0.35rem 0;border-bottom:1px solid rgba(0,0,0,0.05);">'
                    f'<span style="color:#c0392b;font-weight:700;">!</span> '
                    f'<strong style="color:#c0392b;">{name}</strong>'
                    f'{quote_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span style="color:#6a6255">None identified</span>', unsafe_allow_html=True)

    # ================================================================
    # ROW 3: Actions Recommended (left) | Follow-up (right)
    # ================================================================
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_actions, col_followup = st.columns(2)

    with col_actions:
        st.markdown(
            '<p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#6a6255;margin-bottom:0.4rem;">Actions Recommended / Taken</p>',
            unsafe_allow_html=True,
        )
        if recommendations:
            note_lower = selected_note.lower()
            for i, rec in enumerate(recommendations):
                clean_rec = _strip_emojis(rec).strip()
                if not clean_rec:
                    continue
                # Pre-tick if action seems done based on the field note
                keywords = ["gave ors", "referred", "ors", "rdt", "act", "amoxicillin",
                            "zinc", "paracetamol", "referred health center", "referred hospital"]
                pre_ticked = any(kw in note_lower for kw in keywords
                                if kw in clean_rec.lower())
                st.checkbox(clean_rec, key=f"action_{i}", value=pre_ticked)
        else:
            st.info("No recommendations generated.")

    with col_followup:
        st.markdown(
            '<p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#6a6255;margin-bottom:0.4rem;">Follow-up (WHO ICCM)</p>',
            unsafe_allow_html=True,
        )
        if checklist:
            for i, question in enumerate(checklist, 1):
                clean_q = _strip_emojis(question).strip()
                st.markdown(
                    f'<div style="background:rgba(151,67,21,0.06);border-left:3px solid #974315;'
                    f'padding:0.5rem 0.75rem;margin-bottom:0.35rem;border-radius:4px;'
                    f'font-size:0.88rem;line-height:1.5;color:#1a1510;">'
                    f'<strong style="color:#974315;">{i}.</strong> {clean_q}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span style="color:#6a6255">No follow-up items.</span>', unsafe_allow_html=True)

    # ── Pipeline Trace ───────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.expander("Pipeline Trace -- Agent-by-Agent Execution", expanded=False):
        render_pipeline_trace(selected_result.get("agent_trace", []))
