"""Pre-computed demo data for offline Streamlit demonstration.

Contains representative CHW notes, full pipeline outputs (including
agent traces), and a failure mode example showing Strawberry catching
a hallucination. Works without GPU/model access.
"""

# ── Sample CHW Notes ─────────────────────────────────────────
DEMO_NOTES = [
    {
        "id": "demo_001",
        "title": "Respiratory case — pediatric",
        "note_text": (
            "Child 3yo M fever 3 days cough bad rash on chest "
            "no diarrhea mother says not eating gave ORS referred health center"
        ),
        "location_id": "loc_01",
        "week_id": 5,
    },
    {
        "id": "demo_002",
        "title": "Acute watery diarrhea — infant",
        "note_text": (
            "Baby 9 months F watery diarrhea 2 days vomiting unable to drink "
            "sunken eyes mother reports no urine since morning gave ORS "
            "referred urgent health facility"
        ),
        "location_id": "loc_02",
        "week_id": 5,
    },
    {
        "id": "demo_003",
        "title": "Malaria-like symptoms — adult",
        "note_text": (
            "Woman 28 years headache 4 days joint pain high fever sweating at night "
            "no cough no diarrhea took paracetamol not improving RDT positive "
            "referred clinic for ACT"
        ),
        "location_id": "loc_01",
        "week_id": 5,
    },
]

# ── Full Pipeline Outputs (Pre-computed) ─────────────────────
DEMO_RESULTS = [
    {
        "encounter": {
            "encounter_id": "demo_001",
            "location_id": "loc_01",
            "week_id": 5,
            "patient": {"age_years": 3, "sex": "male"},
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "fever 3 days"},
                "cough": {"value": "yes", "evidence_quote": "cough bad"},
                "diarrhea": {"value": "no", "evidence_quote": "no diarrhea"},
                "vomiting": {"value": "unknown", "evidence_quote": ""},
                "rash": {"value": "yes", "evidence_quote": "rash on chest"},
            },
            "other_symptoms": {
                "loss_of_appetite": {"value": "yes", "evidence_quote": "not eating"},
            },
            "onset_days": 3,
            "estimated_onset_week": 5,
            "severity": "moderate",
            "red_flags": [
                {"flag": "not_eating", "evidence_quote": "mother says not eating"},
            ],
            "treatment_given": ["ORS"],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "respiratory_fever",
            "sub_syndrome": "upper-respiratory",
            "confidence": "high",
            "reasoning": "Fever + cough. No difficulty breathing — upper respiratory presentation.",
            "trigger_quotes": ["fever", "cough bad"],
        },
        "recommendations": [
            "🌡️ Give paracetamol for fever; do malaria RDT if available",
            "📋 No urgent action needed — routine follow-up",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.341, "fallback_used": False, "output_summary": "Extracted encounter with 4 positive symptoms"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.002, "fallback_used": False, "output_summary": "0 claims downgraded — fuzzy match accepted"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as respiratory_fever (high) — keyword: fever + cough"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "Sub-syndrome: upper-respiratory (cough, no fast breathing)"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 2 recommendations"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True, overall pass: True"},
        ],
        "processing_time_s": 2.35,
    },
    {
        "encounter": {
            "encounter_id": "demo_002",
            "location_id": "loc_02",
            "week_id": 5,
            "patient": {"age_years": 0, "age_months": 9, "sex": "female"},
            "symptoms": {
                "fever": {"value": "unknown", "evidence_quote": ""},
                "cough": {"value": "unknown", "evidence_quote": ""},
                "diarrhea": {"value": "yes", "evidence_quote": "watery diarrhea 2 days"},
                "vomiting": {"value": "yes", "evidence_quote": "vomiting"},
                "rash": {"value": "unknown", "evidence_quote": ""},
            },
            "other_symptoms": {},
            "onset_days": 2,
            "estimated_onset_week": 5,
            "severity": "severe",
            "red_flags": [
                {"flag": "unable_to_drink", "evidence_quote": "unable to drink"},
                {"flag": "sunken_eyes", "evidence_quote": "sunken eyes"},
                {"flag": "no_urine", "evidence_quote": "no urine since morning"},
            ],
            "treatment_given": ["ORS"],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "acute_watery_diarrhea",
            "sub_syndrome": None,
            "confidence": "high",
            "reasoning": "Watery diarrhea + vomiting + dehydration signs. Multiple red flags.",
            "trigger_quotes": ["watery diarrhea 2 days", "vomiting", "sunken eyes"],
        },
        "recommendations": [
            "🚨 REFER IMMEDIATELY — danger sign(s): unable to drink, dehydration signs",
            "💧 Start ORS immediately; give zinc (10mg if <6mo, 20mg if ≥6mo) for 10 days",
            "⚠️ Persistent vomiting — give ORS in small sips, monitor for dehydration",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.105, "fallback_used": False, "output_summary": "Extracted encounter with 2 positive symptoms"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.001, "fallback_used": False, "output_summary": "0 claims downgraded"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as acute_watery_diarrhea (high)"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "N/A for AWD"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 3 recommendations — REFER IMMEDIATELY"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True"},
        ],
        "processing_time_s": 2.11,
    },
    {
        "encounter": {
            "encounter_id": "demo_003",
            "location_id": "loc_01",
            "week_id": 5,
            "patient": {"age_years": 28, "sex": "female"},
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "high fever"},
                "cough": {"value": "no", "evidence_quote": "no cough"},
                "diarrhea": {"value": "no", "evidence_quote": "no diarrhea"},
                "vomiting": {"value": "unknown", "evidence_quote": ""},
                "rash": {"value": "unknown", "evidence_quote": ""},
            },
            "other_symptoms": {
                "headache": {"value": "yes", "evidence_quote": "headache 4 days"},
                "joint_pain": {"value": "yes", "evidence_quote": "joint pain"},
            },
            "onset_days": 4,
            "estimated_onset_week": 5,
            "severity": "moderate",
            "red_flags": [],
            "treatment_given": ["paracetamol"],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "other",
            "sub_syndrome": "malaria-like",
            "confidence": "medium",
            "reasoning": "Fever + rigors/chills, no cough, positive RDT — malaria-like presentation.",
            "trigger_quotes": ["high fever", "sweating at night"],
        },
        "recommendations": [
            "🌡️ Give paracetamol for fever; do malaria RDT if available",
            "💊 Malaria RDT positive — give ACT (artemisinin-based combination therapy)",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.412, "fallback_used": False, "output_summary": "Extracted encounter with 3 positive symptoms"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.002, "fallback_used": False, "output_summary": "0 claims downgraded"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as other (medium) — fever, no respiratory symptoms"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "Sub-syndrome: malaria-like (fever + RDT positive, no cough)"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 2 recommendations incl. ACT"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True"},
        ],
        "processing_time_s": 2.42,
    },
]

# ── Failure Mode Demo ────────────────────────────────────────
# Shows Strawberry catching a procedural hallucination
FAILURE_MODE = {
    "title": "🚨 Failure Mode: Hallucination Detected",
    "description": (
        "This example shows MedGemma reporting 'rash: yes' even though "
        "the note says 'no rash observed.' The evidence quote exists in "
        "the note (passing enforce_evidence), but Strawberry detects that "
        "the quote contradicts the claim."
    ),
    "note_text": (
        "Child 5yo M fever 2 days cough runny nose no rash observed "
        "no diarrhea drinking well referred health center"
    ),
    "encounter": {
        "encounter_id": "fail_001",
        "location_id": "loc_01",
        "week_id": 5,
        "patient": {"age_years": 5, "sex": "male"},
        "symptoms": {
            "fever": {"value": "yes", "evidence_quote": "fever 2 days"},
            "cough": {"value": "yes", "evidence_quote": "cough"},
            "diarrhea": {"value": "no", "evidence_quote": "no diarrhea"},
            "vomiting": {"value": "unknown", "evidence_quote": ""},
            "rash": {"value": "yes", "evidence_quote": "no rash observed"},
        },
        "other_symptoms": {},
        "onset": "2 days",
        "duration_days": 2,
        "severity": "mild",
        "red_flags": [],
        "treatment_given": [],
        "referral": True,
    },
    "hallucination_check": {
        "flagged": True,
        "claims_checked": 3,
        "flagged_claims": [
            {
                "claim": "Patient has rash [S2]",
                "budget_gap": 9.7,
                "reason": "Evidence 'no rash observed' contradicts 'rash: yes' — quote negates the claim",
            },
        ],
        "budget_gaps": {
            "Patient has fever [S0]": -2.1,
            "Patient has cough [S1]": -1.5,
            "Patient has rash [S2]": 9.7,
        },
        "available": True,
    },
    "agent_trace": [
        {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.105, "fallback_used": False, "output_summary": "Extracted encounter with 3 positive symptoms"},
        {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.001, "fallback_used": False, "output_summary": "0 claims downgraded (quote IS in note)"},
        {"agent": "strawberry_verify", "name": "Hallucination Detector", "duration_s": 1.520, "fallback_used": False, "output_summary": "Checked 3 claims, 1 FLAGGED — rash claim contradicted"},
        {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 1.102, "fallback_used": False, "output_summary": "Tagged as respiratory_fever (high)"},
    ],
}

# ── Surveillance Demo Data ───────────────────────────────────
DEMO_SURVEILLANCE = {
    "weekly_counts": [
        {"week_id": 1, "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 3},
        {"week_id": 1, "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 1},
        {"week_id": 1, "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 2},
        {"week_id": 2, "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 4},
        {"week_id": 2, "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 2},
        {"week_id": 2, "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 3},
        {"week_id": 3, "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 5},
        {"week_id": 3, "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 1},
        {"week_id": 3, "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 2},
        {"week_id": 4, "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 3},
        {"week_id": 4, "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 3},
        {"week_id": 4, "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 4},
        {"week_id": 5, "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 12},
        {"week_id": 5, "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 2},
        {"week_id": 5, "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 3},
        {"week_id": 5, "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 1},
    ],
    "anomalies": [
        {
            "location_id": "loc_01",
            "syndrome_tag": "respiratory_fever",
            "week_id": 5,
            "current_count": 12,
            "baseline_mean": 3.75,
            "ratio": 3.2,
            "alert": True,
            "message": "Respiratory fever cases at Kibera Health Post are 3.2× above baseline (12 vs mean 3.75)",
        },
    ],
    "sitrep": {
        "week_id": 5,
        "generated_by": "template",
        "narrative": (
            "WEEK 5 SITUATION REPORT — CHW Copilot Syndromic Surveillance\n\n"

            "ALERT: Respiratory fever syndrome cases at Kibera Health Post have surged to 12 this week, "
            "representing a 3.2-fold increase above the 4-week baseline mean of 3.75 cases. "
            "This crosses the alert threshold and warrants immediate investigation.\n\n"

            "WHAT THIS SYNDROME COVERS: Respiratory fever syndrome is a syndromic grouping — "
            "it does not represent a single diagnosis. Conditions commonly presenting under this label include:\n"
            "  · Malaria: fever, chills, and rigors without prominent cough; highly likely in rainy season\n"
            "  · Pneumonia (bacterial/viral): fever with cough and fast or difficult breathing\n"
            "  · Influenza-like illness (ILI): fever, body aches, cough, often with cluster spread\n"
            "  · Upper respiratory tract infection (URTI): fever with sore throat or runny nose, milder course\n"
            "  · Pulmonary tuberculosis (TB): chronic cough >2 weeks, weight loss, night sweats\n\n"

            "RECOMMENDED ACTIONS:\n"
            "  1. Deploy additional CHWs to Kibera Health Post for active case finding\n"
            "  2. Conduct malaria RDT testing for all fever presentations in this catchment\n"
            "  3. Ensure adequate stock of ACT, amoxicillin, and ORS at health post\n"
            "  4. Screen for chest indrawing and fast breathing in all paediatric cases\n"
            "  5. Monitor Mathare Community Center for spillover — currently within normal range\n\n"

            "Acute watery diarrhea (AWD) remains within expected baseline across all reporting locations. "
            "AWD covers presentations consistent with cholera, rotavirus, food-borne illness, and "
            "Enterotoxigenic E. coli (ETEC) — community water source monitoring is advised during rainy season.\n\n"

            "Data source: CHW Copilot syndromic surveillance network · 2 locations · 5 weeks of data"
        ),
        "alerts": [
            {
                "severity": "high",
                "location": "loc_01",
                "syndrome": "respiratory_fever",
                "message": "12 cases (3.2× baseline) — investigate immediately",
            }
        ],
    },
}

# Location metadata for map visualization
DEMO_LOCATIONS = {
    "loc_01": {"name": "Kibera Health Post", "lat": -1.3133, "lon": 36.7876},
    "loc_02": {"name": "Mathare Community Center", "lat": -1.2572, "lon": 36.8575},
}
