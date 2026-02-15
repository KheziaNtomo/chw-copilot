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
            "onset": "3 days",
            "duration_days": 3,
            "severity": "moderate",
            "red_flags": [
                {"flag": "not_eating", "evidence_quote": "mother says not eating"},
            ],
            "treatment_given": ["ORS"],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "respiratory_fever",
            "confidence": "high",
            "reasoning": "Fever + cough + rash. No diarrhea rules out acute_watery_diarrhea.",
            "trigger_quotes": ["fever 3 days", "cough bad"],
        },
        "checklist": {
            "questions": [
                {"priority": "high", "question": "What is the child's respiratory rate?"},
                {"priority": "high", "question": "Is there chest indrawing present?"},
                {"priority": "medium", "question": "Has the child been vaccinated for measles?"},
                {"priority": "medium", "question": "What is the child's temperature reading?"},
                {"priority": "low", "question": "Is the rash spreading or changing?"},
            ],
        },
        "validation": {
            "schema_valid": True,
            "schema_errors": [],
            "evidence_downgrades": 0,
            "overall_pass": True,
        },
        "evidence_downgrades": [],
        "invalid_trigger_quotes": [],
        "hallucination_check": {
            "flagged": False,
            "claims_checked": 4,
            "flagged_claims": [],
            "budget_gaps": {
                "Patient has fever [S0]": -2.1,
                "Patient has cough [S1]": -1.8,
                "Patient has rash [S2]": -1.5,
                "Patient has loss of appetite [S3]": -0.9,
            },
            "available": True,
        },
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.341, "fallback_used": False, "output_summary": "Extracted encounter with 4 positive symptoms"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.002, "fallback_used": False, "output_summary": "0 claims downgraded for missing/invalid evidence"},
            {"agent": "strawberry_verify", "name": "Hallucination Detector", "duration_s": 1.205, "fallback_used": False, "output_summary": "Checked 4 claims, 0 flagged"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 1.102, "fallback_used": False, "output_summary": "Tagged as respiratory_fever (high)"},
            {"agent": "checklist", "name": "Checklist Generator", "duration_s": 0.893, "fallback_used": False, "output_summary": "Generated 5 follow-up questions"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True, overall pass: True"},
        ],
        "processing_time_s": 5.54,
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
            "onset": "2 days",
            "duration_days": 2,
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
            "confidence": "high",
            "reasoning": "Watery diarrhea + vomiting + dehydration signs. Multiple red flags.",
            "trigger_quotes": ["watery diarrhea 2 days", "vomiting", "sunken eyes"],
        },
        "checklist": {
            "questions": [
                {"priority": "high", "question": "How many stools in the last 24 hours?"},
                {"priority": "high", "question": "Is the skin pinch slow (>2 seconds)?"},
                {"priority": "medium", "question": "Is the child still breastfeeding?"},
            ],
        },
        "validation": {
            "schema_valid": True,
            "schema_errors": [],
            "evidence_downgrades": 0,
            "overall_pass": True,
        },
        "evidence_downgrades": [],
        "invalid_trigger_quotes": [],
        "hallucination_check": {
            "flagged": False,
            "claims_checked": 5,
            "flagged_claims": [],
            "budget_gaps": {},
            "available": True,
        },
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.105, "fallback_used": False, "output_summary": "Extracted encounter with 2 positive symptoms"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.001, "fallback_used": False, "output_summary": "0 claims downgraded for missing/invalid evidence"},
            {"agent": "strawberry_verify", "name": "Hallucination Detector", "duration_s": 1.310, "fallback_used": False, "output_summary": "Checked 5 claims, 0 flagged"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.982, "fallback_used": False, "output_summary": "Tagged as acute_watery_diarrhea (high)"},
            {"agent": "checklist", "name": "Checklist Generator", "duration_s": 0.745, "fallback_used": False, "output_summary": "Generated 3 follow-up questions"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True, overall pass: True"},
        ],
        "processing_time_s": 5.14,
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
            "onset": "4 days",
            "duration_days": 4,
            "severity": "moderate",
            "red_flags": [],
            "treatment_given": ["paracetamol"],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "other",
            "confidence": "medium",
            "reasoning": "Fever + headache + joint pain + positive RDT suggests malaria. No respiratory or diarrheal syndrome.",
            "trigger_quotes": ["high fever", "headache 4 days", "joint pain"],
        },
        "checklist": {
            "questions": [
                {"priority": "high", "question": "Was the RDT result confirmed?"},
                {"priority": "high", "question": "Has ACT treatment been started?"},
                {"priority": "medium", "question": "Is the patient pregnant or breastfeeding?"},
                {"priority": "low", "question": "Has the patient had previous malaria episodes?"},
            ],
        },
        "validation": {
            "schema_valid": True,
            "schema_errors": [],
            "evidence_downgrades": 0,
            "overall_pass": True,
        },
        "evidence_downgrades": [],
        "invalid_trigger_quotes": [],
        "hallucination_check": {
            "flagged": False,
            "claims_checked": 3,
            "flagged_claims": [],
            "budget_gaps": {},
            "available": True,
        },
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.412, "fallback_used": False, "output_summary": "Extracted encounter with 3 positive symptoms"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.002, "fallback_used": False, "output_summary": "0 claims downgraded for missing/invalid evidence"},
            {"agent": "strawberry_verify", "name": "Hallucination Detector", "duration_s": 1.108, "fallback_used": False, "output_summary": "Checked 3 claims, 0 flagged"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 1.230, "fallback_used": False, "output_summary": "Tagged as other (medium)"},
            {"agent": "checklist", "name": "Checklist Generator", "duration_s": 0.891, "fallback_used": False, "output_summary": "Generated 4 follow-up questions"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True, overall pass: True"},
        ],
        "processing_time_s": 5.64,
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
            "message": "Respiratory fever cases at loc_01 are 3.2x above baseline (12 vs mean 3.75)",
        },
    ],
    "sitrep": {
        "week_id": 5,
        "generated_by": "template",
        "narrative": (
            "Week 5 Situation Report — CHW Copilot Syndromic Surveillance\n\n"
            "ALERT: Respiratory fever cases at loc_01 have surged to 12, "
            "3.2x above the 4-week baseline mean of 3.75. This exceeds "
            "the alert threshold and warrants immediate investigation.\n\n"
            "Acute watery diarrhea remains within normal range across all locations.\n\n"
            "Recommended actions:\n"
            "1. Deploy additional CHWs to loc_01 for active case finding\n"
            "2. Ensure respiratory infection treatment supplies at loc_01 health center\n"
            "3. Investigate potential common exposure source\n"
            "4. Continue monitoring loc_02 for spillover"
        ),
        "alerts": [
            {
                "severity": "high",
                "location": "loc_01",
                "syndrome": "respiratory_fever",
                "message": "12 cases (3.2x baseline) — investigate immediately",
            }
        ],
    },
}

# Location metadata for map visualization
DEMO_LOCATIONS = {
    "loc_01": {"name": "Kibera Health Post", "lat": -1.3133, "lon": 36.7876},
    "loc_02": {"name": "Mathare Community Center", "lat": -1.2572, "lon": 36.8575},
}
