"""Pre-computed demo data for offline Streamlit demonstration.

Contains representative CHW notes, full pipeline outputs (including
agent traces), and a failure mode example showing the hallucination detector catching
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
    {
        "id": "demo_004",
        "title": "Measles-like rash — child",
        "note_text": (
            "Child 6yr rash all over body eyes red fever 4 days measles in village "
            "cough present not vaccinated referred district hospital"
        ),
        "location_id": "loc_02",
        "week_id": 7,
    },
    {
        "id": "demo_005",
        "title": "Severe pneumonia — infant",
        "note_text": (
            "Baby 11 months fever 2 days cough pulling in of chest when breathing "
            "not breastfeeding well restless unable to drink referred urgent"
        ),
        "location_id": "loc_01",
        "week_id": 6,
    },
    {
        "id": "demo_006",
        "title": "Cholera-like AWD — adult cluster",
        "note_text": (
            "Male 25 sudden diarrhea rice-water type cramping vomiting co-workers "
            "also affected ate same food at canteen becoming weak skin pinch slow"
        ),
        "location_id": "loc_02",
        "week_id": 7,
    },
    {
        "id": "demo_007",
        "title": "Unclear presentation — fatigue",
        "note_text": (
            "Woman 23 dizziness and fatigue ate today no vomiting no diarrhea "
            "no cough no fever might be pregnant"
        ),
        "location_id": "loc_01",
        "week_id": 8,
    },
    {
        "id": "demo_008",
        "title": "Malaria in pregnancy",
        "note_text": (
            "Pregnant woman 26 weeks headache high fever chills 3 days "
            "no cough no diarrhea RDT positive needs ACT safe for pregnancy "
            "referred ANC clinic urgent"
        ),
        "location_id": "loc_02",
        "week_id": 8,
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
            " Give paracetamol for fever; do malaria RDT if available",
            " No urgent action needed — routine follow-up",
        ],
        "checklist": [
            "Is the child able to drink or breastfeed? (WHO danger sign if unable)",
            "Count respiratory rate — is breathing fast? (≥40/min for age 12m–5y)",
            "Check for chest indrawing when breathing (WHO ICCM danger sign)",
            "Has a malaria RDT been done? (Fever ≥3 days warrants testing in endemic areas)",
            "Are other children in the household or community also sick? (Cluster detection)",
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
            " REFER IMMEDIATELY — danger sign(s): unable to drink, dehydration signs",
            " Start ORS immediately; give zinc (10mg if <6mo, 20mg if ≥6mo) for 10 days",
            " Persistent vomiting — give ORS in small sips, monitor for dehydration",
        ],
        "checklist": [
            "Is the stool bloody? (Bloody diarrhoea = dysentery, requires antibiotics per WHO ICCM)",
            "How many stools in the last 24 hours? (≥3 watery stools/day = acute diarrhoea)",
            "Can the child drink ORS now? Re-assess hydration after 4 hours of ORS",
            "Is there fever? (Fever + diarrhoea may suggest invasive infection)",
            "Are other family members also affected? (Cluster = possible contaminated water source)",
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
            " Give paracetamol for fever; do malaria RDT if available",
            " Malaria RDT positive — give ACT (artemisinin-based combination therapy)",
        ],
        "checklist": [
            "Has ACT treatment been started? (First dose should be given within 24 hours of positive RDT)",
            "Is the patient vomiting? (If unable to retain oral ACT, refer for parenteral treatment)",
            "Any signs of severe malaria? (Confusion, convulsions, inability to sit/stand)",
            "Is the patient pregnant? (Pregnancy alters treatment protocol — refer)",
            "Has the patient used a bed net? (Counsel on prevention)",
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
    # ── demo_004  Measles-like rash — child ──
    {
        "encounter": {
            "encounter_id": "demo_004",
            "location_id": "loc_02",
            "week_id": 7,
            "patient": {"age_years": 6, "sex": "unknown"},
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "fever 4 days"},
                "cough": {"value": "yes", "evidence_quote": "cough present"},
                "diarrhea": {"value": "unknown", "evidence_quote": ""},
                "vomiting": {"value": "unknown", "evidence_quote": ""},
                "rash": {"value": "yes", "evidence_quote": "rash all over body"},
            },
            "other_symptoms": {
                "red_eyes": {"value": "yes", "evidence_quote": "eyes red"},
            },
            "onset_days": 4,
            "estimated_onset_week": 7,
            "severity": "moderate",
            "red_flags": [],
            "treatment_given": [],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "other",
            "sub_syndrome": "measles-like",
            "confidence": "high",
            "reasoning": "Fever + generalised rash + conjunctivitis (red eyes) + cough. Known measles outbreak in village. Classic measles prodrome.",
            "trigger_quotes": ["rash all over body", "eyes red", "fever 4 days", "measles in village"],
        },
        "recommendations": [
            "REFER — suspect measles (fever + rash + red eyes); isolate from other children",
            "Give vitamin A: 100 000 IU (6–11 mo) or 200 000 IU (≥12 mo) — two doses",
            "Check vaccination status of household contacts; advise catch-up immunisation",
        ],
        "checklist": [
            "Has the child received any measles vaccine doses? (Check vaccination card)",
            "Are there other children with similar rash in the village? (Outbreak confirmation)",
            "Is the child able to eat and drink? (Assess for dehydration and malnutrition complications)",
            "Any mouth ulcers? (Measles can cause oral lesions affecting feeding)",
            "When did the rash start? (Measles rash typically starts 3–5 days after fever onset)",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.208, "fallback_used": False, "output_summary": "Extracted encounter with 3 positive symptoms + red eyes"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.002, "fallback_used": False, "output_summary": "0 claims downgraded"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as other (high) — rash + fever + conjunctivitis"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "Sub-syndrome: measles-like"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 3 recommendations incl. vitamin A"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True"},
        ],
        "processing_time_s": 2.21,
    },
    # ── demo_005  Severe pneumonia — infant ──
    {
        "encounter": {
            "encounter_id": "demo_005",
            "location_id": "loc_01",
            "week_id": 6,
            "patient": {"age_years": 0, "age_months": 11, "sex": "unknown"},
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "fever 2 days"},
                "cough": {"value": "yes", "evidence_quote": "cough"},
                "difficulty_breathing": {"value": "yes", "evidence_quote": "pulling in of chest when breathing"},
                "diarrhea": {"value": "unknown", "evidence_quote": ""},
                "vomiting": {"value": "unknown", "evidence_quote": ""},
                "rash": {"value": "unknown", "evidence_quote": ""},
            },
            "other_symptoms": {},
            "onset_days": 2,
            "estimated_onset_week": 6,
            "severity": "severe",
            "red_flags": [
                {"flag": "chest_indrawing", "evidence_quote": "pulling in of chest when breathing"},
                {"flag": "unable_to_drink", "evidence_quote": "unable to drink"},
                {"flag": "not_feeding", "evidence_quote": "not breastfeeding well"},
            ],
            "treatment_given": [],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "respiratory_fever",
            "sub_syndrome": "pneumonia-like",
            "confidence": "high",
            "reasoning": "Fever + cough + chest indrawing in infant \u003c12 months — severe pneumonia. Multiple danger signs (unable to drink, not feeding).",
            "trigger_quotes": ["fever 2 days", "cough", "pulling in of chest"],
        },
        "recommendations": [
            " REFER IMMEDIATELY — danger sign(s): chest indrawing, unable to drink",
            "Give first dose of amoxicillin if available before referral",
            "Keep infant warm and continue breastfeeding attempts during transport",
            "Count respiratory rate — fast breathing (\u226550/min for \u003c12 mo) confirms pneumonia",
        ],
        "checklist": [
            "Has the infant had convulsions? (WHO danger sign — requires urgent referral)",
            "Is the infant unusually sleepy or difficult to wake? (Altered consciousness = danger sign)",
            "What is the exact respiratory rate? (Count for full 60 seconds when calm)",
            "Is there stridor when calm? (Stridor at rest = severe upper airway obstruction)",
            "Has the infant received any antibiotics already? (Important for referral facility)",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.312, "fallback_used": False, "output_summary": "Extracted encounter with 3 positive symptoms + 3 red flags"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.002, "fallback_used": False, "output_summary": "0 claims downgraded"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as respiratory_fever (high) — fever + cough + chest indrawing"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "Sub-syndrome: pneumonia-like (cough + difficulty breathing)"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 4 recommendations — REFER IMMEDIATELY"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True"},
        ],
        "processing_time_s": 2.32,
    },
    # ── demo_006  Cholera-like AWD — adult cluster ──
    {
        "encounter": {
            "encounter_id": "demo_006",
            "location_id": "loc_02",
            "week_id": 7,
            "patient": {"age_years": 25, "sex": "male"},
            "symptoms": {
                "fever": {"value": "unknown", "evidence_quote": ""},
                "cough": {"value": "unknown", "evidence_quote": ""},
                "diarrhea": {"value": "yes", "evidence_quote": "sudden diarrhea rice-water type"},
                "vomiting": {"value": "yes", "evidence_quote": "vomiting"},
                "rash": {"value": "unknown", "evidence_quote": ""},
            },
            "other_symptoms": {
                "abdominal_cramps": {"value": "yes", "evidence_quote": "cramping"},
            },
            "onset_days": 1,
            "estimated_onset_week": 7,
            "severity": "severe",
            "red_flags": [
                {"flag": "dehydration_signs", "evidence_quote": "skin pinch slow"},
                {"flag": "persistent_vomiting", "evidence_quote": "vomiting"},
                {"flag": "cluster_pattern", "evidence_quote": "co-workers also affected"},
            ],
            "treatment_given": [],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "acute_watery_diarrhea",
            "sub_syndrome": None,
            "confidence": "high",
            "reasoning": "Rice-water diarrhea + vomiting + dehydration signs. Multiple co-workers affected — cluster pattern suggests common-source outbreak (foodborne or waterborne).",
            "trigger_quotes": ["sudden diarrhea rice-water type", "vomiting", "co-workers also affected"],
        },
        "recommendations": [
            "REFER IMMEDIATELY — danger sign(s): dehydration (slow skin pinch), persistent vomiting",
            "Start ORS immediately; give in small frequent sips if vomiting",
            "ALERT: Cluster pattern detected — co-workers affected. Notify district health team for outbreak investigation",
            "Identify common food/water source for all affected individuals",
        ],
        "checklist": [
            "How many co-workers are affected? (Number of cases needed for outbreak report)",
            "What was the common food or water source? (Identify point of exposure)",
            "When did each person's symptoms start? (Timeline helps determine if point-source or propagated)",
            "Has a stool sample been collected? (Lab confirmation needed for cholera notification)",
            "What is the patient's urine output? (Reduced output = severe dehydration)",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.156, "fallback_used": False, "output_summary": "Extracted encounter with 3 positive symptoms + cluster flag"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.001, "fallback_used": False, "output_summary": "0 claims downgraded"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as acute_watery_diarrhea (high) — rice-water stool"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "N/A for AWD"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 4 recommendations — REFER + cluster alert"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True"},
        ],
        "processing_time_s": 2.16,
    },
    # ── demo_007  Unclear presentation — fatigue ──
    {
        "encounter": {
            "encounter_id": "demo_007",
            "location_id": "loc_01",
            "week_id": 8,
            "patient": {"age_years": 23, "sex": "female"},
            "symptoms": {
                "fever": {"value": "no", "evidence_quote": "no fever"},
                "cough": {"value": "no", "evidence_quote": "no cough"},
                "diarrhea": {"value": "no", "evidence_quote": "no diarrhea"},
                "vomiting": {"value": "no", "evidence_quote": "no vomiting"},
                "rash": {"value": "unknown", "evidence_quote": ""},
            },
            "other_symptoms": {
                "dizziness": {"value": "yes", "evidence_quote": "dizziness"},
                "fatigue": {"value": "yes", "evidence_quote": "fatigue"},
            },
            "onset_days": None,
            "estimated_onset_week": 8,
            "severity": "mild",
            "red_flags": [],
            "treatment_given": [],
            "referral": False,
        },
        "syndrome_tag": {
            "syndrome_tag": "unclear",
            "sub_syndrome": None,
            "confidence": "low",
            "reasoning": "No fever, no cough, no diarrhea. Only dizziness and fatigue — insufficient to classify into any syndromic category. Possible pregnancy or anaemia.",
            "trigger_quotes": ["dizziness", "fatigue"],
        },
        "recommendations": [
            "No syndromic classification possible — schedule follow-up in 3 days",
            "Advise pregnancy test if period is late",
            "Screen for anaemia (pallor of palms, conjunctivae) at next visit",
        ],
        "checklist": [
            "When was the last menstrual period? (Rule out pregnancy as cause of dizziness/fatigue)",
            "Assess pallor of palms and inner eyelids — could this be anaemia?",
            "Any history of blood loss or heavy periods? (Iron-deficiency anaemia common in young women)",
            "What is the patient's diet like? (Nutritional deficiencies can cause fatigue + dizziness)",
            "Any weight loss or night sweats? (Screen for TB or HIV if present)",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.089, "fallback_used": False, "output_summary": "Extracted encounter with 0 core symptoms, 2 other symptoms"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.001, "fallback_used": False, "output_summary": "0 claims downgraded"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as unclear (low) — no core symptoms"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "N/A for unclear"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 3 recommendations — follow-up"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True"},
        ],
        "processing_time_s": 2.09,
    },
    # ── demo_008  Malaria in pregnancy ──
    {
        "encounter": {
            "encounter_id": "demo_008",
            "location_id": "loc_02",
            "week_id": 8,
            "patient": {"age_years": 26, "sex": "female", "pregnant": True, "gestational_weeks": 26},
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "high fever"},
                "cough": {"value": "no", "evidence_quote": "no cough"},
                "diarrhea": {"value": "no", "evidence_quote": "no diarrhea"},
                "vomiting": {"value": "unknown", "evidence_quote": ""},
                "rash": {"value": "unknown", "evidence_quote": ""},
            },
            "other_symptoms": {
                "headache": {"value": "yes", "evidence_quote": "headache"},
                "chills": {"value": "yes", "evidence_quote": "chills 3 days"},
            },
            "onset_days": 3,
            "estimated_onset_week": 8,
            "severity": "moderate",
            "red_flags": [
                {"flag": "pregnant_with_fever", "evidence_quote": "pregnant woman 26 weeks headache high fever"},
            ],
            "treatment_given": [],
            "referral": True,
        },
        "syndrome_tag": {
            "syndrome_tag": "other",
            "sub_syndrome": "malaria-like",
            "confidence": "high",
            "reasoning": "Fever + chills + headache with positive RDT in pregnant woman. No respiratory symptoms. Malaria in pregnancy — special population requiring facility-level ACT dosing.",
            "trigger_quotes": ["high fever", "chills 3 days", "RDT positive"],
        },
        "recommendations": [
            "REFER to ANC clinic — malaria in pregnancy requires supervised ACT treatment",
            "Do NOT give doxycycline or primaquine — contraindicated in pregnancy",
            "Insecticide-treated bed net (ITN) — ensure the patient is sleeping under one",
            "Monitor for signs of severe malaria: convulsions, severe anaemia, jaundice",
        ],
        "checklist": [
            "Gestational age is 26 weeks (2nd trimester) — confirm ACT dosing appropriate for this trimester per WHO guidelines",
            "Has the patient been taking IPTp-SP? (Intermittent preventive treatment in pregnancy)",
            "Any vaginal bleeding or abdominal pain? (Malaria increases miscarriage/preterm risk)",
            "Check haemoglobin if possible — severe anaemia in pregnancy is a danger sign",
            "Is the patient sleeping under an ITN every night? (Reinforce prevention)",
        ],
        "agent_trace": [
            {"agent": "extract", "name": "Encounter Extractor", "duration_s": 2.287, "fallback_used": False, "output_summary": "Extracted encounter with 3 positive symptoms, pregnancy flagged"},
            {"agent": "evidence_enforce", "name": "Evidence Grounder", "duration_s": 0.002, "fallback_used": False, "output_summary": "0 claims downgraded"},
            {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 0.001, "fallback_used": False, "output_summary": "Tagged as other (high) — fever + RDT+, no respiratory symptoms"},
            {"agent": "sub_syndrome", "name": "Sub-syndrome Classifier", "duration_s": 0.001, "fallback_used": False, "output_summary": "Sub-syndrome: malaria-like (pregnancy context)"},
            {"agent": "recommend", "name": "ICCM Recommendations", "duration_s": 0.001, "fallback_used": False, "output_summary": "Generated 4 recommendations — ANC referral + pregnancy safety"},
            {"agent": "validate", "name": "Schema Validator", "duration_s": 0.001, "fallback_used": False, "output_summary": "Schema valid: True"},
        ],
        "processing_time_s": 2.29,
    },
]

# ── Failure Mode Demo ────────────────────────────────────────
# Shows the hallucination detector catching a procedural hallucination
FAILURE_MODE = {
    "title": " Failure Mode: Hallucination Detected",
    "description": (
        "This example shows MedGemma reporting 'rash: yes' even though "
        "the note says 'no rash observed.' The evidence quote exists in "
        "the note (passing enforce_evidence), but the hallucination detector finds that "
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
        {"agent": "hallucination_check", "name": "Hallucination Detector", "duration_s": 1.520, "fallback_used": False, "output_summary": "Checked 3 claims, 1 FLAGGED — rash claim contradicted"},
        {"agent": "tag", "name": "Syndrome Tagger", "duration_s": 1.102, "fallback_used": False, "output_summary": "Tagged as respiratory_fever (high)"},
    ],
}

# ── Surveillance Demo Data ───────────────────────────────────
# 12 weeks of data (epi weeks 1–12, starting Mon 6 Jan 2025).
# Each row includes a "date" (ISO Monday of the epi week) for proper axis labels.
DEMO_SURVEILLANCE = {
    "weekly_counts": [
        # Week 1 (Jan 6)
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 34},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 21},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 49},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 12},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 36},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 23},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 22},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 17},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 34},
        {"week_id": 1, "date": "2025-01-06", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 51},
        # Week 2 (Jan 13)
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 45},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 9},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 34},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 24},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 24},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 41},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 45},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 17},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 24},
        {"week_id": 2, "date": "2025-01-13", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 53},
        # Week 3 (Jan 20)
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 63},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 24},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 52},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 13},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 33},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 23},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 39},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 26},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 13},
        {"week_id": 3, "date": "2025-01-20", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 35},
        # Week 4 (Jan 27)
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 48},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 38},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 34},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 10},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 27},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 10},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 50},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 26},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 37},
        {"week_id": 4, "date": "2025-01-27", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 33},
        # Week 5 (Feb 3)
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 64},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 29},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 34},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 27},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 34},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 41},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 25},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 14},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 24},
        {"week_id": 5, "date": "2025-02-03", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 46},
        # Week 6 (Feb 10)
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 69},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 24},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 37},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 10},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 24},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 22},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 39},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 13},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 40},
        {"week_id": 6, "date": "2025-02-10", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 38},
        # Week 7 (Feb 17) -- OUTBREAK: Kibera respiratory, Kayole diarrhoea
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 215},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 38},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 62},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 24},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 37},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 22},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 47},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 17},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 24},
        {"week_id": 7, "date": "2025-02-17", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 83},
        # Week 8 (Feb 24) -- Kibera still elevated, Kayole diarrhoea sustained
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 172},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 27},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 49},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 17},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 24},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 38},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 33},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 12},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 33},
        {"week_id": 8, "date": "2025-02-24", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 74},
        # Week 9 (Mar 3) -- Kibera resolving, Kayole resolving
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 123},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 25},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 34},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 12},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 38},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 24},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 52},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 27},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 28},
        {"week_id": 9, "date": "2025-03-03", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 59},
        # Week 10 (Mar 10) -- NEW: Dandora respiratory spike
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 61},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 11},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 48},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 29},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 149},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 25},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 39},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 27},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 38},
        {"week_id": 10, "date": "2025-03-10", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 36},
        # Week 11 (Mar 17) -- Dandora respiratory elevated, NEW: Langata diarrhoea spike
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 47},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 29},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 40},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 10},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 105},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 22},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 35},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 95},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 27},
        {"week_id": 11, "date": "2025-03-17", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 34},
        # Week 12 (Mar 24) -- current week: Langata diarrhoea worsening
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_01", "syndrome_tag": "respiratory_fever", "count": 51},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_01", "syndrome_tag": "acute_watery_diarrhea", "count": 27},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_02", "syndrome_tag": "respiratory_fever", "count": 40},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_02", "syndrome_tag": "acute_watery_diarrhea", "count": 17},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_03", "syndrome_tag": "respiratory_fever", "count": 61},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_03", "syndrome_tag": "acute_watery_diarrhea", "count": 29},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_04", "syndrome_tag": "respiratory_fever", "count": 33},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "count": 118},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_05", "syndrome_tag": "respiratory_fever", "count": 29},
        {"week_id": 12, "date": "2025-03-24", "location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "count": 49},
    ],
    "anomalies": [
        {"location_id": "loc_01", "syndrome_tag": "respiratory_fever", "week_id": 7,
         "current_count": 18, "baseline_mean": 4.2, "ratio": 4.3, "alert": True,
         "message": "Respiratory fever at Kibera 4.3x above baseline (18 vs mean 4.2)"},
        {"location_id": "loc_01", "syndrome_tag": "respiratory_fever", "week_id": 8,
         "current_count": 14, "baseline_mean": 4.2, "ratio": 3.3, "alert": True,
         "message": "Respiratory fever at Kibera remains elevated (14 cases, 3.3x baseline)"},
        {"location_id": "loc_05", "syndrome_tag": "acute_watery_diarrhea", "week_id": 7,
         "current_count": 7, "baseline_mean": 3.0, "ratio": 2.3, "alert": True,
         "message": "AWD at Kayole 2.3x above baseline (7 vs mean 3.0)"},
        {"location_id": "loc_03", "syndrome_tag": "respiratory_fever", "week_id": 10,
         "current_count": 12, "baseline_mean": 2.5, "ratio": 4.8, "alert": True,
         "message": "Respiratory fever at Dandora 4.8x above baseline (12 vs mean 2.5)"},
        {"location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "week_id": 11,
         "current_count": 8, "baseline_mean": 1.3, "ratio": 6.2, "alert": True,
         "message": "AWD at Langata 6.2x above baseline (8 vs mean 1.3)"},
        {"location_id": "loc_04", "syndrome_tag": "acute_watery_diarrhea", "week_id": 12,
         "current_count": 10, "baseline_mean": 1.3, "ratio": 7.7, "alert": True,
         "message": "AWD at Langata worsening -- 117 cases in W12 (7.7x baseline)"},
    ],
    "sitrep": {
        "week_id": 12,
        "generated_by": "template",
        "narrative": (
            "WEEK 12 SITUATION REPORT -- CHW Copilot Syndromic Surveillance\n\n"
            "ACTIVE ALERTS:\n"
            "  1. LANGATA: Acute watery diarrhoea (AWD) surge -- 117 cases in W12 (7.7x baseline). "
            "Second consecutive week of escalation. Investigate water sources immediately.\n"
            "  2. DANDORA: Respiratory fever cluster resolving -- 12 in W10, down to 57 in W12. "
            "Continue monitoring.\n"
            "  3. KIBERA: Respiratory fever resolved -- peaked at 215 in W7, now at baseline (457 in W12).\n"
            "  4. KAYOLE: AWD resolved -- peaked at 81 in W7, now at baseline (457 in W12).\n\n"
            "Data source: CHW Copilot syndromic surveillance network -- 5 locations -- 12 weeks of data"
        ),
        "alerts": [
            {"severity": "critical", "location": "loc_04", "syndrome": "acute_watery_diarrhea",
             "message": "117 cases in W12 (7.7x baseline) -- worsening trend",
             "action": "Activate WASH response; investigate water sources; collect stool samples"},
            {"severity": "high", "location": "loc_03", "syndrome": "respiratory_fever",
             "message": "141 cases in W10 (4.8x baseline), resolving to 57 in W12",
             "action": "Continue monitoring; verify cases; check ILI sample collection"},
            {"severity": "medium", "location": "loc_01", "syndrome": "respiratory_fever",
             "message": "Resolved: peaked at 215 in W7, now at baseline (457 in W12)",
             "action": "Stand down active response; maintain routine surveillance"},
            {"severity": "medium", "location": "loc_05", "syndrome": "acute_watery_diarrhea",
             "message": "Resolved: peaked at 81 in W7, now at baseline (457 in W12)",
             "action": "Stand down; continue water source monitoring"},
        ],
    },
}

# Location metadata for map visualization
DEMO_LOCATIONS = {
    "loc_01": {"name": "Kibera Health Post", "lat": -1.3133, "lon": 36.7876},
    "loc_02": {"name": "Mathare Community Center", "lat": -1.2572, "lon": 36.8575},
    "loc_03": {"name": "Dandora Community Health Unit", "lat": -1.2456, "lon": 36.9012},
    "loc_04": {"name": "Langata Sub-County Hospital", "lat": -1.3550, "lon": 36.7350},
    "loc_05": {"name": "Kayole Health Centre", "lat": -1.2750, "lon": 36.9150},
}
