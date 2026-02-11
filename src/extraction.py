"""Structured extraction from CHW notes using NuExtract or stub fallback.

Primary: Uses NuExtract model for template-based structured extraction.
Fallback: Uses the keyword-based stub extractor when models are not available.
"""
import json
from pathlib import Path
from typing import Dict, Any

from . import config


# ── NuExtract template matching our encounter schema ────────
EXTRACTION_TEMPLATE = {
    "patient": {
        "patient_id": "",
        "age_group": "",
        "age_years": "",
        "sex": "",
        "pregnancy_status": ""
    },
    "symptoms": {
        "fever": {"value": "", "evidence_quote": ""},
        "cough": {"value": "", "evidence_quote": ""},
        "watery_diarrhea": {"value": "", "evidence_quote": ""},
        "bloody_diarrhea": {"value": "", "evidence_quote": ""},
        "vomiting": {"value": "", "evidence_quote": ""},
        "rash": {"value": "", "evidence_quote": ""},
        "difficulty_breathing": {"value": "", "evidence_quote": ""}
    },
    "other_symptoms": {},
    "onset_days": "",
    "severity": "",
    "red_flags": [],
    "treatments_given": [],
    "referral": {
        "value": "",
        "destination": "",
        "evidence_quote": ""
    },
    "follow_up": {
        "value": "",
        "follow_up_date": "",
        "evidence_quote": ""
    }
}


def _load_prompt() -> str:
    """Load the specialist extraction prompt template."""
    path = config.PROMPT_DIR / "specialist_extraction.txt"
    return path.read_text(encoding="utf-8")


def _postprocess_extraction(raw: Dict[str, Any], note_text: str,
                             encounter_id: str, location_id: str,
                             week_id: int) -> Dict[str, Any]:
    """Clean up and normalize the raw extraction output to match our schema."""

    # Normalize symptom claims
    symptoms = raw.get("symptoms", {})
    for key in ["fever", "cough", "watery_diarrhea", "bloody_diarrhea",
                "vomiting", "rash", "difficulty_breathing"]:
        claim = symptoms.get(key, {})
        val = str(claim.get("value", "unknown")).lower().strip()
        if val not in ("yes", "no"):
            val = "unknown"
        quote = claim.get("evidence_quote")
        if val != "yes":
            quote = None
        elif quote and isinstance(quote, str) and quote.strip():
            quote = quote.strip()
        else:
            quote = None
            val = "unknown"  # can't claim yes without evidence
        symptoms[key] = {"value": val, "evidence_quote": quote}

    # Normalize other_symptoms
    other_symptoms = {}
    for key, claim in raw.get("other_symptoms", {}).items():
        if isinstance(claim, dict):
            val = str(claim.get("value", "unknown")).lower().strip()
            if val not in ("yes", "no"):
                val = "unknown"
            quote = claim.get("evidence_quote")
            if val != "yes":
                quote = None
            elif quote and isinstance(quote, str) and quote.strip():
                quote = quote.strip()
            else:
                quote = None
                val = "unknown"
            other_symptoms[key] = {"value": val, "evidence_quote": quote}

    # Normalize patient
    patient_raw = raw.get("patient", {})
    age_group = str(patient_raw.get("age_group", "unknown")).lower().strip()
    if age_group not in ("infant", "child", "adolescent", "adult", "elderly"):
        age_group = "unknown"

    sex = str(patient_raw.get("sex", "unknown")).lower().strip()
    if sex not in ("male", "female"):
        sex = "unknown"

    age_years = patient_raw.get("age_years")
    try:
        age_years = int(age_years) if age_years else None
    except (ValueError, TypeError):
        age_years = None

    patient = {"age_group": age_group, "sex": sex}
    if age_years is not None:
        patient["age_years"] = age_years
    patient_id = patient_raw.get("patient_id", "unknown")
    if patient_id and str(patient_id).strip():
        patient["patient_id"] = str(patient_id).strip()

    # Normalize onset_days
    onset = raw.get("onset_days")
    try:
        onset = int(onset) if onset else None
    except (ValueError, TypeError):
        onset = None

    # Normalize severity
    severity = str(raw.get("severity", "unknown")).lower().strip()
    if severity not in ("mild", "moderate", "severe"):
        severity = "unknown"

    # Normalize red_flags
    red_flags = []
    valid_flags = {"dehydration_signs", "unable_to_drink", "persistent_vomiting",
                   "blood_in_stool", "high_fever", "convulsions",
                   "altered_consciousness", "severe_malnutrition", "chest_indrawing"}
    for flag in raw.get("red_flags", []):
        if isinstance(flag, dict):
            fname = flag.get("flag", "")
            fquote = flag.get("evidence_quote", "")
            if fname in valid_flags and fquote:
                red_flags.append({"flag": fname, "evidence_quote": fquote})

    # Normalize treatments
    treatments = [str(t).strip() for t in raw.get("treatments_given", [])
                  if t and str(t).strip()]

    # Normalize referral
    referral_raw = raw.get("referral", {})
    referral = None
    if isinstance(referral_raw, dict) and referral_raw.get("value") in ("yes", "no"):
        referral = {
            "value": referral_raw["value"],
            "destination": str(referral_raw.get("destination", "unknown")),
        }
        if referral_raw["value"] == "yes":
            referral["evidence_quote"] = referral_raw.get("evidence_quote")
        else:
            referral["evidence_quote"] = None

    # Normalize follow_up
    followup_raw = raw.get("follow_up", {})
    follow_up = None
    if isinstance(followup_raw, dict) and followup_raw.get("value") in ("yes", "no"):
        follow_up = {
            "value": followup_raw["value"],
            "follow_up_date": followup_raw.get("follow_up_date"),
        }
        if followup_raw["value"] == "yes":
            follow_up["evidence_quote"] = followup_raw.get("evidence_quote")
        else:
            follow_up["evidence_quote"] = None

    # Build final encounter
    encounter = {
        "encounter_id": encounter_id,
        "location_id": location_id,
        "week_id": week_id,
        "note_text": note_text,
        "chw_id": str(raw.get("chw_id", "unknown")),
        "visit_date": raw.get("visit_date"),
        "visit_datetime": raw.get("visit_datetime"),
        "encounter_sequence": raw.get("encounter_sequence"),
        "area_id": str(raw.get("area_id", "unknown")),
        "household_id": str(raw.get("household_id", "unknown")),
        "gps": raw.get("gps"),
        "patient": patient,
        "symptoms": symptoms,
        "other_symptoms": other_symptoms,
        "onset_days": onset,
        "severity": severity,
        "red_flags": red_flags,
        "treatments_given": treatments,
        "referral": referral,
        "follow_up": follow_up,
    }

    return encounter


def extract_with_nuextract(note_text: str, encounter_id: str = "unknown",
                            location_id: str = "unknown",
                            week_id: int = 0) -> Dict[str, Any]:
    """Extract structured encounter from a CHW note using NuExtract.

    Falls back to the MedGemma prompt-based extraction if NuExtract is unavailable.
    """
    from .models import generate_nuextract, parse_json_response

    raw_output = generate_nuextract(note_text, EXTRACTION_TEMPLATE)
    parsed = parse_json_response(raw_output)

    if parsed is None:
        print(f"  WARNING: NuExtract returned unparseable output for {encounter_id}")
        return stub_extract_full(note_text, encounter_id, location_id, week_id)

    return _postprocess_extraction(parsed, note_text, encounter_id, location_id, week_id)


def extract_with_medgemma(note_text: str, encounter_id: str = "unknown",
                           location_id: str = "unknown",
                           week_id: int = 0) -> Dict[str, Any]:
    """Extract structured encounter from a CHW note using MedGemma prompt.

    Uses the specialist_extraction prompt for full structured output.
    """
    from .models import generate_medgemma, parse_json_response

    prompt_template = _load_prompt()
    prompt = prompt_template.replace("{note_text}", note_text)
    raw_output = generate_medgemma(prompt, max_tokens=config.EXTRACTION_MAX_TOKENS)
    parsed = parse_json_response(raw_output)

    if parsed is None:
        print(f"  WARNING: MedGemma returned unparseable output for {encounter_id}")
        return stub_extract_full(note_text, encounter_id, location_id, week_id)

    return _postprocess_extraction(parsed, note_text, encounter_id, location_id, week_id)


def stub_extract_full(note_text: str, encounter_id: str = "unknown",
                       location_id: str = "unknown",
                       week_id: int = 0) -> Dict[str, Any]:
    """Rule-based stub extractor (no model needed). Used as fallback.

    Extended to cover the full schema with demographic extraction.
    """
    from .extract import stub_extract  # original keyword extractor

    basic = stub_extract(note_text)

    # Extend with defaults for new fields
    symptoms = {}
    for key in ["fever", "cough", "watery_diarrhea", "bloody_diarrhea",
                "vomiting", "rash", "difficulty_breathing"]:
        if key in basic.get("symptoms", {}):
            symptoms[key] = basic["symptoms"][key]
        else:
            symptoms[key] = {"value": "unknown", "evidence_quote": None}

    return {
        "encounter_id": encounter_id,
        "location_id": location_id,
        "week_id": week_id,
        "note_text": note_text,
        "chw_id": "unknown",
        "visit_date": None,
        "visit_datetime": None,
        "encounter_sequence": None,
        "area_id": "unknown",
        "household_id": "unknown",
        "gps": None,
        "patient": {"age_group": "unknown", "sex": "unknown"},
        "symptoms": symptoms,
        "other_symptoms": {},
        "onset_days": basic.get("onset"),
        "severity": basic.get("severity") or "unknown",
        "red_flags": [],
        "treatments_given": [],
        "referral": None,
        "follow_up": None,
    }
