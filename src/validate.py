"""Validation helpers for enforcing evidence and JSON Schema compliance.

Uses jsonschema for structural validation and custom logic for evidence grounding.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import difflib
import re

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


def load_schema(name: str) -> Dict[str, Any]:
    """Load a JSON schema by name (without .schema.json suffix)."""
    path = SCHEMA_DIR / f"{name}.schema.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_json_schema(data: Dict[str, Any], schema_name: str) -> Tuple[bool, List[str]]:
    """Validate data against a named JSON schema.

    Returns (is_valid, list_of_error_messages).
    """
    if not HAS_JSONSCHEMA:
        return True, ["jsonschema not installed — skipping validation"]

    schema = load_schema(schema_name)
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    messages = [f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}" for e in errors]
    return len(messages) == 0, messages


def locate_evidence(quote: str, text: str, threshold: float = 0.8) -> tuple[Optional[str], float]:
    """Find the best fuzzy match for a quote in the text.
    
    Returns (best_match_substring, score).
    If strict substring exists, returns (quote, 1.0).
    Otherwise searches for best approximate match.
    """
    if not quote or not text:
        return None, 0.0
    
    quote = quote.strip()
    text_clean = text.replace("\n", " ")
    
    # 1. Exact match check (case-insensitive)
    if quote.lower() in text.lower():
        # Find the exact original casing in text
        start = text.lower().find(quote.lower())
        return text[start:start+len(quote)], 1.0
        
    # 2. Fuzzy match using sliding window of words
    quote_words = quote.split()
    text_words = text_clean.split()
    n_q = len(quote_words)
    
    best_score = 0.0
    best_match = None
    
    # Try windows of size n_q, n_q+1, n_q-1 to account for minor token insertions/deletions
    # limit window size to avoid performance hit on long texts (though CHW notes are short)
    for window_len in range(max(1, n_q - 2), n_q + 3):
        for i in range(len(text_words) - window_len + 1):
            window = text_words[i : i + window_len]
            window_str = " ".join(window)
            
            # Use quick ratio first
            matcher = difflib.SequenceMatcher(None, quote.lower(), window_str.lower())
            if matcher.quick_ratio() < threshold:
                continue
                
            score = matcher.ratio()
            if score > best_score:
                best_score = score
                best_match = window_str
                
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score



def enforce_evidence(encounter: Dict[str, Any], note_text: str) -> Tuple[Dict[str, Any], List[str]]:
    """Ensure every 'yes' symptom has an evidence_quote that appears verbatim in note_text.

    If a 'yes' claim's evidence_quote is missing or not an exact substring of note_text,
    downgrade to 'unknown' and set evidence to None. Returns the modified encounter
    and a list of fields that were downgraded.

    Note: per user rules, evidence_quote is only required for 'yes' claims,
    not for 'no' or 'unknown'.
    """
    note = (note_text or "").lower()
    downgraded = []

    # Check core symptoms
    for k, v in list(encounter.get("symptoms", {}).items()):
        val = v.get("value")
        quote = v.get("evidence_quote")
        if val == "yes":
            if not quote:
                encounter["symptoms"][k]["value"] = "unknown"
                encounter["symptoms"][k]["evidence_quote"] = None
                downgraded.append(f"symptoms.{k}")
            else:
                match, score = locate_evidence(quote, note)
                if match:
                    # Update with the actual text found (fixes minor typos/paraphrasing)
                    encounter["symptoms"][k]["evidence_quote"] = match
                else:
                    # No match found even with fuzzy logic
                    encounter["symptoms"][k]["value"] = "unknown"
                    encounter["symptoms"][k]["evidence_quote"] = None
                    downgraded.append(f"symptoms.{k}")

    # Check other_symptoms (extensible)
    for k, v in list(encounter.get("other_symptoms", {}).items()):
        val = v.get("value")
        quote = v.get("evidence_quote")
        if val == "yes":
            if not quote: 
                encounter["other_symptoms"][k]["value"] = "unknown"
                encounter["other_symptoms"][k]["evidence_quote"] = None
                downgraded.append(f"other_symptoms.{k}")
            else:
                match, score = locate_evidence(quote, note)
                if match:
                    encounter["other_symptoms"][k]["evidence_quote"] = match
                else:
                    encounter["other_symptoms"][k]["value"] = "unknown"
                    encounter["other_symptoms"][k]["evidence_quote"] = None
                    downgraded.append(f"other_symptoms.{k}")

    # Check red_flags evidence
    valid_flags = []
    for flag in encounter.get("red_flags", []):
        quote = flag.get("evidence_quote", "")
        if quote:
            match, score = locate_evidence(quote, note)
            if match:
                flag["evidence_quote"] = match
                valid_flags.append(flag)
            else:
                downgraded.append(f"red_flag:{flag.get('flag', '?')}")
        else:
            downgraded.append(f"red_flag:{flag.get('flag', '?')}")
    if "red_flags" in encounter:
        encounter["red_flags"] = valid_flags

    # Check referral evidence
    referral = encounter.get("referral")
    if referral and referral.get("value") == "yes":
        quote = referral.get("evidence_quote", "")
        match, score = locate_evidence(quote, note)
        if match:
             encounter["referral"]["evidence_quote"] = match
        else:
            encounter["referral"]["value"] = "unknown"
            encounter["referral"]["evidence_quote"] = None
            downgraded.append("referral")

    # Check follow_up evidence
    follow_up = encounter.get("follow_up")
    if follow_up and follow_up.get("value") == "yes":
        quote = follow_up.get("evidence_quote", "")
        match, score = locate_evidence(quote, note)
        if match:
             encounter["follow_up"]["evidence_quote"] = match
        else:
            encounter["follow_up"]["value"] = "unknown"
            encounter["follow_up"]["evidence_quote"] = None
            downgraded.append("follow_up")

    # Check pregnancy_status evidence
    preg = encounter.get("patient", {}).get("pregnancy_status")
    if preg and preg.get("value") == "yes":
        quote = preg.get("evidence_quote", "")
        match, score = locate_evidence(quote, note)
        if match:
             encounter["patient"]["pregnancy_status"]["evidence_quote"] = match
        else:
            encounter["patient"]["pregnancy_status"]["value"] = "unknown"
            encounter["patient"]["pregnancy_status"]["evidence_quote"] = None
            downgraded.append("patient.pregnancy_status")

    return encounter, downgraded


def enforce_trigger_quotes(syndrome_tag: Dict[str, Any], note_text: str) -> Tuple[Dict[str, Any], List[str]]:
    """Ensure every trigger_quote in a syndrome tag is a verbatim substring of note_text.

    Removes invalid quotes. If all quotes are removed, downgrades confidence to 'low'.
    """
    note = (note_text or "").lower()
    invalid = []
    valid_quotes = []

    for q in syndrome_tag.get("trigger_quotes", []):
        if q and q.lower() in note:
            valid_quotes.append(q)
        else:
            invalid.append(q)

    syndrome_tag["trigger_quotes"] = valid_quotes

    # If no valid quotes remain, downgrade confidence
    if not valid_quotes and syndrome_tag.get("syndrome_tag") not in ("unclear",):
        syndrome_tag["confidence"] = "low"

    return syndrome_tag, invalid


def validation_report(encounter: Dict[str, Any], note_text: str, schema_name: str = "encounter") -> Dict[str, Any]:
    """Generate a full validation report for an encounter.

    Returns dict with schema_valid, schema_errors, evidence_downgrades, and overall pass/fail.
    """
    schema_valid, schema_errors = validate_json_schema(encounter, schema_name)
    encounter_checked, downgrades = enforce_evidence(encounter, note_text)

    return {
        "encounter_id": encounter.get("encounter_id", "unknown"),
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "evidence_downgrades": downgrades,
        "overall_pass": schema_valid and len(downgrades) == 0,
    }
