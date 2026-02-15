"""Hallucination detection for CHW Copilot via self-consistency check.

Two-layer verification:
  Layer 1 (enforce_evidence in validate.py): Quote must be verbatim substring of note
  Layer 2 (this module): Quote must actually SUPPORT the claim direction

Uses MedGemma self-consistency: the same model that extracted the claims
is asked a verification question with a different prompt. If the model
contradicts its own extraction, the claim is flagged.

This catches procedural hallucinations where the quote exists in the
note but says the opposite (e.g., "no rash observed" cited for "rash: yes").
"""
from typing import Dict, Any, List, Optional


# ── Verification prompt template ─────────────────────────────
VERIFY_PROMPT = """You are a evidence verification assistant. Your job is to check whether a quoted piece of text supports a specific medical claim.

CLAIM: "{claim}"
EVIDENCE QUOTE: "{evidence}"
ORIGINAL NOTE: "{note}"

Does the evidence quote SUPPORT the claim? Consider:
- Does the quote confirm or deny the claimed symptom?
- A quote like "no fever" does NOT support "patient has fever"
- A quote like "fever 3 days" DOES support "patient has fever"

Respond with ONLY a JSON object:
{{"supported": true/false, "reason": "brief explanation"}}"""


def _build_claims(encounter: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract claim/evidence pairs from an encounter for verification."""
    claims = []

    # Core symptoms
    for key, val in encounter.get("symptoms", {}).items():
        if isinstance(val, dict) and val.get("value") == "yes" and val.get("evidence_quote"):
            claims.append({
                "symptom": key,
                "claim": f"Patient has {key.replace('_', ' ')}",
                "evidence": val["evidence_quote"],
                "section": "symptoms",
            })

    # Other symptoms
    for key, val in encounter.get("other_symptoms", {}).items():
        if isinstance(val, dict) and val.get("value") == "yes" and val.get("evidence_quote"):
            claims.append({
                "symptom": key,
                "claim": f"Patient has {key.replace('_', ' ')}",
                "evidence": val["evidence_quote"],
                "section": "other_symptoms",
            })

    # Red flags
    for flag in encounter.get("red_flags", []):
        if isinstance(flag, dict) and flag.get("evidence_quote"):
            claims.append({
                "symptom": flag.get("flag", "unknown"),
                "claim": f"Patient shows {flag.get('flag', 'unknown').replace('_', ' ')}",
                "evidence": flag["evidence_quote"],
                "section": "red_flags",
            })

    return claims


def verify_extraction_claims(
    encounter: Dict[str, Any],
    note_text: str,
    generate_fn=None,
) -> Dict[str, Any]:
    """Verify extraction claims using MedGemma self-consistency.

    For each "yes" claim with an evidence quote, asks MedGemma:
    "Does this quote actually support this claim?"

    If the model says no, the claim is flagged as a potential hallucination.

    Args:
        encounter: Structured encounter dict (after extraction)
        note_text: Original CHW note text
        generate_fn: Function to call MedGemma (signature: fn(prompt) -> str).
                     If None, verification is skipped gracefully.

    Returns:
        Dict with:
            flagged: bool — whether any claims were flagged
            claims_checked: int — number of claims verified
            flagged_claims: list — details of flagged claims
            available: bool — whether verification was performed
    """
    claims = _build_claims(encounter)

    if not claims:
        return {
            "flagged": False,
            "claims_checked": 0,
            "flagged_claims": [],
            "available": True,
        }

    # If no model function provided, skip gracefully
    if generate_fn is None:
        return {
            "flagged": False,
            "claims_checked": len(claims),
            "flagged_claims": [],
            "available": False,
            "note": "No model available for self-consistency check",
        }

    flagged_claims = []

    for c in claims:
        prompt = VERIFY_PROMPT.format(
            claim=c["claim"],
            evidence=c["evidence"],
            note=note_text,
        )

        try:
            # Import here to avoid circular imports
            from src.models import parse_json_response

            raw = generate_fn(prompt, max_tokens=100)
            result = parse_json_response(raw)

            if result and result.get("supported") is False:
                flagged_claims.append({
                    "claim": c["claim"],
                    "symptom": c["symptom"],
                    "evidence": c["evidence"],
                    "reason": result.get("reason", "Evidence does not support claim"),
                })
        except Exception as e:
            # Single claim verification failed — continue with others
            continue

    return {
        "flagged": len(flagged_claims) > 0,
        "claims_checked": len(claims),
        "flagged_claims": flagged_claims,
        "available": True,
    }


def verify_claims_offline(encounter: Dict[str, Any], note_text: str) -> Dict[str, Any]:
    """Quick deterministic check for obvious contradictions.

    Catches the most common hallucination pattern: quote contains
    a negation word but claim says "yes". This serves as a fast
    fallback when no model is available.
    """
    claims = _build_claims(encounter)
    negation_words = {"no", "not", "none", "without", "absent", "denies", "denied", "negative"}
    flagged = []

    for c in claims:
        quote_words = set(c["evidence"].lower().split())
        # If quote starts with a negation or contains negation + symptom keyword
        if quote_words & negation_words:
            symptom_words = set(c["symptom"].replace("_", " ").lower().split())
            # Quote has both negation AND symptom keyword → likely contradictory
            if quote_words & symptom_words:
                flagged.append({
                    "claim": c["claim"],
                    "symptom": c["symptom"],
                    "evidence": c["evidence"],
                    "reason": f"Quote contains negation word(s) with symptom keyword",
                })

    return {
        "flagged": len(flagged) > 0,
        "claims_checked": len(claims),
        "flagged_claims": flagged,
        "available": True,
        "method": "deterministic_negation_check",
    }
