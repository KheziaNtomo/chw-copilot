"""Syndrome tagging using MedGemma or deterministic fallback.

Tags each encounter with one of the configured syndrome categories.
"""
import json
from typing import Dict, Any

from . import config


def _load_prompt() -> str:
    """Load the syndrome tagger prompt template."""
    path = config.PROMPT_DIR / "syndrome_tagger.txt"
    return path.read_text(encoding="utf-8")


def tag_syndrome_medgemma(encounter: Dict[str, Any]) -> Dict[str, Any]:
    """Tag an encounter with a syndrome using MedGemma.

    Args:
        encounter: Structured encounter dict (after extraction)

    Returns:
        Syndrome tag dict matching syndrome.schema.json
    """
    from .models import generate_medgemma, parse_json_response

    prompt_template = _load_prompt()

    # Build context from encounter
    encounter_summary = json.dumps({
        "symptoms": encounter.get("symptoms", {}),
        "other_symptoms": encounter.get("other_symptoms", {}),
        "red_flags": encounter.get("red_flags", []),
        "severity": encounter.get("severity", "unknown"),
        "onset_days": encounter.get("onset_days"),
    }, indent=2)

    prompt = prompt_template.replace("{encounter_json}", encounter_summary)
    # Note text removed from prompt to enforce safety (rely only on extracted JSON)

    raw_output = generate_medgemma(prompt, max_tokens=config.REASONING_MAX_TOKENS)
    parsed = parse_json_response(raw_output)

    if parsed is None:
        print(f"  WARNING: MedGemma syndrome tagger returned unparseable output")
        return tag_syndrome_deterministic(encounter)

    # Normalize
    tag = str(parsed.get("syndrome_tag", "unclear")).lower().strip()
    if tag not in config.SYNDROMES:
        tag = "unclear"

    confidence = str(parsed.get("confidence", "low")).lower().strip()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    trigger_quotes = parsed.get("trigger_quotes", [])
    if not isinstance(trigger_quotes, list):
        trigger_quotes = []
    trigger_quotes = [str(q).strip() for q in trigger_quotes if q and str(q).strip()]

    # Ensure at least 1 trigger quote for non-unclear tags
    if not trigger_quotes and tag != "unclear":
        trigger_quotes = ["[no evidence extracted]"]

    return {
        "encounter_id": encounter.get("encounter_id", "unknown"),
        "syndrome_tag": tag,
        "confidence": confidence,
        "trigger_quotes": trigger_quotes[:5],  # cap at maxItems
        "reasoning": str(parsed.get("reasoning", ""))[:300],
    }


def tag_syndrome_deterministic(encounter: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based syndrome tagger (no model needed). Used as fallback.

    Implements the same logic as the syndrome_tagger prompt:
    - respiratory_fever: fever AND (cough OR difficulty_breathing)
    - acute_watery_diarrhea: watery_diarrhea (non-bloody)
    - other: clear presentation that doesn't match
    - unclear: insufficient info
    """
    symptoms = encounter.get("symptoms", {})

    def is_yes(key):
        return symptoms.get(key, {}).get("value") == "yes"

    def get_quote(key):
        return symptoms.get(key, {}).get("evidence_quote", "")

    has_fever = is_yes("fever")
    has_cough = is_yes("cough")
    has_breathing = is_yes("difficulty_breathing")
    has_watery_d = is_yes("watery_diarrhea")
    has_bloody_d = is_yes("bloody_diarrhea")

    trigger_quotes = []

    if has_watery_d and not has_bloody_d:
        tag = "acute_watery_diarrhea"
        confidence = "high"
        if get_quote("watery_diarrhea"):
            trigger_quotes.append(get_quote("watery_diarrhea"))
    elif has_fever and (has_cough or has_breathing):
        tag = "respiratory_fever"
        confidence = "high"
        if get_quote("fever"):
            trigger_quotes.append(get_quote("fever"))
        if has_cough and get_quote("cough"):
            trigger_quotes.append(get_quote("cough"))
        if has_breathing and get_quote("difficulty_breathing"):
            trigger_quotes.append(get_quote("difficulty_breathing"))
    elif has_fever:
        tag = "respiratory_fever"
        confidence = "medium"
        if get_quote("fever"):
            trigger_quotes.append(get_quote("fever"))
    elif any(is_yes(k) for k in symptoms) or any(encounter.get("other_symptoms", {}).get(k, {}).get("value") == "yes" for k in encounter.get("other_symptoms", {})):
        tag = "other"
        confidence = "low"
        # Collect quotes from symptoms
        for k in symptoms:
            if is_yes(k) and get_quote(k):
                trigger_quotes.append(get_quote(k))
        # Collect quotes from other_symptoms
        for k, v in encounter.get("other_symptoms", {}).items():
             if v.get("value") == "yes" and v.get("evidence_quote"):
                 trigger_quotes.append(v.get("evidence_quote"))
                 if len(trigger_quotes) >= 5: break
    else:
        tag = "unclear"
        confidence = "low"
        trigger_quotes = ["insufficient data"]

    return {
        "encounter_id": encounter.get("encounter_id", "unknown"),
        "syndrome_tag": tag,
        "confidence": confidence,
        "trigger_quotes": trigger_quotes[:5],
        "reasoning": f"Deterministic tagger: fever={has_fever}, cough={has_cough}, watery_d={has_watery_d}",
    }
