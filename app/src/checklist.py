"""Checklist generation using MedGemma or deterministic fallback.

Identifies missing information in an encounter and generates
follow-up questions for the CHW to ask.
"""
import json
from typing import Dict, Any, List

from . import config


def _load_prompt() -> str:
    """Load the checklist agent prompt template."""
    path = config.PROMPT_DIR / "checklist_agent.txt"
    return path.read_text(encoding="utf-8")


def generate_checklist_medgemma(encounter: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a checklist of missing information using MedGemma.

    Args:
        encounter: Structured encounter dict (after extraction)

    Returns:
        Checklist dict matching checklist.schema.json
    """
    from .models import generate_medgemma, parse_json_response

    prompt_template = _load_prompt()

    encounter_json = json.dumps(encounter, indent=2, default=str)
    note_text = encounter.get("note_text", "")

    prompt = prompt_template.replace("{encounter_json}", encounter_json)
    prompt = prompt_template.replace("{note_text}", note_text)
    # Workaround: apply both replacements on the template
    prompt = _load_prompt()
    prompt = prompt.replace("{encounter_json}", encounter_json)
    prompt = prompt.replace("{note_text}", note_text)

    raw_output = generate_medgemma(prompt, max_tokens=config.REASONING_MAX_TOKENS)
    parsed = parse_json_response(raw_output)

    if parsed is None:
        print(f"  WARNING: MedGemma checklist returned unparseable output")
        return generate_checklist_deterministic(encounter)

    # Normalize questions
    questions = parsed.get("questions", [])
    valid_questions = []
    for q in questions[:5]:  # max 5
        if isinstance(q, dict) and "field" in q and "question" in q:
            priority = str(q.get("priority", "medium")).lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            valid_questions.append({
                "field": str(q["field"]),
                "question": str(q["question"]),
                "priority": priority,
            })

    return {
        "encounter_id": encounter.get("encounter_id", "unknown"),
        "questions": valid_questions,
    }


def generate_checklist_deterministic(encounter: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based checklist generator (no model needed). Used as fallback.

    Checks for unknowns in the encounter and generates questions.
    """
    symptoms = encounter.get("symptoms", {})
    patient = encounter.get("patient", {})
    questions: List[Dict[str, str]] = []

    # High priority: danger signs
    danger_symptoms = ["fever", "difficulty_breathing"]
    for s in danger_symptoms:
        if symptoms.get(s, {}).get("value") == "unknown":
            nice_name = s.replace("_", " ")
            questions.append({
                "field": f"symptoms.{s}",
                "question": f"Does the patient have {nice_name}?",
                "priority": "high",
            })

    # High priority: hydration status for diarrhea cases
    has_diarrhea = symptoms.get("watery_diarrhea", {}).get("value") == "yes"
    if has_diarrhea:
        has_dehydration_flag = any(
            f.get("flag") == "dehydration_signs"
            for f in encounter.get("red_flags", [])
        )
        if not has_dehydration_flag:
            questions.append({
                "field": "red_flags.dehydration_signs",
                "question": "Check for dehydration: are the eyes sunken? Is the mouth dry? Does a skin pinch go back slowly?",
                "priority": "high",
            })

    # Medium priority: onset
    if encounter.get("onset_days") is None:
        questions.append({
            "field": "onset_days",
            "question": "How many days ago did the symptoms start?",
            "priority": "medium",
        })

    # Medium priority: other core symptoms
    for s in ["cough", "vomiting", "rash", "watery_diarrhea", "bloody_diarrhea"]:
        if s not in danger_symptoms and symptoms.get(s, {}).get("value") == "unknown":
            nice_name = s.replace("_", " ")
            questions.append({
                "field": f"symptoms.{s}",
                "question": f"Does the patient have {nice_name}?",
                "priority": "medium",
            })

    # Low priority: demographics
    if patient.get("age_group") == "unknown":
        questions.append({
            "field": "patient.age_years",
            "question": "How old is the patient? (years or months)",
            "priority": "low",
        })

    if patient.get("sex") == "unknown":
        questions.append({
            "field": "patient.sex",
            "question": "Is the patient male or female?",
            "priority": "low",
        })

    # Sort by priority and cap at 5
    priority_order = {"high": 0, "medium": 1, "low": 2}
    questions.sort(key=lambda q: priority_order.get(q["priority"], 1))

    return {
        "encounter_id": encounter.get("encounter_id", "unknown"),
        "questions": questions[:5],
    }
