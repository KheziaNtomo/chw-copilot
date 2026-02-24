"""
pipeline_helpers.py
Helper functions for the CHW Copilot Kaggle notebook.
Keeps the notebook clean by housing all utility and processing logic here.
"""
import re
import json
import time
import torch


# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_json_response(text: str):
    """Extract JSON from a model response, handling code fences and extra text."""
    for attempt in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'```(?:json)?\s*\n(.*?)\n```', t, re.DOTALL).group(1)),
        lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group(0)),
    ]:
        try:
            return attempt(text)
        except Exception:
            continue
    return None


# ── MedGemma inference ─────────────────────────────────────────────────────────

def run_medgemma(prompt: str, model, tokenizer, max_new_tokens: int = 512) -> str:
    """Run a single MedGemma inference with chat template."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ── Symptom normalisation ──────────────────────────────────────────────────────

CORE_SYMPTOMS = [
    "fever", "cough", "watery_diarrhea", "bloody_diarrhea",
    "vomiting", "rash", "difficulty_breathing",
]

def _normalise_claim(claim, note_lower: str) -> dict:
    """Validate a single symptom claim against the source note."""
    if not isinstance(claim, dict):
        claim = {}
    val = str(claim.get("value", "unknown")).lower().strip()
    if val not in ("yes", "no"):
        val = "unknown"
    quote = claim.get("evidence_quote")
    if val == "yes" and (not quote or quote.lower() not in note_lower):
        val, quote = "unknown", None
    return {
        "value": val,
        "evidence_quote": quote,
        "duration": claim.get("duration") if val == "yes" else None,
    }

def normalise_symptoms(raw: dict, note_lower: str) -> dict:
    return {k: _normalise_claim(raw.get(k, {}), note_lower) for k in CORE_SYMPTOMS}

def normalise_other_symptoms(raw: dict, note_lower: str) -> dict:
    out = {}
    for k, v in (raw or {}).items():
        if not isinstance(v, dict):
            continue
        c = _normalise_claim(v, note_lower)
        out[k] = c
    return out

def normalise_patient(raw: dict, note_text: str = "") -> dict:
    if not isinstance(raw, dict):
        raw = {}
    age_group = str(raw.get("age_group", "unknown")).lower().strip()
    if age_group not in ("infant", "child", "adolescent", "adult", "elderly"):
        age_group = "unknown"
    sex = str(raw.get("sex", "unknown")).lower().strip()
    if sex not in ("male", "female"):
        sex = "unknown"

    # Try LLM-extracted value first, then regex fallback from note text
    age_years = raw.get("age_years")
    try:
        age_years = int(age_years) if age_years else None
    except (ValueError, TypeError):
        age_years = None

    if age_years is None and note_text:
        # Match patterns: 3yo, 3yr, 3 yr, 3 years, 3-year-old, 3 year old
        m = re.search(r'\b(\d{1,3})\s*(?:yo|yr|year[s]?(?:\s*old)?|-year-old)\b', note_text, re.IGNORECASE)
        if m:
            age_years = int(m.group(1))

    patient = {"age_group": age_group, "sex": sex}
    if age_years is not None:
        patient["age_years"] = age_years
    return patient



# ── Syndrome tagging (fast keyword classifier) ─────────────────────────────────

SYNDROME_RULES = {
    "respiratory_fever": [
        "fever", "cough", "difficulty breathing", "shortness of breath",
        "respiratory", "pneumonia", "malaria", "flu", "cold",
    ],
    "acute_watery_diarrhea": [
        "watery diarrhea", "diarrhoea", "diarrhea", "loose stool",
        "awd", "cholera", "dehydration",
    ],
}

def keyword_syndrome_tag(note_text: str) -> dict:
    """Fast keyword-based syndrome classifier — no LLM call needed."""
    note = note_text.lower()
    scores = {
        syndrome: sum(1 for kw in kws if kw in note)
        for syndrome, kws in SYNDROME_RULES.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return {"syndrome_tag": "other", "confidence": "low",
                "trigger_quotes": ["no matching keywords"], "reasoning": "No syndrome keywords found in note."}
    triggers = [kw for kw in SYNDROME_RULES[best] if kw in note][:3]
    confidence = "high" if scores[best] >= 2 else "medium"
    return {
        "syndrome_tag": best,
        "confidence": confidence,
        "trigger_quotes": triggers,
        "reasoning": f"Note contains {scores[best]} keyword(s) for {best}: {', '.join(triggers)}.",
    }


# ── Full single-pass pipeline ──────────────────────────────────────────────────

def process_note(
    note_text: str,
    encounter_id: str,
    location_id: str,
    week_id: int,
    combined_prompt: str,
    model,
    tokenizer,
) -> dict:
    """
    Single-pass CHW note processor.
    - One MedGemma call for structured extraction + checklist
    - Fast keyword classifier for syndrome tagging (no extra LLM call)
    """
    result = {"encounter_id": encounter_id, "errors": []}
    t0 = time.time()

    # ── LLM extraction ────────────────────────────────────────────────────────
    prompt = combined_prompt.replace("{note_text}", note_text)
    try:
        raw = run_medgemma(prompt, model, tokenizer, max_new_tokens=768)
        parsed = parse_json_response(raw) or {}
    except Exception as e:
        result["errors"].append(f"generation_error: {e}")
        parsed = {}

    enc_raw = parsed.get("encounter", {})
    cl_raw  = parsed.get("checklist", {})
    note_lower = note_text.lower()

    # ── Build normalised encounter ────────────────────────────────────────────
    onset = enc_raw.get("onset_days")
    try:
        onset = int(onset) if onset else None
    except (ValueError, TypeError):
        onset = None
    severity = str(enc_raw.get("severity", "unknown")).lower().strip()
    if severity not in ("mild", "moderate", "severe"):
        severity = "unknown"

    encounter = {
        "encounter_id":    encounter_id,
        "location_id":     location_id,
        "week_id":         week_id,
        "note_text":       note_text,
        "chw_id":          str(enc_raw.get("chw_id", "unknown")),
        "patient":         normalise_patient(enc_raw.get("patient", {}), note_text),
        "symptoms":        normalise_symptoms(enc_raw.get("symptoms", {}), note_lower),
        "other_symptoms":  normalise_other_symptoms(enc_raw.get("other_symptoms", {}), note_lower),
        "onset_days":      onset,
        "severity":        severity,
        "red_flags":       enc_raw.get("red_flags", []),
        "treatments_given":[str(t) for t in enc_raw.get("treatments_given", []) if t],
        "referral":        enc_raw.get("referral"),
        "follow_up":       None,
    }

    # ── Syndrome tag (keyword, fast) ──────────────────────────────────────────
    syn = keyword_syndrome_tag(note_text)
    syn["encounter_id"] = encounter_id

    # ── Checklist ─────────────────────────────────────────────────────────────
    checklist = {
        "encounter_id": encounter_id,
        "questions":    (cl_raw.get("questions") or [])[:3],
    }

    result.update({
        "encounter":         encounter,
        "syndrome_tag":      syn,
        "checklist":         checklist,
        "processing_time_s": round(time.time() - t0, 2),
    })
    return result
