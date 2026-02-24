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


def run_medgemma_batch(prompts: list, model, tokenizer, max_new_tokens: int = 512) -> list:
    """Run multiple prompts in a single GPU batch for ~batch_size x throughput."""
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    # Pad left so all sequences are the same length (generation works from right)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]



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
    age_years = raw.get("age_years")
    try:
        age_years = int(age_years) if age_years else None
    except (ValueError, TypeError):
        age_years = None
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
        raw = run_medgemma(prompt, model, tokenizer, max_new_tokens=512)
        parsed = parse_json_response(raw) or {}
    except Exception as e:
        result["errors"].append(f"generation_error: {e}")
        parsed = {}

    # Prompt now returns flat extraction JSON (no 'encounter' wrapper)
    note_lower = note_text.lower()


    # ── Build normalised encounter ────────────────────────────────────────────
    onset = parsed.get("onset_days")
    try:
        onset = int(onset) if onset else None
    except (ValueError, TypeError):
        onset = None
    severity = str(parsed.get("severity", "unknown")).lower().strip()
    if severity not in ("mild", "moderate", "severe"):
        severity = "unknown"

    encounter = {
        "encounter_id":    encounter_id,
        "location_id":     location_id,
        "week_id":         week_id,
        "note_text":       note_text,
        "chw_id":          str(parsed.get("chw_id", "unknown")),
        "patient":         normalise_patient(parsed.get("patient", {}), note_text),
        "symptoms":        normalise_symptoms(parsed.get("symptoms", {}), note_lower),
        "other_symptoms":  normalise_other_symptoms(parsed.get("other_symptoms", {}), note_lower),
        "onset_days":      onset,
        "severity":        severity,
        "red_flags":       parsed.get("red_flags", []),
        "treatments_given":[str(t) for t in parsed.get("treatments_given", []) if t],
        "referral":        parsed.get("referral"),
        "follow_up":       None,
    }

    # ── Syndrome tag (keyword, fast) ──────────────────────────────────────────
    syn = keyword_syndrome_tag(note_text)
    syn["encounter_id"] = encounter_id

    # ── Checklist (removed from LLM call for speed) ───────────────────────────
    checklist = {"encounter_id": encounter_id, "questions": []}

    result.update({
        "encounter":         encounter,
        "syndrome_tag":      syn,
        "checklist":         checklist,
        "processing_time_s": round(time.time() - t0, 2),
    })
    return result


def _build_result(parsed: dict, note: dict, t0: float) -> dict:
    """Build a result dict from a parsed LLM response and a note metadata dict."""
    encounter_id = note["encounter_id"]
    location_id  = note.get("location_id", "unknown")
    week_id      = note.get("week_id", 0)
    note_text    = note["note_text"]
    note_lower   = note_text.lower()

    onset = parsed.get("onset_days")
    try:    onset = int(onset) if onset else None
    except: onset = None
    severity = str(parsed.get("severity", "unknown")).lower().strip()
    if severity not in ("mild", "moderate", "severe"): severity = "unknown"

    encounter = {
        "encounter_id":    encounter_id,
        "location_id":     location_id,
        "week_id":         week_id,
        "note_text":       note_text,
        "chw_id":          str(parsed.get("chw_id", "unknown")),
        "patient":         normalise_patient(parsed.get("patient", {}), note_text),
        "symptoms":        normalise_symptoms(parsed.get("symptoms", {}), note_lower),
        "other_symptoms":  normalise_other_symptoms(parsed.get("other_symptoms", {}), note_lower),
        "onset_days":      onset,
        "severity":        severity,
        "red_flags":       parsed.get("red_flags", []),
        "treatments_given":[str(t) for t in parsed.get("treatments_given", []) if t],
        "referral":        parsed.get("referral"),
        "follow_up":       None,
    }
    syn = keyword_syndrome_tag(note_text)
    syn["encounter_id"] = encounter_id
    return {
        "encounter_id":    encounter_id,
        "errors":          [],
        "encounter":       encounter,
        "syndrome_tag":    syn,
        "checklist":       {"encounter_id": encounter_id, "questions": []},
        "processing_time_s": round(time.time() - t0, 2),
    }


def process_notes_batch(
    notes: list,
    combined_prompt: str,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> list:
    """
    Process multiple CHW notes in parallel GPU batches.
    ~batch_size x faster throughput vs sequential processing.

    Args:
        notes:      list of dicts with keys: note_text, encounter_id, location_id, week_id
        batch_size: number of notes to process simultaneously (4 is safe for T4 16GB)
    """
    results = []
    total   = len(notes)
    t_start = time.time()

    for i in range(0, total, batch_size):
        batch     = notes[i : i + batch_size]
        prompts   = [combined_prompt.replace("{note_text}", n["note_text"]) for n in batch]
        t0        = time.time()

        try:
            raw_outputs = run_medgemma_batch(prompts, model, tokenizer, max_new_tokens)
        except Exception as e:
            # Fallback: process individually if batch fails
            print(f"  ⚠️  Batch failed ({e}), falling back to sequential for this batch")
            raw_outputs = [
                run_medgemma(p, model, tokenizer, max_new_tokens) for p in prompts
            ]

        for raw, note in zip(raw_outputs, batch):
            parsed = parse_json_response(raw) or {}
            results.append(_build_result(parsed, note, t0))

        elapsed = time.time() - t_start
        done    = min(i + batch_size, total)
        rate    = elapsed / done
        print(f"  Batch {i//batch_size + 1}: notes {i+1}–{done}/{total}  "
              f"({time.time()-t0:.1f}s batch)  |  avg {rate:.1f}s/note")

    return results
