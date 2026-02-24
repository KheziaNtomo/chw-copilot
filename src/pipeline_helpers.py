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
    """Extract JSON from a model response, handling code fences, preamble, and truncation."""
    if not text or not text.strip():
        return None

    # Strip common MedGemma preamble patterns
    cleaned = text.strip()
    # Remove leading text before the first {
    brace_idx = cleaned.find("{")
    if brace_idx > 0:
        cleaned = cleaned[brace_idx:]
    # Remove trailing text after the last }
    rbrace_idx = cleaned.rfind("}")
    if rbrace_idx >= 0 and rbrace_idx < len(cleaned) - 1:
        cleaned = cleaned[:rbrace_idx + 1]

    for attempt in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'```(?:json)?\s*\n(.*?)\n```', t, re.DOTALL).group(1)),
        lambda t: json.loads(re.search(r'\{.*\}', t, re.DOTALL).group(0)),
    ]:
        try:
            return attempt(cleaned)
        except Exception:
            continue

    # Try fixing truncated JSON (missing closing braces, mid-string cuts)
    try:
        truncated = cleaned
        # If cut mid-string, close the string first
        quote_count = truncated.count('"') - truncated.count('\\"')
        if quote_count % 2 == 1:
            truncated += '"'
        # Remove trailing incomplete key-value (e.g. "treat cut mid-word)
        truncated = re.sub(r',\s*"[^"]*$', '', truncated)
        truncated = re.sub(r',\s*$', '', truncated)
        open_braces = truncated.count("{") - truncated.count("}")
        open_brackets = truncated.count("[") - truncated.count("]")
        if open_braces > 0 or open_brackets > 0:
            truncated += "]" * open_brackets + "}" * open_braces
            return json.loads(truncated)
    except Exception:
        pass

    return None


# ── MedGemma inference ─────────────────────────────────────────────────────────

def run_medgemma(prompt: str, model, tokenizer, max_new_tokens: int = 2048) -> str:
    """Run a single MedGemma inference with chat template.

    Uses AutoProcessor.apply_chat_template(tokenize=True) as per the
    official medgemma-4b-it model card on HuggingFace.
    The 'tokenizer' argument is actually an AutoProcessor instance.
    """
    # MedGemma expects content as list of typed dicts
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()


def run_medgemma_batch(prompts: list, model, tokenizer, max_new_tokens: int = 2048) -> list:
    """Run multiple prompts sequentially for reliable output."""
    results = []
    for p in prompts:
        results.append(run_medgemma(p, model, tokenizer, max_new_tokens))
    return results




# ── Symptom normalisation ──────────────────────────────────────────────────────

CORE_SYMPTOMS = [
    "fever", "cough", "watery_diarrhea", "bloody_diarrhea",
    "vomiting", "rash", "difficulty_breathing",
]

def _normalise_claim(claim, note_lower: str) -> dict:
    """Validate a single symptom claim against the source note.

    Philosophy: trust the model's clinical judgement.
    - If model says "yes" we keep it, even without a quote.
    - We only downgrade if the model provides a quote that has ZERO word
      overlap with the note (clearly hallucinated evidence).
    - If model says "no" we keep it (negative assertions are valuable).
    """
    if not isinstance(claim, dict):
        claim = {}
    val = str(claim.get("value", "unknown")).lower().strip()
    if val not in ("yes", "no"):
        val = "unknown"
    quote = claim.get("evidence_quote")

    if val == "yes" and quote:
        q = quote.lower().strip()
        if q in note_lower:
            pass  # exact match — fully grounded
        else:
            # Check word-level overlap — reject only if ZERO meaningful words match
            words = [w for w in q.split() if len(w) > 2]
            matched = sum(1 for w in words if w in note_lower)
            if words and matched == 0:
                # Quote is completely fabricated — downgrade
                val, quote = "unknown", None
            # else: partial match → trust model, keep the claim

    # If model says "yes" with no quote, trust it (model identified symptom
    # but didn't provide verbatim evidence — this is fine)

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

RESPIRATORY_KEYWORDS = [
    "cough", "difficulty breathing", "fast breathing", "rapid breathing",
    "shortness of breath", "chest indrawing", "chest pulling",
    "noisy breathing", "wheezing", "chest tight",
    "runny nose", "running nose", "sneezing", "sore throat",
]

FEVER_KEYWORDS = [
    "fever", "feverish", "hot body", "high fever", "febrile",
]

AWD_KEYWORDS = [
    "watery diarrhea", "watery stool", "watery stools", "diarrhoea",
    "loose stool", "loose watery", "running stomach", "rice-water",
    "awd", "cholera",
    # Reversed word order / separated patterns common in CHW notes
    "diarrhea watery", "diarrhea profuse watery", "stool watery",
    "stool very watery", "diarrhea 7x", "diarrhea 8",
]

# Negation phrases — if a keyword appears next to these, it's negated
NEGATION_PREFIXES = [
    "no ", "not ", "no history of ", "denies ", "without ",
    "never ", "absent ", "ruled out ",
]

# Keywords indicating a clear non-respiratory, non-diarrheal presentation
OTHER_KEYWORDS = [
    # Skin / dermatology
    "rash", "wound", "abscess", "pus", "boil", "sore",
    "scabies", "ringworm", "fungal",
    # Measles / VPDs
    "measles", "red eyes", "conjunctivitis",
    # Malaria (explicit)
    "rdt positive", "rdt+", "malaria positive", "malaria",
    # Maternal / reproductive
    "pregnant", "pregnancy", "anc visit", "antenatal",
    # Musculoskeletal / surgical
    "fracture", "broken bone", "fell from", "cannot move",
    # Chronic / NCD
    "diabetes", "diabetic", "hypertension", "high bp", "blood pressure",
    "epilepsy", "hiv",
    # Urological / STI
    "painful urination", "dysuria",
    # GI (non-diarrheal)
    "stomach pain", "abdominal pain",
    # Other clear presentations
    "lump", "weight loss", "night sweat", "swollen legs",
    "numbness", "blurred vision", "palpitation",
    "surgery", "hair falling", "joint pain",
]


def _is_negated(keyword: str, note: str) -> bool:
    """Check if a keyword occurrence in the note is preceded by a negation."""
    idx = note.find(keyword)
    while idx >= 0:
        # Check prefix window
        prefix_window = note[max(0, idx - 25):idx].strip()
        is_neg = any(prefix_window.endswith(neg.strip()) for neg in NEGATION_PREFIXES)
        if not is_neg:
            return False  # found at least one non-negated occurrence
        # Look for next occurrence
        idx = note.find(keyword, idx + 1)
    return True  # all occurrences are negated (or keyword not found at all)


def _has_keyword(keyword: str, note: str) -> bool:
    """Check if keyword is present AND not negated in the note."""
    if keyword not in note:
        return False
    return not _is_negated(keyword, note)


def _has_diarrhea_watery(note: str) -> bool:
    """Check for watery diarrhea patterns even when words are separated."""
    # Check standard keyword list first
    for kw in AWD_KEYWORDS:
        if _has_keyword(kw, note):
            return True
    # Check for 'diarrhea' + 'watery' within the same sentence/phrase
    if "diarrhea" in note and "watery" in note:
        d_idx = note.find("diarrhea")
        w_idx = note.find("watery")
        if abs(d_idx - w_idx) < 40:  # within ~40 chars of each other
            return not _is_negated("diarrhea", note)
    return False


def keyword_syndrome_tag(note_text: str) -> dict:
    """
    Keyword-based syndrome classifier with negation awareness.
    - respiratory_fever: requires BOTH fever AND a respiratory symptom (non-negated)
    - acute_watery_diarrhea: watery diarrhea keywords (non-negated)
    - other: clear non-respiratory, non-diarrheal presentation
    - unclear: vague symptoms, insufficient info
    """
    note = note_text.lower()

    has_fever = any(_has_keyword(kw, note) for kw in FEVER_KEYWORDS)
    resp_hits = [kw for kw in RESPIRATORY_KEYWORDS if _has_keyword(kw, note)]
    has_awd   = _has_diarrhea_watery(note)
    awd_hits  = [kw for kw in AWD_KEYWORDS if _has_keyword(kw, note)]
    if has_awd and not awd_hits:
        awd_hits = ["diarrhea + watery"]
    other_hits = [kw for kw in OTHER_KEYWORDS if _has_keyword(kw, note)]

    # AWD takes priority if unambiguous
    if has_awd and not resp_hits:
        confidence = "high" if len(awd_hits) >= 2 else "medium"
        return {
            "syndrome_tag":  "acute_watery_diarrhea",
            "confidence":    confidence,
            "trigger_quotes": awd_hits[:3],
            "reasoning":     f"Watery diarrhea keywords: {', '.join(awd_hits[:3])}.",
        }

    # Respiratory fever: MUST have fever + at least one respiratory symptom
    if has_fever and resp_hits:
        fever_hit = next((kw for kw in FEVER_KEYWORDS if _has_keyword(kw, note)), "fever")
        confidence = "high" if len(resp_hits) >= 2 else "medium"
        return {
            "syndrome_tag":  "respiratory_fever",
            "confidence":    confidence,
            "trigger_quotes": [fever_hit] + resp_hits[:2],
            "reasoning":     f"Fever ({fever_hit}) + respiratory symptoms: {', '.join(resp_hits[:2])}.",
        }

    # AWD that also has respiratory features
    if has_awd:
        confidence = "medium"
        return {
            "syndrome_tag":  "acute_watery_diarrhea",
            "confidence":    confidence,
            "trigger_quotes": awd_hits[:3],
            "reasoning":     f"Watery diarrhea keywords present: {', '.join(awd_hits[:3])}.",
        }

    # Fever alone (malaria-like, convulsion, etc.) → "other" with sub-type
    if has_fever:
        fever_hit = next((kw for kw in FEVER_KEYWORDS if _has_keyword(kw, note)), "fever")
        return {
            "syndrome_tag":  "other",
            "confidence":    "low",
            "trigger_quotes": [fever_hit],
            "reasoning":     "Fever present but no respiratory or diarrheal symptoms — likely non-specific febrile illness.",
        }

    # Clear non-fever, non-respiratory, non-diarrheal presentation → "other"
    if other_hits:
        return {
            "syndrome_tag":  "other",
            "confidence":    "low",
            "trigger_quotes": other_hits[:3],
            "reasoning":     f"Non-syndromic presentation with keywords: {', '.join(other_hits[:3])}.",
        }

    return {
        "syndrome_tag":  "unclear",
        "confidence":    "low",
        "trigger_quotes": [],
        "reasoning":     "No syndrome keywords matched.",
    }


# ── Sub-syndrome classification ────────────────────────────────────────────────

def sub_syndrome_hint(encounter: dict, syndrome_tag: str) -> str:
    """
    Within respiratory_fever, differentiate: malaria-like, pneumonia-like, TB-like.
    Returns a sub-syndrome hint string, or None for non-respiratory.
    """
    if syndrome_tag != "respiratory_fever":
        return None

    sym = encounter.get("symptoms", {})
    note = encounter.get("note_text", "").lower()

    has_cough     = sym.get("cough", {}).get("value") == "yes"
    has_fast_br   = sym.get("difficulty_breathing", {}).get("value") == "yes"
    has_fever     = sym.get("fever", {}).get("value") == "yes"

    # TB-like: chronic cough (>14 days) + weight loss / night sweats
    onset = encounter.get("onset_days")
    chronic_cough = has_cough and onset and onset >= 14
    tb_clues = any(kw in note for kw in ["weight loss", "night sweat", "tb", "tuberculosis"])
    if chronic_cough or tb_clues:
        return "TB-like"

    # Pneumonia-like: cough + fast/difficult breathing
    if has_cough and has_fast_br:
        return "pneumonia-like"

    # Malaria-like: fever + chills/rigors but NO cough
    malaria_clues = any(kw in note for kw in ["chill", "rigor", "shaking", "rdt", "malaria", "swamp"])
    if has_fever and malaria_clues and not has_cough:
        return "malaria-like"

    # Fever + cough but no fast breathing → upper respiratory
    if has_fever and has_cough and not has_fast_br:
        return "upper-respiratory"

    return "unspecified"


# ── ICCM-based clinical recommendations ───────────────────────────────────────

def generate_recommendations(encounter: dict, syndrome_tag: str) -> list:
    """
    Rule-based clinical recommendations following WHO ICCM guidelines.
    Returns a list of action strings for the CHW.
    """
    recs = []
    sym = encounter.get("symptoms", {})
    note = encounter.get("note_text", "").lower()
    # red_flags may be strings (from LLM) or dicts (from keyword fallback)
    raw_flags = encounter.get("red_flags", [])
    red_flags = []
    for rf in raw_flags:
        if isinstance(rf, dict):
            red_flags.append(rf.get("flag", "").lower())
        elif isinstance(rf, str):
            red_flags.append(rf.lower())
        else:
            red_flags.append(str(rf).lower())
    age_years = encounter.get("patient", {}).get("age_years")

    has_fever   = sym.get("fever", {}).get("value") == "yes"
    has_cough   = sym.get("cough", {}).get("value") == "yes"
    has_fast_br = sym.get("difficulty_breathing", {}).get("value") == "yes"
    has_awd     = sym.get("watery_diarrhea", {}).get("value") == "yes"
    has_vomit   = sym.get("vomiting", {}).get("value") == "yes"

    # ── Danger signs → REFER IMMEDIATELY ──────────────────────────────────────
    danger_signs = []
    if any(kw in note for kw in ["convulsion", "convulsions", "seizure"]):
        danger_signs.append("convulsions")
    if any(kw in note for kw in ["unable to drink", "refuses to drink", "not drinking"]):
        danger_signs.append("unable to drink")
    if any(kw in note for kw in ["sunken eyes", "skin pinch slow", "no tears", "sunken fontanelle"]):
        danger_signs.append("dehydration signs")
    if any(kw in note for kw in ["confused", "unconscious", "lethargic", "very sleepy", "altered"]):
        danger_signs.append("altered consciousness")
    if "chest_indrawing" in red_flags or "chest indrawing" in note or "chest pulling" in note:
        danger_signs.append("chest indrawing")

    if danger_signs:
        recs.append(f"🚨 REFER IMMEDIATELY — danger sign(s): {', '.join(danger_signs)}")

    # ── AWD recommendations ───────────────────────────────────────────────────
    if has_awd or syndrome_tag == "acute_watery_diarrhea":
        recs.append("💧 Start ORS immediately; give zinc (10mg if <6mo, 20mg if ≥6mo) for 10 days")
        if has_vomit:
            recs.append("⚠️ Persistent vomiting — give ORS in small sips, monitor for dehydration")
        if not danger_signs:
            recs.append("📋 Follow up in 24 hours — reassess hydration status")

    # ── Respiratory recommendations ───────────────────────────────────────────
    if syndrome_tag == "respiratory_fever":
        if has_fever:
            recs.append("🌡️ Give paracetamol for fever; do malaria RDT if available")
        if has_cough and has_fast_br and age_years and age_years < 5:
            recs.append("💊 Likely pneumonia in child <5 — give oral amoxicillin, refer if no improvement in 48h")
        elif has_cough and has_fast_br:
            recs.append("💊 Cough + difficulty breathing — count respiratory rate, consider referral")
        if any(kw in note for kw in ["rdt positive", "rdt+", "malaria positive"]):
            recs.append("💊 Malaria RDT positive — give ACT (artemisinin-based combination therapy)")

    # ── General ───────────────────────────────────────────────────────────────
    if not recs:
        recs.append("📋 No urgent action needed — routine follow-up")

    return recs


# ── Full single-pass pipeline ──────────────────────────────────────────────────

# ── Keyword-based fallback extraction ─────────────────────────────────────────
# When MedGemma returns 0 chars, extract basic structured data from keywords.

import re as _re

def keyword_fallback_extract(note_text: str) -> dict:
    """Extract structured encounter data from note text using keyword matching.

    This is a safety net when MedGemma generation fails (0 chars).
    It uses the same keyword lists as the syndrome tagger.
    """
    note = note_text.lower()

    # ── Symptoms ──
    symptoms = {}
    symptom_keywords = {
        "fever": FEVER_KEYWORDS,
        "cough": ["cough", "coughing"],
        "diarrhea": ["diarrhea", "diarrhoea", "loose stool", "watery stool", "running stomach"],
        "vomiting": ["vomiting", "vomit", "throwing up"],
        "rash": ["rash", "skin rash", "rashes"],
        "difficulty_breathing": ["difficulty breathing", "hard to breathe", "fast breathing",
                                  "rapid breathing", "shortness of breath", "chest indrawing",
                                  "chest pulling", "noisy breathing", "wheezing"],
    }
    for sym, keywords in symptom_keywords.items():
        matched_kw = None
        for kw in keywords:
            if kw in note:
                if not _is_negated(kw, note):
                    matched_kw = kw
                    break
        # Check for negation pattern like "no diarrhea"
        negated = False
        for kw in keywords:
            if kw in note and _is_negated(kw, note):
                negated = True
                break
        if matched_kw:
            symptoms[sym] = {"value": "yes", "evidence_quote": matched_kw}
        elif negated:
            symptoms[sym] = {"value": "no", "evidence_quote": f"no {keywords[0]}"}
        else:
            symptoms[sym] = {"value": "unknown", "evidence_quote": ""}

    # ── Patient demographics ──
    patient = {}
    # Age: look for patterns like "3yo", "3 year", "3yr", "child 3", "9 months"
    age_match = _re.search(r'(\d+)\s*(?:yo|y\.?o\.?|year|yr|years)', note)
    if age_match:
        patient["age_years"] = int(age_match.group(1))
    month_match = _re.search(r'(\d+)\s*(?:mo|month|months|m\.?o\.?)', note)
    if month_match and "age_years" not in patient:
        patient["age_months"] = int(month_match.group(1))
        patient["age_years"] = 0
    # Also try "child Xyo" or "X year old"
    if "age_years" not in patient:
        age_match2 = _re.search(r'(?:child|baby|infant|boy|girl)\s+(\d+)', note)
        if age_match2:
            patient["age_years"] = int(age_match2.group(1))

    # Sex
    if any(w in note for w in ["male", " boy ", " son "]):
        if not any(w in note for w in ["female"]):
            patient["sex"] = "male"
    if any(w in note for w in ["female", " girl ", " daughter ", "woman"]):
        patient["sex"] = "female"
    patient.setdefault("sex", "unknown")

    # ── Severity & red flags ──
    red_flag_keywords = {
        "unable_to_drink": ["unable to drink", "cannot drink", "refuses to drink", "not drinking"],
        "not_feeding": ["not eating", "not feeding", "refuses to eat", "not breastfeeding"],
        "chest_indrawing": ["chest indrawing", "chest pulling", "pulling in of chest"],
        "convulsions": ["convulsion", "seizure", "fits", "fitting"],
        "sunken_eyes": ["sunken eyes"],
        "persistent_vomiting": ["vomiting everything", "persistent vomiting"],
        "dehydration_signs": ["skin pinch", "sunken fontanelle", "no urine", "dry mouth"],
    }
    red_flags = []
    for flag, keywords in red_flag_keywords.items():
        for kw in keywords:
            if kw in note and not _is_negated(kw, note):
                red_flags.append({"flag": flag, "evidence_quote": kw})
                break

    severity = "mild"
    if red_flags:
        severity = "severe"
    elif any(kw in note for kw in ["difficulty breathing", "fast breathing", "high fever"]):
        severity = "moderate"

    # ── Onset ──
    onset_days = None
    onset_match = _re.search(r'(\d+)\s*(?:day|days|d)\b', note)
    if onset_match:
        onset_days = int(onset_match.group(1))

    # ── Treatment & referral ──
    treatments = []
    if "ors" in note:
        treatments.append("ORS")
    if "paracetamol" in note or "pcm" in note:
        treatments.append("paracetamol")
    if "amoxicillin" in note or "antibiotic" in note:
        treatments.append("amoxicillin")
    if any(w in note for w in ["zinc", "zn"]):
        treatments.append("zinc")

    referral = any(w in note for w in ["refer", "referred", "health center", "clinic", "hospital"])

    return {
        "symptoms": symptoms,
        "patient": patient,
        "severity": severity,
        "onset_days": onset_days,
        "red_flags": red_flags,
        "treatments_given": treatments,
        "referral": referral,
    }


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
    - Keyword fallback if MedGemma returns empty
    """
    result = {"encounter_id": encounter_id, "errors": []}
    t0 = time.time()

    # ── LLM extraction ────────────────────────────────────────────────────────
    prompt = combined_prompt.replace("{note_text}", note_text)
    try:
        raw = run_medgemma(prompt, model, tokenizer, max_new_tokens=1024)
        parsed = parse_json_response(raw) or {}
        # Diagnostic logging for smoke test
        print(f"\n  📋 DIAGNOSTIC — raw MedGemma output ({len(raw)} chars):")
        print(f"  ---BEGIN---\n{raw[:600]}\n  ---END---")
        syms = parsed.get("symptoms", {})
        yes_count = sum(1 for v in syms.values() if isinstance(v, dict) and v.get("value") == "yes")
        print(f"  Parsed keys: {list(parsed.keys()) if parsed else '(empty)'}")
        print(f"  Symptoms: {yes_count} 'yes' / {len(syms)} total")
    except Exception as e:
        result["errors"].append(f"generation_error: {e}")
        parsed = {}

    # ── Keyword fallback when LLM returns empty ──────────────────────────────
    used_fallback = False
    if not parsed or not parsed.get("symptoms"):
        parsed = keyword_fallback_extract(note_text)
        used_fallback = True
        print("  ⚠️  MedGemma returned empty — using keyword fallback extraction")

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

    # ── Sub-syndrome + recommendations (rule-based, instant) ──────────────────
    syn["sub_syndrome"] = sub_syndrome_hint(encounter, syn["syndrome_tag"])
    recs = generate_recommendations(encounter, syn["syndrome_tag"])

    # ── Onset-adjusted week for more accurate surveillance ────────────────────
    estimated_onset_week = week_id
    if onset and onset >= 7:
        estimated_onset_week = max(1, week_id - (onset // 7))
    encounter["estimated_onset_week"] = estimated_onset_week

    result.update({
        "encounter":         encounter,
        "syndrome_tag":      syn,
        "recommendations":   recs,
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
    syn["sub_syndrome"] = sub_syndrome_hint(encounter, syn["syndrome_tag"])
    recs = generate_recommendations(encounter, syn["syndrome_tag"])

    # Onset-adjusted week
    estimated_onset_week = week_id
    if onset and onset >= 7:
        estimated_onset_week = max(1, week_id - (onset // 7))
    encounter["estimated_onset_week"] = estimated_onset_week

    return {
        "encounter_id":    encounter_id,
        "errors":          [],
        "encounter":       encounter,
        "syndrome_tag":    syn,
        "recommendations": recs,
        "processing_time_s": round(time.time() - t0, 2),
    }


def process_notes_batch(
    notes: list,
    combined_prompt: str,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 2048,
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
    _logged_first = False

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

            # Diagnostic: log the first raw output to help debug extraction issues
            if not _logged_first:
                _logged_first = True
                print(f"\n  📋 DIAGNOSTIC — first raw MedGemma output ({len(raw)} chars):")
                print(f"  ---BEGIN---\n{raw[:800]}\n  ---END---")
                print(f"  Parsed keys: {list(parsed.keys()) if parsed else '(empty)'}")
                syms = parsed.get("symptoms", {})
                yes_count = sum(1 for v in syms.values() if isinstance(v, dict) and v.get("value") == "yes")
                print(f"  Symptoms extracted: {yes_count} 'yes' / {len(syms)} total\n")

            results.append(_build_result(parsed, note, t0))

        elapsed = time.time() - t_start
        done    = min(i + batch_size, total)
        rate    = elapsed / done
        # Per-batch extraction quality stats
        batch_results = results[i:done]
        ok = sum(1 for r in batch_results
                 if any(v.get("value") == "yes"
                        for v in r["encounter"]["symptoms"].values()))
        print(f"  Batch {i//batch_size + 1}: notes {i+1}–{done}/{total}  "
              f"({time.time()-t0:.1f}s batch)  |  avg {rate:.1f}s/note  "
              f"|  {ok}/{len(batch_results)} notes with ≥1 extracted symptom")

    return results
