"""Convert AfriMed-QA clinical vignettes into CHW-style gold notes.

Approach:
1. Filter AfriMed-QA for questions that contain patient presentations
   (age, presenting complaints, symptoms described in narrative form).
2. For each vignette, extract the clinical narrative and rewrite it as a
   short, telegraphic CHW field note (messy, abbreviated, realistic).
3. Assign gold labels: syndrome_tag, red_flags, age_group, sex.

The rewriting is done via simple rules (no LLM needed for this step):
- Strip exam-question framing ("A 5-year-old boy presents with...")
- Keep symptoms, demographics, and key clinical details
- Add CHW-style abbreviations ("pt", "yo", "hx", etc.)

Usage:
    python data_synth/convert_afrimedqa.py
"""
import json
import random
import re
from pathlib import Path

random.seed(42)

OUT = Path(__file__).parent
LOCATIONS = ["loc01", "loc02", "loc03", "loc04", "loc05", "loc06", "loc07", "loc08"]
WEEKS = list(range(1, 11))

# ── Helpers ──────────────────────────────────────────────────

def has_clinical_presentation(q: str) -> bool:
    """Check if a question contains a patient presentation vignette."""
    q_lower = q.lower()
    # Must have age/patient reference AND symptoms
    has_patient = any(p in q_lower for p in [
        "year-old", "year old", "yr old", "month-old", "month old",
        "presents with", "presented with", "comes with", "brought to",
        "complains of", "complaining of", "history of", "patient",
        "a child", "an infant", "a woman", "a man", "a boy", "a girl",
    ])
    has_symptoms = any(s in q_lower for s in [
        "fever", "cough", "diarrhea", "diarrhoea", "vomit", "rash",
        "breathing", "headache", "pain", "swelling", "weakness",
        "weight loss", "convulsion", "seizure", "dehydration",
    ])
    return has_patient and has_symptoms and len(q) > 100


def extract_vignette(question: str) -> str:
    """Extract the clinical presentation portion from a question."""
    # Take everything before the first actual question mark or option
    # Common patterns: vignette + "What is the most likely diagnosis?"
    parts = re.split(r'\?\s*(?:A\.|a\.|What|Which|How|The most|Select)', question)
    vignette = parts[0].strip()

    # Also handle "Which of the following..."
    parts2 = re.split(r'Which of the following', vignette, flags=re.IGNORECASE)
    vignette = parts2[0].strip()

    # Remove trailing question if present
    if vignette.endswith("?"):
        vignette = vignette[:-1].strip()

    return vignette


def vignette_to_chw_note(vignette: str) -> str:
    """Convert a clinical vignette into a messy CHW-style note."""
    note = vignette

    # Remove formal medical framing
    note = re.sub(r'^A\s+\d+-year-old\s+(male|female|boy|girl|man|woman|child|infant)',
                  lambda m: f"{m.group(1)}", note, flags=re.IGNORECASE)
    note = re.sub(r'presents? (to the (clinic|hospital|health center|emergency) )?with',
                  '', note, flags=re.IGNORECASE)
    note = re.sub(r'was brought (to the (clinic|hospital|health center))? (by .+? )?(with|for)',
                  '', note, flags=re.IGNORECASE)
    note = re.sub(r'comes? to the (clinic|hospital|health (center|post)) (with|for|complaining)',
                  '', note, flags=re.IGNORECASE)
    note = re.sub(r'complains? of', '', note, flags=re.IGNORECASE)

    # CHW-style abbreviations
    replacements = [
        (r'\btemperature\b', 'temp'),
        (r'\brespiratory rate\b', 'RR'),
        (r'\bblood pressure\b', 'BP'),
        (r'\bphysical examination\b', 'exam'),
        (r'\bexamination reveals?\b', 'exam shows'),
        (r'\bthe patient\b', 'pt'),
        (r'\bpatient\b', 'pt'),
        (r'\byears?\s*old\b', 'yo'),
        (r'\bmonths?\s*old\b', 'mo'),
        (r'\bdiagnosis\b', 'dx'),
        (r'\btreatment\b', 'tx'),
        (r'\bhistory of\b', 'hx'),
        (r'\bmedical history\b', 'hx'),
    ]
    for pattern, repl in replacements:
        note = re.sub(pattern, repl, note, flags=re.IGNORECASE)

    # Clean up extra whitespace and punctuation
    note = re.sub(r'\s+', ' ', note).strip()
    note = re.sub(r'^[\s,.\-]+', '', note).strip()

    # Make it shorter/messier: randomly drop some sentences
    sentences = re.split(r'(?<=[.!])\s+', note)
    if len(sentences) > 4:
        # Keep first 2-4 sentences (the most relevant clinical info)
        keep = min(4, max(2, len(sentences) - 2))
        sentences = sentences[:keep]
        note = ' '.join(sentences)

    return note


def classify_syndrome(text: str) -> str:
    """Simple rule-based syndrome classification for gold labels."""
    t = text.lower()

    has_fever = any(w in t for w in ["fever", "febrile", "hot body", "temperature", "pyrexia"])
    has_respiratory = any(w in t for w in ["cough", "breathing", "respiratory", "chest",
                                            "wheeze", "pneumonia", "dyspnea", "dyspnoea"])
    has_diarrhea = any(w in t for w in ["diarrhea", "diarrhoea", "watery stool", "loose stool",
                                         "running stomach"])

    if has_diarrhea and "blood" not in t:
        return "acute_watery_diarrhea"
    elif has_fever and has_respiratory:
        return "respiratory_fever"
    elif has_fever:
        return "respiratory_fever"  # fever alone → respiratory_fever bucket
    elif has_respiratory:
        return "respiratory_fever"
    else:
        return "other"


def extract_red_flags(text: str) -> list:
    """Extract red flags from text."""
    t = text.lower()
    flags = []
    flag_patterns = {
        "dehydration_signs": ["sunken eyes", "dry mouth", "skin pinch", "dehydrat", "no tears"],
        "unable_to_drink": ["unable to drink", "refuses to drink", "not drinking", "cant drink"],
        "persistent_vomiting": ["persistent vomit", "vomiting everything", "keeps vomiting"],
        "blood_in_stool": ["blood in stool", "bloody stool", "bloody diarr"],
        "high_fever": ["high fever", "39", "40", "very high temp", "burning"],
        "convulsions": ["convulsion", "seizure", "fitting", "fits"],
        "altered_consciousness": ["unconscious", "confused", "altered", "letharg", "very sleepy", "unresponsive"],
        "severe_malnutrition": ["severe malnutrition", "kwashiorkor", "marasmus", "wasting"],
        "chest_indrawing": ["chest indrawing", "chest wall", "subcostal", "intercostal"],
    }
    for flag, patterns in flag_patterns.items():
        if any(p in t for p in patterns):
            flags.append(flag)
    return flags


def extract_demographics(text: str) -> tuple:
    """Extract age_group and sex from text."""
    t = text.lower()

    # Sex
    if any(w in t for w in ["boy", "male", " man ", " his "]):
        sex = "male"
    elif any(w in t for w in ["girl", "female", "woman", " her ", "she "]):
        sex = "female"
    else:
        sex = "unknown"

    # Age group
    age_match = re.search(r'(\d+)[- ]?(year|yr|yo)', t)
    month_match = re.search(r'(\d+)[- ]?(month|mo)', t)
    if month_match:
        months = int(month_match.group(1))
        if months < 12:
            age_group = "infant"
        else:
            age_group = "child"
    elif age_match:
        years = int(age_match.group(1))
        if years < 1:
            age_group = "infant"
        elif years < 10:
            age_group = "child"
        elif years < 18:
            age_group = "adolescent"
        elif years < 65:
            age_group = "adult"
        else:
            age_group = "elderly"
    elif any(w in t for w in ["infant", "baby", "neonate", "newborn"]):
        age_group = "infant"
    elif any(w in t for w in ["child", "toddler"]):
        age_group = "child"
    elif any(w in t for w in ["elderly", "old man", "old woman"]):
        age_group = "elderly"
    else:
        age_group = "unknown"

    return age_group, sex


def main():
    # Load AfriMed-QA relevant entries
    relevant_path = OUT / "afrimedqa_relevant.jsonl"
    with open(relevant_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    print(f"Starting with {len(entries)} relevant AfriMed-QA entries")

    # Filter for actual clinical presentations
    vignettes = []
    for entry in entries:
        q = entry["question"]
        if has_clinical_presentation(q):
            vignette = extract_vignette(q)
            if len(vignette) > 50:  # Skip very short extractions
                vignettes.append({
                    "original_idx": entry["idx"],
                    "vignette": vignette,
                    "specialty": entry.get("specialty", ""),
                    "country": entry.get("country", ""),
                })

    print(f"Found {len(vignettes)} clinical vignettes")

    # Convert to CHW notes and assign gold labels
    gold_notes = []
    for i, v in enumerate(vignettes):
        chw_note = vignette_to_chw_note(v["vignette"])
        syndrome = classify_syndrome(chw_note)
        red_flags = extract_red_flags(chw_note)
        age_group, sex = extract_demographics(v["vignette"])  # use original for better extraction

        gold_notes.append({
            "encounter_id": f"afrimed_{i+1:03d}",
            "location_id": random.choice(LOCATIONS),
            "week_id": random.choice(WEEKS),
            "note_text": chw_note,
            "gold_syndrome_tag": syndrome,
            "gold_red_flags": red_flags,
            "gold_age_group": age_group,
            "gold_sex": sex,
            "source": f"afrimedqa_v2_idx_{v['original_idx']}",
            "specialty": v["specialty"],
        })

    # Save enriched gold encounters
    path = OUT / "gold_encounters_afrimedqa.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in gold_notes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(gold_notes)} enriched gold notes → {path}")

    # Stats
    from collections import Counter
    tag_counts = Counter(n["gold_syndrome_tag"] for n in gold_notes)
    print(f"Syndrome distribution: {dict(tag_counts)}")
    flag_counts = Counter(f for n in gold_notes for f in n["gold_red_flags"])
    print(f"Red flags found: {dict(flag_counts)}")
    age_counts = Counter(n["gold_age_group"] for n in gold_notes)
    print(f"Age groups: {dict(age_counts)}")

    # Print 3 examples
    print("\n=== EXAMPLE CHW NOTES ===")
    for n in gold_notes[:3]:
        print(f"\n[{n['gold_syndrome_tag']}] [{n['gold_age_group']}/{n['gold_sex']}]")
        print(f"  {n['note_text'][:200]}")
        if n['gold_red_flags']:
            print(f"  Red flags: {n['gold_red_flags']}")


if __name__ == "__main__":
    main()
