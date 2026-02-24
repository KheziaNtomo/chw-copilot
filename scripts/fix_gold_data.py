"""
fix_gold_data.py — Adds gold_age_years field extracted from note text.
Replaces gold_age_group (which was the group label e.g. 'child') with
gold_age_years (numeric, e.g. 5) where extractable, null otherwise.
Run from repo root:  python scripts/fix_gold_data.py
"""
import re
import json
from pathlib import Path

AGE_RE = re.compile(
    r'\b(\d{1,3})\s*(?:yo|yr|year[s]?(?:\s*old)?|-year-old|y\.o\.)\b'
    r'|(?:aged?|age)\s*(\d{1,3})\b'
    r'|(\d{1,3})\s*(?:mo(?:nths?)?)\b',  # months — convert to fraction
    re.IGNORECASE
)

def extract_age(note_text):
    m = AGE_RE.search(note_text)
    if not m:
        return None
    yr, aged, mo = m.group(1), m.group(2), m.group(3)
    if yr:   return int(yr)
    if aged: return int(aged)
    if mo:   return None  # sub-year infant — leave as null, age_group covers it
    return None

PATHS = [
    Path("data_synth/gold_encounters.jsonl"),
    Path("data_synth/gold_encounters_merged.jsonl"),
]

for path in PATHS:
    if not path.exists():
        print(f"Skipping {path} (not found)")
        continue

    notes = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    updated = 0
    for note in notes:
        age = extract_age(note.get("note_text", ""))
        note["gold_age_years"] = age   # numeric or null
        # Keep gold_age_group for backwards compat — it describes the category
        updated += 1

    out = "\n".join(json.dumps(n, ensure_ascii=False) for n in notes) + "\n"
    path.write_text(out, encoding="utf-8")
    print(f"✅ {path}: updated {updated} notes with gold_age_years")
