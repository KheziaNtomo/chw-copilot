"""
fix_gold_data.py — Adds gold_age_years field extracted from note text.
Stores age as string: "5" for 5 years, "8m" for 8 months.
Null if not mentioned.
Run from repo root:  python scripts/fix_gold_data.py
"""
import re
import json
from pathlib import Path

# Match years: 3yo, 5yr, 5 years old, aged 3, etc.
YEAR_RE = re.compile(
    r'\b(\d{1,3})\s*(?:yo|yr|year[s]?(?:\s*old)?|-year-old|y\.o\.)\b'
    r'|(?:aged?|age)\s*(\d{1,3})\b'
    r'|(?:male|female|man|woman|boy|girl)\s+(\d{1,3})\b'
    r'|(?:approx)\s+(\d{1,3})\b',
    re.IGNORECASE
)
# Match months: 8mo, 10 months, 14mo, etc.
MONTH_RE = re.compile(
    r'\b(\d{1,2})\s*(?:mo(?:nths?)?)\b',
    re.IGNORECASE
)

def extract_age(note_text):
    """Extract age as string: '5' for years, '8m' for months, None if unknown."""
    m_yr = YEAR_RE.search(note_text)
    if m_yr:
        yr = m_yr.group(1) or m_yr.group(2) or m_yr.group(3) or m_yr.group(4)
        if yr:
            return str(int(yr))

    m_mo = MONTH_RE.search(note_text)
    if m_mo:
        return f"{int(m_mo.group(1))}m"

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
    for note in notes:
        note["gold_age_years"] = extract_age(note.get("note_text", ""))

    out = "\n".join(json.dumps(n, ensure_ascii=False) for n in notes) + "\n"
    path.write_text(out, encoding="utf-8")
    
    # Show a sample
    sample = [(n["encounter_id"], n["gold_age_years"]) for n in notes[:10]]
    print(f"✅ {path}: {len(notes)} notes updated")
    for eid, age in sample:
        print(f"   {eid}: {age}")
