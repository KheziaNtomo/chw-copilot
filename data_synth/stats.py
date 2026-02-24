"""Quick stats on generated gold notes from AfriMed-QA."""
import json
from collections import Counter

notes = [json.loads(l) for l in open("data_synth/gold_encounters_afrimedqa.jsonl", "r", encoding="utf-8")]
print(f"Total AfriMed-QA gold notes: {len(notes)}")
print(f"Syndromes: {dict(Counter(n['gold_syndrome_tag'] for n in notes))}")
print(f"Ages: {dict(Counter(n['gold_age_group'] for n in notes))}")
print(f"Sex: {dict(Counter(n['gold_sex'] for n in notes))}")
print()

for i in [0, 5, 20, 50]:
    if i < len(notes):
        n = notes[i]
        print(f"--- Example {i} [{n['gold_syndrome_tag']}] [{n['gold_age_group']}/{n['gold_sex']}] ---")
        print(f"  Note: {n['note_text'][:250]}")
        if n["gold_red_flags"]:
            print(f"  Red flags: {n['gold_red_flags']}")
        print()
