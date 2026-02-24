"""Merge the handcrafted and AfriMed-QA gold notes into one unified gold dataset."""
import json
from pathlib import Path
from collections import Counter

OUT = Path(__file__).parent

# Load both sets
handcrafted = [json.loads(l) for l in open(OUT / "gold_encounters.jsonl", encoding="utf-8")]
afrimedqa = [json.loads(l) for l in open(OUT / "gold_encounters_afrimedqa.jsonl", encoding="utf-8")]

merged = handcrafted + afrimedqa

# Re-assign encounter IDs for uniqueness
for i, note in enumerate(merged, start=1):
    note["encounter_id"] = f"gold_{i:03d}"

# Save merged
path = OUT / "gold_encounters_merged.jsonl"
with open(path, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Stats
tags = Counter(n["gold_syndrome_tag"] for n in merged)
ages = Counter(n["gold_age_group"] for n in merged)

print(f"Merged gold dataset: {len(merged)} notes")
print(f"  Handcrafted: {len(handcrafted)}")
print(f"  AfriMed-QA: {len(afrimedqa)}")
print(f"  Syndromes: {dict(tags)}")
print(f"  Age groups: {dict(ages)}")
print(f"Saved to {path}")
