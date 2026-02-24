"""Explore the AfriMed-QA dataset and extract relevant clinical vignettes.

Saves filtered vignettes that can be converted to CHW-style notes.
"""
from datasets import load_dataset
import json

ds = load_dataset("afrimedqa/afrimedqa_v2", split="train")
print(f"Total entries: {len(ds)}")
print(f"Columns: {ds.column_names}")
print()

# Find questions with clinical presentations
symptoms = ["fever", "cough", "diarrhea", "diarrhoea", "vomit", "rash",
            "breathing", "headache", "pain", "malaria", "pneumonia",
            "dehydration", "swollen", "weight loss"]

relevant = []
for i, row in enumerate(ds):
    q = (row.get("question", "") or "").lower()
    if any(s in q for s in symptoms) and len(q) > 80:
        relevant.append({
            "idx": i,
            "question": row.get("question", ""),
            "specialty": row.get("specialty", ""),
            "country": row.get("country_of_origin", ""),
            "question_type": row.get("question_type", ""),
        })

print(f"Found {len(relevant)} clinically relevant entries")
print()

# Print 10 samples
for r in relevant[:10]:
    print(f"--- IDX {r['idx']} [{r['specialty']}] [{r['country']}] ---")
    print(r["question"][:500])
    print()

# Save all relevant for later use
with open("data_synth/afrimedqa_relevant.jsonl", "w", encoding="utf-8") as f:
    for r in relevant:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved {len(relevant)} relevant entries to data_synth/afrimedqa_relevant.jsonl")
