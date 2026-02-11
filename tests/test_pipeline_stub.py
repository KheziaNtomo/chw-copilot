"""End-to-end pipeline test using the stub (no model) path."""
import sys
import json
sys.path.insert(0, ".")

from src.pipeline import process_encounter, process_batch, run_surveillance
import pandas as pd

# Test 1: Single encounter
print("=" * 60)
print("TEST 1: Single encounter (stub extractor)")
print("=" * 60)

note = "Child 3yo M fever 3 days cough bad rash on chest no diarrhea mother says not eating gave ORS referred health center"
result = process_encounter(
    note,
    encounter_id="test_001",
    location_id="loc01",
    week_id=5,
    extractor="stub",
    use_model_tagger=False,
    use_model_checklist=False,
)

print("Encounter ID:", result["encounter"]["encounter_id"])
print("Symptoms extracted:", {k: v["value"] for k, v in result["encounter"]["symptoms"].items()})
print("Syndrome tag:", result["syndrome_tag"]["syndrome_tag"])
print("Confidence:", result["syndrome_tag"]["confidence"])
print("Trigger quotes:", result["syndrome_tag"]["trigger_quotes"])
print("Checklist questions:", len(result["checklist"]["questions"]))
for q in result["checklist"]["questions"]:
    print(f"  [{q['priority']}] {q['field']}: {q['question']}")
print("Validation pass:", result["validation"]["overall_pass"])
print("Evidence downgrades:", result["evidence_downgrades"])
print("Processing time:", result["processing_time_s"], "s")

# Test 2: Batch processing
print()
print("=" * 60)
print("TEST 2: Batch processing (3 notes)")
print("=" * 60)

notes = [
    {"note_text": "woman 25yo fever 2 days watery diarrhea vomiting dehydrated sunken eyes", "encounter_id": "batch_001", "location_id": "loc02", "week_id": 7},
    {"note_text": "boy 8yo cough 5 days difficulty breathing chest indrawing high fever", "encounter_id": "batch_002", "location_id": "loc04", "week_id": 7},
    {"note_text": "man 40yo headache back pain no fever no cough no diarrhea", "encounter_id": "batch_003", "location_id": "loc01", "week_id": 7},
]

results = process_batch(notes, extractor="stub", use_model_tagger=False, use_model_checklist=False)
for r in results:
    print(f"  {r['encounter']['encounter_id']}: {r['syndrome_tag']['syndrome_tag']} ({r['syndrome_tag']['confidence']})")

# Test 3: Surveillance pipeline
print()
print("=" * 60)
print("TEST 3: Surveillance (SITREP generation)")
print("=" * 60)

surveillance = run_surveillance(results, use_model_sitrep=False)
print("Weekly counts shape:", surveillance["weekly_counts"].shape)
print("Anomalies found:", len(surveillance["anomalies"]))
for week, sitrep in surveillance["sitreps"].items():
    print(f"  Week {week}: {sitrep['narrative'][:120]}...")

print()
print("ALL TESTS PASSED")
