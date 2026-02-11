"""CHW Copilot pipeline orchestrator.

Runs the full encounter processing pipeline:
  1. Extract structured encounter from CHW note (NuExtract / MedGemma / stub)
  2. Validate and enforce evidence grounding
  3. Tag syndrome (MedGemma / deterministic)
  4. Generate checklist of missing info (MedGemma / deterministic)

Also provides surveillance-level functions:
  5. Aggregate encounters into weekly counts
  6. Detect anomalies
  7. Generate SITREP (MedGemma / template)
"""
import json
import time
from typing import Dict, Any, List, Optional

import pandas as pd

from . import config
from .validate import enforce_evidence, enforce_trigger_quotes, validation_report


def process_encounter(
    note_text: str,
    encounter_id: str = "unknown",
    location_id: str = "unknown",
    week_id: int = 0,
    extractor: str = "nuextract",
    use_model_tagger: bool = True,
    use_model_checklist: bool = True,
) -> Dict[str, Any]:
    """Process a single CHW note through the full pipeline.

    Args:
        note_text: Raw CHW field note
        encounter_id: Unique encounter identifier
        location_id: Location where encounter occurred
        week_id: Epidemiological week number
        extractor: "nuextract", "medgemma", or "stub"
        use_model_tagger: Use MedGemma for syndrome tagging (vs deterministic)
        use_model_checklist: Use MedGemma for checklist (vs deterministic)

    Returns:
        Dict with keys: encounter, syndrome_tag, checklist, validation
    """
    start = time.time()

    # Step 1: Extract structured encounter
    if extractor == "nuextract":
        from .extraction import extract_with_nuextract
        encounter = extract_with_nuextract(note_text, encounter_id, location_id, week_id)
    elif extractor == "medgemma":
        from .extraction import extract_with_medgemma
        encounter = extract_with_medgemma(note_text, encounter_id, location_id, week_id)
    else:
        from .extraction import stub_extract_full
        encounter = stub_extract_full(note_text, encounter_id, location_id, week_id)

    # Step 2: Enforce evidence grounding
    encounter, downgrades = enforce_evidence(encounter, note_text)

    # Step 3: Tag syndrome
    if use_model_tagger:
        from .tagger import tag_syndrome_medgemma
        syndrome = tag_syndrome_medgemma(encounter)
    else:
        from .tagger import tag_syndrome_deterministic
        syndrome = tag_syndrome_deterministic(encounter)

    # Enforce trigger quote evidence
    syndrome, invalid_quotes = enforce_trigger_quotes(syndrome, note_text)

    # Step 4: Generate checklist
    if use_model_checklist:
        from .checklist import generate_checklist_medgemma
        checklist = generate_checklist_medgemma(encounter)
    else:
        from .checklist import generate_checklist_deterministic
        checklist = generate_checklist_deterministic(encounter)

    # Step 5: Validation report
    validation = validation_report(encounter, note_text)

    elapsed = time.time() - start

    return {
        "encounter": encounter,
        "syndrome_tag": syndrome,
        "checklist": checklist,
        "validation": validation,
        "evidence_downgrades": downgrades,
        "invalid_trigger_quotes": invalid_quotes,
        "processing_time_s": round(elapsed, 2),
    }


def process_batch(
    notes: List[Dict[str, Any]],
    extractor: str = "nuextract",
    use_model_tagger: bool = True,
    use_model_checklist: bool = True,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """Process a batch of CHW notes through the pipeline.

    Args:
        notes: List of dicts with at least note_text, encounter_id, location_id, week_id
        extractor: Extraction model to use
        use_model_tagger: Use MedGemma for tagging
        use_model_checklist: Use MedGemma for checklist
        progress_callback: Optional callable(i, total) for progress reporting

    Returns:
        List of pipeline results
    """
    results = []
    total = len(notes)

    for i, note in enumerate(notes):
        if progress_callback:
            progress_callback(i, total)

        result = process_encounter(
            note_text=note["note_text"],
            encounter_id=note.get("encounter_id", f"enc_{i:03d}"),
            location_id=note.get("location_id", "unknown"),
            week_id=note.get("week_id", 0),
            extractor=extractor,
            use_model_tagger=use_model_tagger,
            use_model_checklist=use_model_checklist,
        )
        results.append(result)

    return results


def aggregate_for_surveillance(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate pipeline results into weekly syndrome counts by location.

    Args:
        results: Output from process_batch()

    Returns:
        DataFrame with columns: week_id, location_id, syndrome_tag, count
    """
    records = []
    for r in results:
        enc = r["encounter"]
        syn = r["syndrome_tag"]
        records.append({
            "week_id": enc.get("week_id", 0),
            "location_id": enc.get("location_id", "unknown"),
            "syndrome_tag": syn.get("syndrome_tag", "unclear"),
        })

    df = pd.DataFrame(records)
    counts = df.groupby(["week_id", "location_id", "syndrome_tag"]).size().reset_index(name="count")
    return counts


def run_surveillance(
    results: List[Dict[str, Any]],
    locations: pd.DataFrame = None,
    use_model_sitrep: bool = True,
) -> Dict[str, Any]:
    """Run the full surveillance pipeline on processed encounters.

    Args:
        results: Output from process_batch()
        locations: Optional locations DataFrame
        use_model_sitrep: Use MedGemma for SITREP generation

    Returns:
        Dict with weekly_counts, anomalies, and sitreps by week
    """
    from .detect import detect_anomalies

    # Aggregate
    weekly_counts = aggregate_for_surveillance(results)

    # Detect anomalies
    anomalies = detect_anomalies(weekly_counts)

    # Generate SITREPs
    sitreps = {}
    weeks = sorted(weekly_counts["week_id"].unique())

    for week in weeks:
        if use_model_sitrep:
            from .sitrep import generate_sitrep_medgemma
            sitrep = generate_sitrep_medgemma(anomalies, weekly_counts, week, locations)
        else:
            from .sitrep import generate_sitrep_template
            sitrep = generate_sitrep_template(anomalies, weekly_counts, week, locations)
        sitreps[week] = sitrep

    return {
        "weekly_counts": weekly_counts,
        "anomalies": anomalies,
        "sitreps": sitreps,
    }
