"""CHW Copilot pipeline orchestrator — Agentic Workflow.

Six-agent pipeline with deterministic fallbacks and evidence verification:
  Agent 1: Extract structured encounter from CHW note (MedGemma / stub)
  Agent 2: Enforce evidence grounding (deterministic)
  Agent 3: Self-consistency hallucination detection (MedGemma verification prompt)
  Agent 4: Tag syndrome (MedGemma / deterministic)
  Agent 5: Generate checklist of missing info (MedGemma / deterministic)
  Agent 6: Validate against JSON Schema (deterministic)

Also provides surveillance-level functions:
  - Aggregate encounters into weekly counts
  - Detect anomalies (deterministic z-score)
  - Generate weekly SITREP

Adaptation methods: prompt engineering, agentic orchestration,
evidence grounding enforcement, self-consistency hallucination detection.
"""
import json
import time
from typing import Dict, Any, List, Optional

import pandas as pd

from . import config
from .validate import enforce_evidence, enforce_trigger_quotes, validation_report

# ── Pipeline Agent Metadata ──────────────────────────────────
# Describes the agentic orchestration design for judges + docs
PIPELINE_AGENTS = [
    {
        "id": "extract",
        "name": "Encounter Extractor",
        "type": "llm",
        "model": config.MEDGEMMA_MODEL,
        "description": "Extracts structured encounter JSON from free-text CHW note",
        "input": "Raw CHW field note",
        "output": "Schema-validated encounter with evidence quotes",
        "fallback": "Rule-based keyword extractor",
    },
    {
        "id": "evidence_enforce",
        "name": "Evidence Grounder",
        "type": "deterministic",
        "description": "Enforces verbatim evidence grounding — downgrades ungrounded claims",
        "input": "Extracted encounter + original note",
        "output": "Grounded encounter with downgrades list",
        "fallback": None,
    },
    {
        "id": "hallucination_check",
        "name": "Hallucination Detector",
        "type": "self-consistency",
        "model": config.MEDGEMMA_MODEL,
        "description": "Self-consistency check — asks MedGemma if evidence supports each claim",
        "input": "Grounded encounter + evidence quotes",
        "output": "Verification result with flagged contradictions",
        "fallback": "Deterministic negation-word check",
    },
    {
        "id": "tag",
        "name": "Syndrome Tagger",
        "type": "llm",
        "model": config.MEDGEMMA_MODEL,
        "description": "Tags encounter with syndromic category + confidence",
        "input": "Grounded encounter",
        "output": "Syndrome tag with trigger quotes and reasoning",
        "fallback": "Rule-based symptom-combination tagger",
    },
    {
        "id": "checklist",
        "name": "Checklist Generator",
        "type": "llm",
        "model": config.MEDGEMMA_MODEL,
        "description": "Generates prioritized follow-up questions for missing data",
        "input": "Grounded encounter",
        "output": "Checklist of up to 5 prioritized questions",
        "fallback": "Rule-based missing-field checker",
    },
    {
        "id": "validate",
        "name": "Schema Validator",
        "type": "deterministic",
        "description": "Validates encounter against JSON Schema + generates report",
        "input": "Final encounter",
        "output": "Validation report (schema errors, evidence downgrades)",
        "fallback": None,
    },
]


def process_encounter(
    note_text: str,
    encounter_id: str = "unknown",
    location_id: str = "unknown",
    week_id: int = 0,
    extractor: str = "medgemma",
    use_model_tagger: bool = True,
    use_model_checklist: bool = True,
    run_hallucination_check: bool = True,
) -> Dict[str, Any]:
    """Process a single CHW note through the full agentic pipeline.

    Args:
        note_text: Raw CHW field note
        encounter_id: Unique encounter identifier
        location_id: Location where encounter occurred
        week_id: Epidemiological week number
        extractor: "medgemma" or "stub"
        use_model_tagger: Use MedGemma for syndrome tagging (vs deterministic)
        use_model_checklist: Use MedGemma for checklist (vs deterministic)
        run_hallucination_check: Run self-consistency hallucination check

    Returns:
        Dict with keys: encounter, syndrome_tag, checklist, validation,
                       agent_trace, evidence_downgrades, hallucination_check
    """
    pipeline_start = time.time()
    agent_trace = []

    # ── Agent 1: Extract structured encounter ────────────────
    t0 = time.time()
    fallback_used = False
    if extractor == "medgemma":
        from .extraction import extract_with_medgemma
        encounter = extract_with_medgemma(note_text, encounter_id, location_id, week_id)
    else:
        from .extraction import stub_extract_full
        encounter = stub_extract_full(note_text, encounter_id, location_id, week_id)
        fallback_used = True

    agent_trace.append({
        "agent": "extract",
        "name": "Encounter Extractor",
        "duration_s": round(time.time() - t0, 3),
        "fallback_used": fallback_used,
        "output_summary": f"Extracted encounter with {_count_yes_symptoms(encounter)} positive symptoms",
    })

    # ── Agent 2: Enforce evidence grounding ──────────────────
    t0 = time.time()
    encounter, downgrades = enforce_evidence(encounter, note_text)

    agent_trace.append({
        "agent": "evidence_enforce",
        "name": "Evidence Grounder",
        "duration_s": round(time.time() - t0, 3),
        "fallback_used": False,
        "output_summary": f"{len(downgrades)} claims downgraded for missing/invalid evidence",
    })

    # ── Agent 3: Self-consistency hallucination detection ─────
    hallucination_result = None
    if run_hallucination_check:
        t0 = time.time()
        from .hallucination import verify_extraction_claims, verify_claims_offline

        # Try model-based self-consistency first, fall back to deterministic
        generate_fn = None
        fallback_used = True
        if extractor == "medgemma":
            try:
                from .models import generate_medgemma
                generate_fn = generate_medgemma
                fallback_used = False
            except Exception:
                pass

        if generate_fn:
            hallucination_result = verify_extraction_claims(encounter, note_text, generate_fn=generate_fn)
        else:
            hallucination_result = verify_claims_offline(encounter, note_text)
            fallback_used = True

        agent_trace.append({
            "agent": "hallucination_check",
            "name": "Hallucination Detector",
            "duration_s": round(time.time() - t0, 3),
            "fallback_used": fallback_used,
            "output_summary": (
                f"Checked {hallucination_result.get('claims_checked', 0)} claims, "
                f"{len(hallucination_result.get('flagged_claims', []))} flagged"
                f" ({hallucination_result.get('method', 'self-consistency')})"
            ),
        })

    # ── Agent 4: Tag syndrome ────────────────────────────────
    t0 = time.time()
    fallback_used = False
    if use_model_tagger:
        from .tagger import tag_syndrome_medgemma
        syndrome = tag_syndrome_medgemma(encounter)
    else:
        from .tagger import tag_syndrome_deterministic
        syndrome = tag_syndrome_deterministic(encounter)
        fallback_used = True

    # Enforce trigger quote evidence
    syndrome, invalid_quotes = enforce_trigger_quotes(syndrome, note_text)

    agent_trace.append({
        "agent": "tag",
        "name": "Syndrome Tagger",
        "duration_s": round(time.time() - t0, 3),
        "fallback_used": fallback_used,
        "output_summary": f"Tagged as {syndrome.get('syndrome_tag', '?')} ({syndrome.get('confidence', '?')})",
    })

    # ── Agent 5: Generate checklist ──────────────────────────
    t0 = time.time()
    fallback_used = False
    if use_model_checklist:
        from .checklist import generate_checklist_medgemma
        checklist = generate_checklist_medgemma(encounter)
    else:
        from .checklist import generate_checklist_deterministic
        checklist = generate_checklist_deterministic(encounter)
        fallback_used = True

    agent_trace.append({
        "agent": "checklist",
        "name": "Checklist Generator",
        "duration_s": round(time.time() - t0, 3),
        "fallback_used": fallback_used,
        "output_summary": f"Generated {len(checklist.get('questions', []))} follow-up questions",
    })

    # ── Agent 6: Validation report ───────────────────────────
    t0 = time.time()
    validation = validation_report(encounter, note_text)

    agent_trace.append({
        "agent": "validate",
        "name": "Schema Validator",
        "duration_s": round(time.time() - t0, 3),
        "fallback_used": False,
        "output_summary": f"Schema valid: {validation.get('schema_valid', '?')}, overall pass: {validation.get('overall_pass', '?')}",
    })

    elapsed = time.time() - pipeline_start

    return {
        "encounter": encounter,
        "syndrome_tag": syndrome,
        "checklist": checklist,
        "validation": validation,
        "evidence_downgrades": downgrades,
        "invalid_trigger_quotes": invalid_quotes,
        "hallucination_check": hallucination_result,
        "agent_trace": agent_trace,
        "processing_time_s": round(elapsed, 2),
    }


def process_batch(
    notes: List[Dict[str, Any]],
    extractor: str = "medgemma",
    use_model_tagger: bool = True,
    use_model_checklist: bool = True,
    run_hallucination_check: bool = True,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """Process a batch of CHW notes through the agentic pipeline.

    Args:
        notes: List of dicts with at least note_text, encounter_id, location_id, week_id
        extractor: Extraction model to use
        use_model_tagger: Use MedGemma for tagging
        use_model_checklist: Use MedGemma for checklist
        run_hallucination_check: Run self-consistency hallucination check
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
            run_hallucination_check=run_hallucination_check,
        )
        results.append(result)

    return results


def process_voice_note(
    audio_path: str,
    encounter_id: str = "unknown",
    location_id: str = "unknown",
    week_id: int = 0,
    extractor: str = "medgemma",
    use_model_tagger: bool = True,
    use_model_checklist: bool = True,
    run_hallucination_check: bool = True,
) -> Dict[str, Any]:
    """Process a voice note: transcribe with MedASR → run full pipeline.

    Args:
        audio_path: Path to the audio file
        Other args: same as process_encounter

    Returns:
        Pipeline result dict with additional 'transcription' key
    """
    from .voice import transcribe_audio

    t0 = time.time()
    transcript = transcribe_audio(audio_path)
    transcription_time = round(time.time() - t0, 3)

    result = process_encounter(
        note_text=transcript,
        encounter_id=encounter_id,
        location_id=location_id,
        week_id=week_id,
        extractor=extractor,
        use_model_tagger=use_model_tagger,
        use_model_checklist=use_model_checklist,
        run_hallucination_check=run_hallucination_check,
    )

    # Add transcription metadata to trace
    result["transcription"] = {
        "source": "voice",
        "transcript": transcript,
        "duration_s": transcription_time,
    }
    # Prepend transcription step to agent trace
    result["agent_trace"].insert(0, {
        "agent": "transcribe",
        "name": "Voice Transcriber (MedASR)",
        "duration_s": transcription_time,
        "fallback_used": False,
        "output_summary": f"Transcribed {len(transcript)} characters from audio",
    })

    return result


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
        locations: Optional locations DataFrame with names
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


def _count_yes_symptoms(encounter: Dict[str, Any]) -> int:
    """Count 'yes' symptoms in an encounter."""
    count = 0
    for v in encounter.get("symptoms", {}).values():
        if isinstance(v, dict) and v.get("value") == "yes":
            count += 1
    for v in encounter.get("other_symptoms", {}).values():
        if isinstance(v, dict) and v.get("value") == "yes":
            count += 1
    return count
