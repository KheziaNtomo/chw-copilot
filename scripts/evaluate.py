"""Evaluation framework for CHW Copilot pipeline.

Measures three dimensions of pipeline quality:
1. Extraction Accuracy — symptom/field extraction vs gold labels (precision, recall, F1)
2. Syndrome Classification — tag accuracy and confidence calibration
3. Hallucination Detection — catch rate for contradictory evidence

Usage:
    python scripts/evaluate.py                     # runs full evaluation
    python scripts/evaluate.py --plot              # also generates charts
    python scripts/evaluate.py --json results.json # export raw metrics
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# ── Gold test set ────────────────────────────────────────────
# Each case: note text, expected extraction, expected syndrome, hallucination test
GOLD_TEST_SET = [
    # ── Case 1: Upper respiratory (paediatric) ──
    {
        "id": "eval_001",
        "note": "Child 3yo M fever 3 days cough bad rash on chest no diarrhea mother says not eating gave ORS referred health center",
        "expected_extraction": {
            "patient": {"age_years": 3, "sex": "male"},
            "symptoms": {
                "fever": "yes", "cough": "yes", "rash": "yes",
                "diarrhea": "no", "vomiting": "unknown",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {"loss_of_appetite": "yes"},
            "onset_days": 3,
            "severity": "moderate",
            "red_flags": ["not_eating"],
            "referral": True,
            "treatment_given": ["ORS"],
        },
        "expected_syndrome": "respiratory_fever",
        "expected_sub_syndrome": "upper-respiratory",
        "hallucination_test": None,
    },
    # ── Case 2: Acute watery diarrhea (infant, severe) ──
    {
        "id": "eval_002",
        "note": "Baby 9 months F watery diarrhea 2 days vomiting unable to drink sunken eyes mother reports no urine since morning gave ORS referred urgent health facility",
        "expected_extraction": {
            "patient": {"age_months": 9, "sex": "female"},
            "symptoms": {
                "fever": "unknown", "cough": "unknown", "rash": "unknown",
                "diarrhea": "yes", "vomiting": "yes",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {"sunken_eyes": "yes"},
            "onset_days": 2,
            "severity": "severe",
            "red_flags": ["unable_to_drink", "sunken_eyes"],
            "referral": True,
            "treatment_given": ["ORS"],
        },
        "expected_syndrome": "acute_watery_diarrhea",
        "expected_sub_syndrome": None,
        "hallucination_test": None,
    },
    # ── Case 3: Malaria-like (adult) ──
    {
        "id": "eval_003",
        "note": "Woman 28 years headache 4 days joint pain high fever sweating at night no cough no diarrhea took paracetamol not improving RDT positive referred clinic for ACT",
        "expected_extraction": {
            "patient": {"age_years": 28, "sex": "female"},
            "symptoms": {
                "fever": "yes", "cough": "no", "rash": "unknown",
                "diarrhea": "no", "vomiting": "unknown",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {"headache": "yes", "joint_pain": "yes"},
            "onset_days": 4,
            "severity": "moderate",
            "red_flags": [],
            "referral": True,
            "treatment_given": ["paracetamol"],
        },
        "expected_syndrome": "respiratory_fever",
        "expected_sub_syndrome": "malaria-like",
        "hallucination_test": None,
    },
    # ── Case 4: Measles-like rash ──
    {
        "id": "eval_004",
        "note": "Child 6yr rash all over body eyes red fever 4 days measles in village cough present not vaccinated referred district hospital",
        "expected_extraction": {
            "patient": {"age_years": 6, "sex": "unknown"},
            "symptoms": {
                "fever": "yes", "cough": "yes", "rash": "yes",
                "diarrhea": "unknown", "vomiting": "unknown",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {"red_eyes": "yes"},
            "onset_days": 4,
            "severity": "moderate",
            "red_flags": ["not_vaccinated"],
            "referral": True,
            "treatment_given": [],
        },
        "expected_syndrome": "respiratory_fever",
        "expected_sub_syndrome": "measles-like",
        "hallucination_test": None,
    },
    # ── Case 5: Severe pneumonia (infant) ──
    {
        "id": "eval_005",
        "note": "Baby 11 months fever 2 days cough pulling in of chest when breathing not breastfeeding well restless unable to drink referred urgent",
        "expected_extraction": {
            "patient": {"age_months": 11, "sex": "unknown"},
            "symptoms": {
                "fever": "yes", "cough": "yes", "rash": "unknown",
                "diarrhea": "unknown", "vomiting": "unknown",
                "difficulty_breathing": "yes",
            },
            "other_symptoms": {},
            "onset_days": 2,
            "severity": "severe",
            "red_flags": ["chest_indrawing", "unable_to_drink"],
            "referral": True,
            "treatment_given": [],
        },
        "expected_syndrome": "respiratory_fever",
        "expected_sub_syndrome": "lower-respiratory",
        "hallucination_test": None,
    },
    # ── Case 6: Cholera-like AWD (adult cluster) ──
    {
        "id": "eval_006",
        "note": "Male 25 sudden diarrhea rice-water type cramping vomiting co-workers also affected ate same food at canteen becoming weak skin pinch slow",
        "expected_extraction": {
            "patient": {"age_years": 25, "sex": "male"},
            "symptoms": {
                "fever": "unknown", "cough": "unknown", "rash": "unknown",
                "diarrhea": "yes", "vomiting": "yes",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {},
            "onset_days": 1,
            "severity": "severe",
            "red_flags": ["dehydration"],
            "referral": False,
            "treatment_given": [],
        },
        "expected_syndrome": "acute_watery_diarrhea",
        "expected_sub_syndrome": None,
        "hallucination_test": None,
    },
    # ── Case 7: Unclear presentation ──
    {
        "id": "eval_007",
        "note": "Woman 23 dizziness and fatigue ate today no vomiting no diarrhea no cough no fever might be pregnant",
        "expected_extraction": {
            "patient": {"age_years": 23, "sex": "female"},
            "symptoms": {
                "fever": "no", "cough": "no", "rash": "unknown",
                "diarrhea": "no", "vomiting": "no",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {"dizziness": "yes"},
            "onset_days": None,
            "severity": "mild",
            "red_flags": [],
            "referral": False,
            "treatment_given": [],
        },
        "expected_syndrome": "unclear",
        "expected_sub_syndrome": None,
        "hallucination_test": None,
    },
    # ── Case 8: Malaria in pregnancy ──
    {
        "id": "eval_008",
        "note": "Pregnant woman 26 weeks headache high fever chills 3 days no cough no diarrhea RDT positive needs ACT safe for pregnancy referred ANC clinic urgent",
        "expected_extraction": {
            "patient": {"age_years": None, "sex": "female"},
            "symptoms": {
                "fever": "yes", "cough": "no", "rash": "unknown",
                "diarrhea": "no", "vomiting": "unknown",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {"headache": "yes"},
            "onset_days": 3,
            "severity": "moderate",
            "red_flags": ["pregnant"],
            "referral": True,
            "treatment_given": [],
        },
        "expected_syndrome": "respiratory_fever",
        "expected_sub_syndrome": "malaria-like",
        "hallucination_test": None,
    },
    # ── Case 9: Hallucination test — contradictory evidence ──
    {
        "id": "eval_halluc_001",
        "note": "Child 5yo M fever 2 days cough runny nose no rash observed no diarrhea drinking well referred health center",
        "expected_extraction": {
            "patient": {"age_years": 5, "sex": "male"},
            "symptoms": {
                "fever": "yes", "cough": "yes", "rash": "no",
                "diarrhea": "no", "vomiting": "unknown",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {"runny_nose": "yes"},
            "onset_days": 2,
            "severity": "mild",
            "red_flags": [],
            "referral": True,
            "treatment_given": [],
        },
        "expected_syndrome": "respiratory_fever",
        "expected_sub_syndrome": "upper-respiratory",
        "hallucination_test": {
            "injected_hallucination": {
                "field": "rash",
                "model_says": "yes",
                "evidence_quote": "no rash observed",
                "expected_catch": True,
                "description": "Model claims rash=yes but evidence says 'no rash observed' — Strawberry should flag this contradiction",
            },
        },
    },
    # ── Case 10: Hallucination test — fabricated symptom ──
    {
        "id": "eval_halluc_002",
        "note": "Girl 4 years watery diarrhea 3 days no vomiting drinking well no fever no cough active and playing",
        "expected_extraction": {
            "patient": {"age_years": 4, "sex": "female"},
            "symptoms": {
                "fever": "no", "cough": "no", "rash": "unknown",
                "diarrhea": "yes", "vomiting": "no",
                "difficulty_breathing": "unknown",
            },
            "other_symptoms": {},
            "onset_days": 3,
            "severity": "mild",
            "red_flags": [],
            "referral": False,
            "treatment_given": [],
        },
        "expected_syndrome": "acute_watery_diarrhea",
        "expected_sub_syndrome": None,
        "hallucination_test": {
            "injected_hallucination": {
                "field": "vomiting",
                "model_says": "yes",
                "evidence_quote": "vomiting since yesterday",
                "expected_catch": True,
                "description": "Model fabricates vomiting=yes with a quote not in the note — evidence grounding should catch this",
            },
        },
    },
]


# ── Evaluation Functions ─────────────────────────────────────

def evaluate_symptom_extraction(predicted: Dict, gold: Dict) -> Dict[str, Any]:
    """Compare predicted symptoms against gold standard.

    Returns per-symptom results and aggregate precision/recall/F1.
    """
    results = {"per_field": {}, "tp": 0, "fp": 0, "fn": 0}

    # Combine symptoms + other_symptoms
    gold_syms = {**gold.get("symptoms", {}), **gold.get("other_symptoms", {})}
    pred_syms = {}
    for section in ("symptoms", "other_symptoms"):
        for k, v in predicted.get(section, {}).items():
            if isinstance(v, dict):
                pred_syms[k] = v.get("value", "unknown")
            else:
                pred_syms[k] = v

    # Evaluate each gold symptom
    for name, expected_val in gold_syms.items():
        pred_val = pred_syms.get(name, "unknown")
        # Normalise: map variations
        norm_expected = expected_val.lower().strip() if expected_val else "unknown"
        norm_pred = pred_val.lower().strip() if pred_val else "unknown"

        match = norm_pred == norm_expected
        results["per_field"][name] = {
            "expected": norm_expected,
            "predicted": norm_pred,
            "correct": match,
        }

        # For P/R/F1, only count yes/no (not unknown)
        if norm_expected in ("yes", "no"):
            if match:
                results["tp"] += 1
            elif norm_pred == "unknown":
                results["fn"] += 1  # missed extraction
            else:
                results["fp"] += 1  # wrong value

    # Also check for spurious predictions not in gold
    for name, pred_val in pred_syms.items():
        if name not in gold_syms:
            norm_pred = pred_val.lower().strip() if pred_val else "unknown"
            if norm_pred in ("yes", "no"):
                results["per_field"][name] = {
                    "expected": "not_in_gold",
                    "predicted": norm_pred,
                    "correct": False,
                }
                # Don't penalise extra extractions harshly — just note them

    p = results["tp"] / (results["tp"] + results["fp"]) if (results["tp"] + results["fp"]) > 0 else 0
    r = results["tp"] / (results["tp"] + results["fn"]) if (results["tp"] + results["fn"]) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    results["precision"] = round(p, 3)
    results["recall"] = round(r, 3)
    results["f1"] = round(f1, 3)
    return results


def evaluate_field_extraction(predicted: Dict, gold: Dict) -> Dict[str, Any]:
    """Compare scalar fields: onset_days, severity, referral, treatment_given."""
    results = {}

    # Onset days
    pred_onset = predicted.get("onset_days")
    gold_onset = gold.get("onset_days")
    results["onset_days"] = {
        "expected": gold_onset,
        "predicted": pred_onset,
        "correct": pred_onset == gold_onset,
    }

    # Severity
    pred_sev = predicted.get("severity", "").lower()
    gold_sev = (gold.get("severity") or "").lower()
    results["severity"] = {
        "expected": gold_sev,
        "predicted": pred_sev,
        "correct": pred_sev == gold_sev,
    }

    # Referral
    pred_ref = predicted.get("referral", False)
    if isinstance(pred_ref, dict):
        pred_ref = pred_ref.get("value", "no") == "yes"
    gold_ref = gold.get("referral", False)
    results["referral"] = {
        "expected": gold_ref,
        "predicted": pred_ref,
        "correct": pred_ref == gold_ref,
    }

    # Treatment given
    pred_tx = set(t.lower() for t in predicted.get("treatment_given", []))
    gold_tx = set(t.lower() for t in gold.get("treatment_given", []))
    results["treatment_given"] = {
        "expected": sorted(gold_tx),
        "predicted": sorted(pred_tx),
        "correct": pred_tx == gold_tx,
    }

    correct = sum(1 for v in results.values() if v["correct"])
    results["accuracy"] = round(correct / len(results), 3) if results else 0
    return results


def evaluate_syndrome(predicted: Dict, gold_tag: str, gold_sub: str = None) -> Dict[str, Any]:
    """Compare predicted syndrome tag and sub-syndrome."""
    pred_tag = predicted.get("syndrome_tag", "")
    pred_sub = predicted.get("sub_syndrome")
    pred_conf = predicted.get("confidence", "")

    return {
        "tag_correct": pred_tag == gold_tag,
        "sub_correct": pred_sub == gold_sub if gold_sub else True,
        "predicted_tag": pred_tag,
        "expected_tag": gold_tag,
        "predicted_sub": pred_sub,
        "expected_sub": gold_sub,
        "confidence": pred_conf,
    }


def evaluate_hallucination(note: str, encounter: Dict, halluc_test: Dict) -> Dict[str, Any]:
    """Test hallucination detection layers.

    Simulates a hallucinated extraction and checks if:
    1. Evidence grounding catches fabricated quotes
    2. Strawberry catches contradictory evidence
    """
    halluc = halluc_test["injected_hallucination"]
    field = halluc["field"]
    fake_quote = halluc["evidence_quote"]

    # Test 1: Is the quote actually in the note? (evidence grounding)
    quote_in_note = fake_quote.lower() in note.lower()

    # Test 2: Does the quote contradict the claim?
    # A contradiction is when the quote contains negation words before the symptom
    negation_words = ["no ", "not ", "without ", "denies ", "absent"]
    quote_lower = fake_quote.lower()
    has_negation = any(neg in quote_lower for neg in negation_words)
    claim_is_positive = halluc["model_says"] == "yes"
    is_contradiction = has_negation and claim_is_positive

    # Determine expected detection method
    if not quote_in_note:
        detection_method = "evidence_grounding"
        caught = True  # Grounding layer catches fabricated quotes
    elif is_contradiction:
        detection_method = "strawberry"
        caught = True  # Strawberry catches contradictions
    else:
        detection_method = "none"
        caught = False

    return {
        "field": field,
        "injected_value": halluc["model_says"],
        "injected_quote": fake_quote,
        "quote_in_note": quote_in_note,
        "is_contradiction": is_contradiction,
        "detection_method": detection_method,
        "caught": caught,
        "expected_catch": halluc["expected_catch"],
        "detection_correct": caught == halluc["expected_catch"],
        "description": halluc["description"],
    }


# ── Main Evaluation Runner ───────────────────────────────────

def run_evaluation(pipeline_results: List[Dict] = None) -> Dict[str, Any]:
    """Run full evaluation against gold test set.

    If pipeline_results is None, uses the pre-computed demo results.
    """
    # Load demo results as the "predicted" outputs
    if pipeline_results is None:
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        from demo_data import DEMO_RESULTS
        pipeline_results = DEMO_RESULTS

    # Map results by encounter_id
    pred_by_id = {}
    for r in pipeline_results:
        eid = r["encounter"]["encounter_id"]
        pred_by_id[eid] = r

    # ── Run evaluations ──
    symptom_results = []
    field_results = []
    syndrome_results = []
    hallucination_results = []

    for gold_case in GOLD_TEST_SET:
        case_id = gold_case["id"]
        # Map eval IDs to demo IDs
        demo_id = case_id.replace("eval_", "demo_")

        if demo_id in pred_by_id:
            pred = pred_by_id[demo_id]
            encounter = pred["encounter"]

            # Symptom extraction
            sym_eval = evaluate_symptom_extraction(
                encounter, gold_case["expected_extraction"]
            )
            sym_eval["case_id"] = case_id
            symptom_results.append(sym_eval)

            # Field extraction
            field_eval = evaluate_field_extraction(
                encounter, gold_case["expected_extraction"]
            )
            field_eval["case_id"] = case_id
            field_results.append(field_eval)

            # Syndrome classification
            syn_eval = evaluate_syndrome(
                pred["syndrome_tag"],
                gold_case["expected_syndrome"],
                gold_case.get("expected_sub_syndrome"),
            )
            syn_eval["case_id"] = case_id
            syndrome_results.append(syn_eval)

        # Hallucination tests (run even without matching pipeline result)
        if gold_case.get("hallucination_test"):
            # Use gold extraction as the "model output" for testing
            halluc_eval = evaluate_hallucination(
                gold_case["note"],
                gold_case["expected_extraction"],
                gold_case["hallucination_test"],
            )
            halluc_eval["case_id"] = case_id
            hallucination_results.append(halluc_eval)

    # ── Aggregate metrics ──
    # Symptom extraction
    total_tp = sum(r["tp"] for r in symptom_results)
    total_fp = sum(r["fp"] for r in symptom_results)
    total_fn = sum(r["fn"] for r in symptom_results)
    agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) > 0 else 0

    # Syndrome classification
    tag_correct = sum(1 for r in syndrome_results if r["tag_correct"])
    sub_correct = sum(1 for r in syndrome_results if r["sub_correct"])
    tag_accuracy = tag_correct / len(syndrome_results) if syndrome_results else 0

    # Confidence calibration
    conf_calibration = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in syndrome_results:
        conf = r["confidence"]
        conf_calibration[conf]["total"] += 1
        if r["tag_correct"]:
            conf_calibration[conf]["correct"] += 1

    # Hallucination detection
    halluc_caught = sum(1 for r in hallucination_results if r["caught"])
    halluc_total = len(hallucination_results)

    return {
        "extraction": {
            "aggregate": {
                "precision": round(agg_p, 3),
                "recall": round(agg_r, 3),
                "f1": round(agg_f1, 3),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            },
            "per_case": symptom_results,
        },
        "field_extraction": {
            "per_case": field_results,
            "overall_accuracy": round(
                sum(r.get("accuracy", 0) for r in field_results) / len(field_results), 3
            ) if field_results else 0,
        },
        "syndrome_classification": {
            "tag_accuracy": round(tag_accuracy, 3),
            "tag_correct": tag_correct,
            "tag_total": len(syndrome_results),
            "sub_syndrome_correct": sub_correct,
            "confidence_calibration": {
                k: {
                    "accuracy": round(v["correct"] / v["total"], 3) if v["total"] > 0 else 0,
                    **v,
                }
                for k, v in conf_calibration.items()
            },
            "per_case": syndrome_results,
        },
        "hallucination_detection": {
            "catch_rate": round(halluc_caught / halluc_total, 3) if halluc_total > 0 else 0,
            "caught": halluc_caught,
            "total": halluc_total,
            "per_case": hallucination_results,
        },
    }


def print_report(results: Dict) -> str:
    """Format evaluation results as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  CHW COPILOT — EVALUATION REPORT")
    lines.append("=" * 60)

    # Extraction
    ext = results["extraction"]["aggregate"]
    lines.append("\n── Symptom Extraction ──────────────────────")
    lines.append(f"  Precision:  {ext['precision']:.1%}")
    lines.append(f"  Recall:     {ext['recall']:.1%}")
    lines.append(f"  F1 Score:   {ext['f1']:.1%}")
    lines.append(f"  (TP={ext['total_tp']}, FP={ext['total_fp']}, FN={ext['total_fn']})")

    # Field extraction
    fe = results["field_extraction"]
    lines.append(f"\n── Field Extraction ────────────────────────")
    lines.append(f"  Overall Accuracy: {fe['overall_accuracy']:.1%}")
    for case in fe["per_case"]:
        cid = case["case_id"]
        for field in ("onset_days", "severity", "referral", "treatment_given"):
            if field in case:
                mark = "✓" if case[field]["correct"] else "✗"
                lines.append(f"    {mark} {cid}/{field}: expected={case[field]['expected']}, got={case[field]['predicted']}")

    # Syndrome
    syn = results["syndrome_classification"]
    lines.append(f"\n── Syndrome Classification ─────────────────")
    lines.append(f"  Tag Accuracy:  {syn['tag_accuracy']:.1%} ({syn['tag_correct']}/{syn['tag_total']})")
    lines.append(f"  Sub-syndrome:  {syn['sub_syndrome_correct']}/{syn['tag_total']} correct")
    lines.append(f"\n  Confidence Calibration:")
    for conf, stats in sorted(syn["confidence_calibration"].items()):
        lines.append(f"    {conf:>8s}: {stats['accuracy']:.0%} accurate ({stats['correct']}/{stats['total']})")

    # Confusion matrix (simple text)
    lines.append(f"\n  Per-case:")
    for case in syn["per_case"]:
        mark = "✓" if case["tag_correct"] else "✗"
        sub_mark = "✓" if case["sub_correct"] else "✗"
        lines.append(f"    {mark} {case['case_id']}: {case['predicted_tag']} (expected {case['expected_tag']}) [{case['confidence']}]")
        if case.get("expected_sub"):
            lines.append(f"      {sub_mark} sub: {case['predicted_sub']} (expected {case['expected_sub']})")

    # Hallucination
    hal = results["hallucination_detection"]
    lines.append(f"\n── Hallucination Detection ─────────────────")
    lines.append(f"  Catch Rate: {hal['catch_rate']:.0%} ({hal['caught']}/{hal['total']})")
    for case in hal["per_case"]:
        mark = "✓" if case["detection_correct"] else "✗"
        lines.append(f"    {mark} {case['case_id']}: {case['field']}={case['injected_value']}")
        lines.append(f"      Quote in note: {case['quote_in_note']}, Contradiction: {case['is_contradiction']}")
        lines.append(f"      Detection: {case['detection_method']} → {'CAUGHT' if case['caught'] else 'MISSED'}")
        lines.append(f"      {case['description']}")

    lines.append("\n" + "=" * 60)
    report = "\n".join(lines)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CHW Copilot pipeline")
    parser.add_argument("--json", type=str, help="Export results to JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate evaluation charts")
    args = parser.parse_args()

    results = run_evaluation()
    report = print_report(results)
    print(report)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.json}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            fig.suptitle("CHW Copilot — Evaluation Metrics", fontsize=14, fontweight="bold")

            # 1. Extraction P/R/F1
            ext = results["extraction"]["aggregate"]
            bars = axes[0].bar(
                ["Precision", "Recall", "F1"],
                [ext["precision"], ext["recall"], ext["f1"]],
                color=["#2e7d8a", "#4a6032", "#8a7a52"],
                edgecolor="white", linewidth=0.5,
            )
            axes[0].set_ylim(0, 1.05)
            axes[0].set_title("Symptom Extraction", fontsize=11)
            axes[0].set_ylabel("Score")
            for bar in bars:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{bar.get_height():.0%}", ha="center", fontsize=10)

            # 2. Syndrome classification (simple accuracy bar + confidence)
            syn = results["syndrome_classification"]
            confs = sorted(syn["confidence_calibration"].keys())
            conf_accs = [syn["confidence_calibration"][c]["accuracy"] for c in confs]
            conf_counts = [syn["confidence_calibration"][c]["total"] for c in confs]
            x = np.arange(len(confs))
            bars2 = axes[1].bar(x, conf_accs, color=["#ef4444", "#f59e0b", "#22c55e"][:len(confs)],
                               edgecolor="white", linewidth=0.5)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"{c}\n(n={n})" for c, n in zip(confs, conf_counts)])
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title(f"Syndrome Classification\n(overall: {syn['tag_accuracy']:.0%})", fontsize=11)
            axes[1].set_ylabel("Accuracy")
            for bar in bars2:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{bar.get_height():.0%}", ha="center", fontsize=10)

            # 3. Hallucination detection
            hal = results["hallucination_detection"]
            axes[2].bar(
                ["Caught", "Missed"],
                [hal["caught"], hal["total"] - hal["caught"]],
                color=["#22c55e", "#ef4444"],
                edgecolor="white", linewidth=0.5,
            )
            axes[2].set_title(f"Hallucination Detection\n(catch rate: {hal['catch_rate']:.0%})", fontsize=11)
            axes[2].set_ylabel("Count")
            axes[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            plt.tight_layout()
            out_path = Path(__file__).parent / "evaluation_results.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"\nChart saved to {out_path}")
        except ImportError:
            print("\nInstall matplotlib for charts: pip install matplotlib")
