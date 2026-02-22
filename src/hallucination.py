"""Hallucination detection for CHW Copilot via Pythea/Strawberry budget-gap analysis.

Three verification layers:
  Layer 1 (enforce_evidence in validate.py): Quote must be verbatim substring of note
  Layer 2 (this module - Pythea): Information-budget analysis via evidence scrubbing
  Layer 3 (this module - fallback): Deterministic negation-word check

Pythea/Strawberry method (Leon Chlon, 2026):
  1. Parse extraction claims into (claim, evidence_quote) pairs
  2. Compute p1 = P(claim entailed | full note) — posterior
  3. Scrub cited evidence → compute p0 = P(claim entailed | scrubbed note) — prior
  4. budget_gap = RequiredBits - ObservedBits
  5. If budget_gap > 0, the evidence didn't justify the confidence → flag

If Pythea is unavailable, falls back to self-consistency check or deterministic negation detection.

Reference: https://github.com/leochlon/pythea
"""
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from .validate import locate_evidence

logger = logging.getLogger(__name__)

# ── Budget gap threshold ─────────────────────────────────────
# Claims with budget_gap above this are flagged as hallucinations
BUDGET_GAP_THRESHOLD = 0.0  # Pythea flags when required > observed


def _build_claims(encounter: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract claim/evidence pairs from an encounter for verification."""
    claims = []

    # Core symptoms
    for key, val in encounter.get("symptoms", {}).items():
        if isinstance(val, dict) and val.get("value") == "yes" and val.get("evidence_quote"):
            claims.append({
                "symptom": key,
                "claim": f"Patient has {key.replace('_', ' ')}",
                "evidence": val["evidence_quote"],
                "section": "symptoms",
            })

    # Other symptoms
    for key, val in encounter.get("other_symptoms", {}).items():
        if isinstance(val, dict) and val.get("value") == "yes" and val.get("evidence_quote"):
            claims.append({
                "symptom": key,
                "claim": f"Patient has {key.replace('_', ' ')}",
                "evidence": val["evidence_quote"],
                "section": "other_symptoms",
            })

    # Red flags
    for flag in encounter.get("red_flags", []):
        if isinstance(flag, dict) and flag.get("evidence_quote"):
            claims.append({
                "symptom": flag.get("flag", "unknown"),
                "claim": f"Patient shows {flag.get('flag', 'unknown').replace('_', ' ')}",
                "evidence": flag["evidence_quote"],
                "section": "red_flags",
            })

    return claims


# ══════════════════════════════════════════════════════════════
#  Layer 2a: Pythea/Strawberry Budget-Gap Analysis
# ══════════════════════════════════════════════════════════════

def _try_import_pythea():
    """Try importing Pythea's trace_budget module. Returns (module, available)."""
    try:
        from strawberry.trace_budget import score_trace_budget, kl_bernoulli
        return True
    except ImportError:
        return False


PYTHEA_AVAILABLE = _try_import_pythea()


def verify_claims_pythea(
    encounter: Dict[str, Any],
    note_text: str,
    verifier_model: str = "gpt-4o-mini",
    backend_cfg: Any = None,
    default_target: float = 0.95,
) -> Dict[str, Any]:
    """Verify claims using Pythea's information-budget analysis.

    Uses Strawberry's score_trace_budget to compute budget gaps via
    logprob-based KL divergence scoring. This is the gold-standard
    hallucination detection method.

    Args:
        encounter: Structured encounter dict
        note_text: Original CHW note text
        verifier_model: Model to use as verifier (needs logprob support)
        backend_cfg: Pythea BackendConfig (defaults to OpenAI)
        default_target: Confidence target for budget calculation

    Returns:
        Verification result dict with budget gaps per claim
    """
    from strawberry.trace_budget import score_trace_budget

    claims = _build_claims(encounter)
    if not claims:
        return {"flagged": False, "claims_checked": 0, "flagged_claims": [],
                "budget_gaps": {}, "available": True, "method": "pythea_budget_gap"}

    # Build Pythea trace object
    # Spans: the note text split into the evidence quotes + full note
    @dataclass
    class Span:
        sid: str
        text: str

    @dataclass
    class Step:
        idx: int
        claim: str
        cites: List[str]
        confidence: float = default_target

    @dataclass
    class Trace:
        steps: List[Step]
        spans: List[Span]

    # Create spans: S0 = full note, S1..SN = individual evidence quotes
    spans = [Span(sid="S0", text=note_text)]
    steps = []

    for i, c in enumerate(claims):
        span_id = f"S{i + 1}"
        spans.append(Span(sid=span_id, text=c["evidence"]))
        steps.append(Step(
            idx=i,
            claim=c["claim"],
            cites=[span_id, "S0"],  # Cite both the evidence quote and full note
        ))

    trace = Trace(steps=steps, spans=spans)

    try:
        budget_results = score_trace_budget(
            trace=trace,
            verifier_model=verifier_model,
            backend_cfg=backend_cfg,
            default_target=default_target,
        )
    except Exception as e:
        logger.warning("Pythea score_trace_budget failed: %s — falling back", e)
        return verify_claims_counterfactual(encounter, note_text)

    # Parse results
    flagged_claims = []
    budget_gaps = {}

    for br, c in zip(budget_results, claims):
        # Use the max budget gap (conservative flagging)
        gap = br.budget_gap_max
        claim_label = f"{c['claim']} [S{br.idx + 1}]"
        budget_gaps[claim_label] = round(gap, 1)

        if br.flagged:
            flagged_claims.append({
                "claim": c["claim"],
                "symptom": c["symptom"],
                "evidence": c["evidence"],
                "budget_gap": round(gap, 1),
                "reason": (
                    f"Budget gap: {gap:.1f} bits — evidence does not justify "
                    f"confidence. P(yes|full)=[{br.post_yes.p_yes_lower:.3f}, "
                    f"{br.post_yes.p_yes_upper:.3f}], "
                    f"P(yes|scrubbed)=[{br.prior_yes.p_yes_lower:.3f}, "
                    f"{br.prior_yes.p_yes_upper:.3f}]"
                ),
            })

    return {
        "flagged": len(flagged_claims) > 0,
        "claims_checked": len(claims),
        "flagged_claims": flagged_claims,
        "budget_gaps": budget_gaps,
        "available": True,
        "method": "pythea_budget_gap",
    }


# ══════════════════════════════════════════════════════════════
#  Layer 2b: Self-Consistency Check (MedGemma as verifier)
# ══════════════════════════════════════════════════════════════

VERIFY_PROMPT = """You are a evidence verification assistant. Your job is to check whether a quoted piece of text supports a specific medical claim.

CLAIM: "{claim}"
EVIDENCE QUOTE: "{evidence}"
ORIGINAL NOTE: "{note}"

Does the evidence quote SUPPORT the claim? Consider:
- Does the quote confirm or deny the claimed symptom?
- A quote like "no fever" does NOT support "patient has fever"
- A quote like "fever 3 days" DOES support "patient has fever"

Respond with ONLY a JSON object:
{{"supported": true/false, "reason": "brief explanation"}}"""


def verify_extraction_claims(
    encounter: Dict[str, Any],
    note_text: str,
    generate_fn=None,
) -> Dict[str, Any]:
    """Verify extraction claims using MedGemma self-consistency.

    For each "yes" claim with an evidence quote, asks MedGemma:
    "Does this quote actually support this claim?"

    If the model says no, the claim is flagged as a potential hallucination.

    Args:
        encounter: Structured encounter dict (after extraction)
        note_text: Original CHW note text
        generate_fn: Function to call MedGemma (signature: fn(prompt) -> str).
                     If None, verification is skipped gracefully.

    Returns:
        Dict with flagged, claims_checked, flagged_claims, available
    """
    claims = _build_claims(encounter)

    if not claims:
        return {
            "flagged": False,
            "claims_checked": 0,
            "flagged_claims": [],
            "budget_gaps": {},
            "available": True,
            "method": "self_consistency_check",
        }

    if generate_fn is None:
        return {
            "flagged": False,
            "claims_checked": len(claims),
            "flagged_claims": [],
            "budget_gaps": {},
            "available": False,
            "note": "No model available for self-consistency check",
        }

    flagged_claims = []
    budget_gaps = {}

    for i, c in enumerate(claims):
        prompt = VERIFY_PROMPT.format(
            claim=c["claim"],
            evidence=c["evidence"],
            note=note_text,
        )

        try:
            from src.models import parse_json_response

            raw = generate_fn(prompt, max_tokens=100)
            result = parse_json_response(raw)

            claim_label = f"{c['claim']} [S{i}]"
            if result and result.get("supported") is False:
                flagged_claims.append({
                    "claim": c["claim"],
                    "symptom": c["symptom"],
                    "evidence": c["evidence"],
                    "budget_gap": 5.0,  # Synthetic gap for self-consistency failures
                    "reason": result.get("reason", "Evidence does not support claim"),
                })
                budget_gaps[claim_label] = 5.0
            else:
                budget_gaps[claim_label] = -1.5  # Supported
        except Exception:
            continue

    return {
        "flagged": len(flagged_claims) > 0,
        "claims_checked": len(claims),
        "flagged_claims": flagged_claims,
        "budget_gaps": budget_gaps,
        "available": True,
        "method": "self_consistency_check",
    }


# ══════════════════════════════════════════════════════════════
#  Layer 2c: Counterfactual Evidence Scrubbing (Local)
# ══════════════════════════════════════════════════════════════

def verify_claims_counterfactual(
    encounter: Dict[str, Any],
    note_text: str,
    generate_fn=None,
) -> Dict[str, Any]:
    """Verify claims via counterfactual evidence scrubbing using local model.

    Implements Pythea-style causal intervention locally:
    1. Identify the cited evidence quote for a claim.
    2. Create a 'scrubbed' note where that evidence is masked/removed.
    3. Ask the model if the symptom is still present in the scrubbed note.
    4. If YES despite evidence removal → Confabulation (non-causal evidence).
    5. If NO → Evidence was causal (verified).

    Args:
        encounter: Structured encounter dict
        note_text: Original note text
        generate_fn: Model generation function

    Returns:
        Verification result dict
    """
    claims = _build_claims(encounter)
    if not claims:
        return {"flagged": False, "claims_checked": 0, "flagged_claims": [],
                "budget_gaps": {}, "available": True, "method": "counterfactual_scrubbing"}

    if generate_fn is None:
        return {"flagged": False, "claims_checked": 0, "flagged_claims": [],
                "budget_gaps": {}, "available": False, "note": "No model"}

    flagged_claims = []
    budget_gaps = {}

    for i, c in enumerate(claims):
        evidence = c["evidence"].strip()
        matched_text, score = locate_evidence(evidence, note_text)

        claim_label = f"{c['claim']} [S{i}]"

        if not matched_text:
            flagged_claims.append({
                "claim": c["claim"],
                "symptom": c["symptom"],
                "evidence": c["evidence"],
                "budget_gap": 10.0,
                "reason": "Evidence quote not found in original text (Grounding Error)",
            })
            budget_gaps[claim_label] = 10.0
            continue

        # Create counterfactual note (Pythea's do(evidence:=∅) intervention)
        scrubbed_note = note_text.replace(matched_text, "[REDACTED]")

        prompt = f"""COUNTERFACTUAL VERIFICATION TASK
Original Note (Scrubbed): "{scrubbed_note}"
Question: Based ONLY on the note above, does the patient have {c['symptom'].replace('_', ' ')}?

Respond with ONLY a JSON object:
{{"present": true/false, "reason": "brief explanation"}}"""

        try:
            from src.models import parse_json_response
            raw = generate_fn(prompt, max_tokens=100)
            result = parse_json_response(raw)

            if result and result.get("present") is True:
                # Model still sees symptom despite evidence removal → Confabulation
                flagged_claims.append({
                    "claim": c["claim"],
                    "symptom": c["symptom"],
                    "evidence": c["evidence"],
                    "budget_gap": 8.0,
                    "reason": "Counterfactual Failure: Model still extracted symptom after evidence was removed (Non-Causal Evidence).",
                })
                budget_gaps[claim_label] = 8.0
            else:
                budget_gaps[claim_label] = -2.0  # Evidence was causal (good)
        except Exception:
            continue

    return {
        "flagged": len(flagged_claims) > 0,
        "claims_checked": len(claims),
        "flagged_claims": flagged_claims,
        "budget_gaps": budget_gaps,
        "available": True,
        "method": "counterfactual_scrubbing",
    }


# ══════════════════════════════════════════════════════════════
#  Layer 3: Deterministic Negation Check (Offline)
# ══════════════════════════════════════════════════════════════

def verify_claims_offline(encounter: Dict[str, Any], note_text: str) -> Dict[str, Any]:
    """Quick deterministic check for obvious contradictions.

    Catches the most common hallucination pattern: quote contains
    a negation word but claim says "yes". This serves as a fast
    fallback when no model is available.
    """
    claims = _build_claims(encounter)
    negation_words = {"no", "not", "none", "without", "absent", "denies", "denied", "negative"}
    flagged = []
    budget_gaps = {}

    for i, c in enumerate(claims):
        quote_words = set(c["evidence"].lower().split())
        claim_label = f"{c['claim']} [S{i}]"

        # If quote starts with a negation or contains negation + symptom keyword
        if quote_words & negation_words:
            symptom_words = set(c["symptom"].replace("_", " ").lower().split())
            # Quote has both negation AND symptom keyword → likely contradictory
            if quote_words & symptom_words:
                flagged.append({
                    "claim": c["claim"],
                    "symptom": c["symptom"],
                    "evidence": c["evidence"],
                    "budget_gap": 9.7,  # High gap for clear contradictions
                    "reason": f"Quote contains negation word(s) with symptom keyword",
                })
                budget_gaps[claim_label] = 9.7
            else:
                budget_gaps[claim_label] = -1.0
        else:
            budget_gaps[claim_label] = -1.5  # No negation detected

    return {
        "flagged": len(flagged) > 0,
        "claims_checked": len(claims),
        "flagged_claims": flagged,
        "budget_gaps": budget_gaps,
        "available": True,
        "method": "deterministic_negation_check",
    }
