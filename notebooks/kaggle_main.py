# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     main_language: python
#     notebook_metadata_filter: -all
# ---

# %% [markdown]
# # CHW Copilot — 6-Agent Agentic Pipeline on Kaggle
#
# **Competition:** Google AI Assistants for Data Tasks with Gemma — Agentic Workflow Prize
#
# ---
#
# ## Introduction
#
# **Community Health Workers (CHWs)** are the backbone of primary healthcare in
# sub-Saharan Africa. They conduct household visits, triage sick children, and
# refer urgent cases — all documented in **free-text field notes** written in
# shorthand on paper or basic phones. These notes are rich in clinical signal but
# essentially invisible to the formal health system: no structured data means no
# surveillance, no trend detection, and no quality feedback to the CHW.
#
# **CHW Copilot** solves this by running an **agentic pipeline** powered by
# **MedGemma** (`google/medgemma-4b-it`). Each field note is processed
# through 6 specialised agents that extract structured encounters, verify
# evidence, detect hallucinations, classify syndromes, generate follow-up
# checklists, and validate against JSON schemas. The result is a complete,
# evidence-grounded clinical record — plus actionable recommendations aligned
# with WHO **Integrated Community Case Management (ICCM)** guidelines.
#
# ### Pipeline Overview
#
# | Agent | Method | Purpose |
# |---|---|---|
# | 1. Encounter Extractor | MedGemma (zero-shot JSON) | Structured symptom/patient extraction |
# | 2. Evidence Grounder | Deterministic fuzzy match | Verify every claim links to source text |
# | 3. Hallucination Detector | Pythea/Strawberry budget-gap | Catch contradicted evidence |
# | 4. Syndrome Tagger | MedGemma / keyword fallback | Syndromic classification (ILI, AWD, etc.) |
# | 5. Checklist Generator | MedGemma / rule-based | Recommend follow-up questions for CHW |
# | 6. Schema Validator | JSON Schema Draft 7 | Ensure output conformity |
#
# ### Why MedGemma?
#
# General-purpose LLMs struggle with medical abbreviations, CHW shorthand, and
# clinical reasoning. MedGemma is pre-trained on biomedical text and fine-tuned
# for clinical tasks — it understands that "RDT+" means a positive malaria rapid
# diagnostic test and that "sunken eyes" is a WHO danger sign for dehydration.
# This domain knowledge is critical for accurate extraction from noisy field notes.
#
# **Runtime:** Kaggle T4 GPU (16GB VRAM)
#
# ---

# %% [markdown]
# ## 0. Setup & Dependencies

# %%
# Install dependencies
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40", "accelerate>=0.27", "jsonschema>=4.17",
    "pandas>=2.0", "torch"])
print("Dependencies installed ✅")

# %%
import os, json, time, warnings
from pathlib import Path
import pandas as pd
import torch
warnings.filterwarnings("ignore")

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    ROOT = Path("/kaggle/input/chw-copilot")
    OUT_DIR = Path("/kaggle/working")
else:
    ROOT = Path(".")
    OUT_DIR = Path(".")

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Root (input): {ROOT}")
print(f"Output dir: {OUT_DIR}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 1. Load Prompts and Schemas

# %%
# Note: Prompts are now loaded internally by src modules
print("Prompts managed by src.pipeline ✅")

# %%
# Load schemas for validation
import jsonschema

# Schemas are loaded internally by src.validate
print("Schemas managed by src.validate ✅")

# %% [markdown]
# ## 2. Load MedGemma
#
# **MedGemma 4B-IT** (Google HAI-DEF) — handles all LLM-based agents:
# - **Agent 1:** Structured extraction from typed CHW notes
# - **Agent 4:** Syndrome classification
# - **Agent 5:** Checklist generation
# - SITREP narrative generation
#
# **Adaptation method:** Prompt engineering (zero-shot + JSON schema)
#
# > **Note:** MedGemma is a gated model. You need to:
# > 1. Accept the license at https://huggingface.co/google/medgemma-4b-it
# > 2. Add your HuggingFace token as a Kaggle Secret named `HF_TOKEN`

# %%
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- HuggingFace authentication (for gated models like MedGemma) ---
HF_TOKEN = None
if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    try:
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle Secrets ✅")
    except Exception:
        print("⚠️  HF_TOKEN not found in Kaggle Secrets — MedGemma may fail to load")
else:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        print("HF_TOKEN loaded from environment ✅")
    else:
        print("⚠️  No HF_TOKEN set — MedGemma may fail to load")

# %%
# Load MedGemma
print("Loading MedGemma (4B-IT)...")
t0 = time.time()

MEDGEMMA_ID = "google/medgemma-4b-it"
mg_processor = AutoProcessor.from_pretrained(
    MEDGEMMA_ID, trust_remote_code=True, token=HF_TOKEN
)
mg_model = AutoModelForImageTextToText.from_pretrained(
    MEDGEMMA_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=HF_TOKEN,
)
mg_model.eval()
device = next(mg_model.parameters()).device
print(f"MedGemma loaded on {device} in {time.time()-t0:.1f}s ✅")

# %% [markdown]
# ## 3. Helper Functions

# %%
# %%
# Ensure src is in path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import pipeline, config

# Phase 1: Full Agentic Run (Showcase)
# Run 3 notes with ALL agents enabled (Extraction + Evidence + Hallucination + Tag + Checklist + Validate)
print("=== Phase 1: Full Agentic Pipeline Showcase (3 notes) ===")
print("Running with: Hallucination Check, Model Tagger, Model Checklist")

demo_subset = demo_notes[:3]
results_full = pipeline.process_batch(
    demo_subset,
    extractor="medgemma",
    use_model_tagger=True,
    use_model_checklist=True,
    run_hallucination_check=True,
    progress_callback=lambda i, n: print(f"  Full Agentic Note {i+1}/{n}...")
)

# Print trace for first note
print("\nAgent Trace for Note 1:")
for step in results_full[0]["agent_trace"]:
    print(f"  [{step['agent']}] {step['name']}: {step['duration_s']}s — {step['output_summary']}")

# Phase 2: Fast Batch Run (Performance)
# Run remaining notes with LIGHT pipeline (Extraction + Deterministic Fallbacks)
# This reduces LLM calls from ~5 per note to 1 per note
remaining_notes = demo_notes[3:]
print(f"\n=== Phase 2: High-Throughput Batch ({len(remaining_notes)} notes) ===")
print("Running with: Deterministic Tagger, Deterministic Checklist, No Hallucination Check")

results_fast = pipeline.process_batch(
    remaining_notes,
    extractor="medgemma",
    use_model_tagger=False,     # Optimization: Use deterministic fallback
    use_model_checklist=False,  # Optimization: Use deterministic fallback
    run_hallucination_check=False, # Optimization: Skip expensive verification
    progress_callback=lambda i, n: print(f"  Fast Note {i+1}/{n}...", end="\r")
)

# Combine results
results = results_full + results_fast
print(f"\n✅ Processed {len(results)} notes total")
avg_time = sum(r["processing_time_s"] for r in results) / len(results)
print(f"Average time per note: {avg_time:.1f}s")

# %% [markdown]
# ## 5. Recommended CHW Follow-up Questions
#
# Each encounter produces a **checklist** of missing information — questions the
# CHW should ask on their next visit. These are generated by Agent 5 (MedGemma
# for Phase 1, deterministic rules for Phase 2) and prioritised by clinical
# urgency.

# %%
print("=== Recommended CHW Follow-up Questions ===")
print()
for r in results:
    enc_id = r["encounter"]["encounter_id"]
    checklist = r.get("checklist", {})
    questions = checklist.get("questions", [])
    if not questions:
        print(f"  [{enc_id}] No follow-up questions needed (encounter is complete)")
        continue
    print(f"  [{enc_id}] {len(questions)} follow-up question(s):")
    for q in questions:
        priority = q.get("priority", "medium").upper()
        icon = {"HIGH": "!!", "MEDIUM": " >", "LOW": "  "}.get(priority, "  ")
        print(f"    {icon} [{priority}] {q['question']}")
        print(f"         Field: {q['field']}")
    print()

# %% [markdown]
# ## 6. Evaluation
#
# ### Syndrome Tagger: Hybrid Strategy
#
# The pipeline uses a **hybrid approach** to syndrome classification:
#
# - **Phase 1 (showcase):** MedGemma classifies syndromes using clinical reasoning.
#   It understands context — e.g., "fever with no cough" is NOT respiratory, even
#   though "cough" appears in the text. This is the primary, higher-accuracy method.
#
# - **Phase 2 (batch):** A fast **keyword-based fallback** handles high-throughput
#   processing. This is a deliberate optimisation tradeoff: keyword matching reduces
#   LLM calls from ~5 to 1 per note but can misclassify context-dependent cases
#   (e.g., negated symptoms like "no cough" still match the keyword "cough").
#
# The keyword tagger should be understood as a **speed-optimised fallback**, not
# the primary classification method. Real deployments would use MedGemma for all
# notes where GPU time permits.

# %%
from collections import Counter

# Compare syndrome tags to gold labels
correct = 0
total = 0
confusion = {}

for r, gold in zip(results, demo_notes):
    predicted = r["syndrome_tag"]["syndrome_tag"]
    actual = gold.get("gold_syndrome_tag", "unclear")
    total += 1
    if predicted == actual:
        correct += 1
    key = (actual, predicted)
    confusion[key] = confusion.get(key, 0) + 1

accuracy = correct / total if total > 0 else 0
print(f"=== Syndrome Tag Accuracy ===")
print(f"Correct: {correct}/{total} = {accuracy:.1%}")
print()

# Confusion matrix
print("Confusion matrix (actual → predicted):")
actuals = sorted(set(k[0] for k in confusion))
preds = sorted(set(k[1] for k in confusion))
header = f"{'':>25}" + "".join(f"{p:>20}" for p in preds)
print(header)
for a in actuals:
    row = f"{a:>25}" + "".join(f"{confusion.get((a,p),0):>20}" for p in preds)
    print(row)

# %%
# Evidence quality
total_claims = 0
grounded_claims = 0
hallucinated_claims = 0
total_downgrades = 0

for r in results:
    enc = r["encounter"]
    note_lower = enc.get("note_text", "").lower()

    # Check symptoms
    for k, v in enc.get("symptoms", {}).items():
        if v.get("value") == "yes":
            total_claims += 1
            q = v.get("evidence_quote", "")
            if q and q.lower() in note_lower:
                grounded_claims += 1
            else:
                hallucinated_claims += 1

    # Check other_symptoms
    for k, v in enc.get("other_symptoms", {}).items():
        if v.get("value") == "yes":
            total_claims += 1
            q = v.get("evidence_quote", "")
            if q and q.lower() in note_lower:
                grounded_claims += 1
            else:
                hallucinated_claims += 1

    total_downgrades += len(r.get("evidence_downgrades", []))

print(f"=== Evidence Quality ===")
print(f"Total 'yes' claims: {total_claims}")
print(f"Grounded (quote in note): {grounded_claims}")
print(f"Hallucinated/ungrounded: {hallucinated_claims}")
if total_claims > 0:
    print(f"Grounding rate: {grounded_claims/total_claims:.1%}")
    print(f"Hallucination rate: {hallucinated_claims/total_claims:.1%}")
print(f"Evidence downgrades (post-enforcement): {total_downgrades}")

# %%
# Per-symptom extraction stats
print(f"=== Per-Symptom Extraction ===")
symptom_stats = {}
for r in results:
    for k, v in r["encounter"]["symptoms"].items():
        if k not in symptom_stats:
            symptom_stats[k] = {"yes": 0, "no": 0, "unknown": 0}
        symptom_stats[k][v["value"]] += 1

for k in sorted(symptom_stats):
    s = symptom_stats[k]
    print(f"  {k:25s}: yes={s['yes']:3d}  no={s['no']:3d}  unknown={s['unknown']:3d}")

# %% [markdown]
# ## 8. Surveillance — Anomaly Detection & SITREP

# %%
# Aggregate to weekly counts
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
weekly_counts = df.groupby(["week_id","location_id","syndrome_tag"]).size().reset_index(name="count")
print("Weekly counts:")
print(weekly_counts)

# %%
# Also load the full sim_events for surveillance demo
sim_path = ROOT / "data_synth" / "sim_events.csv"
if sim_path.exists():
    sim_df = pd.read_csv(sim_path)
    print(f"Loaded {len(sim_df)} simulation events for surveillance demo")

    # Anomaly detection
    sys.path.insert(0, str(ROOT))
    from src.detect import detect_anomalies

    anomalies = detect_anomalies(sim_df)
    print(f"\nAnomalies detected: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies.to_string(index=False))
else:
    print("No sim_events.csv found — skip surveillance demo")

# %%
# Generate SITREP for the outbreak week using MedGemma
if sim_path.exists() and not anomalies.empty:
    outbreak_week = anomalies["week_id"].max()
    week_anomalies = anomalies[anomalies["week_id"] == outbreak_week]

    print(f"Generating SITREP for week {outbreak_week}...")
    t0 = time.time()
    
    from src.sitrep import generate_sitrep_medgemma
    sitrep = generate_sitrep_medgemma(week_anomalies, weekly_counts, outbreak_week)
    print(f"SITREP generated in {time.time()-t0:.1f}s")

    if sitrep:
        print(json.dumps(sitrep, indent=2))

# %% [markdown]
# ## 9. Save Results

# %%
# Save processed results
# Compute hallucination stats
hallucination_stats = {"total_checked": 0, "total_flagged": 0, "methods": set()}
for r in results:
    hc = r.get("hallucination_check") or {}
    hallucination_stats["total_checked"] += hc.get("claims_checked", 0)
    hallucination_stats["total_flagged"] += len(hc.get("flagged_claims", []))
    if hc.get("method"):
        hallucination_stats["methods"].add(hc["method"])

hallucination_rate = (
    hallucination_stats["total_flagged"] / max(hallucination_stats["total_checked"], 1)
)

output = {
    "model": MEDGEMMA_ID,
    "n_notes_processed": len(results),
    "avg_processing_time_s": round(avg_time, 2),
    "syndrome_accuracy": round(accuracy, 3),
    "evidence_grounding_rate": round(grounded_claims / max(total_claims, 1), 3),
    "hallucination_rate": round(hallucination_rate, 3),
    "encounters": [r["encounter"] for r in results],
    "syndrome_tags": [r["syndrome_tag"] for r in results],
    "checklists": [r["checklist"] for r in results],
}

out_path = OUT_DIR / "pipeline_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)
print(f"Results saved to {out_path} ✅")

# %%
# Final metrics summary table
print("\n" + "=" * 60)
print("📊 CHW COPILOT — PERFORMANCE METRICS SUMMARY")
print("=" * 60)
print(f"{'Metric':<35} {'Value':>15} {'Target':>10}")
print("-" * 60)
print(f"{'Notes processed':<35} {len(results):>15} {'':>10}")
print(f"{'Avg processing time/note':<35} {avg_time:>14.1f}s {'':>10}")
print(f"{'Syndrome tag accuracy':<35} {accuracy:>14.1%} {'≥80%':>10}")
print(f"{'Evidence grounding rate':<35} {grounded_claims/max(total_claims,1):>14.1%} {'≥98%':>10}")
print(f"{'Hallucination rate':<35} {hallucination_rate:>14.1%} {'≤2%':>10}")
print(f"{'Evidence downgrades':<35} {total_downgrades:>15} {'':>10}")
schema_pass_rate = sum(1 for r in results if r["validation"]["schema_valid"]) / len(results)
print(f"{'Schema validation pass rate':<35} {schema_pass_rate:>14.1%} {'100%':>10}")
print("-" * 60)

# Per-syndrome F1
from collections import Counter
syndrome_tp = Counter()
syndrome_fp = Counter()
syndrome_fn = Counter()

for r, gold in zip(results, demo_notes):
    predicted = r["syndrome_tag"]["syndrome_tag"]
    actual = gold.get("gold_syndrome_tag", "unclear")
    if predicted == actual:
        syndrome_tp[actual] += 1
    else:
        syndrome_fp[predicted] += 1
        syndrome_fn[actual] += 1

print(f"\n{'Syndrome':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)
for tag in sorted(set(list(syndrome_tp.keys()) + list(syndrome_fp.keys()) + list(syndrome_fn.keys()))):
    tp = syndrome_tp[tag]
    fp = syndrome_fp[tag]
    fn = syndrome_fn[tag]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"{tag:<30} {precision:>9.1%} {recall:>9.1%} {f1:>9.1%}")

# %% [markdown]
# ## 10. Limitations
#
# This prototype has several important limitations that should be addressed
# before real-world deployment:
#
# - **Evaluation dataset is synthetic.** The demo notes are hand-crafted to
#   showcase pipeline capabilities. Real CHW notes are noisier, more
#   abbreviated, and may contain local dialects or code-switching.
#
# - **English-only.** CHW notes in sub-Saharan Africa are often written in
#   French, Swahili, Portuguese, or local languages. MedGemma's multilingual
#   medical performance has not been validated here.
#
# - **Keyword tagger limitations.** The fast keyword fallback (Phase 2) cannot
#   handle negation ("no cough" matches "cough"), context-dependent symptoms,
#   or rare syndromes outside the predefined keyword lists. MedGemma-based
#   tagging is more accurate but requires GPU time for each note.
#
# - **No longitudinal validation.** The surveillance anomaly detection has not
#   been validated against confirmed outbreak datasets (e.g., DHIS2 records).
#
# - **Privacy not formally audited.** While the pipeline runs fully offline
#   with no external API calls and strips PII from prompts, it has not
#   undergone a formal privacy impact assessment or IRB review.
#
# - **Small model (4B parameters).** MedGemma 4B is optimised for edge
#   deployment but may under-perform on complex clinical reasoning compared
#   to larger models.

# %% [markdown]
# ## 11. Future Work
#
# CHW Copilot is designed as a **generalizable syndromic surveillance
# framework** — not a single-disease tool. Key directions for expansion:
#
# - **Additional syndromes.** Expand beyond respiratory fever and AWD to cover
#   measles/rubella (fever + rash + cough), meningitis (fever + stiff neck +
#   headache), dengue/arboviral (fever + rash + joint pain), and malnutrition
#   (MUAC-based screening). Each requires new keyword rules and MedGemma
#   prompt templates.
#
# - **Multilingual support.** Add French, Swahili, and Portuguese prompt
#   templates. MedGemma's multilingual capabilities can be leveraged with
#   language-specific few-shot examples.
#
# - **DHIS2 integration.** Push structured encounter data and syndrome counts
#   directly into national HMIS (Health Management Information Systems) via
#   the DHIS2 API, enabling real-time surveillance dashboards at district and
#   national levels.
#
# - **LoRA fine-tuning.** Fine-tune MedGemma on real (de-identified) CHW
#   notes to improve extraction accuracy for region-specific terminology and
#   abbreviations. The `lora_finetune.py` script provides a starting point.
#
# - **Edge deployment.** Package the pipeline for smartphone-based offline
#   inference using MedGemma's 4B parameter size, enabling CHWs to get
#   real-time feedback without connectivity.
#
# - **Feedback loop.** Allow CHWs to correct extracted data, creating a
#   human-in-the-loop training signal for continuous model improvement.

# %%
print("\n" + "=" * 60)
print("🎉 CHW Copilot — 6-Agent Agentic Pipeline Complete!")
print("=" * 60)
print(f"\nModel: {MEDGEMMA_ID}")
print(f"Hallucination detection: Pythea/Strawberry (counterfactual evidence scrubbing)")
print(f"Adaptation: Prompt engineering + agentic orchestration")
print()
print("Agent Pipeline:")
print("  1. Encounter Extractor  — MedGemma (zero-shot JSON extraction)")
print("  2. Evidence Grounder    — Deterministic (fuzzy substring match)")
print("  3. Hallucination Detect — Pythea/Strawberry (budget-gap analysis)")
print("  4. Syndrome Tagger      — MedGemma / keyword fallback (hybrid)")
print("  5. Checklist Generator  — MedGemma / rule-based (follow-up Qs)")
print("  6. Schema Validator     — JSON Schema (Draft 7) validation")
print()
print("Safety: Evidence grounding + Pythea hallucination detection")
print("Privacy: Offline-first, no PII in prompts, no external API calls")
