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
# This notebook runs the full CHW Copilot **agentic pipeline** on Kaggle GPU:
# 1. Load **MedGemma 1.5** (`google/medgemma-1.5-4b-it`) for extraction, tagging, checklist, SITREP
# 2. Run 6-agent pipeline: Extract → Ground → Verify → Tag → Checklist → Validate
# 3. Evaluate extraction quality + evidence grounding
# 4. Run surveillance pipeline with anomaly detection
#
# **Agents:** Encounter Extractor (LLM) → Evidence Grounder (deterministic) →
# Hallucination Detector (Strawberry) → Syndrome Tagger (LLM) →
# Checklist Generator (LLM) → Schema Validator (deterministic)
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
# ## 2. Load MedGemma 1.5
#
# **MedGemma 1.5 4b-it** (Google HAI-DEF) — handles all LLM-based agents:
# - **Agent 1:** Structured extraction from typed CHW notes
# - **Agent 4:** Syndrome classification
# - **Agent 5:** Checklist generation
# - SITREP narrative generation
#
# **Adaptation method:** Prompt engineering (zero-shot + JSON schema)
#
# > **Note:** MedGemma is a gated model. You need to:
# > 1. Accept the license at https://huggingface.co/google/medgemma-1.5-4b-it
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
# Load MedGemma 1.5
print("Loading MedGemma 1.5 (4b-it)...")
t0 = time.time()

MEDGEMMA_ID = "google/medgemma-1.5-4b-it"
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
print(f"MedGemma 1.5 loaded on {device} in {time.time()-t0:.1f}s ✅")

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
# ## 7. Evaluation

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
    week_counts = sim_df[sim_df["week_id"] == outbreak_week]

    anomaly_csv = week_anomalies.to_csv(index=False)
    counts_csv = week_counts.to_csv(index=False)

    sitrep_p = sitrep_prompt.replace("{anomalies_csv}", anomaly_csv)
    sitrep_p = sitrep_p.replace("{weekly_counts_csv}", counts_csv)
    sitrep_p = sitrep_p.replace("{report_week}", str(outbreak_week))

    print(f"Generating SITREP for week {outbreak_week}...")
    t0 = time.time()
    
    # Use src.sitrep 
    from src.sitrep import generate_sitrep_medgemma
    sitrep = generate_sitrep_medgemma(week_anomalies, weekly_counts, outbreak_week)
    print(f"SITREP generated in {time.time()-t0:.1f}s")

    if sitrep:
        print(json.dumps(sitrep, indent=2))

# %% [markdown]
# ## 9. Save Results

# %%
# Save processed results
output = {
    "model": MEDGEMMA_ID,
    "n_notes_processed": len(results),
    "avg_processing_time_s": round(avg_time, 2),
    "syndrome_accuracy": round(accuracy, 3),
    "evidence_grounding_rate": round(grounded_claims / max(total_claims, 1), 3),
    "encounters": [r["encounter"] for r in results],
    "syndrome_tags": [r["syndrome_tag"] for r in results],
    "checklists": [r["checklist"] for r in results],
}

out_path = OUT_DIR / "pipeline_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)
print(f"Results saved to {out_path} ✅")

# %%
print("=" * 60)
print("🎉 CHW Copilot — 6-Agent Agentic Pipeline Complete!")
print("=" * 60)
print(f"Notes processed:        {len(results)}")
print(f"Syndrome accuracy:      {accuracy:.1%}")
print(f"Evidence grounding:     {grounded_claims}/{total_claims} ({grounded_claims/max(total_claims,1):.1%})")
print(f"Avg time per note:      {avg_time:.1f}s")
print(f"Model: {MEDGEMMA_ID}")
print()
print("Agent Pipeline Summary:")
print("  1. Encounter Extractor  — MedGemma 1.5 (zero-shot JSON extraction)")
print("  2. Evidence Grounder    — Deterministic substring check")
print("  3. Hallucination Detect — Strawberry/Pythea budget_gap analysis")
print("  4. Syndrome Tagger      — MedGemma 1.5 (classification)")
print("  5. Checklist Generator   — MedGemma 1.5 (follow-up questions)")
print("  6. Schema Validator      — JSON Schema validation")
print()
print("Adaptation: Prompt engineering — zero-shot + JSON schema")
print("Safety: Evidence grounding + Strawberry hallucination detection")
print("Privacy: Offline-first, no PII in prompts, no external API calls")
