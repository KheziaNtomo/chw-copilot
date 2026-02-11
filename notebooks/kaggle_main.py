# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     main_language: python
#     notebook_metadata_filter: -all
# ---

# %% [markdown]
# # CHW Copilot — End-to-End Pipeline on Kaggle
#
# **Competition:** Google AI Assistants for Data Tasks with Gemma
#
# This notebook runs the full CHW Copilot pipeline on Kaggle GPU using **MedGemma only**:
# 1. Load MedGemma-4b-it for extraction, syndrome tagging, checklist, and SITREP
# 2. Extract structured encounters from typed CHW notes
# 3. Evaluate extraction quality
# 4. Run surveillance pipeline
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
# Load prompt templates
def load_prompt(name):
    path = ROOT / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")

extraction_prompt = load_prompt("specialist_extraction")
syndrome_prompt = load_prompt("syndrome_tagger")
checklist_prompt = load_prompt("checklist_agent")
sitrep_prompt = load_prompt("monitoring_sitrep")

print(f"Extraction prompt: {len(extraction_prompt)} chars")
print(f"Syndrome prompt: {len(syndrome_prompt)} chars")
print(f"Checklist prompt: {len(checklist_prompt)} chars")
print(f"SITREP prompt: {len(sitrep_prompt)} chars")
print("All prompts loaded ✅")

# %%
# Load schemas for validation
import jsonschema

def load_schema(name):
    path = ROOT / "schemas" / f"{name}.schema.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

encounter_schema = load_schema("encounter")
syndrome_schema = load_schema("syndrome")
checklist_schema = load_schema("checklist")
sitrep_schema = load_schema("sitrep")
print("All schemas loaded ✅")

# %% [markdown]
# ## 2. Load MedGemma
#
# **MedGemma-4b-it** (Google) — handles everything:
# - Structured extraction from typed CHW notes
# - Syndrome tagging
# - Checklist generation
# - SITREP generation
#
# > **Note:** MedGemma is a gated model. You need to:
# > 1. Accept the license at https://huggingface.co/google/medgemma-4b-it
# > 2. Add your HuggingFace token as a Kaggle Secret named `HF_TOKEN`

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

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
print("Loading MedGemma-4b-it...")
t0 = time.time()

MEDGEMMA_ID = "google/medgemma-4b-it"
mg_tokenizer = AutoTokenizer.from_pretrained(
    MEDGEMMA_ID, trust_remote_code=True, token=HF_TOKEN
)
mg_model = AutoModelForCausalLM.from_pretrained(
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
import re

def parse_json_response(text):
    """Extract JSON from model response, handling code fences and extra text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def run_medgemma(prompt, max_new_tokens=512):
    """Run MedGemma with chat template."""
    messages = [{"role": "user", "content": prompt}]
    input_text = mg_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = mg_tokenizer(input_text, return_tensors="pt").to(mg_model.device)
    with torch.no_grad():
        outputs = mg_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=mg_tokenizer.eos_token_id,
        )
    return mg_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def enforce_evidence(encounter, note_text):
    """Downgrade 'yes' claims without valid evidence quotes."""
    note = (note_text or "").lower()
    downgrades = []
    for k, v in list(encounter.get("symptoms", {}).items()):
        if v.get("value") == "yes":
            q = v.get("evidence_quote")
            if not q or q.lower() not in note:
                encounter["symptoms"][k] = {"value": "unknown", "evidence_quote": None}
                downgrades.append(f"symptoms.{k}")
    for k, v in list(encounter.get("other_symptoms", {}).items()):
        if v.get("value") == "yes":
            q = v.get("evidence_quote")
            if not q or q.lower() not in note:
                encounter["other_symptoms"][k] = {"value": "unknown", "evidence_quote": None}
                downgrades.append(f"other_symptoms.{k}")
    valid_flags = []
    for flag in encounter.get("red_flags", []):
        q = flag.get("evidence_quote", "")
        if q and q.lower() in note:
            valid_flags.append(flag)
        else:
            downgrades.append(f"red_flag:{flag.get('flag','?')}")
    encounter["red_flags"] = valid_flags
    return encounter, downgrades

# %% [markdown]
# ## 4. Quick Smoke Test (1 Note)

# %%
test_note = "Child 3yo male fever 3 days cough bad difficulty breathing rash on chest no diarrhea mother says not eating gave ORS referred health center"

print("=== MedGemma Extraction ===")
t0 = time.time()
extract_prompt = extraction_prompt.replace("{note_text}", test_note)
raw_extract = run_medgemma(extract_prompt, max_new_tokens=1024)
print(f"MedGemma extraction time: {time.time()-t0:.1f}s")
parsed_extract = parse_json_response(raw_extract)
if parsed_extract:
    print(json.dumps(parsed_extract, indent=2)[:800])
else:
    print("⚠️ Failed to parse extraction output:")
    print(raw_extract[:500])

# %%
print("=== MedGemma Syndrome Tagging ===")
tag_prompt = syndrome_prompt.replace("{encounter_json}", json.dumps(parsed_extract or {}, indent=2))
tag_prompt = tag_prompt.replace("{note_text}", test_note)
t0 = time.time()
raw_tag = run_medgemma(tag_prompt)
print(f"MedGemma time: {time.time()-t0:.1f}s")
parsed_tag = parse_json_response(raw_tag)
if parsed_tag:
    print(json.dumps(parsed_tag, indent=2)[:500])
else:
    print("⚠️ Failed to parse:")
    print(raw_tag[:500])

# %%
print("=== MedGemma Checklist ===")
cl_prompt = checklist_prompt.replace("{encounter_json}", json.dumps(parsed_extract or {}, indent=2))
cl_prompt = cl_prompt.replace("{note_text}", test_note)
t0 = time.time()
raw_cl = run_medgemma(cl_prompt)
print(f"MedGemma time: {time.time()-t0:.1f}s")
parsed_cl = parse_json_response(raw_cl)
if parsed_cl:
    print(json.dumps(parsed_cl, indent=2)[:500])
else:
    print("⚠️ Failed to parse:")
    print(raw_cl[:500])

# %% [markdown]
# ## 5. Full Pipeline Function

# %%
def process_note(note_text, encounter_id, location_id, week_id):
    """Full pipeline: extract → enforce evidence → tag → checklist."""
    result = {"encounter_id": encounter_id, "errors": []}
    t0 = time.time()

    # Step 1: Extract with MedGemma
    try:
        ext_prompt = extraction_prompt.replace("{note_text}", note_text)
        raw = run_medgemma(ext_prompt, max_new_tokens=1024)
        parsed = parse_json_response(raw)
        if parsed is None:
            result["errors"].append("extraction_parse_fail")
            parsed = {}
    except Exception as e:
        result["errors"].append(f"extraction_error: {e}")
        parsed = {}

    # Normalize symptoms
    symptoms = parsed.get("symptoms", {})
    for key in ["fever","cough","watery_diarrhea","bloody_diarrhea","vomiting","rash","difficulty_breathing"]:
        claim = symptoms.get(key, {})
        if not isinstance(claim, dict):
            claim = {}
        val = str(claim.get("value", "unknown")).lower().strip()
        if val not in ("yes", "no"):
            val = "unknown"
        quote = claim.get("evidence_quote")
        if val != "yes":
            quote = None
        elif not (quote and isinstance(quote, str) and quote.strip()):
            quote = None
            val = "unknown"
        symptoms[key] = {"value": val, "evidence_quote": quote}

    # Normalize other_symptoms
    other_symp = {}
    for k, v in parsed.get("other_symptoms", {}).items():
        if isinstance(v, dict):
            val = str(v.get("value","unknown")).lower().strip()
            if val not in ("yes","no"): val = "unknown"
            q = v.get("evidence_quote")
            if val != "yes": q = None
            elif not (q and isinstance(q, str) and q.strip()): q = None; val = "unknown"
            other_symp[k] = {"value": val, "evidence_quote": q}

    # Normalize patient
    pat = parsed.get("patient", {})
    if not isinstance(pat, dict): pat = {}
    age_group = str(pat.get("age_group","unknown")).lower().strip()
    if age_group not in ("infant","child","adolescent","adult","elderly"): age_group = "unknown"
    sex = str(pat.get("sex","unknown")).lower().strip()
    if sex not in ("male","female"): sex = "unknown"
    age_years = pat.get("age_years")
    try: age_years = int(age_years) if age_years else None
    except: age_years = None

    patient = {"age_group": age_group, "sex": sex}
    if age_years is not None: patient["age_years"] = age_years

    # Normalize onset, severity
    onset = parsed.get("onset_days")
    try: onset = int(onset) if onset else None
    except: onset = None
    severity = str(parsed.get("severity","unknown")).lower().strip()
    if severity not in ("mild","moderate","severe"): severity = "unknown"

    # Build encounter
    encounter = {
        "encounter_id": encounter_id,
        "location_id": location_id,
        "week_id": week_id,
        "note_text": note_text,
        "chw_id": str(parsed.get("chw_id", "unknown")),
        "visit_date": parsed.get("visit_date"),
        "visit_datetime": parsed.get("visit_datetime"),
        "encounter_sequence": parsed.get("encounter_sequence"),
        "area_id": str(parsed.get("area_id", "unknown")),
        "household_id": str(parsed.get("household_id", "unknown")),
        "gps": parsed.get("gps"),
        "patient": patient,
        "symptoms": symptoms,
        "other_symptoms": other_symp,
        "onset_days": onset,
        "severity": severity,
        "red_flags": parsed.get("red_flags", []),
        "treatments_given": [str(t) for t in parsed.get("treatments_given",[]) if t],
        "referral": None,
        "follow_up": None,
    }

    # Enforce evidence
    encounter, downgrades = enforce_evidence(encounter, note_text)
    result["evidence_downgrades"] = downgrades

    # Step 2: Syndrome tagging with MedGemma
    try:
        tag_p = syndrome_prompt.replace("{encounter_json}", json.dumps({
            "symptoms": encounter["symptoms"],
            "other_symptoms": encounter.get("other_symptoms", {}),
            "red_flags": encounter.get("red_flags", []),
            "severity": encounter.get("severity"),
            "onset_days": encounter.get("onset_days"),
        }, indent=2))
        tag_p = tag_p.replace("{note_text}", note_text)
        raw_tag = run_medgemma(tag_p)
        syndrome = parse_json_response(raw_tag) or {}
    except Exception as e:
        result["errors"].append(f"tagger_error: {e}")
        syndrome = {}

    tag = str(syndrome.get("syndrome_tag","unclear")).lower().strip()
    if tag not in ("respiratory_fever","acute_watery_diarrhea","other","unclear"):
        tag = "unclear"
    syndrome_result = {
        "encounter_id": encounter_id,
        "syndrome_tag": tag,
        "confidence": str(syndrome.get("confidence","low")).lower().strip(),
        "trigger_quotes": [str(q) for q in syndrome.get("trigger_quotes",[]) if q][:5],
        "reasoning": str(syndrome.get("reasoning",""))[:300],
    }
    if not syndrome_result["trigger_quotes"]:
        syndrome_result["trigger_quotes"] = ["insufficient data"]

    # Step 3: Checklist with MedGemma
    try:
        cl_p = checklist_prompt.replace("{encounter_json}", json.dumps(encounter, indent=2, default=str))
        cl_p = cl_p.replace("{note_text}", note_text)
        raw_cl = run_medgemma(cl_p)
        checklist = parse_json_response(raw_cl) or {"questions": []}
    except Exception as e:
        result["errors"].append(f"checklist_error: {e}")
        checklist = {"questions": []}

    checklist_result = {
        "encounter_id": encounter_id,
        "questions": checklist.get("questions", [])[:5],
    }

    elapsed = time.time() - t0
    result["encounter"] = encounter
    result["syndrome_tag"] = syndrome_result
    result["checklist"] = checklist_result
    result["processing_time_s"] = round(elapsed, 2)

    return result

print("Pipeline function defined ✅")

# %% [markdown]
# ## 6. Run on Gold Notes

# %%
# Load gold notes
gold_path = ROOT / "data_synth" / "gold_encounters_merged.jsonl"
if not gold_path.exists():
    gold_path = ROOT / "data_synth" / "gold_encounters.jsonl"

gold_notes = [json.loads(l) for l in open(gold_path, encoding="utf-8")]
print(f"Loaded {len(gold_notes)} gold notes")

# For demo/time, run on a subset
N_DEMO = 20  # Change to len(gold_notes) for full run
demo_notes = gold_notes[:N_DEMO]
print(f"Running pipeline on {N_DEMO} notes...")

# %%
results = []
for i, note in enumerate(demo_notes):
    print(f"Processing {i+1}/{N_DEMO}: {note['encounter_id']}...", end=" ")
    r = process_note(
        note_text=note["note_text"],
        encounter_id=note["encounter_id"],
        location_id=note.get("location_id", "unknown"),
        week_id=note.get("week_id", 0),
    )
    print(f"→ {r['syndrome_tag']['syndrome_tag']} ({r['processing_time_s']}s)")
    if r["errors"]:
        print(f"  ⚠️ Errors: {r['errors']}")
    results.append(r)

print(f"\n✅ Processed {len(results)} notes")
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
    raw_sitrep = run_medgemma(sitrep_p, max_new_tokens=1024)
    print(f"SITREP generated in {time.time()-t0:.1f}s")

    sitrep = parse_json_response(raw_sitrep)
    if sitrep:
        print(json.dumps(sitrep, indent=2))
    else:
        print("Raw output:")
        print(raw_sitrep[:800])

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
print("🎉 CHW Copilot Pipeline Complete!")
print("=" * 60)
print(f"Notes processed:        {len(results)}")
print(f"Syndrome accuracy:      {accuracy:.1%}")
print(f"Evidence grounding:     {grounded_claims}/{total_claims} ({grounded_claims/max(total_claims,1):.1%})")
print(f"Avg time per note:      {avg_time:.1f}s")
print(f"Model: {MEDGEMMA_ID}")
