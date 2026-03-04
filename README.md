# CHW Copilot

A demo tool that turns Community Health Worker (CHW) field notes into structured encounters and syndromic surveillance signals. Built with MedGemma and deployed as a Streamlit web app.

> **⚠️ Demo only.** This app runs on pre-computed pipeline outputs — no GPU or model access required. All notes and surveillance data are synthetic, created for demonstration purposes. Not for clinical use.

---

## What It Does

CHWs record patient encounters as informal shorthand — abbreviated, multilingual, and unstructured. CHW Copilot runs these notes through a structured extraction pipeline to produce:

- **Structured encounter JSON** — symptoms, demographics, severity, red flags, referral status
- **Syndrome classification** — respiratory fever, acute watery diarrhoea, other, unclear
- **CHW checklist** — follow-up questions for missing or ambiguous clinical information
- **District surveillance dashboard** — weekly syndrome counts, anomaly detection, SITREP

## Demo Data

The app ships with **8 hand-crafted CHW notes** covering a range of clinical presentations (febrile illness, diarrhoea with dehydration, unclear presentations, hallucination test cases). These are pre-processed — the pipeline outputs are baked in so the demo works without a live model.

The surveillance dashboard uses **synthetic weekly encounter counts** across 5 Nairobi locations over 12 weeks, with a simulated respiratory fever cluster injected at weeks 7–8 to demonstrate anomaly detection.

## Pipeline

```
CHW note (free text)
  ↓
[1] Extract       — MedGemma 4B: structured JSON with evidence quotes
  ↓
[2] Ground        — Deterministic: verify every claim is in the note
  ↓
[3] Verify        — Hallucination detection: flag contradictions
  ↓
[4] Tag           — Syndrome classification
  ↓
[5] Checklist     — Follow-up question generation
  ↓
[6] Validate      — JSON Schema validation
```

## Quick Start

```bash
git clone https://github.com/KheziaNtomo/chw-copilot
cd chw-copilot
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app runs fully offline in demo mode — no API keys or GPU required.

## Connecting a Live Model

By default the app runs on pre-computed demo outputs. To run the pipeline live with real MedGemma inference:

**Requirements:** NVIDIA GPU with ≥8 GB VRAM (T4 or better), Python 3.10+

```bash
# 1. Accept the MedGemma licence at huggingface.co/google/medgemma-4b-it
# 2. Set your Hugging Face token
export HF_TOKEN=hf_your_token_here   # Linux/Mac
$env:HF_TOKEN="hf_your_token_here"   # Windows PowerShell

# 3. Install full dependencies
pip install -r app/requirements.txt
pip install transformers>=4.40 accelerate bitsandbytes

# 4. Run — the app will detect the token and load MedGemma on startup
streamlit run app/app.py
```

Without a GPU, the app falls back to demo mode automatically — no errors, just pre-computed results.

The full pipeline (including live MedGemma inference on 60 evaluation notes) is demonstrated in [`notebooks/kaggle_main.ipynb`](notebooks/kaggle_main.ipynb), which runs on a free Kaggle T4 GPU.


## Repo Structure

```
app/
  app.py              — Streamlit entry point
  chw_view.py         — CHW field worker interface
  district_view.py    — District surveillance dashboard
  demo_data.py        — Pre-computed pipeline outputs (demo mode)
  styles.css          — App styling
  src/                — Pipeline modules (extract, tag, validate, detect…)
  prompts/            — MedGemma prompt templates
  schemas/            — JSON Schema definitions
  data_synth/         — Synthetic evaluation data
notebooks/
  kaggle_main.ipynb   — Full pipeline run on Kaggle T4 GPU (MedGemma 4B)
```

## Evaluation

Evaluated on 60 hand-annotated CHW notes run through MedGemma-4b-it on Kaggle (T4 GPU):

| Metric | Result |
|--------|--------|
| Syndrome tag accuracy | **95% (57/60)** |
| Evidence grounding rate | **100% (95/95 claims)** |
| Avg processing time | **27s / note** |

> Results are from a small, hand-authored evaluation set and should be treated as a proof-of-concept baseline, not a clinical validation.

## Limitations

- Demo notes are hand-crafted — real CHW notes will be noisier and more multilingual
- Syndrome categories are limited to respiratory fever and acute watery diarrhoea (plus other/unclear)
- Not validated for clinical decision-making — surveillance support only

## Disclaimer

Not for clinical diagnosis. All outputs require human verification. Surveillance support tool only.
