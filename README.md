# CHW Copilot

Offline-first, privacy-focused tool that turns Community Health Worker (CHW) field notes into schema-validated structured encounters and syndromic surveillance signals.

**Problem statement:** Turn frontline CHW notes into structured encounters and syndromic signals, then aggregate by place/time to surface early anomaly alerts and a weekly SITREP—offline-first, not diagnostic.

## Architecture

```
CHW Note → [NuExtract] → Structured Encounter JSON (schema-validated + evidence-grounded)
                ↓
         [MedGemma] → Syndrome Tag + Checklist of missing questions
                ↓
         Aggregation → Anomaly Detection (deterministic z-score)
                ↓
         [MedGemma] → Weekly SITREP (structured + narrative)
```

### Models
- **NuExtract** (NuMind, HuggingFace): Structured information extraction from free-text notes
- **MedGemma** (Google HAI-DEF): Medical reasoning — syndrome classification, checklist generation, SITREP narrative

### Scope (frozen)
- **Inputs**: Typed CHW notes
- **Outputs**: Structured encounter JSON + missing-questions checklist + syndrome tag
- **Syndromes**: `respiratory_fever` and `acute_watery_diarrhea` only
- **Surveillance**: Weekly aggregation + deterministic anomaly detection
- **Agents**: (A) Checklist agent per encounter, (B) Monitoring agent for SITREPs, (C) Prompt-optimizer loop

## Repo Structure

```
schemas/          → JSON Schemas for all outputs (encounter, checklist, syndrome, sitrep)
prompts/          → Prompt templates for all model calls
data_synth/       → Synthetic gold data (60 CHW notes) + simulation events (672 events)
src/              → Pipeline modules (extract, validate, tag, checklist, aggregate, detect, sitrep)
app/              → Streamlit application
notebooks/        → Kaggle-runnable notebook
```

## Quick Start

```bash
pip install -r requirements.txt
python data_synth/generate.py        # Generate synthetic data
python src/run_pipeline.py           # Run stubbed pipeline
streamlit run app/app.py             # Launch demo app
```

## Success Metrics

| Layer | Metric | Target |
|-------|--------|--------|
| Extraction | Hallucination rate | ≤ 2% |
| Tagging | Syndrome tag F1 | ≥ 0.80 |
| Red flags | Recall | ≥ 0.90 |
| Monitoring | Detection delay | ≤ 1 week |
| Monitoring | False alert rate | < 5% of location-weeks |

## Safety

> **⚠️ Syndromic surveillance support only. NOT diagnosis.**
> Aggregated counts only. No individual patient data displayed.
> All outputs require human verification before action.
