# CHW Copilot

Offline-first, privacy-focused agentic surveillance tool that turns Community Health Worker (CHW) field notes into schema-validated structured encounters and syndromic surveillance signals. Powered by **MedGemma** with **Strawberry** hallucination detection.

**MedGemma Impact Challenge — Agentic Workflow Prize**

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         CHW Copilot Pipeline            │
                    │      6-Agent Agentic Orchestrator       │
                    └─────────────────────────────────────────┘

 CHW Note ──→ [Voice Transcription]  ← MedASR (optional)
       │           │
       ▼           ▼
  ┌──────────────────────────────────────────────────────┐
  │  Agent 1: Encounter Extractor (MedGemma)        │
  │  → Structured JSON with evidence_quote per claim     │
  ├──────────────────────────────────────────────────────┤
  │  Agent 2: Evidence Grounder (Deterministic)          │
  │  → Downgrades unsupported claims to "unknown"        │
  ├──────────────────────────────────────────────────────┤
  │  Agent 3: Hallucination Detector (Strawberry/Pythea) │
  │  → Flags procedural hallucinations via budget_gap    │
  ├──────────────────────────────────────────────────────┤
  │  Agent 4: Syndrome Tagger (MedGemma)            │
  │  → respiratory_fever / acute_watery_diarrhea / other │
  ├──────────────────────────────────────────────────────┤
  │  Agent 5: Checklist Generator (MedGemma)        │
  │  → Priority-ranked follow-up questions               │
  ├──────────────────────────────────────────────────────┤
  │  Agent 6: Schema Validator (Deterministic)           │
  │  → JSON Schema compliance + final pass/fail          │
  └──────────────────────────────────────────────────────┘
       │
       ▼
  Aggregation → Anomaly Detection → Weekly SITREP
```

### Models & Tools

| Component | Model / Tool | Role |
|-----------|-------------|------|
| Extraction | MedGemma (`google/medgemma-4b-it`) | Structured encounter from free-text |
| Evidence Grounding | Deterministic | Verify evidence_quote ⊂ note |
| Hallucination Detection | Strawberry (Pythea) | Budget gap analysis per claim |
| Syndrome Tagging | MedGemma | Syndromic classification |
| Checklist | MedGemma | Follow-up question generation |
| Voice Input | MedASR (optional) | Medical speech-to-text |
| Anomaly Detection | Deterministic (z-score) | Surge detection per location/syndrome |

### MedGemma Adaptation

- **Adaptation method**: Prompt engineering — zero-shot + structured output via JSON schema
- **API**: `AutoModelForImageTextToText` + `AutoProcessor` (chat template with content blocks)
- **Evidence grounding**: Every LLM claim requires `evidence_quote` substring of original note

## Repo Structure

```
schemas/          → JSON Schemas (encounter, checklist, syndrome, sitrep)
prompts/          → Prompt templates for all model calls
data_synth/       → Synthetic gold data (60 CHW notes) + simulation events (672 events)
src/              → Pipeline modules
  ├── config.py         → Centralized configuration
  ├── models.py         → MedGemma loader + inference
  ├── pipeline.py       → 6-agent orchestrator with trace
  ├── hallucination.py  → Strawberry integration
  ├── voice.py          → MedASR voice transcription
  ├── validate.py       → Evidence enforcement + schema validation
  ├── tagger.py         → Syndrome tagging (LLM + deterministic)
  ├── checklist.py      → Checklist generation
  ├── detect.py         → Anomaly detection
  └── sitrep.py         → SITREP generation
app/              → Streamlit demo application
  ├── app.py            → Main entry point
  ├── chw_view.py       → CHW field worker interface
  ├── district_view.py  → District surveillance dashboard
  ├── demo_data.py      → Pre-computed offline demo data
  └── styles.css        → Premium dark medical theme
golden_artifacts/ → Sample inputs, outputs, failure modes
tests/            → Unit tests (30 tests, 4 files)
notebooks/        → Kaggle notebook
```

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py             # Launch demo (offline OK)
python -m pytest tests/ -v                      # Run tests (30 tests)
```

## Privacy & Safety

> **⚠️ Syndromic surveillance support only. NOT for clinical diagnosis.**
> - Offline-first: No data leaves the device
> - Aggregated counts only in surveillance — no individual patient data displayed
> - All outputs require human verification before action
> - Evidence grounding ensures every claim is traceable to the note
> - Strawberry hallucination detection provides a second safety layer

## Impact Model

1. **CHW Level**: Reduces documentation errors, surfaces missing information via checklist
2. **Facility Level**: Structured referrals with evidence-grounded encounter summaries
3. **District Level**: Real-time anomaly detection enables early outbreak response
4. **System Level**: Standardized syndromic data feeds into IDSR-compatible reporting

## HAI-DEF Alignment

- **Privacy by design**: Runs fully offline; no PII in model prompts
- **Evidence grounding**: LLM claims are verified against source text
- **Hallucination detection**: Strawberry catches claims not supported by evidence
- **Human in the loop**: All outputs labelled as decision-support, not diagnosis
- **Deterministic fallbacks**: Pipeline functions without GPU via rule-based alternatives

## Deployment Strategy (Prototype vs. Production)

| Feature | **Competition Prototype** (Current) | **Real-World Pilot** (Vision) |
| :--- | :--- | :--- |
| **Interface** | Streamlit Web App (Browser) | Native Android App (Kotlin/Jetpack Compose) |
| **Compute** | Laptop / Kaggle T4 GPU | On-Device NPU/GPU via **MediaPipe LLM Inference** |
| **Model** | MedGemma (4B) via Transformers | MedGemma (2B/4B) quantized to int4 |
| **Connectivity** | Localhost / Cloud | **Zero Data Cost** (Fully Offline) |
| **Data Sync** | CSV / JSON Export | Opportunistic background sync (WorkManager) |
| **Impact** | Demonstrates logical flow & safety | Enables deployment in remote, disconnected areas |

## Success Metrics

| Layer | Metric | Target |
|-------|--------|--------|
| Extraction | Hallucination rate | ≤ 2% |
| Tagging | Syndrome tag F1 | ≥ 0.80 |
| Red flags | Recall | ≥ 0.90 |
| Monitoring | Detection delay | ≤ 1 week |
| Monitoring | False alert rate | < 5% of location-weeks |
