#!/usr/bin/env python3
"""
update_notebook_narrative.py

Injects rich, research-backed narrative markdown cells into
notebooks/kaggle_main.ipynb, styled similarly to the
medgemma-edge-cardiologist-afib-detection.ipynb reference notebook.

Usage:
    python update_notebook_narrative.py

This script is idempotent — running it multiple times will replace
any previously injected narrative cells.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "notebooks" / "kaggle_main.ipynb"

# ── Sentinel used to identify cells WE injected ──────────────────────────────
NARRATIVE_TAG = "<!-- CHW_COPILOT_NARRATIVE -->"


# ── Narrative Markdown Cells ─────────────────────────────────────────────────

TITLE_BANNER = f"""{NARRATIVE_TAG}
# 🏥 CHW Copilot — MedGemma Clinical Pipeline

<div style="background: linear-gradient(135deg, #2d6a4f 0%, #40916c 50%, #52b788 100%); padding: 20px; border-radius: 15px; color: white;">
    <h2 style="margin: 0;">🎯 MedGemma Impact Challenge Submission</h2>
    <p style="margin-top: 10px;"><strong>Team:</strong> CHW Copilot | <strong>Model:</strong> MedGemma-4b-it (4-bit NF4) | <strong>Track:</strong> Agentic Workflow</p>
    <p style="margin-top: 5px;"><strong>Task:</strong> Structured extraction + syndromic surveillance from Community Health Worker notes</p>
</div>

---

## 📋 Table of Contents

1. [Introduction & Problem Statement](#1-introduction)
2. [Why These Two Illnesses](#2-illness-rationale)
3. [CHWs in Syndromic Surveillance](#3-chws-surveillance)
4. [Solution Architecture](#4-solution)
5. [Performance Targets & Limitations](#5-metrics-limitations)
6. [Setup & Dependencies](#6-setup)
7. [Load Prompts & Schemas](#7-prompts)
8. [HuggingFace Authentication](#8-auth)
9. [Load MedGemma](#9-model)
10. [Pipeline Helpers](#10-helpers)
11. [Smoke Test](#11-smoke)
12. [Run Pipeline on Gold Notes](#12-pipeline)
13. [Evaluation](#13-eval)
14. [Surveillance & SITREP](#14-surveillance)
15. [Save Results](#15-save)

---"""

PROBLEM_STATEMENT = f"""{NARRATIVE_TAG}
<a id="1-introduction"></a>
## 1. 🎯 Introduction & Problem Statement

### The Challenge: Community Health Workers and the Surveillance Gap

**Community Health Workers (CHWs)** are the "first-mile" layer of primary care delivery and case-finding, operating at massive scale:

- 🌍 **3.8 million+** CHWs across 98+ countries (Global Fund estimates)
- 🏠 Those same household visits produce the **earliest observable signals** of outbreaks — diarrheal disease clusters, febrile rash, respiratory symptoms
- 📝 But the signal is typically **trapped in informal notes** and lost to delayed aggregation

### The Data Gap

Public health surveillance frameworks emphasize timely detection and response, including both **Indicator-Based Surveillance (IBS)** and **Event-Based Surveillance (EBS)** for early alerting. Yet real-world timeliness remains a persistent bottleneck:

| Problem | Reality |
|---------|---------|
| 📅 **Weekly/monthly reporting** | Many systems depend on weekly or monthly cycles, missing fast-moving outbreaks |
| ⏰ **Timeliness below target** | Studies of weekly epidemic-prone disease reporting show timeliness consistently below target, even with electronic infrastructure |
| 📄 **Paper-first workflows** | Data loss and lag are often worse; paper-to-electronic transition is a core lever for improvement |
| 📊 **Delayed district picture** | District Health Officers see a smoothed, delayed picture — reducing EBS-style early warning value |

### Impact Hypothesis

If a district has **50 CHWs** completing ~10 household encounters/day over ~250 working days, that is **~125,000 encounter notes/year**.

In a weekly reporting regime, even perfect compliance creates an inherent delay (days to a week), and empirical surveillance literature shows delays can extend into **multiple weeks** depending on system level and disease (median reporting delays can reach ~40 days in some contexts).

> **CHW Copilot's goal:** Same-day syndromic aggregation from the raw note (offline), plausibly shifting actionable detection earlier by **5–14 days** in settings that otherwise rely on weekly rollups — while producing **auditable, schema-valid data** rather than opaque model text."""

ILLNESS_RATIONALE = f"""{NARRATIVE_TAG}
<a id="2-illness-rationale"></a>
## 2. 🔬 Why Respiratory Fever & Acute Watery Diarrhea

### Rationale for Initial Focus

CHW Copilot's initial release deliberately targets **two syndromic categories**: respiratory fever and acute watery diarrhea (AWD). This is not a limitation — it is a strategic design choice driven by three factors:

#### 1. Highest Burden in Sub-Saharan Africa

These are the **two most common syndromic presentations** encountered by CHWs in the regions where this tool is designed to operate. Together, respiratory infections and diarrheal diseases account for the majority of under-5 mortality in low-resource settings.

#### 2. Direct Mapping to IDSR Priority Diseases

Both syndromes map directly to diseases on the **Integrated Disease Surveillance and Response (IDSR)** priority list used across Africa:

| Syndrome Category | IDSR Priority Diseases | Example Pathogens |
|-------------------|----------------------|-------------------|
| 🌡️ **Respiratory Fever** | Measles, Pneumonia, COVID-19, Influenza | *S. pneumoniae*, SARS-CoV-2, Measles virus |
| 💧 **Acute Watery Diarrhea** | Cholera, Typhoid, Dysentery | *V. cholerae*, *S. typhi*, Rotavirus |

#### 3. Clear Case Definitions Enable Reliable Extraction

Both syndromes have well-established, WHO-standard case definitions based on **observable symptoms** (fever + cough + difficulty breathing; watery stool ≥3 times/day). This makes them ideal targets for structured extraction from free-text notes — the symptom vocabulary is consistent and unambiguous.

### Scope to Expand

> **The architecture is deliberately syndrome-agnostic.** Adding a new syndrome category (e.g., acute febrile illness with rash, acute flaccid paralysis, hemorrhagic fever) requires only:
>
> 1. Adding keyword patterns to the deterministic syndrome tagger
> 2. Updating the extraction prompt with relevant symptom fields
> 3. (Optionally) fine-tuning with labeled examples for the new category
>
> No changes to the core pipeline, schema validation, or evidence grounding logic are needed. The system is designed to **grow with surveillance needs**."""

CHWS_SURVEILLANCE = f"""{NARRATIVE_TAG}
<a id="3-chws-surveillance"></a>
## 3. 🏘️ CHWs in Syndromic & Infectious Disease Surveillance

### Why CHWs Are Uniquely Positioned

Community Health Workers are the **only health workforce that operates at the household level** in most low- and middle-income countries. This gives them a unique vantage point:

- 🏠 **Household-level contact** — they observe symptoms, living conditions, and family clusters before anyone presents at a health facility
- 🗣️ **Community trust** — patients share information with CHWs that they might not report to formal healthcare providers
- 🔄 **Regular visits** — routine household visits create a continuous surveillance surface, not just point-in-time facility visits
- 🌍 **Scale** — millions of CHWs conducting millions of visits daily = an enormous, untapped surveillance network

### The Current CHW Journey vs. CHW Copilot

| Step | Today (Paper-first) | With CHW Copilot |
|------|-------------------|-----------------|
| 📝 **Record** | Shorthand / multilingual free-text note | Same — note is the input (typed or voice transcript) |
| 🔄 **Structure** | Manual tallies at end of week | **Instant:** MedGemma extracts structured JSON encounter |
| 🏷️ **Classify** | Manual syndrome assignment (if any) | **Automatic:** keyword + LLM syndrome tagger |
| ✅ **Validate** | None | **Automatic:** JSON Schema Draft-07 + evidence grounding |
| ⚠️ **Alert** | None until district rollup | **Same-day:** red-flag alerts for CHW + anomaly detection |
| 📊 **Aggregate** | Weekly / monthly tallies → district | **Same-day:** structured encounters → weekly counts → anomaly signals |
| 📋 **Report** | District Health Officer manually compiles | **Automated SITREP draft** with MedGemma |

### The Promise of Electronic Surveillance

Pilots of electronic IDSR-style workflows have shown meaningful gains in both **signal capture** and **speed**:
- Higher timeliness and substantially more "rumors" identified/verified from e-reporting sites
- The electronic pathway doesn't just speed up the same process — it **captures signals that paper workflows miss entirely**

CHW Copilot brings this electronic advantage to the **first mile** of the health system, where it has the greatest potential to close the detection gap."""

SOLUTION_ARCHITECTURE = f"""{NARRATIVE_TAG}
<a id="4-solution"></a>
## 4. 🏗️ Solution Architecture

### Why MedGemma?

This submission is built for the **MedGemma Impact Challenge**, where using at least one HAI-DEF model is mandatory. We leverage **MedGemma** as the system's core "medical-language engine" because:

| Feature | Description |
|---------|-------------|
| 🏥 **Medical priors** | HAI-DEF models are designed for healthcare, with documented pathways for prompt-based adaptation |
| 📦 **Compute-efficient** | Available in a 4B-class model, suitable for edge deployment |
| 🔓 **Open-weight** | Distributed via Hugging Face — enables offline, privacy-preserving inference |
| 🔧 **Adaptable** | Supports QLoRA fine-tuning for domain-specific improvements |

### The 6-Agent Pipeline

**MedGemma** handles the parts that require medical priors and flexible inference. **Deterministic agents** handle the parts that must be exact and inspectable.

```
CHW note (typed / voice transcript)
  ↓
[1] Encounter Extractor (MedGemma 4B; schema-guided JSON)
  ↓
[2] Evidence Grounder (deterministic: span-match quotes per field)
  ↓
[3] Hallucination Detector (Pythea/Strawberry: evidence-scrub + KL budget gap)
      ↳ if flagged → 1 targeted retry of [1] with "unsupported claims" feedback
  ↓
[4] Syndrome Tagger (MedGemma; deterministic fallback on low confidence)
  ↓
[5] Checklist Generator (MedGemma; missing-field + risk-trigger prompts)
  ↓
[6] Schema Validator (deterministic: JSON Schema Draft-07)
  ↓
Outputs:
  • Encounter JSON (schema-valid + evidence-quoted)
  • Red-flag alerts for CHW
  • Aggregates (weekly counts by syndrome × location; anomaly signals; SITREP draft)
```

### Key Innovations

1. **Verification as a first-class step** — not an afterthought; integrated hallucination detection via Pythea/Strawberry
2. **Self-correction loop** — when the detector flags insufficient evidence, the extractor re-runs with targeted feedback, prioritizing abstention over fabrication
3. **Evidence anchoring** — every extracted field is tied to a verbatim quote from the original note
4. **Schema-enforced output** — invalid outputs cannot be emitted; 100% schema validation pass rate
5. **QLoRA fine-tuning** — 4-bit quantized LoRA adapters reduce schema failures and improve evidence quoting
6. **DHIS2-aligned aggregation** — structured encounters → aggregate indicators, designed for the world's largest HMIS platform (~80 LMICs, ~3.2B people)"""

METRICS_LIMITATIONS = f"""{NARRATIVE_TAG}
<a id="5-metrics-limitations"></a>
## 5. 📊 Performance Targets & Limitations

### Performance Metrics

| Metric | Target | How Measured / Enforced |
|--------|--------|----------------------|
| 🏷️ **Syndrome tag accuracy** | ≥80% | Labeled eval set; reported in this notebook |
| 📎 **Evidence grounding rate** | ≥98% | Deterministic quote requirement; unsupported fields coerced to `unknown` |
| ⚠️ **Hallucination rate** | ≤2% | Strawberry "budget gap" flags + abstention |
| ✅ **Schema validation pass rate** | 100% | Strict Draft-07 validation gate |
| ⏱️ **Avg processing time / note** | <10s (GPU) | Measured on T4-class GPU |

### Deployment Plan (Offline-first, Privacy-focused)

| Phase | Description |
|-------|-------------|
| 🔬 **Prototype** | Single-machine demo on T4-class GPU, no network dependencies beyond initial model download |
| 🏭 **Production** | Edge workstation or district laptop with quantized model + store-and-forward sync |
| 🔒 **Privacy** | Local inference by default; only structured aggregates synced; minimum cell size of 5 for published aggregates |

### Limitations (Explicit & Testable)

> **CHW Copilot is NOT a diagnosis engine.** It produces surveillance-grade structure from notes and always preserves uncertainty where evidence is missing.

**Known weaknesses:**
1. 🌐 **Multilingual shorthand** — extremely noisy multilingual free-text without reliable transcription is the weakest mode
2. ⏰ **Temporal reasoning** — merging multiple visits in a single note can confuse time anchoring
3. 🌍 **Population-specific** — model adaptation is workflow- and population-specific; HAI-DEF guidance emphasizes use-case validation before deployment

**Mitigations:**
- Checklist-driven clarification prompts the CHW for missing information
- Conservative abstention + re-run logic rather than forced completion
- QLoRA fine-tuning with region-specific medical QA (AfriMed-QA)

---"""


def make_markdown_cell(source_text: str) -> dict:
    """Create a notebook markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_text,
    }


def main():
    # Read notebook
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Remove any previously injected narrative cells
    original_count = len(nb["cells"])
    nb["cells"] = [
        cell for cell in nb["cells"]
        if not (
            cell["cell_type"] == "markdown"
            and NARRATIVE_TAG in (cell.get("source", "") if isinstance(cell.get("source"), str) else "\n".join(cell.get("source", [])))
        )
    ]
    removed = original_count - len(nb["cells"])
    if removed:
        print(f"  Removed {removed} previously injected narrative cell(s)")

    # Also remove the old title cell (first markdown cell) since we replace it
    # The old title starts with "# 🏥 CHW Copilot"
    old_title_idx = None
    for i, cell in enumerate(nb["cells"]):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "\n".join(src)
        if cell["cell_type"] == "markdown" and "CHW Copilot" in src and "Table of Contents" not in src:
            old_title_idx = i
            break

    if old_title_idx is not None:
        nb["cells"].pop(old_title_idx)
        print(f"  Removed old title cell at index {old_title_idx}")

    # Build narrative cells
    narrative_cells = [
        make_markdown_cell(TITLE_BANNER),
        make_markdown_cell(PROBLEM_STATEMENT),
        make_markdown_cell(ILLNESS_RATIONALE),
        make_markdown_cell(CHWS_SURVEILLANCE),
        make_markdown_cell(SOLUTION_ARCHITECTURE),
        make_markdown_cell(METRICS_LIMITATIONS),
    ]

    # Find the "Setup" cell to insert before it
    setup_idx = None
    for i, cell in enumerate(nb["cells"]):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "\n".join(src)
        if cell["cell_type"] == "markdown" and "Setup" in src and ("0." in src or "🔧" in src):
            setup_idx = i
            break

    if setup_idx is None:
        # Fallback: insert at position 0
        setup_idx = 0
        print("  ⚠️  Could not find Setup cell, inserting at beginning")

    # Also update the setup cell to add an anchor
    if setup_idx is not None and setup_idx < len(nb["cells"]):
        cell = nb["cells"][setup_idx]
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "\n".join(src)
        if '<a id="6-setup">' not in src:
            cell["source"] = '<a id="6-setup"></a>\n' + src

    # Insert narrative cells
    for i, cell in enumerate(narrative_cells):
        nb["cells"].insert(setup_idx + i, cell)

    # Write back
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✅ Notebook updated successfully!")
    print(f"   {len(narrative_cells)} narrative cells inserted at position {setup_idx}")
    print(f"   Total cells: {len(nb['cells'])}")
    print(f"   Output: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
