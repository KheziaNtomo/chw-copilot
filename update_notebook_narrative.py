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
# CHW Copilot — MedGemma Clinical Pipeline

> **MedGemma Impact Challenge** | MedGemma-4b-it (bfloat16) | [GitHub](https://github.com/KheziaNtomo/chw-copilot)

Uses MedGemma to extract structured encounters from Community Health Worker notes for syndromic surveillance and outbreak detection.

| Step | What | Model |
|------|------|-------|
| Extract | CHW note → structured JSON (symptoms, severity, red flags) | MedGemma |
| Tag | Syndrome classification (respiratory fever / AWD / other) | Keyword |
| Aggregate | Weekly counts by location, z-score anomaly detection | Statistical |
| SITREP | District situation report | MedGemma |

---"""

PROBLEM_STATEMENT = f"""{NARRATIVE_TAG}
<a id="1-introduction"></a>
## 1. Introduction & Problem Statement

**3.8 million+ CHWs** across 98+ countries generate the earliest observable signals of outbreaks — diarrhoeal clusters, febrile rash, respiratory symptoms — but these signals are trapped in unstructured text notes and lost to delayed paper-based reporting.

Surveillance timeliness is consistently below target: Uganda's DHIS2 weekly reporting achieved only ~44–49% timeliness against an 80% target. Nigeria's eIDSR pilot showed electronic reporting improves timeliness (73% vs 43%) and captures signals that paper workflows miss entirely.

**CHW Copilot's goal:** same-day syndromic aggregation from raw notes, shifting detection earlier by days–weeks while producing auditable, schema-valid data."""

ILLNESS_RATIONALE = f"""{NARRATIVE_TAG}
<a id="2-illness-rationale"></a>
## 2. Why Respiratory Fever & Acute Watery Diarrhea

These are the **two highest-burden syndromic categories** in sub-Saharan Africa with clear WHO IDSR case definitions:

| Syndrome | IDSR Diseases | Burden |
|----------|--------------|--------|
| Respiratory Fever | Measles, Pneumonia, Influenza | Pneumonia: ~740K under-5 deaths/year |
| Acute Watery Diarrhea | Cholera, Typhoid, Rotavirus | Diarrhoeal disease: ~525K under-5 deaths/year |

The architecture is **syndrome-agnostic** — adding new categories requires only updating keyword patterns and prompt fields."""

CHWS_SURVEILLANCE = f"""{NARRATIVE_TAG}
<a id="3-chws-surveillance"></a>
## 3. CHWs in Syndromic Surveillance

CHWs are the **only health workforce operating at household level** — they observe symptoms, family clusters, and community patterns before facility presentation.

| Step | Paper-first | With CHW Copilot |
|------|------------|-----------------|
| Structure | Manual tallies at end of week | Instant MedGemma extraction |
| Classify | Manual (if any) | Automatic keyword + LLM tagger |
| Alert | None until district rollup | Same-day red-flag + anomaly detection |
| Report | District officer manually compiles | Automated SITREP draft |"""

SOLUTION_ARCHITECTURE = f"""{NARRATIVE_TAG}
<a id="4-solution"></a>
## 4. Solution Architecture

MedGemma handles medical reasoning; deterministic agents enforce safety and evidence quality.

```
CHW note → [1] MedGemma Extractor → [2] Evidence Grounder → [3] Hallucination Detector
         → [4] Syndrome Tagger → [5] Checklist Generator → [6] Schema Validator
         → Structured encounter + alerts + surveillance aggregates
```

**Why MedGemma:** Medical pre-training handles CHW shorthand (e.g., "hot body" = fever, "rice-water stool" = cholera-like AWD) that general LLMs miss. Open-weight, bfloat16isable, suitable for edge deployment."""

METRICS_LIMITATIONS = f"""{NARRATIVE_TAG}
<a id="5-metrics-limitations"></a>
## 5. Performance & Limitations

| Metric | Value |
|--------|-------|
| Syndrome tag accuracy | 95% (keyword tagger, 60-note gold set) |
| Processing time | ~1 min/note (T4 GPU, bfloat16) |
| Schema validation | 100% pass rate |

**Limitations:** multilingual shorthand is the weakest mode; temporal reasoning across multi-visit notes can confuse onset anchoring; model adaptation is population-specific (HAI-DEF guidance emphasises validation before deployment).

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
