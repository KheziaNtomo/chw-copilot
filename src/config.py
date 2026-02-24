"""Central configuration for models, paths, and runtime settings.

Switch between local (CPU/small models) and Kaggle (GPU/full models)
by setting the environment variable CHW_ENV=kaggle.
"""
import os
from pathlib import Path

# ── Environment ──────────────────────────────────────────────
ENV = os.getenv("CHW_ENV", "local")  # "local" or "kaggle"
IS_KAGGLE = ENV == "kaggle" or os.path.exists("/kaggle/working")

# ── Paths ────────────────────────────────────────────────────
if IS_KAGGLE:
    ROOT = Path("/kaggle/input/chw-copilot")
else:
    ROOT = Path(__file__).parent.parent

SCHEMA_DIR = ROOT / "schemas"
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data_synth"
OUT_DIR = ROOT / "data_synth"

# ── Model configuration ─────────────────────────────────────
# MedGemma for all tasks: extraction, syndrome tagging, checklist, SITREP
# MedGemma adds EHR understanding and medical document understanding
MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")
MODEL_VERSION = "1.5"
MEDGEMMA_DEVICE = "auto"

# Adaptation methods used (for documentation / judge alignment)
ADAPTATION_METHODS = [
    "prompt_engineering",
    "agentic_orchestration",
    "evidence_grounding_enforcement",
    "hallucination_detection",  # Strawberry/Pythea
]

# Hallucination Check Method: "self_consistency" or "pythea_counterfactual"
HALLUCINATION_METHOD = "pythea_counterfactual"

# Temperature and generation settings
EXTRACTION_MAX_TOKENS = 1024
REASONING_MAX_TOKENS = 512
SITREP_MAX_TOKENS = 1024
TEMPERATURE = 0.1  # Low temperature for deterministic extraction

# ── Syndrome configuration (extensible) ─────────────────────
SYNDROMES = ["respiratory_fever", "acute_watery_diarrhea", "other", "unclear"]

# ── Anomaly detection parameters ────────────────────────────
BASELINE_WINDOW = 4        # weeks of history for baseline
ALERT_THRESHOLD = 3.0      # count must exceed baseline_mean + this
MIN_COUNT_THRESHOLD = 5    # suppress alerts below this count
SMALL_COUNT_SUPPRESS = 5   # privacy: don't display counts < this
