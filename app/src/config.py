"""Central configuration for models, paths, and runtime settings.

Switch between local (CPU/small models), Kaggle (GPU/full models),
and Hugging Face Spaces (GPU/full models) by setting environment
variables or deploying to the respective platform.
"""
import os
from pathlib import Path

# ── Environment ──────────────────────────────────────────────
ENV = os.getenv("CHW_ENV", "local")  # "local", "kaggle", or "hf"
IS_KAGGLE = ENV == "kaggle" or os.path.exists("/kaggle/working")
IS_HF_SPACE = os.getenv("SPACE_ID") is not None  # Set automatically by HF Spaces

# ── Paths ────────────────────────────────────────────────────
if IS_KAGGLE:
    ROOT = Path("/kaggle/input/chw-copilot")
elif IS_HF_SPACE:
    # HF Space root is the app/ directory itself
    ROOT = Path(__file__).parent.parent
else:
    ROOT = Path(__file__).parent.parent

SCHEMA_DIR = ROOT / "schemas"
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data_synth"
OUT_DIR = ROOT / "data_synth"

# ── Model configuration ─────────────────────────────────────
# MedGemma for all tasks: extraction, syndrome tagging, checklist, SITREP
MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")
MODEL_VERSION = "1.5"
MEDGEMMA_DEVICE = "auto"

# Enable 4-bit quantisation on HF Spaces / Kaggle for T4 16GB
USE_4BIT = IS_HF_SPACE or IS_KAGGLE or os.getenv("USE_4BIT", "").lower() in ("1", "true")

# HF token for gated model access — read from env or Streamlit secrets
HF_TOKEN = os.getenv("HF_TOKEN")

# Adaptation methods used (for documentation / judge alignment)
ADAPTATION_METHODS = [
    "prompt_engineering",
    "agentic_orchestration",
    "evidence_grounding_enforcement",
    "hallucination_detection",  # Strawberry/Pythea
]

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
