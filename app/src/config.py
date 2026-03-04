"""Central configuration for models, paths, and runtime settings.

Supports local development, Kaggle GPU, and Streamlit Cloud (API mode).
"""
import os
from pathlib import Path

# ── Environment ──────────────────────────────────────────────
ENV = os.getenv("CHW_ENV", "local")  # "local", "kaggle", or "streamlit"
IS_KAGGLE = ENV == "kaggle" or os.path.exists("/kaggle/working")
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") is not None or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"

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
# MedGemma via Google AI Studio — free tier at https://aistudio.google.com/apikey
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "medgemma-4b-it")
MODEL_VERSION = "medgemma-4b-it"

# Legacy name kept for pipeline agent metadata compatibility
MEDGEMMA_MODEL = GEMINI_MODEL

# Adaptation methods used (for documentation / judge alignment)
ADAPTATION_METHODS = [
    "prompt_engineering",
    "agentic_orchestration",
    "evidence_grounding_enforcement",
    "hallucination_detection",
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
