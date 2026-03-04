"""Model management for CHW Copilot via Google AI Studio.

Uses the Google GenAI SDK (google-genai) for clinical extraction.
Free tier available at https://aistudio.google.com/apikey

Falls back to demo mode (pre-computed results + deterministic pipeline)
when no GOOGLE_API_KEY is configured.
"""
import json
import logging
import os
import re
from typing import Dict, Any, Optional

from . import config

logger = logging.getLogger(__name__)

# ── Globals ──────────────────────────────────────────────────
_client = None
_load_error = None
_api_available = False

# Medical system prompt for clinical extraction tasks
MEDICAL_SYSTEM_PROMPT = (
    "You are MedGemma, a medical AI assistant specialised in community health "
    "surveillance. You extract structured clinical data from community health "
    "worker field notes. You are precise, evidence-based, and always ground "
    "your outputs in the original note text. Respond only with valid JSON "
    "when asked to produce structured output."
)


def _resolve_api_key() -> Optional[str]:
    """Resolve Google AI Studio API key from session, env, or Streamlit secrets."""
    # 1. Check Streamlit session state (user-entered key)
    try:
        import streamlit as st
        session_key = st.session_state.get("google_api_key", "")
        if session_key and session_key.strip():
            return session_key.strip()
    except Exception:
        pass

    # 2. Check environment variable
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    # 3. Check Streamlit secrets
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return None


def _get_client():
    """Get or create the GenAI client (lazy singleton)."""
    global _client, _load_error, _api_available

    if _client is not None:
        return _client

    api_key = _resolve_api_key()
    if not api_key:
        _load_error = "No API key — running in demo mode"
        _api_available = False
        raise RuntimeError(_load_error)

    try:
        from google import genai
    except ImportError as e:
        _load_error = "google-genai not installed — pip install google-genai"
        raise RuntimeError(_load_error) from e

    _client = genai.Client(api_key=api_key)
    _api_available = True
    logger.info("Google AI Studio client ready (model: %s)", config.GEMINI_MODEL)
    return _client


def is_model_available() -> bool:
    """Check if Google AI API is configured and ready."""
    return _api_available


def get_load_error() -> Optional[str]:
    """Return the error message from the last failed load attempt."""
    return _load_error


def try_load_model() -> bool:
    """Validate that the Google AI API is reachable.

    Returns True if the client was created successfully.
    """
    global _load_error, _api_available, _client
    # Reset so we re-check (user may have entered a key)
    _client = None
    try:
        _get_client()
        _api_available = True
        return True
    except Exception as e:
        _load_error = str(e)
        _api_available = False
        return False


def generate_medgemma(prompt: str, max_tokens: int = None) -> str:
    """Run generation via the Google AI Studio API.

    Uses Gemma with a medical system prompt for clinical extraction tasks.
    """
    client = _get_client()
    max_tokens = max_tokens or config.REASONING_MAX_TOKENS

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config={
            "system_instruction": MEDICAL_SYSTEM_PROMPT,
            "max_output_tokens": max_tokens,
            "temperature": config.TEMPERATURE,
        },
    )

    return response.text.strip()


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from a model response."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()
    brace_idx = cleaned.find("{")
    if brace_idx > 0:
        cleaned = cleaned[brace_idx:]
    rbrace_idx = cleaned.rfind("}")
    if rbrace_idx >= 0 and rbrace_idx < len(cleaned) - 1:
        cleaned = cleaned[:rbrace_idx + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    try:
        open_braces = cleaned.count("{") - cleaned.count("}")
        open_brackets = cleaned.count("[") - cleaned.count("]")
        if open_braces > 0 or open_brackets > 0:
            fixed = cleaned + "]" * open_brackets + "}" * open_braces
            return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None
