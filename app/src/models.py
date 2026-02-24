"""Model loading and management for MedGemma.

Lazy-loads MedGemma model using the CausalLM API
(AutoModelForCausalLM + AutoTokenizer). Caches it so it is
only loaded once per session regardless of how many calls are made.

MedGemma uses AutoTokenizer (not AutoTokenizer) and
AutoModelForCausalLM (not AutoModelForCausalLM).

On HF Spaces / Kaggle, uses 4-bit NF4 quantisation to fit in T4 16GB.
"""
import json
import os
import re
from typing import Dict, Any, Optional
from functools import lru_cache

from . import config

# ── Globals for lazy-loaded model ────────────────────────────
_medgemma_model = None
_medgemma_tokenizer = None
_load_error = None


def _load_medgemma():
    """Lazy-load the MedGemma model and tokenizer.

    Uses 4-bit NF4 quantisation when USE_4BIT is set (HF Spaces / Kaggle).
    Reads HF_TOKEN from config for gated model access.
    """
    global _medgemma_model, _medgemma_tokenizer, _load_error
    if _medgemma_model is not None:
        return _medgemma_model, _medgemma_tokenizer

    import torch

    # Check if GPU is available
    if not torch.cuda.is_available():
        _load_error = "No GPU available — running in offline mode"
        raise RuntimeError(_load_error)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Resolve HF token: env > config
    hf_token = config.HF_TOKEN or os.getenv("HF_TOKEN")

    # Try reading from Streamlit secrets if available
    if not hf_token:
        try:
            import streamlit as st
            hf_token = st.secrets.get("HF_TOKEN")
        except Exception:
            pass

    if not hf_token:
        _load_error = "HF_TOKEN not set — cannot access gated model"
        raise RuntimeError(_load_error)

    print(f"Loading MedGemma model: {config.MEDGEMMA_MODEL}")

    # Processor
    _medgemma_tokenizer = AutoTokenizer.from_pretrained(
        config.MEDGEMMA_MODEL,
        trust_remote_code=True,
        token=hf_token,
    )

    # Model — with optional 4-bit quantisation
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": config.MEDGEMMA_DEVICE,
        "token": hf_token,
    }

    if config.USE_4BIT:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("  Using 4-bit NF4 quantisation")
        except ImportError:
            print("  ⚠️  bitsandbytes not available, loading in bfloat16")
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    _medgemma_model = AutoModelForCausalLM.from_pretrained(
        config.MEDGEMMA_MODEL,
        **model_kwargs,
    )
    _medgemma_model.eval()
    print(f"MedGemma loaded on {next(_medgemma_model.parameters()).device}")
    return _medgemma_model, _medgemma_tokenizer


def is_model_available() -> bool:
    """Check if MedGemma is loaded and ready for inference."""
    return _medgemma_model is not None


def get_load_error() -> Optional[str]:
    """Return the error message from the last failed load attempt."""
    return _load_error


def try_load_model() -> bool:
    """Attempt to load MedGemma. Returns True if successful."""
    try:
        _load_medgemma()
        return True
    except Exception as e:
        global _load_error
        _load_error = str(e)
        return False


def generate_medgemma(prompt: str, max_tokens: int = None) -> str:
    """Run MedGemma generation with a text prompt.

    Uses the chat template format via AutoTokenizer for MedGemma.
    """
    import torch

    model, tokenizer = _load_medgemma()
    max_tokens = max_tokens or config.REASONING_MAX_TOKENS

    # Format as chat message
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    # MedGemma uses tokenizer.apply_chat_template which returns
    # tokenized inputs directly
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    generated = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )
    return generated.strip()


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from a model response.

    Handles cases where the model wraps JSON in markdown code fences,
    adds preamble text, or produces truncated JSON.
    """
    if not text or not text.strip():
        return None

    # Strip preamble before first { and trailing text after last }
    cleaned = text.strip()
    brace_idx = cleaned.find("{")
    if brace_idx > 0:
        cleaned = cleaned[brace_idx:]
    rbrace_idx = cleaned.rfind("}")
    if rbrace_idx >= 0 and rbrace_idx < len(cleaned) - 1:
        cleaned = cleaned[:rbrace_idx + 1]

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try fixing truncated JSON
    try:
        open_braces = cleaned.count("{") - cleaned.count("}")
        open_brackets = cleaned.count("[") - cleaned.count("]")
        if open_braces > 0 or open_brackets > 0:
            fixed = cleaned + "]" * open_brackets + "}" * open_braces
            return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None
