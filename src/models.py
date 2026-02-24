"""Model loading and management for MedGemma.

Lazy-loads MedGemma model using the CausalLM API
(AutoModelForCausalLM + AutoTokenizer). Caches it so it is
only loaded once per session regardless of how many calls are made.

MedGemma uses AutoTokenizer (not AutoTokenizer) and
AutoModelForCausalLM (not AutoModelForCausalLM).
"""
import json
import logging
import re
from typing import Dict, Any, Optional

from . import config

logger = logging.getLogger(__name__)

# ── Globals for lazy-loaded model ────────────────────────────
_medgemma_model = None
_medgemma_tokenizer = None


def is_model_available() -> bool:
    """Check if MedGemma model is loaded and available."""
    return _medgemma_model is not None


def _load_medgemma():
    """Lazy-load the MedGemma model and tokenizer.

    Raises RuntimeError if the model cannot be loaded, with a descriptive
    message about the likely cause.
    """
    global _medgemma_model, _medgemma_tokenizer
    if _medgemma_model is not None:
        return _medgemma_model, _medgemma_tokenizer

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"Required packages not installed (transformers, torch): {e}. "
            f"Install with: pip install transformers torch accelerate"
        ) from e

    logger.info("Loading MedGemma model: %s", config.MEDGEMMA_MODEL)
    try:
        _medgemma_tokenizer = AutoTokenizer.from_pretrained(
            config.MEDGEMMA_MODEL,
            trust_remote_code=True,
        )
        _medgemma_model = AutoModelForCausalLM.from_pretrained(
            config.MEDGEMMA_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=config.MEDGEMMA_DEVICE,
        )
        _medgemma_model.eval()
        device = next(_medgemma_model.parameters()).device
        logger.info("MedGemma loaded successfully on %s", device)
    except Exception as e:
        _medgemma_model = None
        _medgemma_tokenizer = None
        raise RuntimeError(
            f"Failed to load MedGemma model '{config.MEDGEMMA_MODEL}': {e}. "
            f"Ensure you have GPU access, sufficient VRAM (~8GB), "
            f"and a valid HuggingFace token for gated model access."
        ) from e

    return _medgemma_model, _medgemma_tokenizer


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

    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Try direct parse first
    try:
        return json.loads(text)
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

    return None
