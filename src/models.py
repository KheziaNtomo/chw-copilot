"""Model loading and management for MedGemma 1.5.

Lazy-loads MedGemma 1.5 model using the Image-Text-to-Text API
(AutoModelForImageTextToText + AutoProcessor). Caches it so it is
only loaded once per session regardless of how many calls are made.

MedGemma 1.5 uses AutoProcessor (not AutoTokenizer) and
AutoModelForImageTextToText (not AutoModelForCausalLM).
"""
import json
import re
from typing import Dict, Any, Optional

from . import config

# ── Globals for lazy-loaded model ────────────────────────────
_medgemma_model = None
_medgemma_processor = None


def _load_medgemma():
    """Lazy-load the MedGemma 1.5 model and processor."""
    global _medgemma_model, _medgemma_processor
    if _medgemma_model is not None:
        return _medgemma_model, _medgemma_processor

    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch

    print(f"Loading MedGemma model: {config.MEDGEMMA_MODEL}")
    _medgemma_processor = AutoProcessor.from_pretrained(
        config.MEDGEMMA_MODEL,
        trust_remote_code=True,
    )
    _medgemma_model = AutoModelForImageTextToText.from_pretrained(
        config.MEDGEMMA_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=config.MEDGEMMA_DEVICE,
    )
    _medgemma_model.eval()
    print(f"MedGemma loaded on {next(_medgemma_model.parameters()).device}")
    return _medgemma_model, _medgemma_processor


def generate_medgemma(prompt: str, max_tokens: int = None) -> str:
    """Run MedGemma generation with a text prompt.

    Uses the chat template format via AutoProcessor for MedGemma 1.5.
    """
    import torch

    model, processor = _load_medgemma()
    max_tokens = max_tokens or config.REASONING_MAX_TOKENS

    # Format as chat message
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    # MedGemma 1.5 uses processor.apply_chat_template which returns
    # tokenized inputs directly
    inputs = processor.apply_chat_template(
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

    generated = processor.decode(
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
