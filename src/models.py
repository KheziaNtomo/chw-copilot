"""Model loading and management.

Lazy-loads NuExtract and MedGemma models. Caches them so they are only
loaded once per session regardless of how many calls are made.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from . import config

# ── Globals for lazy-loaded models ───────────────────────────
_nuextract_model = None
_nuextract_tokenizer = None
_medgemma_model = None
_medgemma_tokenizer = None


def _load_nuextract():
    """Lazy-load the NuExtract model and tokenizer."""
    global _nuextract_model, _nuextract_tokenizer
    if _nuextract_model is not None:
        return _nuextract_model, _nuextract_tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading NuExtract model: {config.NUEXTRACT_MODEL}")
    _nuextract_tokenizer = AutoTokenizer.from_pretrained(
        config.NUEXTRACT_MODEL,
        trust_remote_code=True,
    )
    _nuextract_model = AutoModelForCausalLM.from_pretrained(
        config.NUEXTRACT_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=config.NUEXTRACT_DEVICE,
    )
    _nuextract_model.eval()
    print(f"NuExtract loaded on {next(_nuextract_model.parameters()).device}")
    return _nuextract_model, _nuextract_tokenizer


def _load_medgemma():
    """Lazy-load the MedGemma model and tokenizer."""
    global _medgemma_model, _medgemma_tokenizer
    if _medgemma_model is not None:
        return _medgemma_model, _medgemma_tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading MedGemma model: {config.MEDGEMMA_MODEL}")
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
    print(f"MedGemma loaded on {next(_medgemma_model.parameters()).device}")
    return _medgemma_model, _medgemma_tokenizer


def generate_nuextract(text: str, template: Dict[str, Any], max_tokens: int = None) -> str:
    """Run NuExtract extraction with a JSON template.

    NuExtract expects input formatted as:
        <|input|>\\n{text}\\n<|template|>\\n{template_json}\\n<|output|>
    """
    model, tokenizer = _load_nuextract()
    max_tokens = max_tokens or config.EXTRACTION_MAX_TOKENS

    template_str = json.dumps(template, indent=2)
    prompt = f"<|input|>\n{text}\n<|template|>\n{template_str}\n<|output|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=config.TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def generate_medgemma(prompt: str, max_tokens: int = None) -> str:
    """Run MedGemma generation with a text prompt.

    Uses the chat template format for instruction-tuned models.
    """
    model, tokenizer = _load_medgemma()
    max_tokens = max_tokens or config.REASONING_MAX_TOKENS

    # Format as chat message
    messages = [
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=config.TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
