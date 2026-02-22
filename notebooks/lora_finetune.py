# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     main_language: python
#     notebook_metadata_filter: -all
# ---

# %% [markdown]
# # CHW Copilot — LoRA Fine-Tuning MedGemma for CHW Note Extraction
#
# **Competition:** MedGemma Impact Challenge — Agentic Workflow Prize
#
# This notebook fine-tunes MedGemma 1.5 (4B) using QLoRA on CHW note extraction data.
# The goal: teach MedGemma our exact JSON schema so it produces cleaner extractions
# with fewer parsing failures and better evidence grounding.
#
# **Method:** QLoRA (4-bit quantized LoRA) — trains only ~2M parameters vs 4B total
# **Data:** Gold encounter pairs from synthetic CHW notes + AfriMedQA
# **Runtime:** Kaggle T4 GPU (16GB VRAM) — fits with 4-bit quantization
#
# ---

# %% [markdown]
# ## 0. Setup

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40", "accelerate>=0.27", "peft>=0.10",
    "trl>=0.8", "bitsandbytes>=0.43", "datasets",
    "torch", "jsonschema>=4.17", "pandas>=2.0"])
print("Dependencies installed ✅")

# %%
import os, json, time, warnings
from pathlib import Path
import torch
import pandas as pd
warnings.filterwarnings("ignore")

IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    ROOT = Path("/kaggle/input/chw-copilot")
    OUT_DIR = Path("/kaggle/working")
else:
    ROOT = Path(".")
    OUT_DIR = Path(".")

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 1. Load and Prepare Training Data
#
# Training data format:
# - **Input:** Raw CHW field note text
# - **Output:** Gold-standard structured encounter JSON
#
# We use our merged gold encounters dataset which combines
# synthetic CHW notes + AfriMedQA medical question data.

# %%
# Load gold encounter data
gold_path = ROOT / "data_synth" / "gold_encounters_merged.jsonl"
if not gold_path.exists():
    gold_path = ROOT / "data_synth" / "gold_encounters.jsonl"

records = []
with open(gold_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} gold encounters from {gold_path.name}")

# %%
# Peek at structure
if records:
    sample = records[0]
    print("Keys:", list(sample.keys()))
    print("Note preview:", sample.get("note_text", "")[:200])

# %%
# Load the extraction prompt template
prompt_path = ROOT / "prompts" / "specialist_extraction.txt"
EXTRACTION_PROMPT = prompt_path.read_text(encoding="utf-8")
print(f"Prompt template loaded ({len(EXTRACTION_PROMPT)} chars)")

# %%
# Build training pairs: (prompt, completion)
from datasets import Dataset

def build_training_example(record):
    """Convert a gold encounter record into a training pair."""
    note_text = record.get("note_text", "")
    
    # Build the expected output (the gold encounter JSON)
    # Remove internal fields that aren't part of extraction output
    gold_output = {k: v for k, v in record.items() 
                   if k not in ("note_text", "encounter_id", "location_id", "week_id")}
    
    # Build the full prompt
    prompt = EXTRACTION_PROMPT.replace("{note_text}", note_text)
    completion = json.dumps(gold_output, indent=2)
    
    return {
        "prompt": prompt,
        "completion": completion,
        "note_text": note_text,
    }

train_data = [build_training_example(r) for r in records if r.get("note_text")]
print(f"Built {len(train_data)} training examples")

# Split into train/eval (90/10)
split_idx = int(len(train_data) * 0.9)
train_split = train_data[:split_idx]
eval_split = train_data[split_idx:]
print(f"Train: {len(train_split)}, Eval: {len(eval_split)}")

train_dataset = Dataset.from_list(train_split)
eval_dataset = Dataset.from_list(eval_split)

# %% [markdown]
# ## 2. Load MedGemma with 4-bit Quantization (QLoRA)

# %%
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MEDGEMMA_ID = "google/medgemma-1.5-4b-it"

# HuggingFace authentication
HF_TOKEN = None
if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    try:
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded ✅")
    except Exception:
        HF_TOKEN = os.getenv("HF_TOKEN")
else:
    HF_TOKEN = os.getenv("HF_TOKEN")

# 4-bit quantization config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {MEDGEMMA_ID} with 4-bit quantization...")
t0 = time.time()

processor = AutoProcessor.from_pretrained(
    MEDGEMMA_ID, trust_remote_code=True, token=HF_TOKEN
)
model = AutoModelForImageTextToText.from_pretrained(
    MEDGEMMA_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)

print(f"Model loaded in {time.time()-t0:.1f}s ✅")
print(f"Model dtype: {model.dtype}")
print(f"Trainable params before LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 3. Apply LoRA Adapters

# %%
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Alpha scaling
    target_modules=[           # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# %% [markdown]
# ## 4. Training Setup

# %%
from trl import SFTConfig, SFTTrainer

def formatting_func(example):
    """Format training example as chat messages for MedGemma."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": example["prompt"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": example["completion"]}]},
    ]
    # Use processor's chat template
    text = processor.apply_chat_template(messages, tokenize=False)
    return text

# Training configuration
training_args = SFTConfig(
    output_dir=str(OUT_DIR / "lora_checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=False,
    bf16=torch.cuda.is_available(),
    max_seq_length=2048,
    dataset_text_field=None,  # We use formatting_func
    report_to="none",
)

# %%
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    processing_class=processor,
)

print(f"Trainer initialized ✅")
print(f"Training epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# %% [markdown]
# ## 5. Train!

# %%
print("Starting fine-tuning...")
t0 = time.time()
train_result = trainer.train()
print(f"\nTraining complete in {(time.time()-t0)/60:.1f} minutes ✅")
print(f"Final train loss: {train_result.training_loss:.4f}")

# %%
# Save the LoRA adapter
adapter_path = OUT_DIR / "medgemma_chw_lora"
model.save_pretrained(str(adapter_path))
processor.save_pretrained(str(adapter_path))
print(f"LoRA adapter saved to {adapter_path} ✅")

# %% [markdown]
# ## 6. Evaluate: Before vs After Fine-Tuning

# %%
# Run extraction on eval set with fine-tuned model
from src.models import parse_json_response

def extract_with_model(note_text, model, processor):
    """Run extraction using the given model."""
    prompt = EXTRACTION_PROMPT.replace("{note_text}", note_text)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    generated = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return parse_json_response(generated.strip())

# Evaluate on eval set
print("=== Fine-Tuned Model Evaluation ===")
ft_parse_success = 0
ft_total = len(eval_split)

for ex in eval_split:
    result = extract_with_model(ex["note_text"], model, processor)
    if result is not None:
        ft_parse_success += 1

print(f"Parse success rate: {ft_parse_success}/{ft_total} ({ft_parse_success/max(ft_total,1):.1%})")

# %% [markdown]
# ## 7. Results Summary

# %%
print("=" * 60)
print("🎉 LoRA Fine-Tuning Complete!")
print("=" * 60)
print(f"Base model:           {MEDGEMMA_ID}")
print(f"Adapter:              QLoRA (rank=16, alpha=32)")
print(f"Training examples:    {len(train_split)}")
print(f"Eval examples:        {len(eval_split)}")
print(f"Trainable params:     {trainable:,} ({100*trainable/total:.2f}%)")
print(f"Final train loss:     {train_result.training_loss:.4f}")
print(f"Parse success (FT):   {ft_parse_success}/{ft_total} ({ft_parse_success/max(ft_total,1):.1%})")
print()
print("Adapter saved to:", adapter_path)
print()
print("To use the fine-tuned model:")
print("  from peft import PeftModel")
print(f"  model = PeftModel.from_pretrained(base_model, '{adapter_path}')")
